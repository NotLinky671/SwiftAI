import Foundation
import OpenAI

/// A provider-neutral LLM backend for OpenAI-compatible Chat Completions APIs.
public struct ChatCompletionsLLM: LLM {
  public typealias Session = ChatCompletionsSession

  public let model: String
  public let provider: ChatCompletionsProvider

  private let client: any ChatCompletionsClient

  /// Creates a Chat Completions backend.
  ///
  /// When `apiToken` is nil, the provider's configured environment variable is used.
  public init(
    apiToken: String? = nil,
    model: String,
    provider: ChatCompletionsProvider = .openAI,
    organizationIdentifier: String? = nil,
    customHeaders: [String: String] = [:],
    timeoutInterval: TimeInterval = 60
  ) {
    let providerConfiguration = provider.configuration
    let token =
      apiToken
      ?? providerConfiguration.apiKeyEnvironmentVariable.flatMap {
        ProcessInfo.processInfo.environment[$0]
      }
    let parsingOptions: ParsingOptions =
      providerConfiguration.responseParsing == .relaxed ? .relaxed : []
    let configuration = OpenAI.Configuration(
      token: token,
      organizationIdentifier: organizationIdentifier,
      host: providerConfiguration.host,
      port: providerConfiguration.port,
      scheme: providerConfiguration.scheme,
      basePath: providerConfiguration.basePath,
      timeoutInterval: timeoutInterval,
      customHeaders: customHeaders,
      parsingOptions: parsingOptions
    )

    self.client = OpenAI(configuration: configuration)
    self.model = model
    self.provider = provider
  }

  init(
    model: String,
    provider: ChatCompletionsProvider,
    client: any ChatCompletionsClient
  ) {
    self.model = model
    self.provider = provider
    self.client = client
  }

  public var isAvailable: Bool { true }

  public var availability: LLMAvailability { .available }

  public func makeSession(
    tools: [any Tool],
    messages: [Message]
  ) -> ChatCompletionsSession {
    ChatCompletionsSession(
      messages: messages,
      tools: tools,
      client: client,
      model: model,
      configuration: provider.configuration
    )
  }

  public func reply<T: Generable>(
    to messages: [Message],
    returning type: T.Type,
    tools: [any Tool],
    options: LLMReplyOptions
  ) async throws -> LLMReply<T> {
    guard let lastMessage = messages.last, lastMessage.role == .user else {
      throw LLMError.generalError("Conversation must end with a user message")
    }

    let session = makeSession(tools: tools, messages: Array(messages.dropLast()))
    return try await reply(
      to: Prompt(chunks: lastMessage.chunks),
      returning: type,
      in: session,
      options: options
    )
  }

  public func reply<T: Generable>(
    to prompt: Prompt,
    returning type: T.Type,
    in session: ChatCompletionsSession,
    options: LLMReplyOptions
  ) async throws -> LLMReply<T> {
    try await session.generateResponse(to: prompt, returning: type, options: options)
  }

  public func replyStream<T: Generable>(
    to messages: [Message],
    returning type: T.Type,
    tools: [any Tool],
    options: LLMReplyOptions
  ) -> AsyncThrowingStream<T.Partial, Error> where T: Sendable {
    guard let lastMessage = messages.last, lastMessage.role == .user else {
      return AsyncThrowingStream { continuation in
        continuation.finish(
          throwing: LLMError.generalError("Conversation must end with a user message"))
      }
    }

    let session = makeSession(tools: tools, messages: Array(messages.dropLast()))
    return replyStream(
      to: Prompt(chunks: lastMessage.chunks),
      returning: type,
      in: session,
      options: options
    )
  }

  public func replyStream<T: Generable>(
    to prompt: Prompt,
    returning type: T.Type,
    in session: ChatCompletionsSession,
    options: LLMReplyOptions
  ) -> AsyncThrowingStream<T.Partial, Error> where T: Sendable {
    session.generateResponseStream(to: prompt, returning: type, options: options)
  }
}

protocol ChatCompletionsClient: Sendable {
  func chatsStream(query: ChatQuery) -> AsyncThrowingStream<ChatStreamResult, Error>
}

extension OpenAI: ChatCompletionsClient {}
