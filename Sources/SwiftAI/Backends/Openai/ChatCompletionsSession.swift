import Foundation
import OpenAI

/// Maintains a full-message-history Chat Completions conversation.
public final actor ChatCompletionsSession: LLMSession {
  private(set) var messages: [Message]

  private let tools: [any SwiftAI.Tool]
  private let client: any ChatCompletionsClient
  private let model: String
  private let configuration: ChatCompletionsProviderConfiguration

  init(
    messages: [Message] = [],
    tools: [any SwiftAI.Tool] = [],
    client: any ChatCompletionsClient,
    model: String,
    configuration: ChatCompletionsProviderConfiguration
  ) {
    self.messages = messages
    self.tools = tools
    self.client = client
    self.model = model
    self.configuration = configuration
  }

  public nonisolated func prewarm(promptPrefix: Prompt?) {}

  func generateResponse<T: Generable>(
    to prompt: Prompt,
    returning type: T.Type,
    options: LLMReplyOptions
  ) async throws -> LLMReply<T> {
    var finalPartial: T.Partial?
    for try await partial in generateResponseStream(
      to: prompt,
      returning: type,
      options: options
    ) {
      finalPartial = partial
    }

    guard let finalPartial else {
      throw LLMError.generalError("No response received from Chat Completions API")
    }

    let content: T
    if T.self == String.self {
      content = unsafeBitCast(finalPartial, to: T.self)
    } else {
      content = try T(from: finalPartial.generableContent)
    }

    return LLMReply(content: content, history: messages)
  }

  nonisolated func generateResponseStream<T: Generable>(
    to prompt: Prompt,
    returning type: T.Type,
    options: LLMReplyOptions
  ) -> AsyncThrowingStream<T.Partial, Error> where T: Sendable {
    AsyncThrowingStream { continuation in
      let task = Task(name: "ChatCompletionsSession.generateResponseStream") {
        do {
          try await self.runGeneration(
            prompt: prompt,
            type: type,
            options: options,
            continuation: continuation
          )
          continuation.finish()
        } catch is CancellationError {
          continuation.finish()
        } catch let error as LLMError {
          continuation.finish(throwing: error)
        } catch {
          continuation.finish(
            throwing: LLMError.generalError("Chat Completions request failed: \(error)"))
        }
      }

      continuation.onTermination = { @Sendable _ in
        task.cancel()
      }
    }
  }

  private func runGeneration<T: Generable>(
    prompt: Prompt,
    type: T.Type,
    options: LLMReplyOptions,
    continuation: AsyncThrowingStream<T.Partial, Error>.Continuation
  ) async throws where T: Sendable {
    messages.append(.user(.init(chunks: prompt.chunks)))

    let structuredOutput =
      try type == String.self
      ? (
        responseFormat: Optional<ChatQuery.ResponseFormat>.none, instruction: Optional<String>.none
      )
      : makeChatStructuredOutput(for: type, strategy: configuration.structuredOutput)

    while true {
      try Task.checkCancellation()

      var requestMessages = try messages.asChatCompletionMessages
      if let instruction = structuredOutput.instruction {
        requestMessages.insert(
          .system(.init(content: .textContent(instruction))),
          at: 0
        )
      }

      var query = ChatQuery(
        messages: requestMessages,
        model: model,
        maxCompletionTokens: configuration.tokenLimitParameter == .maxCompletionTokens
          ? options.maximumTokens : nil,
        responseFormat: structuredOutput.responseFormat,
        temperature: options.temperature.map { $0 * 2 },
        tools: try tools.isEmpty
          ? nil
          : makeChatCompletionTools(tools, strict: configuration.usesStrictToolSchemas),
        topP: topP(from: options.samplingMode),
        stream: true
      )
      if configuration.tokenLimitParameter == .maxTokens {
        query.maxTokens = options.maximumTokens
      }

      var accumulatedText = ""
      var accumulatedToolCalls: [Int: AccumulatedToolCall] = [:]
      var sawChoice = false

      for try await chunk in client.chatsStream(query: query) {
        try Task.checkCancellation()

        for choice in chunk.choices where choice.index == 0 {
          sawChoice = true
          if let delta = choice.delta.content {
            accumulatedText += delta
            if let partial = try makePartial(type: type, accumulatedText: accumulatedText) {
              continuation.yield(partial)
            }
          }

          for toolCall in choice.delta.toolCalls ?? [] {
            guard let index = toolCall.index else {
              throw LLMError.generalError("A streamed tool call is missing its index")
            }
            accumulatedToolCalls[index, default: AccumulatedToolCall()].merge(toolCall)
          }

          switch choice.finishReason {
          case .contentFilter:
            throw LLMError.generalError(
              "Chat Completions response was blocked by content filtering")
          case .error:
            throw LLMError.generalError("Chat Completions provider ended the stream with an error")
          default:
            break
          }
        }
      }

      guard sawChoice else {
        throw LLMError.generalError("Chat Completions returned no choices")
      }

      let toolCalls =
        try accumulatedToolCalls
        .sorted { $0.key < $1.key }
        .map { try $0.value.makeToolCall() }
      guard !accumulatedText.isEmpty || !toolCalls.isEmpty else {
        throw LLMError.generalError("Chat Completions returned an empty response")
      }

      let chunks: [ContentChunk]
      if accumulatedText.isEmpty {
        chunks = []
      } else if type != String.self,
        let content = try? StructuredContent(json: accumulatedText)
      {
        chunks = [.structured(content)]
      } else {
        chunks = [.text(accumulatedText)]
      }
      messages.append(.ai(.init(chunks: chunks, toolCalls: toolCalls)))

      if toolCalls.isEmpty {
        return
      }

      for toolCall in toolCalls {
        messages.append(.toolOutput(try await execute(toolCall: toolCall)))
      }
    }
  }

  private func makePartial<T: Generable>(
    type: T.Type,
    accumulatedText: String
  ) throws -> T.Partial? {
    if T.self == String.self {
      return accumulatedText as? T.Partial
    }

    let repairedJSON = repair(json: accumulatedText)
    let content = try StructuredContent(json: repairedJSON)
    return try? T.Partial(from: content)
  }

  private func execute(toolCall: Message.ToolCall) async throws -> Message.ToolOutput {
    guard let tool = tools.first(where: { $0.name == toolCall.toolName }) else {
      throw LLMError.generalError("Tool '\(toolCall.toolName)' not found")
    }

    do {
      let data = toolCall.arguments.jsonString.data(using: .utf8) ?? Data()
      let result = try await tool.call(data)
      return .init(
        id: toolCall.id,
        toolName: toolCall.toolName,
        chunks: result.chunks
      )
    } catch {
      throw LLMError.toolExecutionFailed(tool: tool, underlyingError: error)
    }
  }

  private func topP(from samplingMode: LLMReplyOptions.SamplingMode?) -> Double? {
    switch samplingMode {
    case .topP(let threshold): threshold
    case .greedy: 0
    case nil: nil
    }
  }
}

private struct AccumulatedToolCall {
  var id: String?
  var name: String?
  var arguments = ""

  mutating func merge(_ delta: ChatStreamResult.Choice.ChoiceDelta.ChoiceDeltaToolCall) {
    if let id = delta.id, !id.isEmpty {
      self.id = id
    }
    if let name = delta.function?.name, !name.isEmpty {
      self.name = name
    }
    if let arguments = delta.function?.arguments {
      self.arguments += arguments
    }
  }

  func makeToolCall() throws -> Message.ToolCall {
    guard let id, !id.isEmpty else {
      throw LLMError.generalError("A streamed tool call is missing its id")
    }
    guard let name, !name.isEmpty else {
      throw LLMError.generalError("A streamed tool call is missing its function name")
    }

    return Message.ToolCall(
      id: id,
      toolName: name,
      arguments: try StructuredContent(json: arguments)
    )
  }
}
