import Foundation
import OpenAI
import SwiftAILLMTesting
import Testing

@testable import SwiftAI

@Suite("Chat Completions offline contract tests")
struct ChatCompletionsContractTests {
  @Test("Provider presets expose the expected compatibility behavior")
  func providerPresets() {
    let openAI = ChatCompletionsProvider.openAI.configuration
    #expect(openAI.host == "api.openai.com")
    #expect(openAI.basePath == "/v1")
    #expect(openAI.apiKeyEnvironmentVariable == "OPENAI_API_KEY")
    #expect(openAI.responseParsing == .strict)
    #expect(openAI.tokenLimitParameter == .maxCompletionTokens)
    #expect(openAI.structuredOutput == .jsonSchema(strict: true))
    #expect(openAI.usesStrictToolSchemas)

    let deepSeek = ChatCompletionsProvider.deepSeek.configuration
    #expect(deepSeek.host == "api.deepseek.com")
    #expect(deepSeek.basePath.isEmpty)
    #expect(deepSeek.apiKeyEnvironmentVariable == "DEEPSEEK_API_KEY")
    #expect(deepSeek.responseParsing == .relaxed)
    #expect(deepSeek.tokenLimitParameter == .maxTokens)
    #expect(deepSeek.structuredOutput == .jsonObject)
    #expect(!deepSeek.usesStrictToolSchemas)
  }

  @Test("Custom provider preserves every capability")
  func customProvider() {
    let configuration = ChatCompletionsProviderConfiguration(
      scheme: "http",
      host: "localhost",
      port: 8080,
      basePath: "/compatible/v1",
      apiKeyEnvironmentVariable: "LOCAL_API_KEY",
      responseParsing: .strict,
      tokenLimitParameter: .maxCompletionTokens,
      structuredOutput: .promptOnly,
      usesStrictToolSchemas: true
    )

    #expect(ChatCompletionsProvider.custom(configuration).configuration == configuration)
  }

  @Test("OpenAI query uses strict JSON Schema and max_completion_tokens")
  func openAIQueryContract() async throws {
    let client = MockChatCompletionsClient(responses: [
      try [.text("{\"value\":\"ok\"}")]
    ])
    let llm = ChatCompletionsLLM(model: "test-model", provider: .openAI, client: client)

    let reply: LLMReply<ContractResponse> = try await llm.reply(
      to: "Return a value",
      returning: ContractResponse.self,
      tools: [FakeTool()],
      options: LLMReplyOptions(maximumTokens: 42)
    )

    #expect(reply.content.value == "ok")
    let body = try encodedJSONObject(try #require(client.queries.first))
    #expect(body["max_completion_tokens"] as? Int == 42)
    #expect(body["max_tokens"] == nil)

    let responseFormat = try #require(body["response_format"] as? [String: Any])
    #expect(responseFormat["type"] as? String == "json_schema")
    let jsonSchema = try #require(responseFormat["json_schema"] as? [String: Any])
    #expect(jsonSchema["strict"] as? Bool == true)

    let tools = try #require(body["tools"] as? [[String: Any]])
    let function = try #require(tools.first?["function"] as? [String: Any])
    #expect(function["strict"] as? Bool == true)
  }

  @Test("DeepSeek query uses JSON Object, schema instructions, and max_tokens")
  func deepSeekQueryContract() async throws {
    let client = MockChatCompletionsClient(responses: [
      try [.text("{\"value\":\"ok\"}")]
    ])
    let llm = ChatCompletionsLLM(model: "test-model", provider: .deepSeek, client: client)

    let reply: LLMReply<ContractResponse> = try await llm.reply(
      to: "Return a value",
      returning: ContractResponse.self,
      tools: [FakeTool()],
      options: LLMReplyOptions(maximumTokens: 42)
    )

    #expect(reply.content.value == "ok")
    let body = try encodedJSONObject(try #require(client.queries.first))
    #expect(body["max_tokens"] as? Int == 42)
    #expect(body["max_completion_tokens"] == nil)

    let responseFormat = try #require(body["response_format"] as? [String: Any])
    #expect(responseFormat["type"] as? String == "json_object")
    let messages = try #require(body["messages"] as? [[String: Any]])
    #expect(messages.first?["role"] as? String == "system")
    let instruction = try #require(messages.first?["content"] as? String)
    #expect(instruction.contains("json"))
    #expect(instruction.contains("JSON Schema"))

    let tools = try #require(body["tools"] as? [[String: Any]])
    let function = try #require(tools.first?["function"] as? [String: Any])
    #expect(function["strict"] == nil)
  }

  @Test("Streaming emits progressive text snapshots")
  func streamingText() async throws {
    let client = MockChatCompletionsClient(responses: [
      try [.text("Hel", finished: false), .text("lo")]
    ])
    let llm = ChatCompletionsLLM(model: "test-model", provider: .openAI, client: client)

    var partials: [String] = []
    for try await partial in llm.replyStream(to: "Say hello") {
      partials.append(partial)
    }

    #expect(partials == ["Hel", "Hello"])
  }

  @Test("Fragmented tool calls execute and preserve full history")
  func fragmentedToolCallLoop() async throws {
    let client = MockChatCompletionsClient(responses: [
      try [
        .toolCall(
          id: "call_1",
          name: "fake_tool",
          arguments: "{\"input\":",
          finished: false
        ),
        .toolCall(name: "", arguments: "\"weather\"}", finished: true),
      ],
      try [.text("Tool output received")],
    ])
    let llm = ChatCompletionsLLM(model: "test-model", provider: .deepSeek, client: client)

    let reply = try await llm.reply(
      to: "Use the tool",
      tools: [FakeTool()]
    )

    #expect(reply.content == "Tool output received")
    #expect(reply.history.map(\.role) == [.user, .ai, .toolOutput, .ai])
    #expect(reply.history[1].role == .ai)
    #expect(reply.history[2].text == "Tool output: weather")
    #expect(client.queries.count == 2)

    let secondBody = try encodedJSONObject(client.queries[1])
    let messages = try #require(secondBody["messages"] as? [[String: Any]])
    #expect(messages.contains { $0["role"] as? String == "assistant" })
    #expect(messages.contains { $0["role"] as? String == "tool" })
  }

  @Test("No choices is reported as an LLM error")
  func noChoices() async {
    let client = MockChatCompletionsClient(responses: [
      [try! .emptyChoices()]
    ])
    let llm = ChatCompletionsLLM(model: "test-model", provider: .openAI, client: client)

    await #expect(throws: LLMError.self) {
      _ = try await llm.reply(to: "Hello")
    }
  }

  @Test("Malformed and unknown tool calls are reported as LLM errors")
  func invalidToolCalls() async {
    let malformedClient = MockChatCompletionsClient(responses: [
      [
        try! .toolCall(
          id: "call_1",
          name: "fake_tool",
          arguments: "not-json",
          finished: true
        )
      ]
    ])
    let malformedLLM = ChatCompletionsLLM(
      model: "test-model",
      provider: .deepSeek,
      client: malformedClient
    )
    await #expect(throws: LLMError.self) {
      _ = try await malformedLLM.reply(to: "Use a tool", tools: [FakeTool()])
    }

    let unknownClient = MockChatCompletionsClient(responses: [
      [
        try! .toolCall(
          id: "call_2",
          name: "missing_tool",
          arguments: "{}",
          finished: true
        )
      ]
    ])
    let unknownLLM = ChatCompletionsLLM(
      model: "test-model",
      provider: .deepSeek,
      client: unknownClient
    )
    await #expect(throws: LLMError.self) {
      _ = try await unknownLLM.reply(to: "Use a tool", tools: [FakeTool()])
    }
  }

  @Test("Client and content-filter failures are reported as LLM errors")
  func providerFailures() async {
    let failingClient = MockChatCompletionsClient(
      responses: [],
      error: MockClientError.requestFailed
    )
    let failingLLM = ChatCompletionsLLM(
      model: "test-model",
      provider: .openAI,
      client: failingClient
    )
    await #expect(throws: LLMError.self) {
      _ = try await failingLLM.reply(to: "Hello")
    }

    let filteredClient = MockChatCompletionsClient(responses: [
      [try! .contentFiltered()]
    ])
    let filteredLLM = ChatCompletionsLLM(
      model: "test-model",
      provider: .openAI,
      client: filteredClient
    )
    await #expect(throws: LLMError.self) {
      _ = try await filteredLLM.reply(to: "Hello")
    }
  }
}

@Suite("Chat Completions OpenAI integration tests")
struct ChatCompletionsOpenAIIntegrationTests {
  @Test("OpenAI text and streaming", .enabled(if: environmentHas("OPENAI_API_KEY")))
  func textAndStreaming() async throws {
    let llm = ChatCompletionsLLM(model: "gpt-4.1-mini", provider: .openAI)
    let reply = try await llm.reply(to: "Reply with exactly: hello")
    #expect(reply.content.lowercased().contains("hello"))

    var final = ""
    for try await partial in llm.replyStream(to: "Reply with exactly: streamed") {
      final = partial
    }
    #expect(final.lowercased().contains("streamed"))
  }

  @Test("OpenAI strict structured output", .enabled(if: environmentHas("OPENAI_API_KEY")))
  func structuredOutput() async throws {
    let llm = ChatCompletionsLLM(model: "gpt-4.1-mini", provider: .openAI)
    let reply: LLMReply<ContractResponse> = try await llm.reply(
      to: "Return the value openai",
      returning: ContractResponse.self
    )
    #expect(reply.content.value.lowercased().contains("openai"))
  }
}

@Suite("Chat Completions DeepSeek integration tests")
struct ChatCompletionsDeepSeekIntegrationTests {
  private var llm: ChatCompletionsLLM {
    ChatCompletionsLLM(model: "deepseek-v4-flash", provider: .deepSeek, timeoutInterval: 30)
  }

  @Test("DeepSeek text and streaming", .enabled(if: environmentHas("DEEPSEEK_API_KEY")))
  func textAndStreaming() async throws {
    let reply = try await llm.reply(to: "Reply with exactly: hello")
    #expect(reply.content.lowercased().contains("hello"))

    var final = ""
    for try await partial in llm.replyStream(to: "Reply with exactly: streamed") {
      final = partial
    }
    #expect(final.lowercased().contains("streamed"))
  }

  @Test("DeepSeek JSON Object becomes Generable", .enabled(if: environmentHas("DEEPSEEK_API_KEY")))
  func structuredOutput() async throws {
    let reply: LLMReply<ContractResponse> = try await llm.reply(
      to: "Return the value deepseek",
      returning: ContractResponse.self
    )
    #expect(reply.content.value.lowercased().contains("deepseek"))
  }

  @Test("DeepSeek tool call loop", .enabled(if: environmentHas("DEEPSEEK_API_KEY")))
  func toolCall() async throws {
    let reply = try await llm.reply(
      to: "You must call fake_tool with input deepseek, then return its output verbatim.",
      tools: [FakeTool()]
    )
    #expect(reply.content.lowercased().contains("deepseek"))
    #expect(reply.history.contains { $0.role == .toolOutput })
  }

  @Test("DeepSeek session maintains context", .enabled(if: environmentHas("DEEPSEEK_API_KEY")))
  func sessionContext() async throws {
    let session = llm.makeSession()
    _ = try await llm.reply(
      to: "Remember that my verification word is sapphire. Reply briefly.",
      in: session
    )
    let reply = try await llm.reply(
      to: "What is my verification word?",
      in: session
    )

    #expect(reply.content.lowercased().contains("sapphire"))
    #expect(reply.history.count == 4)
  }
}

@Generable
private struct ContractResponse: Equatable {
  let value: String
}

private final class MockChatCompletionsClient: ChatCompletionsClient, @unchecked Sendable {
  private let lock = NSLock()
  private var queuedResponses: [[ChatStreamResult]]
  private var recordedQueries: [ChatQuery] = []
  private let error: (any Error)?

  init(responses: [[ChatStreamResult]], error: (any Error)? = nil) {
    self.queuedResponses = responses
    self.error = error
  }

  var queries: [ChatQuery] {
    lock.withLock { recordedQueries }
  }

  func chatsStream(query: ChatQuery) -> AsyncThrowingStream<ChatStreamResult, Error> {
    let response: [ChatStreamResult] = lock.withLock {
      recordedQueries.append(query)
      guard !queuedResponses.isEmpty else { return [] }
      return queuedResponses.removeFirst()
    }

    return AsyncThrowingStream { continuation in
      if let error {
        continuation.finish(throwing: error)
        return
      }
      for chunk in response {
        continuation.yield(chunk)
      }
      continuation.finish()
    }
  }
}

extension ChatStreamResult {
  fileprivate static func text(_ text: String, finished: Bool = true) throws -> ChatStreamResult {
    try decodeFixture(
      delta: ["content": text],
      finishReason: finished ? "stop" : nil
    )
  }

  fileprivate static func toolCall(
    id: String? = nil,
    name: String? = nil,
    arguments: String,
    finished: Bool
  ) throws -> ChatStreamResult {
    var function: [String: Any] = ["arguments": arguments]
    if let name {
      function["name"] = name
    }
    var toolCall: [String: Any] = ["index": 0, "function": function]
    if let id {
      toolCall["id"] = id
      toolCall["type"] = "function"
    }
    return try decodeFixture(
      delta: ["tool_calls": [toolCall]],
      finishReason: finished ? "tool_calls" : nil
    )
  }

  fileprivate static func emptyChoices() throws -> ChatStreamResult {
    try decodeFixture(choices: [])
  }

  fileprivate static func contentFiltered() throws -> ChatStreamResult {
    try decodeFixture(finishReason: "content_filter")
  }

  private static func decodeFixture(
    delta: [String: Any] = [:],
    finishReason: String? = nil,
    choices: [[String: Any]]? = nil
  ) throws -> ChatStreamResult {
    let resolvedChoices =
      choices ?? [
        [
          "index": 0,
          "delta": delta,
          "finish_reason": finishReason as Any,
          "logprobs": NSNull(),
        ]
      ]
    let json: [String: Any] = [
      "id": "chatcmpl-test",
      "object": "chat.completion.chunk",
      "created": 1,
      "model": "test-model",
      "choices": resolvedChoices,
      "system_fingerprint": NSNull(),
    ]
    let data = try JSONSerialization.data(withJSONObject: json)
    return try JSONDecoder().decode(ChatStreamResult.self, from: data)
  }
}

private func encodedJSONObject(_ query: ChatQuery) throws -> [String: Any] {
  let data = try JSONEncoder().encode(query)
  return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
}

private func environmentHas(_ name: String) -> Bool {
  ProcessInfo.processInfo.environment[name]?.isEmpty == false
}

private enum MockClientError: Error {
  case requestFailed
}
