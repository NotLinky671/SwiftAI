import Foundation

/// Describes the provider-specific differences between OpenAI-compatible Chat Completions APIs.
public struct ChatCompletionsProviderConfiguration: Sendable, Equatable {
  /// Controls how required fields in provider responses are decoded.
  public enum ResponseParsing: Sendable, Equatable {
    case strict
    case relaxed
  }

  /// Selects the request field used for the output-token limit.
  public enum TokenLimitParameter: Sendable, Equatable {
    case maxCompletionTokens
    case maxTokens
  }

  /// Selects how structured output is requested from the provider.
  public enum StructuredOutput: Sendable, Equatable {
    /// Sends an OpenAI JSON Schema response format.
    case jsonSchema(strict: Bool)
    /// Sends JSON Object mode and adds the schema to the system instructions.
    case jsonObject
    /// Adds the schema to the system instructions without sending a response format.
    case promptOnly
  }

  public let scheme: String
  public let host: String
  public let port: Int
  public let basePath: String
  public let apiKeyEnvironmentVariable: String?
  public let responseParsing: ResponseParsing
  public let tokenLimitParameter: TokenLimitParameter
  public let structuredOutput: StructuredOutput
  public let usesStrictToolSchemas: Bool

  public init(
    scheme: String = "https",
    host: String,
    port: Int = 443,
    basePath: String = "/v1",
    apiKeyEnvironmentVariable: String? = nil,
    responseParsing: ResponseParsing = .relaxed,
    tokenLimitParameter: TokenLimitParameter = .maxTokens,
    structuredOutput: StructuredOutput = .jsonObject,
    usesStrictToolSchemas: Bool = false
  ) {
    self.scheme = scheme
    self.host = host
    self.port = port
    self.basePath = basePath
    self.apiKeyEnvironmentVariable = apiKeyEnvironmentVariable
    self.responseParsing = responseParsing
    self.tokenLimitParameter = tokenLimitParameter
    self.structuredOutput = structuredOutput
    self.usesStrictToolSchemas = usesStrictToolSchemas
  }
}

/// A Chat Completions provider preset or a custom OpenAI-compatible endpoint.
public enum ChatCompletionsProvider: Sendable, Equatable {
  case openAI
  case deepSeek
  case custom(ChatCompletionsProviderConfiguration)

  public var configuration: ChatCompletionsProviderConfiguration {
    switch self {
    case .openAI:
      return ChatCompletionsProviderConfiguration(
        host: "api.openai.com",
        basePath: "/v1",
        apiKeyEnvironmentVariable: "OPENAI_API_KEY",
        responseParsing: .strict,
        tokenLimitParameter: .maxCompletionTokens,
        structuredOutput: .jsonSchema(strict: true),
        usesStrictToolSchemas: true
      )
    case .deepSeek:
      return ChatCompletionsProviderConfiguration(
        host: "api.deepseek.com",
        basePath: "",
        apiKeyEnvironmentVariable: "DEEPSEEK_API_KEY",
        responseParsing: .relaxed,
        tokenLimitParameter: .maxTokens,
        structuredOutput: .jsonObject,
        usesStrictToolSchemas: false
      )
    case .custom(let configuration):
      return configuration
    }
  }
}
