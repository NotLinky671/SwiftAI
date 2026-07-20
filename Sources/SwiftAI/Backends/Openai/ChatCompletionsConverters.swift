import Foundation
import OpenAI

extension Message {
  var asChatCompletionMessage: ChatQuery.ChatCompletionMessageParam {
    get throws {
      switch self {
      case .system(let message):
        return .system(.init(content: .textContent(message.text)))
      case .user(let message):
        return .user(.init(content: .string(message.text)))
      case .ai(let message):
        let toolCalls = message.toolCalls.map { toolCall in
          ChatQuery.ChatCompletionMessageParam.AssistantMessageParam.ToolCallParam(
            id: toolCall.id,
            function: .init(
              arguments: toolCall.arguments.jsonString,
              name: toolCall.toolName
            )
          )
        }
        return .assistant(
          .init(
            content: message.chunks.isEmpty ? nil : .textContent(message.text),
            toolCalls: toolCalls.isEmpty ? nil : toolCalls
          ))
      case .toolOutput(let message):
        return .tool(
          .init(
            content: .textContent(message.text),
            toolCallId: message.id
          ))
      }
    }
  }
}

extension Array where Element == Message {
  var asChatCompletionMessages: [ChatQuery.ChatCompletionMessageParam] {
    get throws {
      try map { try $0.asChatCompletionMessage }
    }
  }
}

func makeChatCompletionTools(
  _ tools: [any SwiftAI.Tool],
  strict: Bool
) throws -> [ChatQuery.ChatCompletionToolParam] {
  try tools.map { tool in
    ChatQuery.ChatCompletionToolParam(
      function: .init(
        name: tool.name,
        description: tool.description,
        parameters: try convertRootSchemaToOpenaiSupportedJsonSchema(type(of: tool).parameters),
        strict: strict ? true : nil
      ))
  }
}

func makeChatStructuredOutput<T: Generable>(
  for type: T.Type,
  strategy: ChatCompletionsProviderConfiguration.StructuredOutput
) throws -> (responseFormat: ChatQuery.ResponseFormat?, instruction: String?) {
  let schema = try convertRootSchemaToOpenaiSupportedJsonSchema(type.schema)

  switch strategy {
  case .jsonSchema(let strict):
    let configuration = ChatQuery.StructuredOutputConfigurationOptions(
      name: String(describing: type),
      description: nil,
      schema: .jsonSchema(schema),
      strict: strict
    )
    return (.jsonSchema(configuration), nil)
  case .jsonObject:
    return (
      .jsonObject,
      try structuredOutputInstruction(type: type, schema: schema)
    )
  case .promptOnly:
    return (
      nil,
      try structuredOutputInstruction(type: type, schema: schema)
    )
  }
}

private func structuredOutputInstruction<T: Generable>(
  type: T.Type,
  schema: JSONSchema
) throws -> String {
  let data = try JSONEncoder().encode(schema)
  guard let schemaJSON = String(data: data, encoding: .utf8) else {
    throw LLMError.generalError("Unable to encode the JSON schema")
  }

  return """
    Return only valid json matching the following JSON Schema for \(String(describing: type)).
    Do not include Markdown fences, commentary, or properties not defined by the schema.
    JSON Schema: \(schemaJSON)
    """
}
