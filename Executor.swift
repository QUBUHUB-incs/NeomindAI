// LanguageModel conformance
public struct MyLanguageModel: LanguageModel {
    typealias Executor = MyLanguageModelExecutor

    public var capabilities: LanguageModelCapabilities {
        LanguageModelCapabilities(capabilities: [
            .toolCalling, .guidedGeneration, .reasoning
        ])
    }

    public var executorConfiguration: Executor.Configuration {
        Executor.Configuration(/* ... */)
    }
}

// Executor conformance
public struct MyLanguageModelExecutor: LanguageModelExecutor {
    public typealias Model = MyLanguageModel

    public struct Configuration: Hashable, Sendable { /* ... */ }

    public init(configuration: Configuration) throws { /* ... */ }

    public func respond(
        to request: LanguageModelExecutorGenerationRequest,
        model: MyLanguageModel,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws { /* ... */ }
}
