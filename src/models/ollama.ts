/**
 * Ollama model provider implementation.
 *
 * This module provides integration with Ollama's local inference server,
 * supporting streaming responses, tool use, and configurable model parameters.
 *
 * @see https://ollama.com/
 */

import { Ollama } from 'ollama'
import type {
  ChatRequest,
  ChatResponse,
  Config,
  Message as OllamaMessage,
  Options as OllamaOptions,
  Tool as OllamaTool,
} from 'ollama'
import { Model } from './model.js'
import type { BaseModelConfig, StreamOptions } from './model.js'
import type { ContentBlock, Message, StopReason, ToolResultBlock } from '../types/messages.js'
import type { ImageBlock } from '../types/media.js'
import { encodeBase64 } from '../types/media.js'
import type { ModelStreamEvent } from './streaming.js'
import { ContextWindowOverflowError } from '../errors.js'
import { logger } from '../logging/logger.js'

const DEFAULT_OLLAMA_MODEL_ID = 'llama3.2'

/**
 * Error message patterns that indicate context window overflow.
 * Used to detect when input exceeds the model's context window.
 */
const OLLAMA_CONTEXT_WINDOW_OVERFLOW_PATTERNS = ['context length', 'context window']

/**
 * Configuration interface for Ollama model provider.
 *
 * Extends BaseModelConfig with Ollama-specific configuration options
 * for model parameters and request settings.
 *
 * @example
 * ```typescript
 * const config: OllamaModelConfig = {
 *   modelId: 'llama3.2',
 *   temperature: 0.7,
 *   maxTokens: 1024
 * }
 * ```
 */
export interface OllamaModelConfig extends BaseModelConfig {
  /**
   * Ollama model identifier (e.g., llama3.2, mistral, qwen2.5).
   */
  modelId?: string

  /**
   * Controls randomness in generation.
   * Higher values produce more random output.
   */
  temperature?: number

  /**
   * Maximum number of tokens to generate in the response.
   * Maps to Ollama's `num_predict` option.
   */
  maxTokens?: number

  /**
   * Controls diversity via nucleus sampling.
   */
  topP?: number

  /**
   * List of sequences that will stop generation when encountered.
   */
  stopSequences?: string[]

  /**
   * Controls how long the model will stay loaded into memory following the request.
   * Examples: "5m", "30s", "1h", "0" (unload immediately), "-1" (keep loaded forever).
   */
  keepAlive?: string

  /**
   * Additional model parameters passed through to Ollama's options object.
   * Use this for knobs like top_k, repeat_penalty, num_ctx, etc.
   *
   * @example
   * ```typescript
   * { options: { top_k: 40, repeat_penalty: 1.1 } }
   * ```
   */
  options?: Record<string, unknown>

  /**
   * Additional parameters to pass through at the top level of the Ollama API request.
   * This field provides forward compatibility for any new parameters that Ollama introduces.
   *
   * @example
   * ```typescript
   * { params: { think: true } }
   * ```
   */
  params?: Record<string, unknown>
}

/**
 * Options interface for creating an OllamaModel instance.
 */
export interface OllamaModelOptions extends OllamaModelConfig {
  /**
   * The address of the Ollama server.
   * Defaults to http://127.0.0.1:11434 if not specified.
   */
  host?: string

  /**
   * Pre-configured Ollama client instance.
   * If provided, this client will be used instead of creating a new one.
   */
  client?: Ollama

  /**
   * Additional Ollama client configuration (fetch, proxy, headers).
   * Only used if client is not provided.
   */
  clientConfig?: Partial<Config>
}

/**
 * Ollama model provider implementation.
 *
 * Implements the Model interface for Ollama's local inference server.
 * Supports streaming responses, tool use, and comprehensive configuration.
 *
 * @example
 * ```typescript
 * import { Agent } from '@strands-agents/sdk'
 * import { OllamaModel } from '@strands-agents/sdk/models/ollama'
 *
 * const model = new OllamaModel({
 *   host: 'http://localhost:11434',
 *   modelId: 'llama3.2',
 *   temperature: 0.7,
 *   maxTokens: 1024
 * })
 *
 * const agent = new Agent({ model })
 * const result = await agent.invoke('Hello!')
 * ```
 */
export class OllamaModel extends Model<OllamaModelConfig> {
  private _config: OllamaModelConfig
  private _client: Ollama

  /**
   * Creates a new OllamaModel instance.
   *
   * @param options - Configuration for model and client
   *
   * @example
   * ```typescript
   * // Minimal configuration (uses default host and model)
   * const model = new OllamaModel()
   *
   * // With custom host and model
   * const model = new OllamaModel({
   *   host: 'http://localhost:11434',
   *   modelId: 'llama3.2',
   *   temperature: 0.7
   * })
   *
   * // With a pre-configured client
   * import { Ollama } from 'ollama'
   * const client = new Ollama({ host: 'http://remote:11434' })
   * const model = new OllamaModel({ client, modelId: 'mistral' })
   * ```
   */
  constructor(options?: OllamaModelOptions) {
    super()
    const { host, client, clientConfig, ...modelConfig } = options ?? {}

    this._config = modelConfig

    if (client) {
      this._client = client
    } else {
      this._client = new Ollama({
        ...(host ? { host } : {}),
        ...clientConfig,
      })
    }
  }

  /**
   * Updates the model configuration.
   * Merges the provided configuration with existing settings.
   *
   * @param modelConfig - Configuration object with model-specific settings to update
   *
   * @example
   * ```typescript
   * model.updateConfig({ temperature: 0.9, maxTokens: 2048 })
   * ```
   */
  updateConfig(modelConfig: OllamaModelConfig): void {
    this._config = { ...this._config, ...modelConfig }
  }

  /**
   * Retrieves the current model configuration.
   *
   * @returns The current configuration object
   *
   * @example
   * ```typescript
   * const config = model.getConfig()
   * console.log(config.modelId)
   * ```
   */
  getConfig(): OllamaModelConfig {
    return this._config
  }

  /**
   * Streams a conversation with the Ollama model.
   * Returns an async iterable that yields streaming events as they occur.
   *
   * @param messages - Array of conversation messages
   * @param options - Optional streaming configuration
   * @returns Async iterable of streaming events
   *
   * @example
   * ```typescript
   * const model = new OllamaModel({ modelId: 'llama3.2' })
   * const messages: Message[] = [
   *   { role: 'user', content: [{ type: 'textBlock', text: 'Hello!' }] }
   * ]
   *
   * for await (const event of model.stream(messages)) {
   *   if (event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta') {
   *     process.stdout.write(event.delta.text)
   *   }
   * }
   * ```
   */
  async *stream(messages: Message[], options?: StreamOptions): AsyncIterable<ModelStreamEvent> {
    if (!messages || messages.length === 0) {
      throw new Error('At least one message is required')
    }

    try {
      const request = this._formatRequest(messages, options)
      const response = await this._client.chat({ ...request, stream: true })

      let toolRequested = false
      let textBlockStarted = false
      let lastChunk: ChatResponse | undefined

      yield { type: 'modelMessageStartEvent', role: 'assistant' as const }

      for await (const chunk of response) {
        lastChunk = chunk

        // Process text before tool calls so that text preceding a tool call
        // stays in the same block rather than being split across blocks.
        if (chunk.message.content && chunk.message.content.length > 0) {
          if (!textBlockStarted) {
            yield { type: 'modelContentBlockStartEvent' }
            textBlockStarted = true
          }
          yield {
            type: 'modelContentBlockDeltaEvent',
            delta: {
              type: 'textDelta' as const,
              text: chunk.message.content,
            },
          }
        }

        // Ollama sends complete tool calls per chunk, not incremental deltas.
        if (chunk.message.tool_calls) {
          // Close any open text block before emitting tool blocks
          if (textBlockStarted) {
            yield { type: 'modelContentBlockStopEvent' }
            textBlockStarted = false
          }

          for (const toolCall of chunk.message.tool_calls) {
            const toolName = toolCall.function.name
            const toolUseId = (toolCall as unknown as { id?: string }).id ?? `tooluse_${globalThis.crypto.randomUUID()}`

            yield {
              type: 'modelContentBlockStartEvent',
              start: {
                type: 'toolUseStart' as const,
                name: toolName,
                toolUseId,
              },
            }
            yield {
              type: 'modelContentBlockDeltaEvent',
              delta: {
                type: 'toolUseInputDelta' as const,
                input: JSON.stringify(toolCall.function.arguments),
              },
            }
            yield { type: 'modelContentBlockStopEvent' }
            toolRequested = true
          }
        }
      }

      // Close text block if still open
      if (textBlockStarted) {
        yield { type: 'modelContentBlockStopEvent' }
      }

      // Emit metadata before message stop, following SDK convention
      if (lastChunk) {
        const inputTokens = lastChunk.prompt_eval_count ?? 0
        const outputTokens = lastChunk.eval_count ?? 0
        yield {
          type: 'modelMetadataEvent',
          usage: {
            inputTokens,
            outputTokens,
            totalTokens: inputTokens + outputTokens,
          },
          metrics: {
            latencyMs: lastChunk.total_duration ? lastChunk.total_duration / 1e6 : 0,
          },
        }
      }

      yield {
        type: 'modelMessageStopEvent',
        stopReason: this._mapStopReason(lastChunk?.done_reason, toolRequested),
      }
    } catch (error) {
      const err = error as Error

      if (OLLAMA_CONTEXT_WINDOW_OVERFLOW_PATTERNS.some((pattern) => err.message?.toLowerCase().includes(pattern))) {
        throw new ContextWindowOverflowError(err.message)
      }

      throw error
    }
  }

  /**
   * Formats a request for the Ollama chat API.
   *
   * @param messages - Conversation messages
   * @param options - Stream options
   * @returns Formatted Ollama chat request (without stream field)
   */
  private _formatRequest(messages: Message[], options?: StreamOptions): Omit<ChatRequest, 'stream'> {
    // Build computed options, only including defined values
    const computedOptions: Record<string, unknown> = {}
    if (this._config.maxTokens !== undefined) computedOptions.num_predict = this._config.maxTokens
    if (this._config.temperature !== undefined) computedOptions.temperature = this._config.temperature
    if (this._config.topP !== undefined) computedOptions.top_p = this._config.topP
    if (this._config.stopSequences !== undefined) computedOptions.stop = this._config.stopSequences

    const tools = this._formatTools(options?.toolSpecs)

    const request: Omit<ChatRequest, 'stream'> = {
      model: this._config.modelId ?? DEFAULT_OLLAMA_MODEL_ID,
      messages: this._formatMessages(messages, options?.systemPrompt),
      ...(tools.length > 0 ? { tools } : {}),
      options: {
        ...(this._config.options ?? {}),
        ...computedOptions,
      } as Partial<OllamaOptions>,
    }

    if (this._config.keepAlive !== undefined) {
      request.keep_alive = this._config.keepAlive
    }

    if (options?.toolChoice) {
      logger.warn('tool_choice is not supported by ollama, ignoring')
    }

    // Spread params for forward compatibility
    if (this._config.params) {
      Object.assign(request, this._config.params)
    }

    return request
  }

  /**
   * Formats tool specifications for the Ollama API.
   *
   * @param toolSpecs - Array of tool specifications
   * @returns Ollama-formatted tool array
   */
  private _formatTools(toolSpecs?: StreamOptions['toolSpecs']): OllamaTool[] {
    if (!toolSpecs || toolSpecs.length === 0) return []

    return toolSpecs.map(
      (spec) =>
        ({
          type: 'function',
          function: {
            name: spec.name,
            description: spec.description,
            ...(spec.inputSchema ? { parameters: spec.inputSchema } : {}),
          },
        }) as OllamaTool
    )
  }

  /**
   * Formats SDK messages to Ollama message format.
   * Each content block becomes a separate Ollama message because Ollama
   * does not support arrays of mixed content types in a single message.
   *
   * @param messages - SDK messages
   * @param systemPrompt - Optional system prompt
   * @returns Ollama-formatted messages
   */
  private _formatMessages(messages: Message[], systemPrompt?: StreamOptions['systemPrompt']): OllamaMessage[] {
    const ollamaMessages: OllamaMessage[] = []

    const systemText = this._extractSystemPromptText(systemPrompt)
    if (systemText) {
      ollamaMessages.push({ role: 'system', content: systemText })
    }

    for (const message of messages) {
      for (const block of message.content) {
        const formatted = this._formatContentBlock(message.role, block)
        ollamaMessages.push(...formatted)
      }
    }

    return ollamaMessages
  }

  /**
   * Extracts text from a system prompt (string or array format).
   *
   * @param systemPrompt - System prompt configuration
   * @returns Extracted text, or undefined if empty
   */
  private _extractSystemPromptText(systemPrompt?: StreamOptions['systemPrompt']): string | undefined {
    if (systemPrompt === undefined) return undefined

    if (typeof systemPrompt === 'string') {
      return systemPrompt.trim().length > 0 ? systemPrompt : undefined
    }

    if (Array.isArray(systemPrompt) && systemPrompt.length > 0) {
      const textBlocks: string[] = []

      for (const block of systemPrompt) {
        if (block.type === 'textBlock') {
          textBlocks.push(block.text)
        } else if (block.type === 'cachePointBlock') {
          logger.warn('cache points are not supported in ollama system prompts, ignoring cache points')
        } else if (block.type === 'guardContentBlock') {
          logger.warn('guard content is not supported in ollama system prompts, removing guard content block')
        }
      }

      return textBlocks.length > 0 ? textBlocks.join('') : undefined
    }

    return undefined
  }

  /**
   * Formats a single content block into Ollama message(s).
   *
   * @param role - Message role
   * @param block - Content block to format
   * @returns Array of Ollama messages (may be multiple for tool results)
   */
  private _formatContentBlock(role: string, block: ContentBlock): OllamaMessage[] {
    switch (block.type) {
      case 'textBlock':
        return [{ role, content: block.text }]

      case 'imageBlock':
        return this._formatImageBlock(role, block)

      case 'toolUseBlock':
        return [
          {
            role,
            content: '',
            tool_calls: [
              {
                function: {
                  name: block.name,
                  arguments: block.input as Record<string, unknown>,
                },
              },
            ],
          },
        ]

      case 'toolResultBlock':
        return this._formatToolResult(block)

      case 'reasoningBlock':
        if (block.text) {
          return [{ role, content: block.text }]
        }
        return []

      case 'cachePointBlock':
        return []

      case 'guardContentBlock':
        logger.warn('block_type=<guardContentBlock> | guard content not supported by ollama | skipping')
        return []

      case 'documentBlock':
        logger.warn('block_type=<documentBlock> | documents not supported by ollama | skipping')
        return []

      case 'videoBlock':
        logger.warn('block_type=<videoBlock> | videos not supported by ollama | skipping')
        return []

      case 'citationsBlock':
        logger.warn('block_type=<citationsBlock> | citations not supported by ollama | skipping')
        return []

      default:
        logger.warn(`block_type=<${(block as ContentBlock).type}> | unsupported content type for ollama | skipping`)
        return []
    }
  }

  /**
   * Formats an image block to Ollama message format.
   *
   * @param role - Message role
   * @param imageBlock - Image block to format
   * @returns Ollama message with image, or empty array if unsupported source
   */
  private _formatImageBlock(role: string, imageBlock: ImageBlock): OllamaMessage[] {
    if (imageBlock.source.type === 'imageSourceBytes') {
      const base64 = encodeBase64(imageBlock.source.bytes)
      return [{ role, content: '', images: [base64] }]
    } else if (imageBlock.source.type === 'imageSourceUrl') {
      return [{ role, content: '', images: [imageBlock.source.url] }]
    }

    logger.warn(`source_type=<${imageBlock.source.type}> | image source not supported by ollama | skipping image block`)
    return []
  }

  /**
   * Formats a tool result block into Ollama messages.
   * Each sub-content becomes a separate 'tool' role message.
   *
   * @param toolResult - Tool result block to format
   * @returns Array of Ollama tool messages
   */
  private _formatToolResult(toolResult: ToolResultBlock): OllamaMessage[] {
    const messages: OllamaMessage[] = []

    for (const content of toolResult.content) {
      switch (content.type) {
        case 'textBlock':
          messages.push({ role: 'tool', content: content.text })
          break
        case 'jsonBlock':
          messages.push({ role: 'tool', content: JSON.stringify(content.json) })
          break
        case 'imageBlock': {
          const formatted = this._formatImageBlock('tool', content)
          messages.push(...formatted)
          break
        }
        case 'documentBlock':
          logger.warn('block_type=<documentBlock> | documents not supported in ollama tool results | skipping')
          break
        case 'videoBlock':
          logger.warn('block_type=<videoBlock> | videos not supported in ollama tool results | skipping')
          break
        default:
          logger.warn(
            `block_type=<${(content as { type: string }).type}> | unsupported content in ollama tool result | skipping`
          )
      }
    }

    return messages
  }

  /**
   * Maps Ollama's done_reason to SDK StopReason.
   *
   * @param doneReason - Ollama's done_reason string
   * @param hasToolCalls - Whether tool calls were made during the response
   * @returns SDK stop reason
   */
  private _mapStopReason(doneReason: string | undefined, hasToolCalls: boolean): StopReason {
    if (hasToolCalls) return 'toolUse'
    if (doneReason === 'length') return 'maxTokens'
    return 'endTurn'
  }
}
