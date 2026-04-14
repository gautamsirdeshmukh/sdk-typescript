import { describe, it, expect, vi, beforeEach } from 'vitest'
import { Ollama } from 'ollama'
import type { ChatResponse } from 'ollama'
import { OllamaModel } from '../ollama.js'
import { ContextWindowOverflowError } from '../../errors.js'
import { collectIterator } from '../../__fixtures__/model-test-helpers.js'
import {
  Message,
  TextBlock,
  ToolUseBlock,
  ToolResultBlock,
  ReasoningBlock,
  CachePointBlock,
  GuardContentBlock,
  JsonBlock,
} from '../../types/messages.js'
import type { SystemContentBlock } from '../../types/messages.js'
import { ImageBlock, DocumentBlock, VideoBlock } from '../../types/media.js'
import type { ModelStreamEvent } from '../streaming.js'
import { logger } from '../../logging/logger.js'

// Mock the Ollama SDK
vi.mock('ollama', () => {
  const mockConstructor = vi.fn(function (this: Record<string, unknown>) {
    return {}
  })
  return {
    Ollama: mockConstructor,
  }
})

/**
 * Helper to create a mock Ollama client with streaming chat support.
 */
function createMockClient(streamGenerator: () => AsyncGenerator<Partial<ChatResponse>>): Ollama {
  return {
    chat: vi.fn(async () => {
      const gen = streamGenerator()
      return {
        [Symbol.asyncIterator]: () => gen,
        abort: vi.fn(),
      }
    }),
  } as unknown as Ollama
}

/**
 * Creates a basic text streaming response.
 */
function createTextStream(text: string, doneReason = 'stop'): () => AsyncGenerator<Partial<ChatResponse>> {
  return async function* () {
    yield {
      model: 'llama3.2',
      message: { role: 'assistant', content: text },
      done: false,
      done_reason: '',
    } as Partial<ChatResponse>
    yield {
      model: 'llama3.2',
      message: { role: 'assistant', content: '' },
      done: true,
      done_reason: doneReason,
      prompt_eval_count: 10,
      eval_count: 20,
      total_duration: 1_000_000_000, // 1 second in nanoseconds
    } as Partial<ChatResponse>
  }
}

/**
 * Creates a mock client that captures the request for assertion.
 */
function createMockClientWithCapture(captureContainer: { request: Record<string, unknown> }): Ollama {
  return {
    chat: vi.fn(async (request: Record<string, unknown>) => {
      captureContainer.request = request
      const gen = createTextStream('test')()
      return {
        [Symbol.asyncIterator]: () => gen,
        abort: vi.fn(),
      }
    }),
  } as unknown as Ollama
}

describe('OllamaModel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.restoreAllMocks()
  })

  describe('constructor', () => {
    it('creates an instance with default settings', () => {
      const model = new OllamaModel()
      const config = model.getConfig()
      expect(config).toStrictEqual({})
    })

    it('stores model config correctly', () => {
      const model = new OllamaModel({
        modelId: 'mistral',
        temperature: 0.7,
        maxTokens: 1024,
        topP: 0.9,
      })
      const config = model.getConfig()
      expect(config.modelId).toBe('mistral')
      expect(config.temperature).toBe(0.7)
      expect(config.maxTokens).toBe(1024)
      expect(config.topP).toBe(0.9)
    })

    it('does not include host, client, or clientConfig in model config', () => {
      const model = new OllamaModel({
        host: 'http://localhost:11434',
        modelId: 'llama3.2',
      })
      const config = model.getConfig()
      expect(config).toStrictEqual({ modelId: 'llama3.2' })
      expect(config).not.toHaveProperty('host')
      expect(config).not.toHaveProperty('client')
      expect(config).not.toHaveProperty('clientConfig')
    })

    it('passes host to Ollama constructor', () => {
      new OllamaModel({ host: 'http://remote:11434' })
      expect(Ollama).toHaveBeenCalledWith(expect.objectContaining({ host: 'http://remote:11434' }))
    })

    it('passes clientConfig to Ollama constructor', () => {
      new OllamaModel({ clientConfig: { proxy: true } })
      expect(Ollama).toHaveBeenCalledWith(expect.objectContaining({ proxy: true }))
    })

    it('uses provided client instance instead of creating new one', () => {
      const mockClient = {} as Ollama
      new OllamaModel({ client: mockClient, modelId: 'test' })
      // Ollama constructor should not be called when client is provided
      expect(Ollama).not.toHaveBeenCalled()
    })
  })

  describe('updateConfig', () => {
    it('merges new config with existing', () => {
      const model = new OllamaModel({ modelId: 'llama3.2', temperature: 0.5 })
      model.updateConfig({ temperature: 0.9, maxTokens: 2048 })
      const config = model.getConfig()
      expect(config.modelId).toBe('llama3.2')
      expect(config.temperature).toBe(0.9)
      expect(config.maxTokens).toBe(2048)
    })

    it('preserves fields not in update', () => {
      const model = new OllamaModel({ modelId: 'llama3.2', keepAlive: '5m' })
      model.updateConfig({ temperature: 0.7 })
      expect(model.getConfig().keepAlive).toBe('5m')
    })
  })

  describe('stream', () => {
    describe('validation', () => {
      it('throws error when messages array is empty', async () => {
        const mockClient = createMockClient(createTextStream(''))
        const model = new OllamaModel({ client: mockClient })

        await expect(async () => {
          await collectIterator(model.stream([]))
        }).rejects.toThrow('At least one message is required')
      })
    })

    describe('basic text streaming', () => {
      it('yields correct event sequence for text response', async () => {
        const mockClient = createMockClient(createTextStream('Hello, world!'))
        const model = new OllamaModel({ client: mockClient, modelId: 'llama3.2' })
        const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

        const events = await collectIterator(model.stream(messages))

        expect(events[0]).toStrictEqual({ type: 'modelMessageStartEvent', role: 'assistant' })
        expect(events[1]).toStrictEqual({ type: 'modelContentBlockStartEvent' })
        expect(events[2]).toStrictEqual({
          type: 'modelContentBlockDeltaEvent',
          delta: { type: 'textDelta', text: 'Hello, world!' },
        })
        // Text content block stop
        expect(events[events.length - 3]).toStrictEqual({ type: 'modelContentBlockStopEvent' })
        // Metadata
        expect(events[events.length - 2]).toMatchObject({
          type: 'modelMetadataEvent',
          usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
        })
        // Message stop
        expect(events[events.length - 1]).toStrictEqual({
          type: 'modelMessageStopEvent',
          stopReason: 'endTurn',
        })
      })

      it('skips empty text content chunks', async () => {
        const mockClient = createMockClient(async function* () {
          yield {
            model: 'llama3.2',
            message: { role: 'assistant', content: '' },
            done: false,
          } as Partial<ChatResponse>
          yield {
            model: 'llama3.2',
            message: { role: 'assistant', content: 'Hello' },
            done: false,
          } as Partial<ChatResponse>
          yield {
            model: 'llama3.2',
            message: { role: 'assistant', content: '' },
            done: true,
            done_reason: 'stop',
            prompt_eval_count: 5,
            eval_count: 10,
            total_duration: 500_000_000,
          } as Partial<ChatResponse>
        })
        const model = new OllamaModel({ client: mockClient })
        const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

        const events = await collectIterator(model.stream(messages))
        const textDeltas = events.filter(
          (e: ModelStreamEvent) => e.type === 'modelContentBlockDeltaEvent' && e.delta.type === 'textDelta'
        )

        // Only the non-empty chunk should produce a text delta
        expect(textDeltas).toHaveLength(1)
      })
    })

    describe('tool calling', () => {
      it('emits tool call events with start, delta, and stop', async () => {
        const mockClient = createMockClient(async function* () {
          yield {
            model: 'llama3.2',
            message: {
              role: 'assistant',
              content: '',
              tool_calls: [
                {
                  function: {
                    name: 'calculator',
                    arguments: { expression: '2+2' },
                  },
                },
              ],
            },
            done: false,
          } as Partial<ChatResponse>
          yield {
            model: 'llama3.2',
            message: { role: 'assistant', content: '' },
            done: true,
            done_reason: 'stop',
            prompt_eval_count: 10,
            eval_count: 5,
            total_duration: 500_000_000,
          } as Partial<ChatResponse>
        })
        const model = new OllamaModel({ client: mockClient })
        const messages = [new Message({ role: 'user', content: [new TextBlock('What is 2+2?')] })]

        const events = await collectIterator(model.stream(messages))

        const toolStart = events.find(
          (e: ModelStreamEvent) => e.type === 'modelContentBlockStartEvent' && e.start?.type === 'toolUseStart'
        )
        expect(toolStart).toBeDefined()
        expect(toolStart).toMatchObject({
          type: 'modelContentBlockStartEvent',
          start: { type: 'toolUseStart', name: 'calculator' },
        })
        if (toolStart?.type === 'modelContentBlockStartEvent' && toolStart.start) {
          expect(toolStart.start.toolUseId).toMatch(/^tooluse_/)
        }

        const toolDelta = events.find(
          (e: ModelStreamEvent) => e.type === 'modelContentBlockDeltaEvent' && e.delta.type === 'toolUseInputDelta'
        )
        expect(toolDelta).toMatchObject({
          type: 'modelContentBlockDeltaEvent',
          delta: { type: 'toolUseInputDelta', input: '{"expression":"2+2"}' },
        })

        expect(events).toContainEqual(expect.objectContaining({ type: 'modelMessageStopEvent', stopReason: 'toolUse' }))

        // Tool-only response must not produce a text block
        const textBlockStarts = events.filter(
          (e: ModelStreamEvent) => e.type === 'modelContentBlockStartEvent' && !e.start
        )
        expect(textBlockStarts).toHaveLength(0)
      })

      it('handles multiple tool calls in same chunk', async () => {
        const mockClient = createMockClient(async function* () {
          yield {
            model: 'llama3.2',
            message: {
              role: 'assistant',
              content: '',
              tool_calls: [
                { function: { name: 'tool_a', arguments: { x: 1 } } },
                { function: { name: 'tool_b', arguments: { y: 2 } } },
              ],
            },
            done: false,
          } as Partial<ChatResponse>
          yield {
            model: 'llama3.2',
            message: { role: 'assistant', content: '' },
            done: true,
            done_reason: 'stop',
            prompt_eval_count: 10,
            eval_count: 5,
            total_duration: 500_000_000,
          } as Partial<ChatResponse>
        })
        const model = new OllamaModel({ client: mockClient })
        const messages = [new Message({ role: 'user', content: [new TextBlock('test')] })]

        const events = await collectIterator(model.stream(messages))
        const toolStarts = events.filter(
          (e: ModelStreamEvent) => e.type === 'modelContentBlockStartEvent' && e.start?.type === 'toolUseStart'
        )

        expect(toolStarts).toHaveLength(2)
      })

      it('closes text block before tool blocks when chunk has both text and tool_calls', async () => {
        const mockClient = createMockClient(async function* () {
          yield {
            model: 'llama3.2',
            message: {
              role: 'assistant',
              content: "I'll calculate that",
              tool_calls: [{ function: { name: 'calculator', arguments: { expr: '2+2' } } }],
            },
            done: false,
          } as Partial<ChatResponse>
          yield {
            model: 'llama3.2',
            message: { role: 'assistant', content: '' },
            done: true,
            done_reason: 'stop',
            prompt_eval_count: 10,
            eval_count: 5,
            total_duration: 500_000_000,
          } as Partial<ChatResponse>
        })
        const model = new OllamaModel({ client: mockClient })
        const messages = [new Message({ role: 'user', content: [new TextBlock('test')] })]

        const events = await collectIterator(model.stream(messages))

        const toolStarts = events.filter(
          (e: ModelStreamEvent) => e.type === 'modelContentBlockStartEvent' && e.start?.type === 'toolUseStart'
        )
        const textDeltas = events.filter(
          (e: ModelStreamEvent) => e.type === 'modelContentBlockDeltaEvent' && e.delta.type === 'textDelta'
        )
        expect(toolStarts).toHaveLength(1)
        expect(textDeltas).toHaveLength(1)

        // Text delta must appear before tool start (text block is closed before tools)
        const textDeltaIdx = events.findIndex(
          (e: ModelStreamEvent) => e.type === 'modelContentBlockDeltaEvent' && e.delta.type === 'textDelta'
        )
        const toolStartIdx = events.findIndex(
          (e: ModelStreamEvent) => e.type === 'modelContentBlockStartEvent' && e.start?.type === 'toolUseStart'
        )
        expect(textDeltaIdx).toBeLessThan(toolStartIdx)
      })

      it('uses server-provided tool call ID when available', async () => {
        const mockClient = createMockClient(async function* () {
          yield {
            model: 'llama3.2',
            message: {
              role: 'assistant',
              content: '',
              tool_calls: [
                {
                  id: 'call_abc123',
                  function: { name: 'calculator', arguments: { expression: '2+2' } },
                },
              ],
            },
            done: false,
          } as unknown as Partial<ChatResponse>
          yield {
            model: 'llama3.2',
            message: { role: 'assistant', content: '' },
            done: true,
            done_reason: 'stop',
            prompt_eval_count: 10,
            eval_count: 5,
            total_duration: 500_000_000,
          } as Partial<ChatResponse>
        })
        const model = new OllamaModel({ client: mockClient })
        const messages = [new Message({ role: 'user', content: [new TextBlock('test')] })]

        const events = await collectIterator(model.stream(messages))
        const toolStart = events.find(
          (e: ModelStreamEvent) => e.type === 'modelContentBlockStartEvent' && e.start?.type === 'toolUseStart'
        )

        expect(toolStart).toBeDefined()
        if (toolStart && toolStart.type === 'modelContentBlockStartEvent' && toolStart.start) {
          expect(toolStart.start.toolUseId).toBe('call_abc123')
          expect(toolStart.start.name).toBe('calculator')
        }
      })
    })

    describe('stop reasons', () => {
      it('maps "stop" to "endTurn"', async () => {
        const mockClient = createMockClient(createTextStream('test', 'stop'))
        const model = new OllamaModel({ client: mockClient })
        const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

        const events = await collectIterator(model.stream(messages))

        expect(events).toContainEqual(expect.objectContaining({ type: 'modelMessageStopEvent', stopReason: 'endTurn' }))
      })

      it('maps "length" to "maxTokens"', async () => {
        const mockClient = createMockClient(createTextStream('test', 'length'))
        const model = new OllamaModel({ client: mockClient })
        const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

        const events = await collectIterator(model.stream(messages))

        expect(events).toContainEqual(
          expect.objectContaining({ type: 'modelMessageStopEvent', stopReason: 'maxTokens' })
        )
      })

      it('uses "toolUse" when tool calls are present regardless of done_reason', async () => {
        const mockClient = createMockClient(async function* () {
          yield {
            model: 'llama3.2',
            message: {
              role: 'assistant',
              content: '',
              tool_calls: [{ function: { name: 'test', arguments: {} } }],
            },
            done: false,
          } as Partial<ChatResponse>
          yield {
            model: 'llama3.2',
            message: { role: 'assistant', content: '' },
            done: true,
            done_reason: 'stop',
            prompt_eval_count: 5,
            eval_count: 5,
            total_duration: 100_000_000,
          } as Partial<ChatResponse>
        })
        const model = new OllamaModel({ client: mockClient })
        const messages = [new Message({ role: 'user', content: [new TextBlock('test')] })]

        const events = await collectIterator(model.stream(messages))

        expect(events).toContainEqual(expect.objectContaining({ type: 'modelMessageStopEvent', stopReason: 'toolUse' }))
      })

      it('defaults to "endTurn" for undefined done_reason', async () => {
        const mockClient = createMockClient(async function* () {
          yield {
            model: 'llama3.2',
            message: { role: 'assistant', content: 'test' },
            done: true,
            prompt_eval_count: 5,
            eval_count: 5,
            total_duration: 100_000_000,
          } as Partial<ChatResponse>
        })
        const model = new OllamaModel({ client: mockClient })
        const messages = [new Message({ role: 'user', content: [new TextBlock('test')] })]

        const events = await collectIterator(model.stream(messages))

        expect(events).toContainEqual(expect.objectContaining({ type: 'modelMessageStopEvent', stopReason: 'endTurn' }))
      })
    })

    describe('metadata', () => {
      it('emits usage with token counts from final chunk', async () => {
        const mockClient = createMockClient(createTextStream('test'))
        const model = new OllamaModel({ client: mockClient })
        const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

        const events = await collectIterator(model.stream(messages))

        expect(events).toContainEqual(
          expect.objectContaining({
            type: 'modelMetadataEvent',
            usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
          })
        )
      })

      it('emits latency from total_duration (nanoseconds to milliseconds)', async () => {
        const mockClient = createMockClient(createTextStream('test'))
        const model = new OllamaModel({ client: mockClient })
        const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

        const events = await collectIterator(model.stream(messages))

        expect(events).toContainEqual(
          expect.objectContaining({
            type: 'modelMetadataEvent',
            metrics: { latencyMs: 1000 }, // 1_000_000_000 ns = 1000 ms
          })
        )
      })

      it('defaults to zero when eval counts are missing', async () => {
        const mockClient = createMockClient(async function* () {
          yield {
            model: 'llama3.2',
            message: { role: 'assistant', content: 'test' },
            done: true,
            done_reason: 'stop',
          } as Partial<ChatResponse>
        })
        const model = new OllamaModel({ client: mockClient })
        const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

        const events = await collectIterator(model.stream(messages))

        expect(events).toContainEqual(
          expect.objectContaining({
            type: 'modelMetadataEvent',
            usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
          })
        )
      })
    })
  })

  describe('request formatting', () => {
    it('uses default model ID when none specified', async () => {
      const capture = { request: {} as Record<string, unknown> }
      const mockClient = createMockClientWithCapture(capture)
      const model = new OllamaModel({ client: mockClient })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

      await collectIterator(model.stream(messages))

      expect(capture.request.model).toBe('llama3.2')
    })

    it('uses configured model ID', async () => {
      const capture = { request: {} as Record<string, unknown> }
      const mockClient = createMockClientWithCapture(capture)
      const model = new OllamaModel({ client: mockClient, modelId: 'mistral' })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

      await collectIterator(model.stream(messages))

      expect(capture.request.model).toBe('mistral')
    })

    it('maps config options to Ollama options', async () => {
      const capture = { request: {} as Record<string, unknown> }
      const mockClient = createMockClientWithCapture(capture)
      const model = new OllamaModel({
        client: mockClient,
        maxTokens: 100,
        temperature: 0.5,
        topP: 0.9,
        stopSequences: ['END'],
      })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

      await collectIterator(model.stream(messages))

      const options = capture.request.options as Record<string, unknown>
      expect(options.num_predict).toBe(100)
      expect(options.temperature).toBe(0.5)
      expect(options.top_p).toBe(0.9)
      expect(options.stop).toStrictEqual(['END'])
    })

    it('omits undefined config options', async () => {
      const capture = { request: {} as Record<string, unknown> }
      const mockClient = createMockClientWithCapture(capture)
      const model = new OllamaModel({ client: mockClient, temperature: 0.5 })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

      await collectIterator(model.stream(messages))

      const options = capture.request.options as Record<string, unknown>
      expect(options.temperature).toBe(0.5)
      expect(options).not.toHaveProperty('num_predict')
      expect(options).not.toHaveProperty('top_p')
      expect(options).not.toHaveProperty('stop')
    })

    it('merges custom options with computed options', async () => {
      const capture = { request: {} as Record<string, unknown> }
      const mockClient = createMockClientWithCapture(capture)
      const model = new OllamaModel({
        client: mockClient,
        temperature: 0.5,
        options: { top_k: 40, repeat_penalty: 1.1 },
      })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

      await collectIterator(model.stream(messages))

      const options = capture.request.options as Record<string, unknown>
      expect(options.temperature).toBe(0.5)
      expect(options.top_k).toBe(40)
      expect(options.repeat_penalty).toBe(1.1)
    })

    it('includes keepAlive when set', async () => {
      const capture = { request: {} as Record<string, unknown> }
      const mockClient = createMockClientWithCapture(capture)
      const model = new OllamaModel({ client: mockClient, keepAlive: '10m' })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

      await collectIterator(model.stream(messages))

      expect(capture.request.keep_alive).toBe('10m')
    })

    it('does not include keepAlive when not set', async () => {
      const capture = { request: {} as Record<string, unknown> }
      const mockClient = createMockClientWithCapture(capture)
      const model = new OllamaModel({ client: mockClient })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

      await collectIterator(model.stream(messages))

      expect(capture.request).not.toHaveProperty('keep_alive')
    })

    it('spreads params for forward compatibility', async () => {
      const capture = { request: {} as Record<string, unknown> }
      const mockClient = createMockClientWithCapture(capture)
      const model = new OllamaModel({
        client: mockClient,
        params: { think: true },
      })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

      await collectIterator(model.stream(messages))

      expect(capture.request.think).toBe(true)
    })

    it('warns when toolChoice is provided', async () => {
      const warnSpy = vi.spyOn(logger, 'warn')
      const mockClient = createMockClient(createTextStream('test'))
      const model = new OllamaModel({ client: mockClient })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

      await collectIterator(model.stream(messages, { toolChoice: { auto: {} } }))

      expect(warnSpy).toHaveBeenCalledWith('tool_choice is not supported by ollama, ignoring')
    })

    it('omits tools field when no tool specs provided', async () => {
      const capture = { request: {} as Record<string, unknown> }
      const mockClient = createMockClientWithCapture(capture)
      const model = new OllamaModel({ client: mockClient })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

      await collectIterator(model.stream(messages))

      expect(capture.request).not.toHaveProperty('tools')
    })

    it('formats tool specs without inputSchema', async () => {
      const capture = { request: {} as Record<string, unknown> }
      const mockClient = createMockClientWithCapture(capture)
      const model = new OllamaModel({ client: mockClient })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

      await collectIterator(
        model.stream(messages, {
          toolSpecs: [
            {
              name: 'no_input_tool',
              description: 'A tool with no input schema',
            },
          ],
        })
      )

      const tools = capture.request.tools as Array<Record<string, unknown>>
      expect(tools).toHaveLength(1)
      expect(tools[0]).toStrictEqual({
        type: 'function',
        function: {
          name: 'no_input_tool',
          description: 'A tool with no input schema',
        },
      })
    })

    it('formats tool specs correctly', async () => {
      const capture = { request: {} as Record<string, unknown> }
      const mockClient = createMockClientWithCapture(capture)
      const model = new OllamaModel({ client: mockClient })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

      await collectIterator(
        model.stream(messages, {
          toolSpecs: [
            {
              name: 'calculator',
              description: 'Calculate math expressions',
              inputSchema: {
                type: 'object',
                properties: { expression: { type: 'string' } },
                required: ['expression'],
              },
            },
          ],
        })
      )

      const tools = capture.request.tools as Array<Record<string, unknown>>
      expect(tools).toHaveLength(1)
      expect(tools[0]).toStrictEqual({
        type: 'function',
        function: {
          name: 'calculator',
          description: 'Calculate math expressions',
          parameters: {
            type: 'object',
            properties: { expression: { type: 'string' } },
            required: ['expression'],
          },
        },
      })
    })
  })

  describe('message formatting', () => {
    it('formats text blocks', async () => {
      const capture = { request: {} as Record<string, unknown> }
      const mockClient = createMockClientWithCapture(capture)
      const model = new OllamaModel({ client: mockClient })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hello')] })]

      await collectIterator(model.stream(messages))

      const formatted = capture.request.messages as Array<Record<string, unknown>>
      expect(formatted).toContainEqual({ role: 'user', content: 'Hello' })
    })

    it('formats image blocks with bytes source', async () => {
      const capture = { request: {} as Record<string, unknown> }
      const mockClient = createMockClientWithCapture(capture)
      const model = new OllamaModel({ client: mockClient })
      const imageBytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47])
      const messages = [
        new Message({
          role: 'user',
          content: [new ImageBlock({ format: 'png', source: { bytes: imageBytes } })],
        }),
      ]

      await collectIterator(model.stream(messages))

      const formatted = capture.request.messages as Array<Record<string, unknown>>
      const imageMsg = formatted.find((m) => Array.isArray(m.images))
      expect(imageMsg).toBeDefined()
      expect(imageMsg?.role).toBe('user')
      expect(imageMsg?.content).toBe('')
      expect((imageMsg?.images as string[])?.length).toBe(1)
    })

    it('formats image blocks with URL source', async () => {
      const capture = { request: {} as Record<string, unknown> }
      const mockClient = createMockClientWithCapture(capture)
      const model = new OllamaModel({ client: mockClient })
      const messages = [
        new Message({
          role: 'user',
          content: [new ImageBlock({ format: 'png', source: { url: 'https://example.com/img.png' } })],
        }),
      ]

      await collectIterator(model.stream(messages))

      const formatted = capture.request.messages as Array<Record<string, unknown>>
      const imageMsg = formatted.find((m) => Array.isArray(m.images))
      expect(imageMsg).toBeDefined()
      expect((imageMsg?.images as string[])?.[0]).toBe('https://example.com/img.png')
    })

    it('skips S3 source images with warning', async () => {
      const warnSpy = vi.spyOn(logger, 'warn')
      const capture = { request: {} as Record<string, unknown> }
      const mockClient = createMockClientWithCapture(capture)
      const model = new OllamaModel({ client: mockClient })
      const messages = [
        new Message({
          role: 'user',
          content: [
            new TextBlock('test'),
            new ImageBlock({
              format: 'png',
              source: { location: { type: 's3', uri: 's3://bucket/key' } },
            }),
          ],
        }),
      ]

      await collectIterator(model.stream(messages))

      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('image source not supported by ollama'))
    })

    it('formats tool use blocks with block.name as function name', async () => {
      const capture = { request: {} as Record<string, unknown> }
      const mockClient = createMockClientWithCapture(capture)
      const model = new OllamaModel({ client: mockClient })
      const messages = [
        new Message({
          role: 'assistant',
          content: [new ToolUseBlock({ name: 'calculator', toolUseId: 'calc-123', input: { expression: '2+2' } })],
        }),
      ]

      await collectIterator(model.stream(messages))

      const formatted = capture.request.messages as Array<Record<string, unknown>>
      const toolMsg = formatted.find((m) => m.tool_calls)
      expect(toolMsg).toBeDefined()
      expect(toolMsg?.role).toBe('assistant')
      const toolCalls = toolMsg!.tool_calls as Array<{ function: { name: string; arguments: unknown } }>
      expect(toolCalls[0]!.function.name).toBe('calculator')
      expect(toolCalls[0]!.function.arguments).toStrictEqual({ expression: '2+2' })
    })

    it('flattens tool result blocks into separate tool messages', async () => {
      const capture = { request: {} as Record<string, unknown> }
      const mockClient = createMockClientWithCapture(capture)
      const model = new OllamaModel({ client: mockClient })
      const messages = [
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'tool-1',
              status: 'success',
              content: [new TextBlock('result text'), new JsonBlock({ json: { answer: 42 } })],
            }),
          ],
        }),
      ]

      await collectIterator(model.stream(messages))

      const formatted = capture.request.messages as Array<Record<string, unknown>>
      const toolMsgs = formatted.filter((m) => m.role === 'tool')
      expect(toolMsgs).toHaveLength(2)
      expect(toolMsgs[0]!.content).toBe('result text')
      expect(toolMsgs[1]!.content).toBe('{"answer":42}')
    })

    it('formats reasoning blocks with text as text messages', async () => {
      const capture = { request: {} as Record<string, unknown> }
      const mockClient = createMockClientWithCapture(capture)
      const model = new OllamaModel({ client: mockClient })
      const messages = [
        new Message({
          role: 'assistant',
          content: [new ReasoningBlock({ text: 'thinking...' })],
        }),
      ]

      await collectIterator(model.stream(messages))

      const formatted = capture.request.messages as Array<Record<string, unknown>>
      expect(formatted).toContainEqual({ role: 'assistant', content: 'thinking...' })
    })

    it('skips reasoning blocks without text', async () => {
      const capture = { request: {} as Record<string, unknown> }
      const mockClient = createMockClientWithCapture(capture)
      const model = new OllamaModel({ client: mockClient })
      const messages = [
        new Message({
          role: 'assistant',
          content: [new TextBlock('hello'), new ReasoningBlock({ signature: 'sig-123' })],
        }),
      ]

      await collectIterator(model.stream(messages))

      const formatted = capture.request.messages as Array<Record<string, unknown>>
      // Should only have the text message, not the reasoning block
      expect(formatted).toHaveLength(1)
      expect(formatted[0]!.content).toBe('hello')
    })

    it('silently skips cache point blocks', async () => {
      const warnSpy = vi.spyOn(logger, 'warn')
      const capture = { request: {} as Record<string, unknown> }
      const mockClient = createMockClientWithCapture(capture)
      const model = new OllamaModel({ client: mockClient })
      const messages = [
        new Message({
          role: 'user',
          content: [new TextBlock('test'), new CachePointBlock({ cacheType: 'default' })],
        }),
      ]

      await collectIterator(model.stream(messages))

      const formatted = capture.request.messages as Array<Record<string, unknown>>
      expect(formatted).toHaveLength(1) // Only text, no cache point
      // No warning for cache points (silent skip)
      expect(warnSpy).not.toHaveBeenCalledWith(expect.stringContaining('cachePoint'))
    })

    it('skips unsupported block types with warning', async () => {
      const warnSpy = vi.spyOn(logger, 'warn')
      const capture = { request: {} as Record<string, unknown> }
      const mockClient = createMockClientWithCapture(capture)
      const model = new OllamaModel({ client: mockClient })
      const messages = [
        new Message({
          role: 'user',
          content: [
            new TextBlock('test'),
            new GuardContentBlock({ text: { qualifiers: ['grounding_source'], text: 'guard' } }),
            new DocumentBlock({
              format: 'pdf',
              name: 'doc.pdf',
              source: { text: 'content' },
            }),
            new VideoBlock({
              format: 'mp4',
              source: { bytes: new Uint8Array([1, 2, 3]) },
            }),
          ],
        }),
      ]

      await collectIterator(model.stream(messages))

      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('guard content not supported'))
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('documents not supported'))
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('videos not supported'))
    })

    describe('system prompt', () => {
      it('formats string system prompt', async () => {
        const capture = { request: {} as Record<string, unknown> }
        const mockClient = createMockClientWithCapture(capture)
        const model = new OllamaModel({ client: mockClient })
        const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

        await collectIterator(model.stream(messages, { systemPrompt: 'You are helpful' }))

        const formatted = capture.request.messages as Array<Record<string, unknown>>
        expect(formatted[0]).toStrictEqual({ role: 'system', content: 'You are helpful' })
      })

      it('skips empty string system prompt', async () => {
        const capture = { request: {} as Record<string, unknown> }
        const mockClient = createMockClientWithCapture(capture)
        const model = new OllamaModel({ client: mockClient })
        const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

        await collectIterator(model.stream(messages, { systemPrompt: '   ' }))

        const formatted = capture.request.messages as Array<Record<string, unknown>>
        expect(formatted[0]?.role).not.toBe('system')
      })

      it('formats array system prompt extracting text blocks', async () => {
        const capture = { request: {} as Record<string, unknown> }
        const mockClient = createMockClientWithCapture(capture)
        const model = new OllamaModel({ client: mockClient })
        const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

        const systemPrompt: SystemContentBlock[] = [new TextBlock('You are '), new TextBlock('helpful')]

        await collectIterator(model.stream(messages, { systemPrompt }))

        const formatted = capture.request.messages as Array<Record<string, unknown>>
        expect(formatted[0]).toStrictEqual({ role: 'system', content: 'You are helpful' })
      })

      it('warns on cache points in system prompt', async () => {
        const warnSpy = vi.spyOn(logger, 'warn')
        const capture = { request: {} as Record<string, unknown> }
        const mockClient = createMockClientWithCapture(capture)
        const model = new OllamaModel({ client: mockClient })
        const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

        const systemPrompt: SystemContentBlock[] = [
          new TextBlock('You are helpful'),
          new CachePointBlock({ cacheType: 'default' }),
        ]

        await collectIterator(model.stream(messages, { systemPrompt }))

        expect(warnSpy).toHaveBeenCalledWith(
          'cache points are not supported in ollama system prompts, ignoring cache points'
        )
      })

      it('warns on guard content in system prompt', async () => {
        const warnSpy = vi.spyOn(logger, 'warn')
        const capture = { request: {} as Record<string, unknown> }
        const mockClient = createMockClientWithCapture(capture)
        const model = new OllamaModel({ client: mockClient })
        const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

        const systemPrompt: SystemContentBlock[] = [
          new TextBlock('You are helpful'),
          new GuardContentBlock({ text: { qualifiers: ['grounding_source'], text: 'guard' } }),
        ]

        await collectIterator(model.stream(messages, { systemPrompt }))

        expect(warnSpy).toHaveBeenCalledWith(
          'guard content is not supported in ollama system prompts, removing guard content block'
        )
      })
    })
  })

  describe('error handling', () => {
    it('throws ContextWindowOverflowError for context length errors', async () => {
      const mockClient = {
        chat: vi.fn(async () => {
          throw new Error('context length exceeded: 12345 > 8192')
        }),
      } as unknown as Ollama
      const model = new OllamaModel({ client: mockClient })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

      await expect(async () => {
        await collectIterator(model.stream(messages))
      }).rejects.toThrow(ContextWindowOverflowError)
    })

    it('throws ContextWindowOverflowError for various context window error patterns', async () => {
      const patterns = ['context length', 'context window']

      for (const pattern of patterns) {
        const mockClient = {
          chat: vi.fn(async () => {
            throw new Error(`Error: ${pattern} limit reached`)
          }),
        } as unknown as Ollama
        const model = new OllamaModel({ client: mockClient })
        const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

        await expect(async () => {
          await collectIterator(model.stream(messages))
        }).rejects.toThrow(ContextWindowOverflowError)
      }
    })

    it('passes through non-context-window errors unchanged', async () => {
      const mockClient = {
        chat: vi.fn(async () => {
          throw new Error('Connection refused')
        }),
      } as unknown as Ollama
      const model = new OllamaModel({ client: mockClient })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

      await expect(async () => {
        await collectIterator(model.stream(messages))
      }).rejects.toThrow('Connection refused')

      // Should NOT be a ContextWindowOverflowError
      try {
        await collectIterator(model.stream(messages))
      } catch (e) {
        expect(e).not.toBeInstanceOf(ContextWindowOverflowError)
      }
    })
  })
})
