import { createErrorResult, Tool, type ToolContext, type ToolStreamGenerator } from './tool.js'
import type { ToolSpec } from './types.js'
import type { JSONSchema, JSONValue } from '../types/json.js'
import { JsonBlock, TextBlock, ToolResultBlock, type ToolResultContent } from '../types/messages.js'
import { ImageBlock, decodeBase64 } from '../types/media.js'
import { toMediaFormat, toMimeType, type ImageFormat } from '../mime.js'
import { logger } from '../logging/logger.js'
import type { McpClient } from '../mcp.js'

export interface McpToolConfig {
  name: string
  description: string
  inputSchema: JSONSchema
  client: McpClient
}

/**
 * A Tool implementation that proxies calls to a remote MCP server.
 *
 * Unlike FunctionTool, which wraps local logic, McpTool delegates execution
 * to the connected McpClient and translates the SDK's response format
 * directly into ToolResultBlocks.
 */
export class McpTool extends Tool {
  readonly name: string
  readonly description: string
  readonly toolSpec: ToolSpec
  private readonly mcpClient: McpClient

  constructor(config: McpToolConfig) {
    super()
    this.name = config.name
    this.description = config.description
    this.toolSpec = {
      name: config.name,
      description: config.description,
      inputSchema: config.inputSchema,
    }
    this.mcpClient = config.client
  }

  // eslint-disable-next-line require-yield
  async *stream(toolContext: ToolContext): ToolStreamGenerator {
    const { toolUseId, input } = toolContext.toolUse

    try {
      // Input is validated by MCP Client before invocation
      const rawResult: unknown = await this.mcpClient.callTool(this, input as JSONValue)

      if (!this._isMcpToolResult(rawResult)) {
        throw new Error('Invalid tool result from MCP Client: missing content array')
      }

      const content: ToolResultContent[] = rawResult.content.map((item: unknown) => this._mapMcpContent(item))

      if (content.length === 0) {
        content.push(new TextBlock('Tool execution completed successfully with no output.'))
      }

      return new ToolResultBlock({
        toolUseId,
        status: rawResult.isError ? 'error' : 'success',
        content,
      })
    } catch (error) {
      return createErrorResult(error, toolUseId)
    }
  }

  /**
   * Type Guard: Checks if value matches the expected MCP SDK result shape.
   * \{ content: unknown[]; isError?: boolean \}
   */
  private _isMcpToolResult(value: unknown): value is { content: unknown[]; isError?: boolean } {
    if (typeof value !== 'object' || value === null) {
      return false
    }

    // Safe cast to generic record to check properties
    const record = value as Record<string, unknown>

    return Array.isArray(record.content)
  }

  /**
   * Maps a single MCP content item to the corresponding SDK ToolResultContent block.
   */
  private _mapMcpContent(item: unknown): ToolResultContent {
    if (this._isMcpTextContent(item)) {
      return new TextBlock(item.text)
    }

    if (this._isMcpImageContent(item)) {
      const format = toMediaFormat(item.mimeType)
      if (format && this._isImageFormat(format)) {
        return new ImageBlock({ format, source: { bytes: decodeBase64(item.data) } })
      }
      logger.warn(`mimeType=<${item.mimeType}> | unsupported MCP image content MIME type, falling back to JSON`)
      return new JsonBlock({ json: item as JSONValue })
    }

    if (this._isMcpResourceContent(item)) {
      const resource = item.resource as Record<string, unknown>

      if (typeof resource.text === 'string') {
        return new TextBlock(resource.text)
      }

      if (typeof resource.blob === 'string') {
        const mimeType = typeof resource.mimeType === 'string' ? resource.mimeType : undefined
        const format = mimeType ? toMediaFormat(mimeType) : undefined
        if (format && this._isImageFormat(format)) {
          return new ImageBlock({ format, source: { bytes: decodeBase64(resource.blob) } })
        }
        logger.warn(
          `mimeType=<${mimeType ?? 'unknown'}> | unsupported MCP embedded resource MIME type, falling back to JSON`
        )
      }
    }

    return new JsonBlock({ json: item as JSONValue })
  }

  /**
   * Type Guard: Checks if an item is a Text content block.
   * \{ type: 'text'; text: string \}
   */
  private _isMcpTextContent(value: unknown): value is { type: 'text'; text: string } {
    if (typeof value !== 'object' || value === null) {
      return false
    }

    const record = value as Record<string, unknown>

    return record.type === 'text' && typeof record.text === 'string'
  }

  /**
   * Type Guard: Checks if an item is an Image content block.
   * \{ type: 'image'; data: string; mimeType: string \}
   */
  private _isMcpImageContent(value: unknown): value is { type: 'image'; data: string; mimeType: string } {
    if (typeof value !== 'object' || value === null) {
      return false
    }

    const record = value as Record<string, unknown>

    return record.type === 'image' && typeof record.data === 'string' && typeof record.mimeType === 'string'
  }

  /**
   * Type Guard: Checks if an item is an EmbeddedResource content block.
   * \{ type: 'resource'; resource: Record\<string, unknown\> \}
   */
  private _isMcpResourceContent(value: unknown): value is { type: 'resource'; resource: Record<string, unknown> } {
    if (typeof value !== 'object' || value === null) {
      return false
    }

    const record = value as Record<string, unknown>

    return record.type === 'resource' && typeof record.resource === 'object' && record.resource !== null
  }

  /**
   * Type Guard: Checks if a media format string is a supported image format.
   */
  private _isImageFormat(format: string): format is ImageFormat {
    return toMimeType(format)?.startsWith('image/') ?? false
  }
}
