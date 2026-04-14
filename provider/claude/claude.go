package claude

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/dreaifekks/hebi8/llm"
)

type Config struct {
	APIKey     string
	BaseURL    string
	Model      string
	APIVersion string
	HTTPClient *http.Client
	DefaultMax int
}

type Client struct {
	apiKey     string
	baseURL    string
	model      string
	apiVersion string
	httpClient *http.Client
	defaultMax int
}

func New(cfg Config) *Client {
	baseURL := strings.TrimRight(cfg.BaseURL, "/")
	if baseURL == "" {
		baseURL = "https://api.anthropic.com/v1"
	}

	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 90 * time.Second}
	}

	apiVersion := cfg.APIVersion
	if apiVersion == "" {
		apiVersion = "2023-06-01"
	}

	defaultMax := cfg.DefaultMax
	if defaultMax <= 0 {
		defaultMax = 1024
	}

	return &Client{
		apiKey:     cfg.APIKey,
		baseURL:    baseURL,
		model:      cfg.Model,
		apiVersion: apiVersion,
		httpClient: httpClient,
		defaultMax: defaultMax,
	}
}

func (c *Client) Generate(ctx context.Context, req llm.Request) (llm.Response, error) {
	model := req.Model
	if model == "" {
		model = c.model
	}
	if model == "" {
		return llm.Response{}, fmt.Errorf("claude model is required")
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = c.defaultMax
	}

	wireReq := anthropicRequest{
		Model:       model,
		System:      req.System,
		MaxTokens:   maxTokens,
		Messages:    buildMessages(req.Messages),
		Temperature: req.Temperature,
	}
	if len(req.Tools) > 0 {
		wireReq.Tools = make([]anthropicTool, 0, len(req.Tools))
		for _, tool := range req.Tools {
			wireReq.Tools = append(wireReq.Tools, anthropicTool{
				Name:        tool.Name,
				Description: tool.Description,
				InputSchema: tool.Parameters,
			})
		}
	}

	body, err := json.Marshal(wireReq)
	if err != nil {
		return llm.Response{}, fmt.Errorf("marshal claude request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpointURL(), bytes.NewReader(body))
	if err != nil {
		return llm.Response{}, fmt.Errorf("build claude request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("anthropic-version", c.apiVersion)
	if c.apiKey != "" {
		httpReq.Header.Set("x-api-key", c.apiKey)
	}

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return llm.Response{}, fmt.Errorf("send claude request: %w", err)
	}
	defer httpResp.Body.Close()

	payload, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return llm.Response{}, fmt.Errorf("read claude response: %w", err)
	}
	if httpResp.StatusCode >= 300 {
		return llm.Response{}, fmt.Errorf("claude api error (%d): %s", httpResp.StatusCode, strings.TrimSpace(string(payload)))
	}

	var wireResp anthropicResponse
	if err := json.Unmarshal(payload, &wireResp); err != nil {
		return llm.Response{}, fmt.Errorf("decode claude response: %w", err)
	}

	message := llm.Message{Role: llm.RoleAssistant}
	for _, block := range wireResp.Content {
		switch block.Type {
		case "text":
			message.Content += block.Text
		case "tool_use":
			message.ToolCalls = append(message.ToolCalls, llm.ToolCall{
				ID:        block.ID,
				Name:      block.Name,
				Arguments: normalizeObject(block.Input),
			})
		}
	}

	return llm.Response{
		Message:    message,
		StopReason: wireResp.StopReason,
		Usage: llm.Usage{
			InputTokens:  wireResp.Usage.InputTokens,
			OutputTokens: wireResp.Usage.OutputTokens,
			TotalTokens:  wireResp.Usage.InputTokens + wireResp.Usage.OutputTokens,
		},
	}, nil
}

func (c *Client) endpointURL() string {
	if strings.HasSuffix(c.baseURL, "/messages") {
		return c.baseURL
	}
	return c.baseURL + "/messages"
}

func buildMessages(messages []llm.Message) []anthropicMessage {
	wireMessages := make([]anthropicMessage, 0, len(messages))

	for i := 0; i < len(messages); {
		msg := messages[i]
		switch msg.Role {
		case llm.RoleUser:
			wireMessages = append(wireMessages, anthropicMessage{
				Role:    "user",
				Content: msg.Content,
			})
			i++
		case llm.RoleAssistant:
			if len(msg.ToolCalls) == 0 {
				wireMessages = append(wireMessages, anthropicMessage{
					Role:    "assistant",
					Content: msg.Content,
				})
				i++
				continue
			}

			blocks := make([]anthropicBlock, 0, len(msg.ToolCalls)+1)
			if msg.Content != "" {
				blocks = append(blocks, anthropicBlock{Type: "text", Text: msg.Content})
			}
			for _, call := range msg.ToolCalls {
				blocks = append(blocks, anthropicBlock{
					Type:  "tool_use",
					ID:    call.ID,
					Name:  call.Name,
					Input: normalizeObject(call.Arguments),
				})
			}
			wireMessages = append(wireMessages, anthropicMessage{
				Role:    "assistant",
				Content: blocks,
			})
			i++
		case llm.RoleTool:
			blocks := make([]anthropicBlock, 0)
			for i < len(messages) && messages[i].Role == llm.RoleTool {
				toolMsg := messages[i]
				blocks = append(blocks, anthropicBlock{
					Type:      "tool_result",
					ToolUseID: toolMsg.ToolCallID,
					Content:   toolResultContent(toolMsg),
					IsError:   toolMsg.IsError,
				})
				i++
			}
			if i < len(messages) && messages[i].Role == llm.RoleUser && messages[i].Content != "" {
				blocks = append(blocks, anthropicBlock{
					Type: "text",
					Text: messages[i].Content,
				})
				i++
			}
			wireMessages = append(wireMessages, anthropicMessage{
				Role:    "user",
				Content: blocks,
			})
		default:
			i++
		}
	}

	return wireMessages
}

func normalizeObject(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 || !json.Valid(raw) {
		return json.RawMessage(`{}`)
	}
	return raw
}

func toolResultContent(msg llm.Message) string {
	if msg.Content != "" {
		return msg.Content
	}
	if len(msg.Data) > 0 {
		return string(msg.Data)
	}
	return ""
}

type anthropicRequest struct {
	Model       string             `json:"model"`
	System      string             `json:"system,omitempty"`
	MaxTokens   int                `json:"max_tokens"`
	Messages    []anthropicMessage `json:"messages"`
	Tools       []anthropicTool    `json:"tools,omitempty"`
	Temperature *float64           `json:"temperature,omitempty"`
}

type anthropicMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

type anthropicTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema,omitempty"`
}

type anthropicBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   string          `json:"content,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
}

type anthropicResponse struct {
	Content []anthropicBlock `json:"content"`
	Usage   struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
	StopReason string `json:"stop_reason"`
}
