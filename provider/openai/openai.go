package openai

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
	HTTPClient *http.Client
}

type Client struct {
	apiKey     string
	baseURL    string
	model      string
	httpClient *http.Client
}

func New(cfg Config) *Client {
	baseURL := strings.TrimRight(cfg.BaseURL, "/")
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}

	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 90 * time.Second}
	}

	return &Client{
		apiKey:     cfg.APIKey,
		baseURL:    baseURL,
		model:      cfg.Model,
		httpClient: httpClient,
	}
}

func (c *Client) Generate(ctx context.Context, req llm.Request) (llm.Response, error) {
	model := req.Model
	if model == "" {
		model = c.model
	}
	if model == "" {
		return llm.Response{}, fmt.Errorf("openai model is required")
	}

	wireReq := openAIRequest{
		Model:       model,
		Messages:    buildMessages(req.System, req.Messages),
		Temperature: req.Temperature,
	}
	if req.MaxTokens > 0 {
		wireReq.MaxCompletionTokens = req.MaxTokens
	}
	if len(req.Tools) > 0 {
		wireReq.ToolChoice = "auto"
		wireReq.Tools = make([]openAITool, 0, len(req.Tools))
		for _, tool := range req.Tools {
			wireReq.Tools = append(wireReq.Tools, openAITool{
				Type: "function",
				Function: openAIFunction{
					Name:        tool.Name,
					Description: tool.Description,
					Parameters:  tool.Parameters,
					Strict:      tool.Strict,
				},
			})
		}
	}

	body, err := json.Marshal(wireReq)
	if err != nil {
		return llm.Response{}, fmt.Errorf("marshal openai request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpointURL(), bytes.NewReader(body))
	if err != nil {
		return llm.Response{}, fmt.Errorf("build openai request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return llm.Response{}, fmt.Errorf("send openai request: %w", err)
	}
	defer httpResp.Body.Close()

	payload, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return llm.Response{}, fmt.Errorf("read openai response: %w", err)
	}
	if httpResp.StatusCode >= 300 {
		return llm.Response{}, fmt.Errorf("openai api error (%d): %s", httpResp.StatusCode, strings.TrimSpace(string(payload)))
	}

	var wireResp openAIResponse
	if err := json.Unmarshal(payload, &wireResp); err != nil {
		return llm.Response{}, fmt.Errorf("decode openai response: %w", err)
	}
	if len(wireResp.Choices) == 0 {
		return llm.Response{}, fmt.Errorf("openai response did not contain choices")
	}

	message := llm.Message{
		Role:    llm.RoleAssistant,
		Content: decodeOpenAIText(wireResp.Choices[0].Message.Content),
	}
	for _, call := range wireResp.Choices[0].Message.ToolCalls {
		message.ToolCalls = append(message.ToolCalls, llm.ToolCall{
			ID:        call.ID,
			Name:      call.Function.Name,
			Arguments: normalizeArgumentString(call.Function.Arguments),
		})
	}

	return llm.Response{
		Message:    message,
		StopReason: wireResp.Choices[0].FinishReason,
		Usage: llm.Usage{
			InputTokens:  wireResp.Usage.PromptTokens,
			OutputTokens: wireResp.Usage.CompletionTokens,
			TotalTokens:  wireResp.Usage.TotalTokens,
		},
	}, nil
}

func (c *Client) endpointURL() string {
	if strings.HasSuffix(c.baseURL, "/chat/completions") {
		return c.baseURL
	}
	return c.baseURL + "/chat/completions"
}

func buildMessages(system string, messages []llm.Message) []openAIMessage {
	wireMessages := make([]openAIMessage, 0, len(messages)+1)
	if system != "" {
		wireMessages = append(wireMessages, openAIMessage{
			Role:    "system",
			Content: system,
		})
	}

	for _, msg := range messages {
		switch msg.Role {
		case llm.RoleUser:
			wireMessages = append(wireMessages, openAIMessage{
				Role:    "user",
				Content: msg.Content,
			})
		case llm.RoleAssistant:
			wireMsg := openAIMessage{
				Role: "assistant",
			}
			if msg.Content != "" {
				wireMsg.Content = msg.Content
			}
			for _, call := range msg.ToolCalls {
				wireMsg.ToolCalls = append(wireMsg.ToolCalls, openAIToolCall{
					ID:   call.ID,
					Type: "function",
					Function: openAIFunctionCall{
						Name:      call.Name,
						Arguments: string(normalizeArguments(call.Arguments)),
					},
				})
			}
			wireMessages = append(wireMessages, wireMsg)
		case llm.RoleTool:
			content := msg.Content
			if content == "" && len(msg.Data) > 0 {
				content = string(msg.Data)
			}
			wireMessages = append(wireMessages, openAIMessage{
				Role:       "tool",
				Content:    content,
				Name:       msg.Name,
				ToolCallID: msg.ToolCallID,
			})
		}
	}

	return wireMessages
}

func normalizeArguments(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return json.RawMessage(`{}`)
	}
	if json.Valid(raw) {
		return raw
	}
	return json.RawMessage(`{}`)
}

func normalizeArgumentString(raw string) json.RawMessage {
	if strings.TrimSpace(raw) == "" {
		return json.RawMessage(`{}`)
	}

	data := []byte(raw)
	if json.Valid(data) {
		return data
	}

	return json.RawMessage(`{}`)
}

func decodeOpenAIText(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}

	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text
	}

	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err == nil {
		var builder strings.Builder
		for _, part := range parts {
			if part.Type == "text" {
				builder.WriteString(part.Text)
			}
		}
		return builder.String()
	}

	return string(raw)
}

type openAIRequest struct {
	Model               string          `json:"model"`
	Messages            []openAIMessage `json:"messages"`
	Tools               []openAITool    `json:"tools,omitempty"`
	ToolChoice          any             `json:"tool_choice,omitempty"`
	Temperature         *float64        `json:"temperature,omitempty"`
	MaxCompletionTokens int             `json:"max_completion_tokens,omitempty"`
}

type openAIMessage struct {
	Role       string           `json:"role"`
	Content    any              `json:"content,omitempty"`
	Name       string           `json:"name,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
	ToolCalls  []openAIToolCall `json:"tool_calls,omitempty"`
}

type openAITool struct {
	Type     string         `json:"type"`
	Function openAIFunction `json:"function"`
}

type openAIFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
	Strict      bool           `json:"strict,omitempty"`
}

type openAIToolCall struct {
	ID       string             `json:"id,omitempty"`
	Type     string             `json:"type"`
	Function openAIFunctionCall `json:"function"`
}

type openAIFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type openAIResponse struct {
	Choices []struct {
		Message struct {
			Content   json.RawMessage  `json:"content"`
			ToolCalls []openAIToolCall `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}
