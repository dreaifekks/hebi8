package gemini

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
		baseURL = "https://generativelanguage.googleapis.com/v1beta"
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
		return llm.Response{}, fmt.Errorf("gemini model is required")
	}

	wireReq := geminiRequest{
		Contents: buildContents(req.Messages),
	}
	if req.System != "" {
		wireReq.SystemInstruction = &geminiContent{
			Parts: []geminiPart{{Text: req.System}},
		}
	}
	if req.Temperature != nil || req.MaxTokens > 0 {
		wireReq.GenerationConfig = &geminiGenerationConfig{}
		if req.Temperature != nil {
			wireReq.GenerationConfig.Temperature = *req.Temperature
		}
		if req.MaxTokens > 0 {
			wireReq.GenerationConfig.MaxOutputTokens = req.MaxTokens
		}
	}
	if len(req.Tools) > 0 {
		declarations := make([]geminiFunctionDeclaration, 0, len(req.Tools))
		for _, tool := range req.Tools {
			declarations = append(declarations, geminiFunctionDeclaration{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.Parameters,
			})
		}
		wireReq.Tools = []geminiTool{{FunctionDeclarations: declarations}}
		wireReq.ToolConfig = &geminiToolConfig{
			FunctionCallingConfig: geminiFunctionCallingConfig{Mode: "AUTO"},
		}
	}

	body, err := json.Marshal(wireReq)
	if err != nil {
		return llm.Response{}, fmt.Errorf("marshal gemini request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpointURL(model), bytes.NewReader(body))
	if err != nil {
		return llm.Response{}, fmt.Errorf("build gemini request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		httpReq.Header.Set("x-goog-api-key", c.apiKey)
	}

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return llm.Response{}, fmt.Errorf("send gemini request: %w", err)
	}
	defer httpResp.Body.Close()

	payload, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return llm.Response{}, fmt.Errorf("read gemini response: %w", err)
	}
	if httpResp.StatusCode >= 300 {
		return llm.Response{}, fmt.Errorf("gemini api error (%d): %s", httpResp.StatusCode, strings.TrimSpace(string(payload)))
	}

	var wireResp geminiResponse
	if err := json.Unmarshal(payload, &wireResp); err != nil {
		return llm.Response{}, fmt.Errorf("decode gemini response: %w", err)
	}
	if len(wireResp.Candidates) == 0 {
		return llm.Response{}, fmt.Errorf("gemini response did not contain candidates")
	}

	message := llm.Message{Role: llm.RoleAssistant}
	for _, part := range wireResp.Candidates[0].Content.Parts {
		switch {
		case part.Text != "":
			message.Content += part.Text
		case part.FunctionCall != nil:
			args, _ := json.Marshal(part.FunctionCall.Args)
			message.ToolCalls = append(message.ToolCalls, llm.ToolCall{
				ID:        part.FunctionCall.ID,
				Name:      part.FunctionCall.Name,
				Arguments: normalizeArgs(args),
			})
		}
	}

	return llm.Response{
		Message:    message,
		StopReason: wireResp.Candidates[0].FinishReason,
		Usage: llm.Usage{
			InputTokens:  wireResp.UsageMetadata.PromptTokenCount,
			OutputTokens: wireResp.UsageMetadata.CandidatesTokenCount,
			TotalTokens:  wireResp.UsageMetadata.TotalTokenCount,
		},
	}, nil
}

func (c *Client) endpointURL(model string) string {
	modelPath := model
	if !strings.HasPrefix(modelPath, "models/") {
		modelPath = "models/" + modelPath
	}
	return c.baseURL + "/" + modelPath + ":generateContent"
}

func buildContents(messages []llm.Message) []geminiContent {
	contents := make([]geminiContent, 0, len(messages))

	for i := 0; i < len(messages); {
		msg := messages[i]
		switch msg.Role {
		case llm.RoleUser:
			parts := []geminiPart{}
			if msg.Content != "" {
				parts = append(parts, geminiPart{Text: msg.Content})
			}
			if len(parts) > 0 {
				contents = append(contents, geminiContent{
					Role:  "user",
					Parts: parts,
				})
			}
			i++
		case llm.RoleAssistant:
			parts := make([]geminiPart, 0, len(msg.ToolCalls)+1)
			if msg.Content != "" {
				parts = append(parts, geminiPart{Text: msg.Content})
			}
			for _, call := range msg.ToolCalls {
				parts = append(parts, geminiPart{
					FunctionCall: &geminiFunctionCall{
						Name: call.Name,
						Args: decodeJSONValue(call.Arguments),
						ID:   call.ID,
					},
				})
			}
			if len(parts) > 0 {
				contents = append(contents, geminiContent{
					Role:  "model",
					Parts: parts,
				})
			}
			i++
		case llm.RoleTool:
			parts := make([]geminiPart, 0)
			for i < len(messages) && messages[i].Role == llm.RoleTool {
				toolMsg := messages[i]
				parts = append(parts, geminiPart{
					FunctionResponse: &geminiFunctionResponse{
						Name:     toolMsg.Name,
						ID:       toolMsg.ToolCallID,
						Response: toolResponsePayload(toolMsg),
					},
				})
				i++
			}
			if i < len(messages) && messages[i].Role == llm.RoleUser && messages[i].Content != "" {
				parts = append(parts, geminiPart{Text: messages[i].Content})
				i++
			}
			if len(parts) > 0 {
				contents = append(contents, geminiContent{
					Role:  "user",
					Parts: parts,
				})
			}
		default:
			i++
		}
	}

	return contents
}

func decodeJSONValue(raw json.RawMessage) any {
	if len(raw) == 0 || !json.Valid(raw) {
		return map[string]any{}
	}
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return map[string]any{}
	}
	return value
}

func normalizeArgs(raw []byte) json.RawMessage {
	if len(raw) == 0 || !json.Valid(raw) {
		return json.RawMessage(`{}`)
	}
	return raw
}

func toolResponsePayload(msg llm.Message) any {
	if len(msg.Data) > 0 && json.Valid(msg.Data) {
		var value any
		if err := json.Unmarshal(msg.Data, &value); err == nil {
			return value
		}
	}
	if msg.IsError {
		return map[string]any{"error": msg.Content}
	}
	if msg.Content != "" {
		return map[string]any{"result": msg.Content}
	}
	return map[string]any{}
}

type geminiRequest struct {
	SystemInstruction *geminiContent          `json:"systemInstruction,omitempty"`
	Contents          []geminiContent         `json:"contents"`
	Tools             []geminiTool            `json:"tools,omitempty"`
	ToolConfig        *geminiToolConfig       `json:"toolConfig,omitempty"`
	GenerationConfig  *geminiGenerationConfig `json:"generationConfig,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text             string                  `json:"text,omitempty"`
	FunctionCall     *geminiFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResponse `json:"functionResponse,omitempty"`
}

type geminiFunctionCall struct {
	Name string `json:"name"`
	Args any    `json:"args,omitempty"`
	ID   string `json:"id,omitempty"`
}

type geminiFunctionResponse struct {
	Name     string `json:"name"`
	ID       string `json:"id,omitempty"`
	Response any    `json:"response,omitempty"`
}

type geminiTool struct {
	FunctionDeclarations []geminiFunctionDeclaration `json:"functionDeclarations"`
}

type geminiFunctionDeclaration struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type geminiToolConfig struct {
	FunctionCallingConfig geminiFunctionCallingConfig `json:"functionCallingConfig"`
}

type geminiFunctionCallingConfig struct {
	Mode string `json:"mode"`
}

type geminiGenerationConfig struct {
	Temperature     float64 `json:"temperature,omitempty"`
	MaxOutputTokens int     `json:"maxOutputTokens,omitempty"`
}

type geminiResponse struct {
	Candidates []struct {
		Content      geminiContent `json:"content"`
		FinishReason string        `json:"finishReason"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
}
