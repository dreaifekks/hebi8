package agent_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/dreaifekks/hebi8/agent"
	"github.com/dreaifekks/hebi8/llm"
	"github.com/dreaifekks/hebi8/skill"
)

type stubProvider struct {
	calls int
}

func (s *stubProvider) Generate(_ context.Context, req llm.Request) (llm.Response, error) {
	switch s.calls {
	case 0:
		s.calls++
		return llm.Response{
			Message: llm.Message{
				Role: llm.RoleAssistant,
				ToolCalls: []llm.ToolCall{
					{
						ID:        "call_1",
						Name:      "echo",
						Arguments: json.RawMessage(`{"text":"ping"}`),
					},
				},
			},
		}, nil
	case 1:
		s.calls++
		if len(req.Messages) < 3 {
			t := testing.T{}
			t.Fatalf("expected tool result in transcript, got %d messages", len(req.Messages))
		}
		return llm.Response{
			Message: llm.Message{
				Role:    llm.RoleAssistant,
				Content: "pong",
			},
		}, nil
	default:
		return llm.Response{}, nil
	}
}

func TestRunExecutesSkillCalls(t *testing.T) {
	registry := skill.NewRegistry()
	registry.MustRegister(skill.NewFuncSkill(
		llm.ToolDefinition{
			Name:        "echo",
			Description: "Echo input",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"text": map[string]any{"type": "string"},
				},
				"required": []string{"text"},
			},
		},
		func(_ context.Context, input json.RawMessage) (any, error) {
			var args struct {
				Text string `json:"text"`
			}
			if err := json.Unmarshal(input, &args); err != nil {
				return nil, err
			}
			return map[string]any{"echo": args.Text}, nil
		},
	))

	a, err := agent.New(agent.Config{
		Provider: &stubProvider{},
		Tools:    registry,
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	result, err := a.Run(context.Background(), "say hello")
	if err != nil {
		t.Fatalf("run agent: %v", err)
	}

	if result.Message != "pong" {
		t.Fatalf("unexpected final message: %q", result.Message)
	}
	if result.Steps != 2 {
		t.Fatalf("unexpected steps: %d", result.Steps)
	}
	if len(result.Transcript) != 4 {
		t.Fatalf("unexpected transcript length: %d", len(result.Transcript))
	}
	if result.Transcript[2].Role != llm.RoleTool {
		t.Fatalf("expected third message to be tool result, got %s", result.Transcript[2].Role)
	}
}
