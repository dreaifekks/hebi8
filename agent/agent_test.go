package agent_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/dreaifekks/hebi8/agent"
	"github.com/dreaifekks/hebi8/llm"
	"github.com/dreaifekks/hebi8/skill"
)

type scriptedProvider struct {
	t     *testing.T
	calls int
	steps []func(context.Context, llm.Request) (llm.Response, error)
}

func (s *scriptedProvider) Generate(ctx context.Context, req llm.Request) (llm.Response, error) {
	if s.calls >= len(s.steps) {
		s.t.Fatalf("unexpected provider call %d", s.calls+1)
	}

	step := s.steps[s.calls]
	s.calls++
	return step(ctx, req)
}

func TestRunExecutesSkillCallsUntilEndSignal(t *testing.T) {
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

	provider := &scriptedProvider{
		t: t,
		steps: []func(context.Context, llm.Request) (llm.Response, error){
			func(_ context.Context, _ llm.Request) (llm.Response, error) {
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
					StopReason: "tool_calls",
				}, nil
			},
			func(_ context.Context, req llm.Request) (llm.Response, error) {
				if len(req.Messages) < 3 {
					t.Fatalf("expected tool result in transcript, got %d messages", len(req.Messages))
				}
				if req.Messages[2].Role != llm.RoleTool {
					t.Fatalf("expected third message to be tool result, got %s", req.Messages[2].Role)
				}
				return llm.Response{
					Message: llm.Message{
						Role:    llm.RoleAssistant,
						Content: agent.DefaultEndSignal,
					},
					StopReason: "stop",
				}, nil
			},
		},
	}

	a, err := agent.New(agent.Config{
		Provider: provider,
		Tools:    registry,
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	result, err := a.Run(context.Background(), "say hello")
	if err != nil {
		t.Fatalf("run agent: %v", err)
	}

	if !result.Completed {
		t.Fatalf("expected completed result")
	}
	if result.CompletionText != agent.DefaultEndSignal {
		t.Fatalf("unexpected completion text: %q", result.CompletionText)
	}
	if result.Steps != 2 {
		t.Fatalf("unexpected steps: %d", result.Steps)
	}
	if len(result.Transcript) != 4 {
		t.Fatalf("unexpected transcript length: %d", len(result.Transcript))
	}
}

func TestRunRequiresExactEndSignalToTerminate(t *testing.T) {
	provider := &scriptedProvider{
		t: t,
		steps: []func(context.Context, llm.Request) (llm.Response, error){
			func(_ context.Context, _ llm.Request) (llm.Response, error) {
				return llm.Response{
					Message: llm.Message{
						Role:    llm.RoleAssistant,
						Content: "all set",
					},
					StopReason: "stop",
				}, nil
			},
			func(_ context.Context, req llm.Request) (llm.Response, error) {
				last := req.Messages[len(req.Messages)-1]
				if last.Role != llm.RoleUser {
					t.Fatalf("expected protocol observation as last message, got %s", last.Role)
				}
				if !strings.Contains(last.Content, "did not satisfy the completion protocol") {
					t.Fatalf("expected protocol violation message, got %q", last.Content)
				}
				if !strings.Contains(last.Content, agent.DefaultEndSignal) {
					t.Fatalf("expected end signal reminder, got %q", last.Content)
				}
				return llm.Response{
					Message: llm.Message{
						Role:    llm.RoleAssistant,
						Content: agent.DefaultEndSignal,
					},
					StopReason: "stop",
				}, nil
			},
		},
	}

	a, err := agent.New(agent.Config{Provider: provider})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	result, err := a.Run(context.Background(), "finish the task")
	if err != nil {
		t.Fatalf("run agent: %v", err)
	}

	if !result.Completed {
		t.Fatalf("expected completed result")
	}
	if result.Steps != 2 {
		t.Fatalf("unexpected steps: %d", result.Steps)
	}
}

func TestRunCarriesProviderErrorForward(t *testing.T) {
	provider := &scriptedProvider{
		t: t,
		steps: []func(context.Context, llm.Request) (llm.Response, error){
			func(_ context.Context, _ llm.Request) (llm.Response, error) {
				return llm.Response{}, errors.New("temporary upstream failure")
			},
			func(_ context.Context, req llm.Request) (llm.Response, error) {
				last := req.Messages[len(req.Messages)-1]
				if last.Role != llm.RoleUser {
					t.Fatalf("expected provider error observation as last message, got %s", last.Role)
				}
				if !strings.Contains(last.Content, "temporary upstream failure") {
					t.Fatalf("expected provider error details, got %q", last.Content)
				}
				return llm.Response{
					Message: llm.Message{
						Role:    llm.RoleAssistant,
						Content: agent.DefaultEndSignal,
					},
					StopReason: "stop",
				}, nil
			},
		},
	}

	a, err := agent.New(agent.Config{Provider: provider})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	result, err := a.Run(context.Background(), "recover from errors")
	if err != nil {
		t.Fatalf("run agent: %v", err)
	}

	if !result.Completed {
		t.Fatalf("expected completed result")
	}
	if result.Steps != 2 {
		t.Fatalf("unexpected steps: %d", result.Steps)
	}
}

func TestRunCarriesRoundTimeoutForward(t *testing.T) {
	provider := &scriptedProvider{
		t: t,
		steps: []func(context.Context, llm.Request) (llm.Response, error){
			func(ctx context.Context, _ llm.Request) (llm.Response, error) {
				<-ctx.Done()
				return llm.Response{}, ctx.Err()
			},
			func(_ context.Context, req llm.Request) (llm.Response, error) {
				last := req.Messages[len(req.Messages)-1]
				if last.Role != llm.RoleUser {
					t.Fatalf("expected timeout observation as last message, got %s", last.Role)
				}
				if !strings.Contains(last.Content, "context deadline exceeded") {
					t.Fatalf("expected timeout details, got %q", last.Content)
				}
				return llm.Response{
					Message: llm.Message{
						Role:    llm.RoleAssistant,
						Content: agent.DefaultEndSignal,
					},
					StopReason: "stop",
				}, nil
			},
		},
	}

	a, err := agent.New(agent.Config{
		Provider:     provider,
		RoundTimeout: 10 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	result, err := a.Run(context.Background(), "retry after timeout")
	if err != nil {
		t.Fatalf("run agent: %v", err)
	}

	if !result.Completed {
		t.Fatalf("expected completed result")
	}
	if result.Steps != 2 {
		t.Fatalf("unexpected steps: %d", result.Steps)
	}
}

func TestRunRejectsEndSignalWithIncompleteStopReason(t *testing.T) {
	provider := &scriptedProvider{
		t: t,
		steps: []func(context.Context, llm.Request) (llm.Response, error){
			func(_ context.Context, _ llm.Request) (llm.Response, error) {
				return llm.Response{
					Message: llm.Message{
						Role:    llm.RoleAssistant,
						Content: agent.DefaultEndSignal,
					},
					StopReason: "length",
				}, nil
			},
			func(_ context.Context, req llm.Request) (llm.Response, error) {
				last := req.Messages[len(req.Messages)-1]
				if !strings.Contains(last.Content, "stop_reason=length") {
					t.Fatalf("expected stop reason in observation, got %q", last.Content)
				}
				return llm.Response{
					Message: llm.Message{
						Role:    llm.RoleAssistant,
						Content: agent.DefaultEndSignal,
					},
					StopReason: "stop",
				}, nil
			},
		},
	}

	a, err := agent.New(agent.Config{Provider: provider})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	result, err := a.Run(context.Background(), "do not stop on truncated sentinel")
	if err != nil {
		t.Fatalf("run agent: %v", err)
	}

	if !result.Completed {
		t.Fatalf("expected completed result")
	}
	if result.Steps != 2 {
		t.Fatalf("unexpected steps: %d", result.Steps)
	}
}
