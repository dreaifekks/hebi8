package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/dreaifekks/hebi8/llm"
	"github.com/dreaifekks/hebi8/skill"
)

var ErrMaxStepsExceeded = errors.New("agent reached the maximum number of reasoning steps")

const (
	DefaultEndSignal    = "<Hebi8End />"
	defaultRoundTimeout = 60 * time.Second
)

type Config struct {
	Provider     llm.Provider
	Tools        *skill.Registry
	SystemPrompt string
	Model        string
	MaxSteps     int
	Temperature  *float64
	MaxTokens    int
	RoundTimeout time.Duration
	EndSignal    string
}

type Agent struct {
	provider     llm.Provider
	tools        *skill.Registry
	systemPrompt string
	model        string
	maxSteps     int
	temperature  *float64
	maxTokens    int
	roundTimeout time.Duration
	endSignal    string
}

type Result struct {
	Message        string
	Transcript     []llm.Message
	Steps          int
	Usage          llm.Usage
	Completed      bool
	CompletionText string
}

func New(cfg Config) (*Agent, error) {
	if cfg.Provider == nil {
		return nil, errors.New("agent provider is required")
	}

	tools := cfg.Tools
	if tools == nil {
		tools = skill.NewRegistry()
	}

	maxSteps := cfg.MaxSteps
	if maxSteps <= 0 {
		maxSteps = 8
	}

	roundTimeout := cfg.RoundTimeout
	if roundTimeout <= 0 {
		roundTimeout = defaultRoundTimeout
	}

	endSignal := strings.TrimSpace(cfg.EndSignal)
	if endSignal == "" {
		endSignal = DefaultEndSignal
	}

	systemPrompt := cfg.SystemPrompt
	if strings.TrimSpace(systemPrompt) == "" {
		systemPrompt = defaultSystemPrompt(endSignal)
	}

	return &Agent{
		provider:     cfg.Provider,
		tools:        tools,
		systemPrompt: systemPrompt,
		model:        cfg.Model,
		maxSteps:     maxSteps,
		temperature:  cfg.Temperature,
		maxTokens:    cfg.MaxTokens,
		roundTimeout: roundTimeout,
		endSignal:    endSignal,
	}, nil
}

func (a *Agent) Run(ctx context.Context, prompt string) (Result, error) {
	return a.RunWithMessages(ctx, []llm.Message{
		{Role: llm.RoleUser, Content: prompt},
	})
}

func (a *Agent) RunWithMessages(ctx context.Context, messages []llm.Message) (Result, error) {
	history := append([]llm.Message(nil), messages...)
	usage := llm.Usage{}

	for step := 1; step <= a.maxSteps; step++ {
		if err := ctx.Err(); err != nil {
			return Result{Transcript: history, Steps: step - 1, Usage: usage}, err
		}

		resp, err := a.generateRound(ctx, history)
		if err != nil {
			if ctx.Err() != nil {
				return Result{Transcript: history, Steps: step - 1, Usage: usage}, ctx.Err()
			}

			history = append(history, llmErrorObservation(step, err, a.endSignal))
			continue
		}

		usage = usage.Add(resp.Usage)
		history = append(history, resp.Message)

		if len(resp.Message.ToolCalls) > 0 {
			history = append(history, a.tools.ExecuteCalls(ctx, resp.Message.ToolCalls)...)
			continue
		}

		if a.isCompletionSignal(resp) {
			return Result{
				Message:        a.endSignal,
				Transcript:     history,
				Steps:          step,
				Usage:          usage,
				Completed:      true,
				CompletionText: a.endSignal,
			}, nil
		}

		history = append(history, protocolViolationObservation(step, resp, a.endSignal))
	}

	return Result{
		Transcript: history,
		Steps:      a.maxSteps,
		Usage:      usage,
	}, ErrMaxStepsExceeded
}

func DefaultSystemPrompt() string {
	return defaultSystemPrompt(DefaultEndSignal)
}

func defaultSystemPrompt(endSignal string) string {
	return fmt.Sprintf(
		"You are a ReAct-style agent. Think through the task, call tools when they are useful, and never invent tool results.\n"+
			"Completion protocol:\n"+
			"- Only reply with %s when every requested task is fully complete.\n"+
			"- When you emit %s, it must be the entire assistant response with no extra text.\n"+
			"- If work remains, do not emit %s.\n"+
			"- If a SYSTEM OBSERVATION message reports a timeout, provider error, or protocol violation, use it to continue the task.\n",
		endSignal,
		endSignal,
		endSignal,
	)
}

func (a *Agent) generateRound(ctx context.Context, history []llm.Message) (llm.Response, error) {
	roundCtx := ctx
	cancel := func() {}
	if a.roundTimeout > 0 {
		roundCtx, cancel = context.WithTimeout(ctx, a.roundTimeout)
	}
	defer cancel()

	return a.provider.Generate(roundCtx, llm.Request{
		Model:       a.model,
		System:      a.systemPrompt,
		Messages:    history,
		Tools:       a.tools.Definitions(),
		Temperature: a.temperature,
		MaxTokens:   a.maxTokens,
	})
}

func (a *Agent) isCompletionSignal(resp llm.Response) bool {
	if strings.TrimSpace(resp.Message.Content) != a.endSignal {
		return false
	}

	switch normalizeStopReason(resp.StopReason) {
	case "", "stop", "end_turn":
		return true
	default:
		return false
	}
}

func llmErrorObservation(step int, err error, endSignal string) llm.Message {
	return llm.Message{
		Role: llm.RoleUser,
		Content: fmt.Sprintf(
			"SYSTEM OBSERVATION: round %d failed before producing a valid assistant message.\n"+
				"error=%s\n"+
				"Continue the task from the existing transcript. Only reply with %s when all tasks are fully complete.",
			step,
			err.Error(),
			endSignal,
		),
	}
}

func protocolViolationObservation(step int, resp llm.Response, endSignal string) llm.Message {
	content := truncateForObservation(strings.TrimSpace(resp.Message.Content), 400)
	stopReason := resp.StopReason
	if strings.TrimSpace(stopReason) == "" {
		stopReason = "<empty>"
	}

	return llm.Message{
		Role: llm.RoleUser,
		Content: fmt.Sprintf(
			"SYSTEM OBSERVATION: round %d did not satisfy the completion protocol.\n"+
				"stop_reason=%s\n"+
				"assistant_content=%q\n"+
				"The assistant must either call tools or reply with %s and nothing else once all tasks are fully complete. Continue the task.",
			step,
			stopReason,
			content,
			endSignal,
		),
	}
}

func normalizeStopReason(reason string) string {
	return strings.ToLower(strings.TrimSpace(reason))
}

func truncateForObservation(value string, limit int) string {
	if limit <= 0 || len(value) <= limit {
		return value
	}
	if limit <= 3 {
		return value[:limit]
	}
	return value[:limit-3] + "..."
}
