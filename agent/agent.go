package agent

import (
	"context"
	"errors"
	"fmt"

	"github.com/dreaifekks/hebi8/llm"
	"github.com/dreaifekks/hebi8/skill"
)

var ErrMaxStepsExceeded = errors.New("agent reached the maximum number of reasoning steps")

type Config struct {
	Provider     llm.Provider
	Tools        *skill.Registry
	SystemPrompt string
	Model        string
	MaxSteps     int
	Temperature  *float64
	MaxTokens    int
}

type Agent struct {
	provider     llm.Provider
	tools        *skill.Registry
	systemPrompt string
	model        string
	maxSteps     int
	temperature  *float64
	maxTokens    int
}

type Result struct {
	Message    string
	Transcript []llm.Message
	Steps      int
	Usage      llm.Usage
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

	return &Agent{
		provider:     cfg.Provider,
		tools:        tools,
		systemPrompt: cfg.SystemPrompt,
		model:        cfg.Model,
		maxSteps:     maxSteps,
		temperature:  cfg.Temperature,
		maxTokens:    cfg.MaxTokens,
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
		resp, err := a.provider.Generate(ctx, llm.Request{
			Model:       a.model,
			System:      a.systemPrompt,
			Messages:    history,
			Tools:       a.tools.Definitions(),
			Temperature: a.temperature,
			MaxTokens:   a.maxTokens,
		})
		if err != nil {
			return Result{Transcript: history, Steps: step - 1, Usage: usage}, err
		}

		usage = usage.Add(resp.Usage)
		history = append(history, resp.Message)

		if len(resp.Message.ToolCalls) == 0 {
			return Result{
				Message:    resp.Message.Content,
				Transcript: history,
				Steps:      step,
				Usage:      usage,
			}, nil
		}

		if a.tools.Len() == 0 {
			return Result{Transcript: history, Steps: step, Usage: usage}, fmt.Errorf(
				"provider requested %d tool call(s), but no skills are registered",
				len(resp.Message.ToolCalls),
			)
		}

		history = append(history, a.tools.ExecuteCalls(ctx, resp.Message.ToolCalls)...)
	}

	return Result{
		Transcript: history,
		Steps:      a.maxSteps,
		Usage:      usage,
	}, ErrMaxStepsExceeded
}

func DefaultSystemPrompt() string {
	return "You are a ReAct-style agent. Think through the task, call tools when they are useful, never invent tool results, and provide a final answer once you have enough information."
}
