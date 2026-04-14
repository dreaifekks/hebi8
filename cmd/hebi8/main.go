package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/dreaifekks/hebi8/agent"
	"github.com/dreaifekks/hebi8/llm"
	"github.com/dreaifekks/hebi8/provider/claude"
	"github.com/dreaifekks/hebi8/provider/gemini"
	"github.com/dreaifekks/hebi8/provider/openai"
	"github.com/dreaifekks/hebi8/skill"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run() error {
	providerName := flag.String("provider", envOr("HEBI8_PROVIDER", "openai"), "Provider name: openai, claude, gemini")
	model := flag.String("model", os.Getenv("HEBI8_MODEL"), "Model name")
	prompt := flag.String("prompt", "", "Prompt text. If empty, remaining args or stdin are used.")
	maxSteps := flag.Int("max-steps", 8, "Maximum ReAct iterations")
	maxTokens := flag.Int("max-tokens", 0, "Provider output token cap")
	timeout := flag.Duration("timeout", 10*time.Minute, "End-to-end agent timeout across all rounds")
	roundTimeout := flag.Duration("round-timeout", 60*time.Second, "Per-round LLM timeout")
	allowShell := flag.Bool("shell", true, "Register the built-in run_command skill")
	workdir := flag.String("workdir", ".", "Root working directory for run_command")
	verbose := flag.Bool("verbose", false, "Print transcript and tool activity to stderr")
	flag.Parse()

	input, err := resolvePrompt(*prompt, flag.Args(), os.Stdin)
	if err != nil {
		return err
	}

	if strings.TrimSpace(*model) == "" {
		*model = defaultModel(*providerName)
	}

	client, err := buildProvider(*providerName, *model)
	if err != nil {
		return err
	}

	registry := skill.NewRegistry()
	registry.MustRegister(skill.NewFuncSkill(
		llm.ToolDefinition{
			Name:        "current_time",
			Description: "Return the current local time in RFC3339 format.",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
		},
		func(_ context.Context, _ json.RawMessage) (any, error) {
			return map[string]any{
				"now": time.Now().Format(time.RFC3339),
			}, nil
		},
	))

	if *allowShell {
		registry.MustRegister(skill.NewShellSkill(skill.ShellConfig{
			Workdir:        *workdir,
			DefaultTimeout: 15 * time.Second,
			MaxTimeout:     60 * time.Second,
			MaxOutputBytes: 24 * 1024,
		}))
	}

	reactAgent, err := agent.New(agent.Config{
		Provider:     client,
		Tools:        registry,
		SystemPrompt: agent.DefaultSystemPrompt(),
		Model:        *model,
		MaxSteps:     *maxSteps,
		MaxTokens:    *maxTokens,
		RoundTimeout: *roundTimeout,
	})
	if err != nil {
		return err
	}

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()

	result, err := reactAgent.Run(ctx, input)
	if err != nil {
		return err
	}

	if *verbose {
		printTranscript(os.Stderr, result.Transcript)
		fmt.Fprintf(os.Stderr, "\nsteps=%d completed=%t completion_signal=%q tokens_in=%d tokens_out=%d total=%d\n\n",
			result.Steps,
			result.Completed,
			result.CompletionText,
			result.Usage.InputTokens,
			result.Usage.OutputTokens,
			result.Usage.TotalTokens,
		)
	}

	if result.Message != "" && result.Message != result.CompletionText {
		fmt.Println(result.Message)
	}
	return nil
}

func resolvePrompt(explicit string, args []string, stdin io.Reader) (string, error) {
	if strings.TrimSpace(explicit) != "" {
		return explicit, nil
	}
	if len(args) > 0 {
		return strings.Join(args, " "), nil
	}

	data, err := io.ReadAll(stdin)
	if err != nil {
		return "", fmt.Errorf("read stdin: %w", err)
	}
	text := strings.TrimSpace(string(data))
	if text == "" {
		return "", fmt.Errorf("prompt is required")
	}
	return text, nil
}

func buildProvider(name, model string) (llm.Provider, error) {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "openai":
		return openai.New(openai.Config{
			APIKey:  os.Getenv("OPENAI_API_KEY"),
			BaseURL: os.Getenv("OPENAI_BASE_URL"),
			Model:   model,
		}), nil
	case "claude", "anthropic":
		return claude.New(claude.Config{
			APIKey:  firstNonEmpty(os.Getenv("ANTHROPIC_API_KEY"), os.Getenv("CLAUDE_API_KEY")),
			BaseURL: os.Getenv("CLAUDE_BASE_URL"),
			Model:   model,
		}), nil
	case "gemini", "google":
		return gemini.New(gemini.Config{
			APIKey:  firstNonEmpty(os.Getenv("GEMINI_API_KEY"), os.Getenv("GOOGLE_API_KEY")),
			BaseURL: os.Getenv("GEMINI_BASE_URL"),
			Model:   model,
		}), nil
	default:
		return nil, fmt.Errorf("unsupported provider %q", name)
	}
}

func defaultModel(provider string) string {
	switch strings.ToLower(strings.TrimSpace(provider)) {
	case "claude", "anthropic":
		return "claude-sonnet-4-5"
	case "gemini", "google":
		return "gemini-2.5-flash"
	default:
		return "gpt-4.1-mini"
	}
}

func envOr(key, fallback string) string {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	return value
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func printTranscript(w io.Writer, transcript []llm.Message) {
	for _, msg := range transcript {
		switch msg.Role {
		case llm.RoleUser:
			fmt.Fprintf(w, "[user] %s\n", msg.Content)
		case llm.RoleAssistant:
			if msg.Content != "" {
				fmt.Fprintf(w, "[assistant] %s\n", msg.Content)
			}
			for _, call := range msg.ToolCalls {
				fmt.Fprintf(w, "[tool-call] %s %s\n", call.Name, string(call.Arguments))
			}
		case llm.RoleTool:
			fmt.Fprintf(w, "[tool-result] %s => %s\n", msg.Name, msg.Content)
		}
	}
}
