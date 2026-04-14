package skill

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/dreaifekks/hebi8/llm"
)

type ShellConfig struct {
	Name           string
	Description    string
	Shell          string
	Workdir        string
	Env            []string
	DefaultTimeout time.Duration
	MaxTimeout     time.Duration
	MaxOutputBytes int
}

type ShellSkill struct {
	definition llm.ToolDefinition
	config     ShellConfig
}

type shellRequest struct {
	Command    string `json:"command"`
	Workdir    string `json:"workdir,omitempty"`
	TimeoutSec int    `json:"timeout_sec,omitempty"`
}

type ShellResult struct {
	Command    string `json:"command"`
	Workdir    string `json:"workdir"`
	ExitCode   int    `json:"exit_code"`
	Stdout     string `json:"stdout,omitempty"`
	Stderr     string `json:"stderr,omitempty"`
	DurationMs int64  `json:"duration_ms"`
	TimedOut   bool   `json:"timed_out,omitempty"`
}

func NewShellSkill(cfg ShellConfig) *ShellSkill {
	if cfg.Name == "" {
		cfg.Name = "run_command"
	}
	if cfg.Description == "" {
		cfg.Description = "Execute a shell command inside the configured working directory and return stdout, stderr, exit code, and timing."
	}
	if cfg.Shell == "" {
		cfg.Shell = "/bin/sh"
	}
	if cfg.DefaultTimeout <= 0 {
		cfg.DefaultTimeout = 15 * time.Second
	}
	if cfg.MaxTimeout <= 0 {
		cfg.MaxTimeout = 60 * time.Second
	}
	if cfg.MaxOutputBytes <= 0 {
		cfg.MaxOutputBytes = 16 * 1024
	}

	return &ShellSkill{
		definition: llm.ToolDefinition{
			Name:        cfg.Name,
			Description: cfg.Description,
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"command": map[string]any{
						"type":        "string",
						"description": "Shell command to execute.",
					},
					"workdir": map[string]any{
						"type":        "string",
						"description": "Optional working directory. Relative paths are resolved under the configured root.",
					},
					"timeout_sec": map[string]any{
						"type":        "integer",
						"description": "Optional timeout in seconds. Values above the configured maximum are rejected.",
					},
				},
				"required": []string{"command"},
			},
		},
		config: cfg,
	}
}

func (s *ShellSkill) Definition() llm.ToolDefinition {
	return s.definition
}

func (s *ShellSkill) Execute(ctx context.Context, input json.RawMessage) (any, error) {
	var req shellRequest
	if err := json.Unmarshal(input, &req); err != nil {
		return nil, fmt.Errorf("decode shell input: %w", err)
	}
	if strings.TrimSpace(req.Command) == "" {
		return nil, errors.New("command cannot be empty")
	}

	workdir, err := resolveWorkdir(s.config.Workdir, req.Workdir)
	if err != nil {
		return nil, err
	}

	timeout := s.config.DefaultTimeout
	if req.TimeoutSec > 0 {
		timeout = time.Duration(req.TimeoutSec) * time.Second
	}
	if timeout > s.config.MaxTimeout {
		return nil, fmt.Errorf("requested timeout %s exceeds max timeout %s", timeout, s.config.MaxTimeout)
	}

	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	start := time.Now()
	cmd := exec.CommandContext(runCtx, s.config.Shell, "-lc", req.Command)
	cmd.Dir = workdir
	cmd.Env = append(os.Environ(), s.config.Env...)

	stdout := newLimitedBuffer(s.config.MaxOutputBytes)
	stderr := newLimitedBuffer(s.config.MaxOutputBytes)
	cmd.Stdout = stdout
	cmd.Stderr = stderr

	err = cmd.Run()
	result := ShellResult{
		Command:    req.Command,
		Workdir:    workdir,
		Stdout:     stdout.String(),
		Stderr:     stderr.String(),
		DurationMs: time.Since(start).Milliseconds(),
	}

	switch {
	case err == nil:
		result.ExitCode = 0
		return result, nil
	case errors.Is(runCtx.Err(), context.DeadlineExceeded):
		result.ExitCode = -1
		result.TimedOut = true
		return result, nil
	default:
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			result.ExitCode = exitErr.ExitCode()
			return result, nil
		}
		return nil, fmt.Errorf("run command: %w", err)
	}
}

func resolveWorkdir(root, requested string) (string, error) {
	if root == "" && requested == "" {
		return os.Getwd()
	}

	if root == "" {
		return filepath.Clean(requested), nil
	}

	root = filepath.Clean(root)
	if requested == "" {
		return root, nil
	}

	var candidate string
	if filepath.IsAbs(requested) {
		candidate = filepath.Clean(requested)
	} else {
		candidate = filepath.Join(root, requested)
	}

	rel, err := filepath.Rel(root, candidate)
	if err != nil {
		return "", fmt.Errorf("resolve workdir: %w", err)
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("requested workdir %q escapes configured root %q", candidate, root)
	}

	return candidate, nil
}

type limitedBuffer struct {
	buf       bytes.Buffer
	max       int
	truncated bool
}

func newLimitedBuffer(max int) *limitedBuffer {
	return &limitedBuffer{max: max}
}

func (b *limitedBuffer) Write(p []byte) (int, error) {
	if b.max <= 0 {
		return len(p), nil
	}

	remaining := b.max - b.buf.Len()
	if remaining <= 0 {
		b.truncated = true
		return len(p), nil
	}

	if len(p) > remaining {
		_, _ = b.buf.Write(p[:remaining])
		b.truncated = true
		return len(p), nil
	}

	return b.buf.Write(p)
}

func (b *limitedBuffer) String() string {
	if !b.truncated {
		return b.buf.String()
	}
	return b.buf.String() + "\n[output truncated]"
}
