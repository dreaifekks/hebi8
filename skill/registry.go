package skill

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/dreaifekks/hebi8/llm"
)

type Skill interface {
	Definition() llm.ToolDefinition
	Execute(ctx context.Context, input json.RawMessage) (any, error)
}

type Registry struct {
	order  []string
	skills map[string]Skill
}

func NewRegistry() *Registry {
	return &Registry{
		order:  make([]string, 0),
		skills: make(map[string]Skill),
	}
}

func (r *Registry) Register(skill Skill) error {
	def := skill.Definition()
	if def.Name == "" {
		return fmt.Errorf("skill name cannot be empty")
	}
	if _, exists := r.skills[def.Name]; exists {
		return fmt.Errorf("skill %q is already registered", def.Name)
	}
	r.skills[def.Name] = skill
	r.order = append(r.order, def.Name)
	return nil
}

func (r *Registry) MustRegister(skill Skill) {
	if err := r.Register(skill); err != nil {
		panic(err)
	}
}

func (r *Registry) Definitions() []llm.ToolDefinition {
	definitions := make([]llm.ToolDefinition, 0, len(r.order))
	for _, name := range r.order {
		definitions = append(definitions, r.skills[name].Definition())
	}
	return definitions
}

func (r *Registry) ExecuteCalls(ctx context.Context, calls []llm.ToolCall) []llm.Message {
	results := make([]llm.Message, 0, len(calls))
	for _, call := range calls {
		skill, ok := r.skills[call.Name]
		if !ok {
			results = append(results, toolErrorMessage(call, fmt.Errorf("unknown skill %q", call.Name)))
			continue
		}

		value, err := skill.Execute(ctx, call.Arguments)
		if err != nil {
			results = append(results, toolErrorMessage(call, err))
			continue
		}

		results = append(results, toolSuccessMessage(call, value))
	}
	return results
}

func (r *Registry) Len() int {
	return len(r.order)
}

func toolSuccessMessage(call llm.ToolCall, value any) llm.Message {
	msg := llm.Message{
		Role:       llm.RoleTool,
		Name:       call.Name,
		ToolCallID: call.ID,
	}

	switch v := value.(type) {
	case nil:
		return msg
	case string:
		msg.Content = v
		return msg
	case []byte:
		msg.Content = string(v)
		return msg
	case json.RawMessage:
		msg.Content = string(v)
		msg.Data = v
		return msg
	default:
		raw, err := json.Marshal(v)
		if err != nil {
			return toolErrorMessage(call, fmt.Errorf("encode skill result: %w", err))
		}
		msg.Content = string(raw)
		msg.Data = raw
		return msg
	}
}

func toolErrorMessage(call llm.ToolCall, err error) llm.Message {
	payload, _ := json.Marshal(map[string]any{
		"error": err.Error(),
	})
	return llm.Message{
		Role:       llm.RoleTool,
		Content:    err.Error(),
		Data:       payload,
		Name:       call.Name,
		ToolCallID: call.ID,
		IsError:    true,
	}
}
