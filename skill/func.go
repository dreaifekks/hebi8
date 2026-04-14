package skill

import (
	"context"
	"encoding/json"

	"github.com/dreaifekks/hebi8/llm"
)

type Func func(ctx context.Context, input json.RawMessage) (any, error)

type FuncSkill struct {
	definition llm.ToolDefinition
	run        Func
}

func NewFuncSkill(definition llm.ToolDefinition, run Func) *FuncSkill {
	return &FuncSkill{
		definition: definition,
		run:        run,
	}
}

func (s *FuncSkill) Definition() llm.ToolDefinition {
	return s.definition
}

func (s *FuncSkill) Execute(ctx context.Context, input json.RawMessage) (any, error) {
	return s.run(ctx, input)
}
