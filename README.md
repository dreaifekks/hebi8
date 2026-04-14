# hebi8

一个用 Go 写的轻量 ReAct Agent 骨架，支持：

- ReAct 式的 think -> act -> observe 循环
- 显式终止信号协议，只有 `<Hebi8End />` 才视为完成
- 自定义 skill 注册与调用
- 内置 shell / 代码执行能力
- OpenAI / Claude / Gemini 三种主流工具调用接口格式
- 纯标准库实现，不依赖第三方 SDK

## 项目结构

```text
.
├── agent/              # ReAct 主循环
├── llm/                # Provider-neutral 消息和工具协议
├── provider/
│   ├── claude/         # Anthropic Messages API 适配
│   ├── gemini/         # Gemini generateContent 适配
│   └── openai/         # OpenAI Chat Completions 适配
├── skill/              # Skill 注册、函数技能、shell 执行技能
└── cmd/hebi8/          # 最小 CLI 示例
```

## 快速开始

```bash
go test ./...
go run ./cmd/hebi8 -provider openai -model gpt-4.1-mini "列出当前目录并解释这个项目结构"
```

如果想让 agent 能执行命令，CLI 默认会注册 `run_command` skill。你也可以显式指定工作目录：

```bash
go run ./cmd/hebi8 -provider claude -workdir . "查看 go.mod 并总结项目的包设计"
```

## 环境变量

### OpenAI

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1
```

### Claude / Anthropic

```bash
export ANTHROPIC_API_KEY=...
export CLAUDE_BASE_URL=https://api.anthropic.com/v1
```

### Gemini

```bash
export GEMINI_API_KEY=...
export GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
```

## 作为库使用

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/dreaifekks/hebi8/agent"
	"github.com/dreaifekks/hebi8/llm"
	"github.com/dreaifekks/hebi8/provider/openai"
	"github.com/dreaifekks/hebi8/skill"
)

func main() {
	registry := skill.NewRegistry()
	registry.MustRegister(skill.NewFuncSkill(
		llm.ToolDefinition{
			Name:        "lookup_ticket",
			Description: "Lookup an internal ticket by id.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"id": map[string]any{"type": "string"},
				},
				"required": []string{"id"},
			},
		},
		func(_ context.Context, input json.RawMessage) (any, error) {
			return map[string]any{"ticket": "demo"}, nil
		},
	))

	registry.MustRegister(skill.NewShellSkill(skill.ShellConfig{
		Workdir: ".",
	}))

	provider := openai.New(openai.Config{
		APIKey: "your-api-key",
		Model:  "gpt-4.1-mini",
	})

	a, _ := agent.New(agent.Config{
		Provider:     provider,
		Tools:        registry,
		SystemPrompt: agent.DefaultSystemPrompt(),
		MaxSteps:     8,
	})

	result, _ := a.Run(context.Background(), "先查工单，再给我一段总结")
	fmt.Println(result.Message)
}
```

## 设计说明

### 1. Provider-neutral transcript

内部统一使用 `llm.Message` 表示对话，再在 provider 边界上转成各自的 wire format：

- OpenAI: `messages` + `tools` + `tool_calls`
- Claude: `messages[].content[]` + `tool_use` / `tool_result`
- Gemini: `contents[].parts[]` + `functionCall` / `functionResponse`

### 1.1 Completion protocol

- assistant 只有在所有任务都完成时，才允许返回纯 `<Hebi8End />`
- 任何非终止文本都不会被当作完成，而会被追加成下一轮可见的 `SYSTEM OBSERVATION`
- 每轮 LLM 请求默认 60s 超时；provider 错误或超时不会直接中断，而会作为 observation 传到下一轮
- `StopReason` 是截断类状态（如 `length` / `max_tokens`）时，即使内容是 `<Hebi8End />` 也不会结束

### 2. Skill 机制

`skill.Registry` 负责：

- 维护可调用技能列表
- 输出 provider 所需的工具定义
- 执行模型返回的 tool call
- 把结果封装回统一 transcript

### 3. 代码执行

内置 `skill.NewShellSkill(...)`，特性包括：

- 限制最大执行时长
- 限制工作目录范围
- 限制输出长度
- 返回结构化的 `stdout` / `stderr` / `exit_code`

## 后续可扩展点

- 加入流式输出
- 支持并发 tool 调用调度
- 增加 memory / planner / RAG 层
- 暴露 HTTP API，把这个 agent 服务化
