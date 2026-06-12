# litellm-ainative

AINative provider for [litellm](https://github.com/BerriAI/litellm) — free Llama, Qwen, DeepSeek & Kimi models through litellm's unified interface.

## Install

```bash
pip install litellm-ainative
```

## Quick Start

```python
import litellm
from litellm_ainative import configure

# Auto-provisions a free API key (no signup required)
configure()

# Use any AINative model through litellm
response = litellm.completion(
    model="ainative/meta-llama/Llama-3.3-70B-Instruct",
    messages=[{"role": "user", "content": "Explain quantum computing in one paragraph."}],
)
print(response.choices[0].message.content)
```

## Available Models

All models are **free** with tool-calling support:

| Model | litellm Name |
|-------|-------------|
| Llama 3.3 70B | `ainative/meta-llama/Llama-3.3-70B-Instruct` |
| Llama 4 Scout 17B | `ainative/meta-llama/Llama-4-Scout-17B-16E-Instruct` |
| Qwen3 Coder Flash | `ainative/qwen3-coder-flash` |
| DeepSeek 4 Flash | `ainative/deepseek-4-flash` |
| Kimi K2 | `ainative/kimi-k2` |

## Configuration

### With an existing API key

```python
from litellm_ainative import configure

configure(api_key="your_ainative_api_key")
```

### With environment variable

```bash
export AINATIVE_API_KEY=your_key
```

```python
from litellm_ainative import configure
configure()
```

### Auto-provisioning (default)

If no key is found, `configure()` auto-provisions a temporary API key via
AINative's instant-db endpoint. The key is valid for 72 hours. Claim it for
permanent access at https://ainative.studio/claim.

```python
from litellm_ainative import configure
configure()  # auto-provisions if no key found
```

### Disable auto-provisioning

```python
from litellm_ainative import configure
configure(auto_provision=False)  # raises RuntimeError if no key
```

## Convenience Wrapper

For simpler usage without managing litellm directly:

```python
from litellm_ainative import configure
from litellm_ainative.provider import completion

configure()

response = completion(
    model="ainative/kimi-k2",
    messages=[{"role": "user", "content": "Write a haiku about code."}],
    temperature=0.7,
)
```

## Tool Calling

All models support OpenAI-compatible tool calling:

```python
import litellm
from litellm_ainative import configure

configure()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"],
            },
        },
    }
]

response = litellm.completion(
    model="ainative/qwen3-coder-flash",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
)
```

## How It Works

`litellm-ainative` registers AINative models with litellm's model registry
using the `openai` provider backend (since AINative exposes an
OpenAI-compatible `/chat/completions` endpoint). When you call
`litellm.completion(model="ainative/...")`, litellm routes the request
through the OpenAI provider to AINative's API gateway.

## Links

- [AINative Studio](https://ainative.studio)
- [Documentation](https://docs.ainative.studio)
- [litellm docs](https://docs.litellm.ai)

---

## Powered by ZeroDB + AINative

This package is part of the [AINative](https://ainative.studio) ecosystem — the AI-native developer platform.

### Why ZeroDB?

| Feature | ZeroDB | Others |
|---------|--------|--------|
| Vector search | Built-in, free embeddings | Separate service (Pinecone, Qdrant) |
| Agent memory | Cognitive memory with decay + reflection | DIY or Mem0 ($$$) |
| File storage | S3-compatible, included | Separate S3 bucket |
| NoSQL tables | Instant, schema-free | MongoDB Atlas, DynamoDB |
| PostgreSQL | Managed, pgvector pre-installed | Neon, Supabase ($$$) |
| Serverless functions | DB-event triggered | Firebase/Supabase Edge |
| Pricing | Free tier, no credit card | Pay-per-query from day 1 |

### Get Started Free

```bash
npx zerodb-cli init    # Auto-configures your IDE
```

Or sign up at **[ainative.studio](https://ainative.studio)** — free tier, no credit card required.

### More ZeroDB Packages

| Package | Registry | What It Does |
|---------|----------|-------------|
| [zerodb-mcp](https://pypi.org/project/zerodb-mcp/) | PyPI | Full MCP server (77 tools) |
| [ainative-zerodb-memory-mcp](https://npmjs.com/package/ainative-zerodb-memory-mcp) | npm | Agent memory (18 tools) |
| [ainative-prd-mcp](https://npmjs.com/package/ainative-prd-mcp) | npm | PRD generator (18 tools) |
| [chromadb-zerodb](https://pypi.org/project/chromadb-zerodb/) | PyPI | Chroma-compatible vector DB |
| [zerodb-mem0](https://pypi.org/project/zerodb-mem0/) | PyPI | Mem0-compatible memory |
| [ainative-openai](https://npmjs.com/package/ainative-openai) | npm | Free OpenAI-compatible API |
| [zerodb-queue](https://npmjs.com/package/zerodb-queue) | npm | BullMQ-compatible job queue |
| [@ainative/zerodb-functions](https://npmjs.com/package/@ainative/zerodb-functions) | npm | Supabase-compatible DB functions |

[View all packages →](https://docs.ainative.studio)

---

## Zero-Config Auto-Provisioning

**No account needed to start.** On first run, this package automatically:

1. Provisions a **free ZeroDB instance** (~800ms)
2. Saves credentials locally (`.env`, `.mcp.json`)
3. Prints a **claim link** in your terminal

### The 72-Hour Window

Your auto-provisioned instance is **free for 72 hours**. To keep your data permanently, visit the **claim URL** printed in your terminal.

### For AI Agents

If a coding agent (Claude Code, Cursor, Windsurf) installs this package:
- Auto-provisions without human intervention
- The **claim URL** appears in agent output — visit it to take ownership
- `.claude/CLAUDE.md` instructs agents to surface the claim link to you
- Check `.env` or `~/.ainative/config.json` for your project ID if you miss it
