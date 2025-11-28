# ComfyUI-Z-Image-Utilities

A collection of utility nodes for ComfyUI designed specifically for the [Z-Image](https://github.com/Tongyi-MAI/Z-Image) model.

![ComfyUI-Z-Image-Utilities](https://i.imgur.com/n2Jh9PD.png)

## Installation

1. Navigate to your ComfyUI custom nodes directory and clone the repository:

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/yourusername/ComfyUI-Z-Image-Utilities.git
```

2. Restart ComfyUI

---

## Included Utilities

| Utility | Description | Status |
|---------|-------------|--------|
| [Prompt Enhancer](#prompt-enhancer) | LLM-powered prompt enhancement using the official Z-Image system prompt | ✅ Available |

---

## Prompt Enhancer

Transform simple prompts into detailed visual descriptions optimized for Z-Image.

This node uses the official prompt enhancement system prompt from [Z-Image Turbo Space](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo/blob/main/pe.py) to expand your prompts via an LLM. Z-Image works best with long, detailed prompts, and this node automates that process.

### Example

**Input:**
```
a cat
```

**Output:**
```
A domestic shorthair cat with orange and white fur sits on a wooden floor. 
The cat has green eyes and is looking directly at the camera with an alert 
expression. Soft natural light from a nearby window illuminates the scene 
from the left, creating gentle shadows. The background shows a blurred 
living room interior with warm tones.
```

### Features

- **Free to use** — Uses OpenRouter's free tier models (Qwen 235B by default)
- **Bilingual** — Automatically detects and handles Chinese and English prompts
- **Reliable** — Smart retry logic with exponential backoff and rate limit handling
- **Transparent** — Debug output shows full API request/response details
- **CLIP integration** — Optional direct conditioning output for streamlined workflows

### Setup

Get a free API key from [OpenRouter](https://openrouter.ai/keys)

### Nodes

#### Z-Image OpenRouter API Router

Configure your API connection once and reuse it across your workflow.

| Parameter | Description |
|-----------|-------------|
| `api_key` | Your OpenRouter API key |
| `model` | Model ID (default: `qwen/qwen3-235b-a22b:free`) |

**Output:** `api_config`

---

#### Z-Image Prompt Enhancer

The core enhancement node.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `api_config` | Configuration from the API Router node | — |
| `prompt` | Your input text | — |
| `output_language` | `auto`, `english`, or `chinese` | `auto` |
| `temperature` | Creativity level (0.1–1.5) | 0.7 |
| `max_tokens` | Maximum output length (256–8192) | 2048 |
| `retry_count` | Retry attempts on failure (0–10) | 1 |

**Outputs:** `enhanced_prompt`, `debug_log`

---

#### Z-Image Prompt Enhancer + CLIP

Same as above, but outputs CLIP conditioning directly.

**Additional Input:** `clip` — CLIP model from your checkpoint loader

**Outputs:** `conditioning`, `enhanced_prompt`, `debug_log`

### Example Workflows

**Standard:**
```
[API Router] → [Prompt Enhancer] → [CLIP Text Encode] → [KSampler]
```

**Streamlined:**
```
[Checkpoint] → [Prompt Enhancer + CLIP] → [KSampler]
                        ↑
                  [API Router]
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Empty response errors | Increase `retry_count` or verify your API key |
| Rate limiting | The node respects `Retry-After` headers automatically; wait a few minutes if persistent |
| Unexpected output | Check `debug_log` for full API request/response details |

---

## Roadmap

More Z-Image utilities coming soon.

*Have a suggestion? Open an issue!*

---

## Credits

- **System prompt:** [Z-Image Turbo Space](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo/blob/main/pe.py) by Tongyi-MAI
- **Author:** Kokoboy

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.