# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

PaperBanana is a reference-driven multi-agent framework for automated academic illustration generation. It orchestrates five specialized agents (Retriever → Planner → Stylist → Visualizer → Critic) to transform scientific content into publication-quality diagrams.

## Setup

Uses `uv` for package management:

```bash
uv venv
source .venv/bin/activate
uv python install 3.12
uv pip install -r requirements.txt
```

**Configuration**: Copy `configs/model_config.template.yaml` to `configs/model_config.yaml` and fill in model names and at least one API key. This file is gitignored. API keys can also be set via environment variables (`GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`).

## Running

```bash
# Interactive Streamlit demo
streamlit run demo.py

# CLI batch processing
python main.py --dataset_name PaperBananaBench --task_name diagram --exp_mode dev_full --retrieval_setting auto

# Visualization tools
streamlit run visualize/show_pipeline_evolution.py
streamlit run visualize/show_referenced_eval.py
```

**Experiment modes**: `vanilla`, `dev_planner`, `dev_planner_stylist`, `dev_planner_critic`, `dev_full`, `demo_planner_critic`, `demo_full`

**Retrieval settings**: `auto`, `manual`, `random`, `none`

## Architecture

### Agent Pipeline (`agents/`)

All agents extend `BaseAgent` (abstract, async `process()` method) and receive `ExpConfig`. The pipeline is orchestrated by `utils/paperviz_processor.py`:

- **RetrieverAgent**: Finds relevant reference diagrams from the dataset using generative retrieval
- **PlannerAgent**: Converts method content + caption into a detailed textual description using retrieved references as in-context examples
- **StylistAgent**: Refines the description to meet academic aesthetic standards using style guides from `style_guides/`
- **VisualizerAgent**: Calls the image generation model to produce the diagram
- **CriticAgent**: Evaluates the generated image and provides feedback; loops with Visualizer for iterative refinement
- **VanillaAgent**: Direct generation without planning (baseline)
- **PolishAgent**: High-resolution upscaling/refinement for the Streamlit "Refine Image" tab

### Multi-Provider LLM Routing (`utils/generation_utils.py`)

`call_model_with_retry_async()` is the unified entry point for all LLM calls. It routes based on model name prefix:
- `openrouter/...` → OpenRouter
- `claude-...` → Anthropic
- `gpt-`/`o1-`/`o3-`/`o4-` → OpenAI
- No prefix → auto-detect by which API key is configured (OpenRouter > Gemini > Anthropic > OpenAI)

All calls use async with exponential backoff retry. The content format uses a generic list structure (Claude's format as the canonical internal format); `_convert_to_gemini_parts()` and `_convert_to_openai_format()` handle provider-specific conversion.

### Configuration (`utils/config.py`)

`ExpConfig` dataclass loads model names from: CLI args → `configs/model_config.yaml` → environment variables → hardcoded defaults. Results are saved to `results/{dataset_name}_{task_name}/`.

### Data

The benchmark dataset (`data/PaperBananaBench/`) contains `diagram/` and `plot/` subdirectories, each with `test.json`, `ref.json` (references), and `images/`. The framework degrades gracefully without it (Retriever is bypassed).
