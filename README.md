# 💀 Grim Reaper

> A tiny, mischievous LLM research studio. It doesn't collect souls — it collects tokens, hunts bugs, and explores language.

Grim Reaper is an educational and portfolio project for understanding language models by **building, training, experimenting, and measuring** them, not just using them. The character brings humour and personality; the engineering and research behind it stay serious.

This is **not** a claim that a small model will outperform frontier models.

**Welcome to the Graveyard.**

> **Status:** 🚧 Planning / Stage 1. Nothing below is implemented yet. Capabilities listed are goals; results will only be reported once measured.

---

## Objectives

1. Train a small decoder-only language model **from scratch**.
2. **Fine-tune** an existing compact open model.
3. Experiment with **retrieval-augmented generation (RAG)**.
4. Run all available models through a **common interface**.
5. **Compare** model quality, speed, resource use, and failure modes.
6. **Inspect** experiments and understand how each result was produced.

## System overview

| Area | Responsibilities |
|---|---|
| **Model development** | Model definition, tokenization, datasets, training, checkpoints, inference, evaluation |
| **Fine-tuning** | Base-model loading, dataset prep, supervised fine-tuning, adapter management, evaluation |
| **Knowledge / RAG** | Ingestion, parsing, chunking, embedding, indexing, retrieval, grounded generation, source display |
| **Backend** | Consistent API for models, conversations, experiments, documents, evaluation jobs |
| **Web app** | Chat playground, model comparison, experiment views, RAG inspection |
| **Persistence & ops** | Metadata storage, artifact management, config, logging, reproducible runs |

Final architecture and tooling will be chosen based on actual hardware, budget, and constraints, preferring the simplest reliable solution.

## Model development

### Scratch language model
A small decoder-only Transformer, with:
- Tokenizer training, encoding/decoding
- Token and positional representations
- Causal, multi-head self-attention
- Feed-forward blocks, residual connections, normalization
- Transformer block stack and LM output head
- Autoregressive generation

Experimental size targets are roughly **10M, 30M, and 50M** parameters (targets, not requirements; the actual count is recorded for every configuration).

Training capabilities: dataset loading, train/validation split, sequence packing and batching, next-token loss, optimizer updates, periodic validation, checkpoint save/resume, logs and sample generation, and configurable context length, batch size, learning rate, and steps. Training and inference are **decoupled from the web UI**.

### Fine-tuned model
Fine-tune a compact open model for developer-oriented tasks (initial candidate: **Qwen2.5-Coder-0.5B**; availability, licence, compatibility, and hardware fit to be verified first). Includes instruction/response data prep and validation, supervised fine-tuning with LoRA/QLoRA where appropriate, adapter saving, and base-vs-fine-tuned comparison on the same evaluation set. The base model's original tokenizer is kept, and the exact model revision, data version, hyperparameters, and adapter details are recorded.

### Common inference interface
Every model is called the same way.

- **Input:** model/config ID, prompt or messages, max generation length, sampling settings (where supported), optional system instruction
- **Output:** generated text, model/config ID, duration and token counts (where available), warnings/errors

Real model limits (context length, supported settings) are exposed rather than hidden.

## Retrieval-augmented generation

Document intake → text extraction (failures shown) → chunking with metadata → embedding (separately identified embedding model) → vector index → configurable retrieval → prompt assembly → grounded answer → source inspection.

Behaviour requirements:
- Retrieved passages and sources are inspectable beside the answer
- No fabricated citations
- Says so when the documents don't support an answer
- Normal responses are clearly distinguishable from RAG-grounded ones

## Evaluation and benchmarking

A repeatable area for fair comparison, not impressive-looking numbers.

- **Model quality:** train/val loss, perplexity on a documented held-out corpus, coding and instruction-following tasks, custom examples (generation, debugging, explanation, test generation). Candidate suites: HumanEval, MBPP (subject to licence and suitability).
- **RAG quality:** retrieval relevance/recall on a small labelled set, answer correctness, claim support, citation correctness, behaviour when the answer is absent.
- **Runtime & resources:** time to first token, end-to-end latency, tokens/sec, peak memory, checkpoint size, training duration and hardware.

Ground rules: held-out data only, consistent prompts and settings across comparisons, stored configs and results, visible sample counts, reported failures and limitations, **never invent or inflate scores**, and generated code runs in an isolated sandbox with resource and network restrictions.

## Web application

| Module | Purpose |
|---|---|
| 🪦 **The Graveyard** | Playground: pick a model, set supported generation options, chat, inspect turns and runtime info, see loading/unavailable states |
| ⚔️ **Model Arena** | Same prompt to two or more configs, side-by-side responses, latency/token stats, visible config differences, saved eval sets |
| 🔥 **Soul Forge** | Training/fine-tuning runs, status, loss curves, checkpoints, errors and interruptions, select a checkpoint for inference |
| 📜 **Knowledge Crypt** | Manage document collections, ingestion status and errors, chunk metadata, ask questions, view retrieved evidence separately from the answer |
| ⚖️ **Final Judgement** | Evaluation runs, consistent metric comparison, task-level results, runtime/memory, export |

The first version may monitor externally launched training jobs rather than control them.

## Backend and data

The backend is a stable boundary between UI and model/retrieval code. It exposes model configs and status, inference, conversations, run metadata, checkpoint/adapter references, document collections and ingestion jobs, retrieval/RAG requests, evaluation jobs, health checks, logs, and clear errors. Long-running work (training, ingestion, benchmarks) runs in the background with visible progress and failure states. Experiment metadata is stored separately from large weights and datasets.

**Likely entities:** Model, Model configuration, Training run, Fine-tuning run, Conversation, Document collection, Document/chunk, Evaluation run. Enough provenance is kept to reproduce or explain any result.

## Non-functional goals

- **Reproducibility:** configs, seeds, dataset versions, model revisions, software versions
- **Transparency:** measured results are distinguished from estimates and targets
- **Reliability:** useful errors for failed jobs, missing models, invalid documents
- **Resource awareness:** documented hardware needs; never load every model at once
- **Security:** validated uploads, restricted file access, protected secrets, sandboxed code execution
- **Maintainability:** training, inference, retrieval, API, and UI kept separate
- **Usability:** clear first-run experience, explained metrics
- **Portability:** documented local setup; containerization optional

## Roadmap

- [ ] **Stage 1: Working foundation**: repo and dev setup, dataset and tokenizer pipeline, first scratch model, small successful training run, basic generation
- [ ] **Stage 2: Model experiments**: larger scratch models, checkpointing and experiment tracking, fine-tuning a compact open model, baseline vs fine-tuned evaluation
- [ ] **Stage 3: Knowledge retrieval**: ingestion, embeddings and vector retrieval, grounded generation with sources, small RAG eval set
- [ ] **Stage 4: Research studio**: backend integration, Graveyard, Model Arena, Soul Forge, Knowledge Crypt, Final Judgement
- [ ] **Stage 5: Release quality**: reproducible benchmarks, setup guide, model/data cards and limitations, demos, UI polish

Order and duration will be revised after assessing hardware, experience, budget, and the first prototype.

## Definition of done (v1)

- [ ] A scratch Transformer can be trained and used for generation
- [ ] A compact open model can be fine-tuned and loaded for inference
- [ ] A RAG pipeline retrieves passages and shows their sources
- [ ] The app can interact with the available model configurations
- [ ] At least two configurations can be compared on identical prompts or a fixed eval set
- [ ] Training and evaluation results are recorded with enough metadata to explain them
- [ ] The UI honestly shows errors, limitations, and actual measurements
- [ ] A new developer can follow the setup instructions and reproduce a small end-to-end run

## Getting started

_Setup instructions will be added as Stage 1 lands._

## Guiding principle

Grim Reaper should be a **real, inspectable LLM experimentation system**, not merely a themed chatbot UI.

## Documentation

The full requirements live in [`docs/technical-brief.md`](docs/technical-brief.md).
