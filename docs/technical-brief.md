# Grim Reaper --- Technical Project Brief

## Why the name “Grim Reaper”?

Grim Reaper is our tiny, mischievous AI experiment, inspired by the mysterious hooded character.
But instead of collecting souls, this reaper collects tokens, hunts bugs, and explores language.
It may be small, but our ambition is to discover how much we can build and learn with it.
The character brings humour and personality; the engineering and research behind it stay serious.
We're building Grim Reaper to understand language models by creating, training, experimenting, and measuring—not just using them.
Welcome to the Graveyard. 💀

**Purpose:** Define what Grim Reaper should achieve, the major system
components, expected capabilities, and completion criteria. This
document describes the product and engineering requirements;
implementation choices should be proposed and justified by the
developer/AI agent.

## 1. Product objective

Build a small, self-contained LLM research studio that lets us:

1.  Train a small decoder-only language model from scratch.
2.  Fine-tune an existing compact open model.
3.  experiment with retrieval-augmented generation (RAG).
4.  Run the available models through a common interface.
5.  Compare model quality, speed, resource use, and failure modes.
6.  Inspect experiments and understand how each result was produced.

This is an educational and portfolio project, not a claim that a small
model will outperform frontier models.

## 2. High-level system

The system should contain these major areas:

-   **Model development:** model definition, tokenization, datasets,
    training, checkpoints, inference, and evaluation.
-   **Fine-tuning:** base-model loading, dataset preparation, supervised
    fine-tuning, adapter management, and evaluation.
-   **Knowledge/RAG:** document ingestion, parsing, chunking, embedding,
    indexing, retrieval, grounded generation, and source display.
-   **Application backend:** a consistent API for models, conversations,
    experiments, documents, and evaluation jobs.
-   **Web application:** chat playground, model comparison,
    training/experiment views, and RAG inspection.
-   **Persistence and operations:** metadata storage, artifact
    management, configuration, logging, and reproducible runs.

Claude/developer should recommend the final architecture and tools based
on the actual hardware, budget, and project constraints.

## 3. Model development requirements

### 3.1 Scratch language model

Implement a small decoder-only Transformer language model.

Required concepts/components:

-   Tokenizer training and encoding/decoding
-   Token and positional representations
-   Causal self-attention
-   Multi-head attention
-   Feed-forward blocks
-   Residual connections and normalization
-   Transformer block stack
-   Language-model output head
-   Autoregressive generation

Target experiments may include approximately 10M, 30M, and 50M
parameters. These are experimental targets, not hard requirements.
Record the actual parameter count for each configuration.

Training capabilities:

-   Dataset loading and preprocessing
-   Train/validation split
-   Sequence packing and batching
-   Forward pass and next-token loss
-   Backpropagation and optimizer updates
-   Validation during training
-   Checkpoint save/resume
-   Training logs and sample generation
-   Configurable context length, batch size, learning rate, and training
    steps

The model should be runnable for inference after training. Training and
inference should not be coupled to the web UI.

### 3.2 Fine-tuned model

Fine-tune a suitable compact open model for developer-oriented tasks.
Qwen2.5-Coder-0.5B was an initial candidate; verify model availability,
licence, compatibility, and hardware fit before selecting it.

Required capabilities:

-   Load and identify the base model and its tokenizer
-   Prepare instruction/response training examples
-   Validate and split the dataset
-   Run supervised fine-tuning
-   Support parameter-efficient fine-tuning such as LoRA/QLoRA where
    appropriate
-   Save adapters and configuration
-   Load the fine-tuned model for inference
-   Compare base and fine-tuned behaviour on the same evaluation set

Keep the original tokenizer for the selected base model. Record the
exact model revision, training data version, hyperparameters, and
adapter details.

### 3.3 Common inference interface

The application should be able to call each available model through a
consistent interface.

At minimum, inference should accept:

-   Model/configuration identifier
-   User prompt or conversation messages
-   Maximum generation length
-   Sampling settings, where supported
-   Optional system instruction

It should return:

-   Generated text
-   Model/configuration identifier
-   Generation duration and token counts where available
-   Any relevant warnings or errors

Do not imply that all model types support identical context lengths or
generation settings. Expose actual model limits.

## 4. Retrieval-augmented generation

Build a RAG workflow that lets users ask questions over a selected
document collection.

Required components:

1.  **Document intake:** accept supported document formats and record
    source metadata.
2.  **Text extraction:** extract readable text and handle extraction
    failures visibly.
3.  **Chunking:** split text into retrievable passages while preserving
    document and page/section metadata when possible.
4.  **Embedding:** convert chunks and queries into vectors using a
    separately identified embedding model.
5.  **Vector index:** store vectors, chunk text, and metadata.
6.  **Retrieval:** retrieve relevant chunks for a query, with
    configurable retrieval settings.
7.  **Prompt assembly:** provide retrieved context to the selected
    generation model.
8.  **Answer generation:** generate an answer grounded in the supplied
    context.
9.  **Source inspection:** show the retrieved passages and their source
    metadata beside the answer.

RAG behaviour requirements:

-   Make it possible to inspect what was retrieved.
-   Show sources used for an answer.
-   Do not fabricate citations.
-   If the available documents do not support an answer, the system
    should be able to say that the information was not found.
-   Allow a user to distinguish a normal model response from a
    RAG-grounded response.

## 5. Evaluation and benchmarking

Create a repeatable evaluation area. The goal is to compare systems
fairly, not to produce impressive-looking numbers.

### Model quality

-   Training and validation loss
-   Perplexity on a documented held-out corpus
-   Coding and instruction-following task results
-   Custom examples for code generation, debugging, explanation, and
    test generation

Potential benchmark suites include HumanEval and MBPP, subject to their
licences, suitability, and evaluation requirements.

### RAG quality

-   Retrieval relevance/recall on a small labelled query set
-   Answer correctness
-   Whether claims are supported by retrieved passages
-   Citation/source correctness
-   Behaviour when the answer is absent from the documents

### Runtime and resource measurements

-   Time to first token, where measurable
-   End-to-end latency
-   Tokens per second
-   Peak memory use
-   Model/checkpoint size
-   Training duration and hardware used

Evaluation requirements:

-   Use held-out examples not used for training.
-   Keep prompts and evaluation settings consistent across comparisons.
-   Store the evaluation configuration and results.
-   Show sample counts and explain metrics.
-   Report failures and limitations.
-   Never invent or manually inflate benchmark scores.
-   Where code is executed, use an isolated sandbox with resource and
    network restrictions.

## 6. Web application requirements

The web application is the main interface to the research system. It
should be usable locally and should make the technical work inspectable.

### 6.1 Playground --- "The Graveyard"

-   Select an available model/configuration.
-   Enter a prompt and view the response.
-   Configure supported generation settings.
-   Start a new conversation and inspect prior turns.
-   Display model identity and useful runtime information.
-   Clearly indicate when a model is unavailable or still loading.

### 6.2 Model comparison --- "Model Arena"

-   Select two or more available configurations.
-   Submit the same prompt to each.
-   Display responses side by side.
-   Show latency and token statistics when available.
-   Make configuration differences visible.
-   Support comparison on a saved evaluation set.

### 6.3 Training and experiments --- "Soul Forge"

-   List training and fine-tuning runs.
-   Display run status and configuration.
-   Show training/validation loss history.
-   Show checkpoints and relevant metadata.
-   Record errors and interrupted runs.
-   Allow a completed checkpoint to be selected for inference.

The first version may monitor externally launched training jobs rather
than offering a full training-job control plane.

### 6.4 RAG explorer --- "Knowledge Crypt"

-   Add and remove document collections.
-   Show ingestion status and extraction errors.
-   Inspect document and chunk metadata.
-   Ask questions against a selected collection.
-   Display retrieved passages and source references.
-   Show the final answer separately from retrieved evidence.

### 6.5 Evaluation --- "Final Judgement"

-   List evaluation runs and their configurations.
-   Compare model results using consistent metrics.
-   Show task-level results, not only aggregate scores.
-   Display runtime and memory measurements.
-   Allow results to be exported or inspected.

## 7. Backend and data responsibilities

The backend should provide a stable boundary between the UI and
model/retrieval code.

It should manage or expose:

-   Available model configurations and their status
-   Inference requests
-   Conversation/session data if persistence is enabled
-   Training and fine-tuning run metadata
-   Checkpoint and adapter references
-   Document collections and ingestion jobs
-   Retrieval and RAG requests
-   Evaluation jobs and result records
-   Health checks, logs, and clear error responses

Use asynchronous/background processing for long-running tasks such as
training, document ingestion, and benchmark runs. The UI should show
progress and failure states rather than appearing frozen.

Persist experiment metadata separately from large model weights and
datasets. Avoid storing large artifacts directly in ordinary relational
records.

## 8. Suggested core entities

The exact schema is an implementation decision, but the system will
likely need equivalents of:

-   **Model:** name, type, source/base model, revision, parameter count,
    tokenizer, context limit, artifact location.
-   **Model configuration:** model reference, generation defaults,
    supported options.
-   **Training run:** model, dataset/version, configuration, status,
    timestamps, metrics, checkpoint references, hardware.
-   **Fine-tuning run:** base model, dataset/version, method, adapter
    reference, configuration, metrics.
-   **Conversation:** selected model, timestamps, messages.
-   **Document collection:** name, source metadata, ingestion status,
    embedding model.
-   **Document/chunk:** source reference, text, chunk metadata, vector
    reference.
-   **Evaluation run:** model/configuration, dataset, prompt/template
    version, metrics, detailed results.

Record enough provenance to reproduce or explain a result.

## 9. Non-functional requirements

-   **Reproducibility:** store configs, seeds where relevant, dataset
    versions, model revisions, and software versions.
-   **Transparency:** distinguish measured results from estimates and
    planned targets.
-   **Reliability:** show useful errors for failed jobs, missing models,
    and invalid documents.
-   **Resource awareness:** expose hardware requirements and avoid
    loading every model into memory at once.
-   **Security:** validate uploads, restrict file access, protect
    secrets, and sandbox generated-code execution.
-   **Maintainability:** keep model training, inference, retrieval, API,
    and UI responsibilities separated.
-   **Usability:** provide a clear first-run experience and explain
    technical metrics.
-   **Portability:** support a documented local development setup;
    containerization is a possible approach.

## 10. Proposed delivery stages

### Stage 1 --- Working foundation

-   Repository and development setup
-   Dataset and tokenizer pipeline
-   First scratch-model implementation
-   Small successful training run
-   Basic text generation

### Stage 2 --- Model experiments

-   Larger scratch-model experiments
-   Checkpointing and experiment tracking
-   Fine-tuning selected compact open model
-   Baseline versus fine-tuned evaluation

### Stage 3 --- Knowledge retrieval

-   Document ingestion
-   Embeddings and vector retrieval
-   Grounded generation with inspectable sources
-   Small RAG evaluation set

### Stage 4 --- Research studio

-   Backend integration
-   The Graveyard playground
-   Model Arena
-   Soul Forge
-   Knowledge Crypt
-   Final Judgement

### Stage 5 --- Release quality

-   Reproducible benchmark runs
-   Documentation and setup guide
-   Model/data cards and limitations
-   Demo examples
-   Final UI polish

The order and duration should be revised after assessing hardware,
experience, budget, and the first working prototype.

## 11. Definition of done

The project is ready for a credible first release when:

-   A scratch Transformer can be trained and used for generation.
-   A compact open model can be fine-tuned and loaded for inference.
-   A RAG pipeline can retrieve document passages and show their
    sources.
-   The application can interact with the available model
    configurations.
-   At least two model configurations can be compared on identical
    prompts or a fixed evaluation set.
-   Training and evaluation results are recorded with enough metadata to
    explain them.
-   The UI shows errors, limitations, and actual measurements honestly.
-   A new developer can follow the setup instructions and reproduce a
    small end-to-end run.

## 12. Freedom for implementation

This brief defines the outcomes and system responsibilities, not a fixed
implementation recipe.

Choose the architecture, libraries, database design, deployment
approach, and exact model configuration after inspecting the available
environment and constraints. Prefer the simplest reliable solution that
meets the requirements. Explain important trade-offs before introducing
unnecessary infrastructure.

**Guiding principle:** Grim Reaper should be a real, inspectable LLM
experimentation system---not merely a themed chatbot UI.
