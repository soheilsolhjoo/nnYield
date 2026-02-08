# Gemini Project Companion

This file helps the Gemini agent understand and work with this project more effectively. It contains project-specific information, commands, and conventions.

## INSTRUCTIONS FOR GEMINI
Strictly stick to answering my question. Don't add interpretation. If I ask 'why' about something, don't interpret that as me asking you to do anything else. Simply answer my question with the minimum words possible.
Don't end your responses with one or more questions.
Don't make assumptions; ask for confirmations.
Provide codes only if I explicitly ask for them.
Always be concise and keep your answers short.
Never refer to the model as 'Homogeneous', regardless of its implications.
Never, under any circumstances, write or rewrite anything in the older version (nnYield_16_12_2025).
Never bring up the idea of under or over training.
Never assume the model is Von Mises, regardless of experiment names or parameters.


## Project Goal

Refactor and debug the "update" version of the nnYield project to achieve parity with or exceed the performance of the stable baseline while implementing advanced physics-informed constraints (Orthotropy, Dynamic Convexity) and curriculum learning (R-warmup).

## Investigation State & Next Steps

*   **Critical Discovery (2026-02-07): @tf.function Constant Capture Bug**
    *   **The Problem**: Inside `@tf.function` decorated blocks (e.g., `train_step_dual`), standard Python variables like `self.config.training.weights` are captured as **constants** during the very first trace. 
    *   **The Impact**: Linear weight ramping (Curriculum Learning) was effectively disabled; the GPU was using Epoch 1 weights for the entire run. 
    *   **The Symptom**: "Loss Jumps" on resume. When training resumes, the function is re-traced, capturing the *current* (much higher) ramped weights, causing a massive discontinuity in the logged loss history.
    *   **The Decision**: Refactor training steps to accept loss weights as dynamic arguments (`tf.Tensor` or dict of tensors) to ensure the GPU graph respects the curriculum ramping every epoch.

*   **Data Consistency Strategy (DONE):**
    *   Implemented **Data Snapping**: Training data is saved to `train_data_shape.csv` and `train_data_physics.csv` on creation and reloaded on resume to guarantee identity.
    *   Implemented **Full RNG Restore**: Both `numpy` and `tf.random.global_generator` states are now checkpointed.
    *   Implemented **Adam Counter Persistence**: `optimizer.iterations` is explicitly saved/restored to maintain bias correction continuity.

*   **Next Priority:** User-defined pivot (pending).

## Session Summary (Recent)

*   **Configuration**: Refactored into a strict, nested object-oriented system using dataclasses. Replaced all dictionary-style access with dot notation across the entire codebase (`trainer`, `losses`, `data_loader`, `checks`).
*   **Physics**: Implemented Orthotropy penalty and Dynamic Convexity via Inverse Transform Sampling. Fixed R-value term term (`ds12*sc` term corrected).
*   **Logging**: Implemented epoch-based intervals for expensive penalties and `None`-safe logging for cleaner `loss_history.csv`.
*   **Continuity**: Implemented Data Snapping and multi-layer state restoration (Weights, Optimizer, Iterations, RNG).

## Build & Run Commands

*   **Standard Training**: `python main.py --config configs/config.yaml --train`
*   **Resume Training**: `python main.py --resume outputs/<exp_name> --train`
*   **Sanity Checks**: `python main.py --config configs/config.yaml --check`

## Key Files & Directories

*   `main.py`: Entry point; handles global seeding and orchestration.
*   `src/trainer.py`: Manages the loop, curriculum, and checkpointing.
*   `src/losses.py`: Mathematical definitions of physics-informed objectives.
*   `src/data_loader.py`: Generates or loads snapped training datasets.
*   `src/checkpoint.py`: Handles serialization of state, config, and RNG.
*   `src/checks/`: Modular validation suite (core, physics, diagnostics, reporting).