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

The primary goal is to find and fix bugs in the refactored `update` codebase by comparing it to the older, working `nnYield_16_12_2025` version. The key symptom of the bug is that the `update` version produces incorrect `yield_loci_slices.png` plots, even though the training process reports a low error and completes successfully.

## Investigation State & Next Steps

This section tracks the immediate next step to resume the conversation.

*   **Last Finding:** Confirmed a fundamental formulation dissimilarity between training data and benchmark plotting logic:
    *   **Numeric Evidence**: In `yield_loci_slices.png`, the Benchmark ("Ground Truth") shear stress is not constant despite labels. For a target `S12 = 0.5485`, the actual plotted shear ranges from `0.3539` to `0.4631`.
    *   **Formulation Dissimilarity**:
        *   **Training Data (`data_loader.py`)**: Uses a **Cylindrical** formulation. It fixes absolute shear stress and solves for the in-plane radius. This creates true constant-shear slices.
        *   **Benchmark Plot (`physics.py`)**: Uses a **Spherical/Conical** formulation. It defines a direction vector and scales the entire vector by the yield radius. This causes the shear component to vary along the curve.
    *   **Impact**: The "Ground Truth" plotted is not a constant shear slice, making it inconsistent with the training data and the figure labels.
    *   **Secondary Finding**: Identified an extra factor of 2 in the R-value shear term across `physics.py`, `reporting.py`, and `trainer.py`.
    *   **Current State**: Model inaccurate (Max Rel Error ~21.7%) and R-values are wrong (50-170% error).
*   **Pending Question:** We need to align the plotting formulation with the training data (Cylindrical) to get accurate "Ground Truth" slices. Should we modify `physics.py` to use the same logic as `data_loader.py`?

## Session Summary (2026-01-28)

This section summarizes the debugging progress.

*   **`main.py`**: Compared both versions. Differences noted but deemed not critical.
*   **`data_loader.py`**: Differences in data generation sampling were confirmed as acceptable/intentional.
*   **`model.py`**: User has manually synchronized the feature calculation logic.
*   **`losses.py`**: Corrected the stress loss calculation from a `sum` to a `weighted average`.
*   **`config.py`**: Confirmed `get_model_architecture` was removed in the `update` version and is unused.
*   **`utils.py`**: Files are functionally identical.

## Build & Run Commands

*   **Run main application:** `python main.py --config-path <path_to_config>`
*   **Setup/Installation:** `pip install -r requirements.txt`

## Test Commands

*(How to run the test suite.)*

## Linting & Formatting

*(Commands for running linters and formatters, e.g., `ruff check .` or `black .`)*

## Key Files & Directories

*   `main.py`: Main entry point for the application.
*   `configs/`: Directory for configuration files.
*   `src/`: Contains the core source code.
*   `src/model.py`: Defines the neural network architecture.
*   `src/trainer.py`: Handles the model training loop.
*   `src/data_loader.py`: Manages loading and preprocessing of data.

## Architectural Notes & Conventions

*(Notes on coding style, architectural patterns, or other conventions to follow.)*
