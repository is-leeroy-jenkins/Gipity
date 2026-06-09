# User Interface

## Purpose

This page documents the main Gipity UI application at a high level.

`app.py` is the application entry point. It coordinates the user interface, workflow selection,
session state, file upload handling, local document processing, result rendering, token tracking,
and calls into the provider wrappers.

## Responsibilities

`app.py` is responsible for:

- Creating the Streamlit application shell.
- Initializing session state.
- Loading configuration values.
- Rendering mode-specific controls.
- Coordinating text, image, audio, embeddings, document, file, vector-store, prompt, and data
  workflows.
- Handling uploaded files.
- Rendering results and metrics.
- Tracking token usage where supported.
- Connecting the interface to provider wrappers in `gpt.py`.

## Import-Safety Notice

This page intentionally does not use a `mkdocstrings` directive for `app.py`.

The main application module performs Streamlit-oriented setup and state initialization. Since
`mkdocstrings` imports modules during documentation generation, importing the Streamlit runtime
module can cause documentation build failures or side effects.

## Current Documentation Strategy

The documentation site uses `mkdocstrings` for:

    ::: config
    ::: gpt

The documentation site describes `app.py` manually.

## Build Guidance

If the documentation build fails after adding `::: app`, remove that directive and keep this manual
page.

The safe initial API documentation targets are:

- `config.py`
- `gpt.py`