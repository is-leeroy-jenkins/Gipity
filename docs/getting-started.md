# Getting Started

## Prerequisites

Before running Gipity or building the documentation, confirm that the workstation has:

- Python 3.10 or newer.
- Git.
- PowerShell.
- A Python virtual environment.
- Project dependencies from `requirements.txt`.
- Documentation dependencies from `requirements-docs.txt`.
- An OpenAI API key for OpenAI-backed workflows.

## Clone the Repository

Run the following from the parent folder where you keep local repositories:

    git clone https://github.com/is-leeroy-jenkins/Gipity.git
    cd Gipity

## Create a Virtual Environment

Create a local Python virtual environment from the repository root:

    python -m venv .venv

## Activate the Virtual Environment

Activate the virtual environment in PowerShell:

    .\.venv\Scripts\Activate.ps1

If PowerShell blocks activation, run this once:

    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

Then activate the virtual environment again:

    .\.venv\Scripts\Activate.ps1

## Install Application Dependencies

Install the application dependencies first:

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

## Install Documentation Dependencies

Install the documentation dependencies next:

    python -m pip install -r requirements-docs.txt

## Run Gipity

Run the Streamlit application from the repository root:

    python -m streamlit run app.py

Streamlit should open the application in a browser.

## Serve Documentation Locally

Serve the documentation locally:

    mkdocs serve

Then open:

    http://127.0.0.1:8000/

## Build Documentation Locally

Build the documentation site:

    mkdocs build --strict

A successful build writes the generated static documentation site to:

    site/

## Publish Documentation to GitHub Pages

After the local build succeeds, publish the site:

    mkdocs gh-deploy --force

The expected GitHub Pages settings are:

    Source: Deploy from a branch
    Branch: gh-pages
    Folder: / root

The expected public documentation URL is:

    https://is-leeroy-jenkins.github.io/Gipity/

## Verification Checklist

Before considering the documentation setup complete, verify that:

- `mkdocs.yml` exists in the repository root.
- `requirements-docs.txt` exists in the repository root.
- The `docs/` folder exists.
- The Markdown files referenced in `mkdocs.yml` exist under `docs/`.
- `docs/api/config.md` uses `::: config`.
- `docs/api/gpt.md` uses `::: gpt`.
- `docs/api/app.md` does not use `::: app`.
- `mkdocs build --strict` succeeds.
- `mkdocs serve` opens the local documentation site.
- `mkdocs gh-deploy --force` publishes the generated site to `gh-pages`.

## Troubleshooting

| Problem                            | Cause                                                                     | Correction                                                                  |
|------------------------------------|---------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| `mkdocs` is not recognized         | Documentation dependencies are not installed.                             | Run `python -m pip install -r requirements-docs.txt`.                       |
| PowerShell cannot activate `.venv` | Execution policy blocks local scripts.                                    | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`. |
| A page is missing from navigation  | `mkdocs.yml` references a file that does not exist under `docs/`.         | Create the missing Markdown file or remove the navigation entry.            |
| API docs do not render             | `mkdocstrings` cannot import the module.                                  | Confirm the module name and import-safety of the source file.               |
| GitHub Pages opens the README      | Pages is pointed at the wrong branch or the static site was not deployed. | Use `gh-pages` branch and `/ root`, then run `mkdocs gh-deploy --force`.    |