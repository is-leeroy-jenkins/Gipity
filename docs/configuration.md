# Configuration

Gipity configuration is centralized primarily in `config.py`.

The configuration layer defines paths, environment-variable helpers, API key lookups, model lists,
user-interface constants, prompt constants, database paths, logging paths, and workflow defaults.

## Environment Variables

| Variable             | Purpose                                                                                     |
|----------------------|---------------------------------------------------------------------------------------------|
| `OPENAI_API_KEY`     | Required for OpenAI-backed text, image, audio, embedding, file, and vector-store workflows. |
| `GOOGLE_API_KEY`     | Optional Google service key where configured.                                               |
| `GOOGLE_CSE_ID`      | Optional Google Custom Search Engine identifier where configured.                           |
| `GOOGLEMAPS_API_KEY` | Optional Google Maps API key where configured.                                              |
| `GEOCODING_API_KEY`  | Optional geocoding API key where configured.                                                |
| `LOG_DIR`            | Optional logging directory override.                                                        |
| `LOG_PATH`           | Optional exception database path override.                                                  |
| `LOG_FILE`           | Optional exception table name override.                                                     |

## Set an OpenAI API Key in PowerShell

Set the API key for the current PowerShell session:

    $env:OPENAI_API_KEY="your_api_key_here"

## Application Paths

The active repository uses root-relative paths for resources, storage, and logging.

| Path                                  | Purpose                                                |
|---------------------------------------|--------------------------------------------------------|
| `resources/images`                    | Application images, logos, and favicon assets.         |
| `resources/audio`                     | Audio test assets.                                     |
| `stores/sqlite/gipity.db`             | Local SQLite database path.                            |
| `logging/Exceptions.db`               | Local exception logging database path when configured. |
| `models/gipity-3-270m-it-Q4_K_M.gguf` | Optional local GGUF model path.                        |

## Documentation Configuration

The documentation site is controlled by:

    mkdocs.yml

The documentation dependency file is:

    requirements-docs.txt

The documentation source folder is:

    docs/

The generated static output folder is:

    site/

## GitHub Pages Configuration

The `mkdocs.yml` file should use the actual GitHub owner and repository name:

    site_url: https://is-leeroy-jenkins.github.io/Gipity/
    repo_url: https://github.com/is-leeroy-jenkins/Gipity
    repo_name: is-leeroy-jenkins/Gipity

GitHub Pages should be configured as:

    Source: Deploy from a branch
    Branch: gh-pages
    Folder: / root

## Secret Handling

Do not commit API keys, access tokens, local secrets, `.env` files, local databases, private
documents, or user data to GitHub.

Use environment variables, Streamlit secrets, GitHub Actions secrets, or another appropriate
secret-management mechanism for deployment.

## Documentation Build Notes

The documentation build imports `config.py` and `gpt.py` for API reference generation.

For reliable builds, install both application dependencies and documentation dependencies before
running:

    mkdocs build --strict