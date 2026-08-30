###### Gipity

![](https://github.com/is-leeroy-jenkins/Gipity/blob/main/resources/images/gipity_project.gif)

<p align="center">
  <a href="#-key-features">Features</a> ·
  <a href="#-cloud-demos">Demo</a> ·
  <a href="#-application-modes">Modes</a> ·
  <a href="#-architecture">Architecture</a> ·
  <a href="#-repository-structure">Structure</a> ·
  <a href="#-installation--setup">Install</a> ·
  <a href="#custom-llm">LLM</a> ·
  <a href="#-running-gipity">Running</a> ·
  <a href="#-api-key-setup">Keys</a> ·
  <a href="#-document-and-retrieval-workflows">Workflows</a> ·
  <a href="#-data-management">Data</a> ·
  <a href="#-requirements">Requirements</a> ·
</p>

___


Gipity is a Streamlit application for multimodal AI workflows centered on OpenAI GPT models,
OpenAI platform services, local document retrieval, vector search, prompt engineering, and local
SQLite-backed data management. It provides a single workspace for text generation, image generation
and vision, audio transcription and speech generation, embeddings, document-grounded question
answering, OpenAI Files, OpenAI Vector Stores, prompt templates, export workflows, and structured
local data operations.


[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-0078FC?style=for-the-badge&logo=github)](https://is-leeroy-jenkins.github.io/Gipity/)

## 🎥 Demo

![](https://github.com/is-leeroy-jenkins/Gipity/blob/main/resources/images/gipity-demo.gif)

___



## ☁️ Cloud

<table>
<tr>
<td align="center">
<img width="190" height="1" alt=""><br>
<a href="https://gipity.lemonglacier-5339eed8.eastus.azurecontainerapps.io">
<img src="https://img.shields.io/badge/Docker-App-2496ED?logo=docker&logoColor=white" alt="Docker App">
</a>
</td>

<td align="center">
<img width="190" height="1" alt=""><br>
<a href="https://gipity-py.streamlit.app/">
<img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit App">
</a>
</td>

<td align="center">
<img width="190" height="1" alt=""><br>
<a href="https://dbc-a0c21f80-7bb3.cloud.databricks.com/browse/folders/home?addGit=&o=7474645703081351&gitUrl=https%3A%2F%2Fgithub.com%2Fis-leeroy-jenkins%2FGipity.git&gitProvider=gitHub">
<img src="https://img.shields.io/badge/Databricks%20Repo-Gipity--Py-FF3621?logo=databricks&logoColor=white" alt="Databricks Notebook">
</a>
</td>

<td align="center">
<a href="https://leeroy.usw-16.palantirfoundry.com/shares/links/deytlv6kvnhxo">
<img width="190" height="1" alt=""><br>
<img src="https://img.shields.io/badge/Palantir%20Foundry-Repo-101113?logo=palantir&logoColor=white" alt="Palantir Repo">
</a>
</td>
</tr>
</table>



Gipity combines:

* Cloud-hosted OpenAI workflows for text, images, audio, embeddings, files, and vector stores.
* Local document ingestion and retrieval for Document Q&A.
* SQLite and sqlite-vec infrastructure for local persistence and vector search.
* Sentence-transformers support for semantic retrieval workflows.
* Prompt engineering tools backed by local prompt-template storage.
* Data export and local data-management utilities for operational workflows.

## Custom LLM

Gipity uses a local LLM available on Hugging Face based on OpenAI's ChatGPT 5.x

[![HuggingFace](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/leeroy-jankins/gipity) 

-  Fine-tuned
-  Post-trained
  
### 📥 Local Model Setup

The README and configuration reference a local GGUF fallback model path. Place the GGUF model file
under the project `models` directory when local fallback support is enabled.

```text
models/gipity-3-270m-it-Q4_K_M.gguf
```

## ✨ Key Features

| Feature            | Description                                                                                                                    |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| Text generation    | Chat, drafting, reasoning, structured output, web search, file search, and conversation-state controls.                        |
| Image workflows    | Image generation, editing, and visual analysis with image-model, quality, size, background, compression, and detail controls.  |
| Audio workflows    | Text-to-speech, transcription, and translation using OpenAI audio wrappers.                                                    |
| Embeddings         | Token-aware chunking, embedding generation, dimensionality controls, encoding format selection, metrics, and dataframe output. |
| Document Q&A       | Ask grounded questions over local uploads, OpenAI File IDs, or OpenAI Vector Store IDs.                                        |
| Files API          | Upload, list, retrieve, inspect, extract, delete, and analyze OpenAI Files.                                                    |
| Vector Stores      | Create, list, retrieve, update, delete, attach files, create batches, search, and answer with file search.                     |
| Prompt Engineering | Create, search, sort, edit, cascade, and reuse prompt templates stored in SQLite.                                              |
| Data Export        | Export application outputs and generated artifacts.                                                                            |
| Data Management    | Browse, import, query, and manage local SQLite application data.                                                               |
| Local retrieval    | Use sentence-transformers and sqlite-vec for local semantic workflows where available.                                         |
| Usage tracking     | Track last-call and cumulative token usage for supported responses.                                                            |



## 🧭 Application Modes

The attached `app.py` exposes the following Streamlit modes.

| Mode                   | Purpose                                                                             | Representative Controls / Outputs                                                                                                                     |
| ---------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Text**               | Generate text and analytical responses through the GPT wrapper.                     | Model, reasoning, temperature, top-p, max tokens, tools, include fields, vector store IDs, JSON output format, previous response ID, conversation ID. |
| **Images**             | Generate, edit, and analyze images.                                                 | Image mode, image model, analysis model, size, quality, background, compression, detail, uploaded image context, generated image output.              |
| **Audio**              | Run speech and language audio workflows.                                            | Task selection, transcription model, translation model, TTS model, voice, speed, language, uploaded audio, generated audio bytes.                     |
| **Document Q&A**       | Ask questions against local documents, OpenAI File IDs, or OpenAI Vector Store IDs. | Source controls, local uploads, file/vector IDs, top-k, chunk size, chunk overlap, diagnostics, retrieval hits, grounded answers.                     |
| **Embeddings**         | Generate vector embeddings from input text.                                         | Embedding model, dimensions, chunk size, overlap, encoding format, user tag, metrics, embedding dataframe.                                            |
| **Files**              | Manage OpenAI Files API assets.                                                     | Upload purpose, file filters, list/retrieve/extract/delete operations, metadata, content preview, selected-file analysis.                             |
| **Vector Stores**      | Manage OpenAI Vector Store resources.                                               | Store create/list/retrieve/update/delete, file attachment, file batches, native search, answer-with-file-search workflows.                            |
| **Prompt Engineering** | Maintain reusable prompt templates.                                                 | Prompt search, sorting, editing, cascade into system instructions, version and ID fields.                                                             |
| **Data Export**        | Export application outputs and generated artifacts.                                 | Export controls for persisted results and generated content.                                                                                          |
| **Data Management**    | Work with local SQLite data.                                                        | Tables, SQL operations, local persistence, application metadata, and data inspection workflows.                                                       |

## 🏛️ Architecture

```text
Streamlit UI
    │
    ├── Text / Images / Audio ───────────────► OpenAI platform wrappers in gpt.py
    │
    ├── Document Q&A ────────────────────────► Local files, OpenAI Files, or Vector Stores
    │                                            │
    │                                            ├── PyMuPDF / text extraction
    │                                            ├── sentence-transformers
    │                                            ├── sqlite-vec
    │                                            └── OpenAI file_search
    │
    ├── Embeddings ──────────────────────────► OpenAI Embeddings API + token-aware chunking
    │
    ├── Files / Vector Stores ───────────────► OpenAI Files API and Vector Stores API
    │
    ├── Prompt Engineering ──────────────────► SQLite prompt template table
    │
    └── Data Management / Export ────────────► SQLite, pandas, ReportLab, Streamlit outputs
```

## 🗂 Repository Structure

```text
Gipity/
├─ app.py                         # Main Streamlit application
├─ config.py                      # Application configuration, model lists, paths, constants
├─ gpt.py                         # OpenAI wrappers: Chat, Images, Embeddings, Audio, Files, Vector Stores
├─ requirements.txt               # Python dependencies
├─ models/
│  └─ gipity-3-270m-it-Q4_K_M.gguf # Optional local fallback model path
├─ resources/
│  ├─ images/
│  │  ├─ favicon.ico
│  │  ├─ gipity_logo.png
│  │  └─ gpt.png
│  └─ audio/
│     └─ conditions.mp3           # Audio API test asset
├─ stores/
│  └─ sqlite/
│     └─ gipity.db                # Local SQLite database
└─ README.md
```

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/is-leeroy-jenkins/Gipity.git
cd Gipity
```

### 2️⃣ Create and Activate a Virtual Environment

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## ▶️ Running Gipity

Run Streamlit through Python so the active virtual environment is used.

```bash
python -m streamlit run app.py
```

The application starts with a collapsed sidebar and a wide layout. Select the active workflow from
**AI Mode** in the sidebar.

## 🔑 API KEY SETUP

| Provider | Setup Link                                                                                       |
| -------- | ------------------------------------------------------------------------------------------------ |
| OpenAI   | [OpenAI API Key](https://github.com/is-leeroy-jenkins/Buddy/blob/main/resources/setup/openai.md) |

### Data Services

| Service      | Link                                                                              | Service       | Link                                                                                       |
| ------------ | --------------------------------------------------------------------------------- | ------------- | ------------------------------------------------------------------------------------------ |
| OpenAI       | [Platform](https://platform.openai.com/home)                                      | Google Search | [Custom Search](https://developers.google.com/custom-search/v1/introduction)               |
| Google Maps  | [Google Maps](https://developers.google.com/maps/documentation/embed/get-api-key) | Geocoding     | [Google Geocoding](https://developers.google.com/maps/documentation/geocoding/get-api-key) |
| Hugging Face | [Models](https://huggingface.co/models)                                           | Streamlit     | [Cloud](https://streamlit.io/cloud)                                                        |

## 🔑 Environment Variables

At minimum, Gipity supports the following environment variables and configuration values.

| Variable             | Purpose                                                                                |
| -------------------- | -------------------------------------------------------------------------------------- |
| `OPENAI_API_KEY`     | Required for OpenAI text, image, audio, embeddings, files, and vector store workflows. |
| `GOOGLE_API_KEY`     | Optional Google service key used where configured.                                     |
| `GOOGLE_CSE_ID`      | Optional Google Custom Search Engine identifier used where configured.                 |
| `GOOGLEMAPS_API_KEY` | Optional Google Maps key used where configured.                                        |
| `GEOCODING_API_KEY`  | Optional geocoding key used where configured.                                          |

### Example: OpenAI API Key

#### Windows PowerShell

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

#### Linux / macOS

```bash
export OPENAI_API_KEY="your_api_key_here"
```

## 🧩 Mode Details

### Text

Text mode is the primary GPT workflow. It supports model selection, response formatting,
structured JSON schema output, conversation IDs, previous response IDs, tool selection, include
fields, vector store IDs for file search, and configurable generation parameters.

### Images

Images mode supports generation, editing, and analysis. It exposes model, analysis model, output
size, quality, MIME type, background, detail, and compression controls. It can render bytes, URLs,
or markdown-style image outputs.

### Audio

Audio mode supports transcription, translation, and text-to-speech. It manages audio task selection,
model selection, language, voice, rate/speed, uploaded audio files, generated output bytes, and
response metadata.

### Document Q&A

Document Q&A supports three source paths:

| Source                 | Description                                                                                                             |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Local Upload           | Extracts text from uploaded documents, chunks content, retrieves relevant chunks, and grounds answers in local context. |
| OpenAI File ID         | Queries an OpenAI File ID through the Files wrapper.                                                                    |
| OpenAI Vector Store ID | Uses OpenAI Vector Store file search for grounded answers.                                                              |

Supported local document extraction includes PDF, DOCX, TXT, Markdown, CSV, JSON, XML, Python,
C#, SQL, YAML, HTML, CSS, JavaScript, and TypeScript-style text files.

### Embeddings

Embeddings mode normalizes text, performs token-aware chunking, calls the OpenAI Embeddings API,
and displays metrics and embedding vectors. It supports float and base64 encoding formats and
model-specific dimension limits.

### Files

Files mode supports OpenAI Files API operations:

| Operation | Description                                              |
| --------- | -------------------------------------------------------- |
| Upload    | Upload a local file with a selected OpenAI file purpose. |
| List      | List files with optional purpose filtering.              |
| Retrieve  | Retrieve selected file metadata.                         |
| Extract   | Retrieve file content where supported.                   |
| Delete    | Delete a selected OpenAI file.                           |
| Analyze   | Ask a model to analyze a selected file.                  |

### Vector Stores

Vector Stores mode supports OpenAI vector store lifecycle and retrieval workflows:

| Operation   | Description                                                                      |
| ----------- | -------------------------------------------------------------------------------- |
| Create      | Create vector stores with metadata, expiration, file IDs, and chunking strategy. |
| List        | List available vector stores.                                                    |
| Retrieve    | Inspect vector store metadata.                                                   |
| Update      | Update store name, description, metadata, or expiration policy.                  |
| Delete      | Delete a vector store.                                                           |
| Attach File | Attach OpenAI file IDs to a vector store.                                        |
| File Batch  | Create, retrieve, cancel, or inspect file batches.                               |
| Search      | Run native vector store search with ranker and score threshold options.          |
| Answer      | Answer prompts using file search over selected vector stores.                    |

### Prompt Engineering

Prompt Engineering mode stores prompt templates in SQLite and supports search, sorting, editing,
versioning, and cascade into mode-specific system instructions.

### Data Export

Data Export mode supports export-oriented application workflows for generated outputs and stored
content.

## 📄 Document and Retrieval Workflows

Gipity includes local and cloud-backed retrieval paths. Local retrieval extracts text, creates chunks,
and ranks chunks against a user query. Cloud-backed retrieval can use OpenAI Files and OpenAI Vector
Stores for file search. This allows Gipity to support both lightweight local document analysis and
OpenAI-managed retrieval backends.

## 🗄️ Data Management

Gipity uses SQLite for local persistence. The application initializes and uses local database paths
from `config.py`, stores prompt templates, tracks chat history, and supports local data workflows
through the Data Management mode.

| Component        | Purpose                                                                          |
| ---------------- | -------------------------------------------------------------------------------- |
| SQLite database  | Local persistence for prompts, chat history, embeddings, and application tables. |
| Prompt table     | Stores reusable prompt captions, names, text, versions, and IDs.                 |
| Chat history     | Stores role/content pairs for conversation history where enabled.                |
| Embeddings table | Stores chunk/vector records for local workflows.                                 |
| sqlite-vec       | Enables vector search where the extension is available.                          |


## 📦 Requirements

The table below reflects the active imports and runtime features used by the attached `app.py`.
Version pins should follow the repository `requirements.txt` when present.

| Requirement           | Package / Import               | Purpose                                                                      | Used By                                                        |
| --------------------- | ------------------------------ | ---------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Python                | `python>=3.10`                 | Runtime for modern type hints and Streamlit execution.                       | Entire application.                                            |
| Streamlit             | `streamlit`                    | Web UI framework, widgets, layout, session state, file upload, data editors. | All modes.                                                     |
| Streamlit Components  | `streamlit.components.v1.html` | Inline HTML/component rendering.                                             | UI rendering and custom output sections.                       |
| OpenAI                | `openai`                       | OpenAI client for text, image, audio, embeddings, files, and vector stores.  | `gpt.py` wrappers and OpenAI workflows.                        |
| NumPy                 | `numpy`                        | Numeric arrays, vector operations, cosine similarity.                        | Embeddings and Document Q&A retrieval.                         |
| Pandas                | `pandas`                       | Dataframes, tables, data editors, local data views.                          | Embeddings, Files, Vector Stores, Data Management.             |
| Plotly                | `plotly.graph_objects`         | Chart and visualization support.                                             | Data Management and metrics visualization.                     |
| ReportLab             | `reportlab`                    | PDF page/canvas generation.                                                  | Data Export and report output workflows.                       |
| SQLite                | `sqlite3`                      | Local application database.                                                  | Prompt Engineering, chat history, Data Management, embeddings. |
| sqlite-vec            | `sqlite_vec`                   | SQLite vector extension for semantic search.                                 | Document Q&A local retrieval.                                  |
| Sentence Transformers | `sentence-transformers`        | Local embedding model loading and semantic vectors.                          | Document Q&A local retrieval.                                  |
| Tiktoken              | `tiktoken`                     | Token counting and token-aware chunking.                                     | Text metrics and Embeddings mode.                              |
| PyMuPDF               | `fitz` / `pymupdf`             | PDF text extraction.                                                         | Document Q&A and PDF extraction workflows.                     |
| Pillow                | `pillow`                       | Image processing support where required by image workflows.                  | Images mode and supporting image utilities.                    |
| Requests / HTTPX      | `requests`, `httpx`            | HTTP transport used by API clients and supporting wrappers.                  | Provider and API calls.                                        |
| Pydantic              | `pydantic`                     | Validation and structured model support for SDKs/wrappers.                   | OpenAI SDK and wrapper models.                                 |
| Typing Extensions     | `typing_extensions`            | Backported typing support.                                                   | Compatibility.                                                 |
| PyYAML                | `pyyaml`                       | YAML parsing where configuration or uploaded files require it.               | Document Q&A text extraction and configuration workflows.      |
| OpenPyXL              | `openpyxl`                     | Excel workbook support.                                                      | Data Management and file workflows.                            |
| Llama CPP Python      | `llama-cpp-python`             | Optional local GGUF inference runtime.                                       | Local fallback workflows where enabled.                        |
| Hugging Face Hub      | `huggingface_hub`              | Model download and repository access.                                        | Local model setup.                                             |
| Transformers          | `transformers`                 | Hugging Face model/tokenizer support.                                        | Local and embedding-adjacent workflows.                        |
| Torch                 | `torch`                        | PyTorch runtime for sentence-transformers and local ML components.           | Sentence-transformers.                                         |
| Scikit-learn          | `scikit-learn`                 | Classical ML utilities and similarity/data tooling where required.           | Supporting analysis workflows.                                 |
| SciPy                 | `scipy`                        | Scientific computing support.                                                | ML and embedding dependencies.                                 |
| Regex                 | `regex`                        | Enhanced regular expression support.                                         | Text utilities.                                                |
| Python Dotenv         | `python-dotenv`                | Optional `.env`-based configuration.                                         | Local development.                                             |



## 🔒 Privacy and Deployment Notes

| Workflow                | Data Location / Consideration                                                                                     |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------- |
| OpenAI-backed workflows | Prompts, files, images, audio, embeddings, and vector store data may be sent to OpenAI APIs.                      |
| Local Document Q&A      | Uploaded files can be extracted and chunked locally before a model call is made.                                  |
| SQLite storage          | Prompt templates, chat history, embeddings, and application tables are stored locally according to `cfg.DB_PATH`. |
| Local GGUF fallback     | Local inference can run on-machine when the GGUF model and runtime are configured.                                |
| Streamlit deployment    | Secrets should be configured through environment variables or Streamlit secrets rather than hard-coded values.    |

## 🧪 Local Inference Notes

The local GGUF fallback is a practical companion path for offline or lower-cost text workflows, but
OpenAI-backed workflows remain the primary path for multimodal capabilities, Files API operations,
Vector Stores, audio processing, and managed embeddings.

## 📜 License

Gipity is published under the MIT license for open-source use:

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/is-leeroy-jenkins/Gipity/blob/main/LICENSE.txt)

## 🙌 Acknowledgements

* OpenAI
* Hugging Face
* llama.cpp and llama-cpp-python
* Streamlit
* sentence-transformers
* sqlite-vec
* The broader open-source Python and machine-learning tooling ecosystem

