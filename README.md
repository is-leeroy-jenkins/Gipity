###### Gipity

![](https://github.com/is-leeroy-jenkins/Gipity/blob/main/resources/images/gipity_project.png)

Gipity is a Python application for multimodal AI workflows centered primarily on **OpenAI GPT-5.x**,
with a lightweight **local GGUF fallback** for text generation. It is designed to provide a unified
workspace for text, image and vision, audio, embeddings, files, vector stores, prompt engineering,
and document-grounded analysis in a single Streamlit application.

Rather than being a local-only application, Gipity combines:

* **cloud-hosted GPT workflows** for its primary multimodal experience
* **local GGUF inference** through `llama-cpp-python` as a fallback path
* **retrieval and vector workflows** backed by SQLite and `sqlite-vec`
* **embedding pipelines** using `sentence-transformers` and related NLP tooling

## [![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit\&logoColor=white)](https://gipity-py.streamlit.app/)

![](https://github.com/is-leeroy-jenkins/Gipity/blob/main/resources/images/Gipity-streamlit.gif)

## ✨ Key Features

* 🧠 **GPT-5.x-first application design** for modern multimodal workflows
* 🖥️ **Local GGUF fallback model** via `llama-cpp-python`
* 📝 **Text generation and chat**
* 🖼️ **Image generation, editing, and vision workflows**
* 🔊 **Audio workflows** including text-to-speech, translation, and transcription
* 🔍 **Embeddings and semantic search**
* 📄 **Document Q&A and retrieval-augmented generation**
* 🗂️ **Files and vector store integrations**
* 🧩 **Prompt engineering utilities**
* 🗄️ **SQLite-backed local data management**
* 📊 **Configurable inference and workflow controls**

## 🧠 Model Architecture

Gipity is designed around two complementary inference paths:

### Primary Path: OpenAI GPT Models

The current configuration defines GPT-family models including:

* `gpt-5-nano-2025-08-07`
* `gpt-4.1-nano-2025-04-14`
* `gpt-5-mini`

The application is intended to support multimodal workflows across:

* Text
* Images / Vision
* Audio
* Embeddings
* Document Q&A
* Files
* Vector Stores
* Prompt Engineering
* Data Management

### Local Fallback Path

For local text-generation fallback, Gipity also points to a GGUF model at:

```text
models/gipity-3-270m-it-Q4_K_M.gguf
```

This local artifact is intended for use with `llama.cpp` / `llama-cpp-python` and provides a small,
portable inference option when a local path is preferred.

## 🗂 Repository Structure

```text
Gipity/
├─ app.py                     # Main Streamlit application
├─ config.py                  # Application configuration and provider settings
├─ gpt.py                     # GPT / provider integration logic
├─ requirements.txt           # Python dependencies
├─ models/
│  └─ gipity-3-270m-it-Q4_K_M.gguf
├─ resources/
│  ├─ images/
│  │  ├─ favicon.ico
│  │  ├─ gipity_logo.png
│  │  └─ gpt.png
│  └─ audio/
│     └─ conditions.mp3
├─ stores/
│  └─ sqlite/
│     └─ gipity.db
└─ README.md
```

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/is-leeroy-jenkins/Gipity.git
cd Gipity
```

### 2️⃣ Create and Activate a Virtual Environment

#### Windows (Git Bash / PowerShell)

```bash
python -m venv .venv
source .venv/Scripts/activate
```

You should see:

```text
(.venv)
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 🔑 Environment Variables

At minimum, Gipity is configured to use environment variables such as:

* `OPENAI_API_KEY`
* `GOOGLEMAPS_API_KEY`
* `GOOGLE_API_KEY`
* `GEOCODING_API_KEY`
* `GOOGLE_CSE_ID`

### Example: OpenAI API Key

#### Windows PowerShell

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

#### Windows System Environment Variables

Set a user or system environment variable named:

```text
OPENAI_API_KEY
```

Then restart your terminal or IDE.

## 📥 Local Model Setup

If you want to use Gipity's local fallback model path, place the GGUF model file here:

```text
models/gipity-3-270m-it-Q4_K_M.gguf
```

The default application configuration already points to that relative path.

### Option A: Download from the Hugging Face Web UI

1. Open the Hugging Face model repository for Gipity's backup GGUF model.
2. Sign in to Hugging Face if the repository or license requires authentication.
3. Open the **Files and versions** section.
4. Download `gipity-3-270m-it-Q4_K_M.gguf`.
5. Create the `models` folder in the project root if it does not already exist.
6. Move the downloaded file into:

```text
Gipity/models/gipity-3-270m-it-Q4_K_M.gguf
```

### Option B: Download with the Hugging Face CLI

Install the CLI support:

```bash
python -m pip install huggingface_hub
```

If the model repo is private or gated, authenticate first:

```bash
hf auth login
```

Then download the GGUF file directly into Gipity's `models` folder:

```bash
hf download <your-hf-username-or-org>/<your-gipity-model-repo> \
  gipity-3-270m-it-Q4_K_M.gguf \
  --local-dir models
```

### Option C: Download with Python

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="<your-hf-username-or-org>/<your-gipity-model-repo>",
    filename="gipity-3-270m-it-Q4_K_M.gguf",
    local_dir="models",
)
```

### Folder Layout After Download

```text
Gipity/
├─ app.py
├─ config.py
├─ gpt.py
├─ requirements.txt
├─ models/
│  └─ gipity-3-270m-it-Q4_K_M.gguf
└─ ...
```

### Verification

Before launching Gipity, confirm that:

* the `models` folder exists at the project root
* the file name is exactly `gipity-3-270m-it-Q4_K_M.gguf`
* the file path matches the configured relative path in the application

## ▶️ Running Gipity

Always run Streamlit through Python so the correct virtual environment is used:

```bash
python -m streamlit run app.py
```

If your configuration is valid, Gipity should launch in your browser.

## 🧭 Application Modes

Based on the current configuration and source structure, Gipity supports the following major
workflow
areas:

| Mode               | Purpose                                                  |
| ------------------ | -------------------------------------------------------- |
| Text               | Chat, drafting, reasoning, and analytical generation     |
| Images / Vision    | Image generation, editing, and visual analysis workflows |
| Audio              | Text-to-speech, translation, and transcription           |
| Embeddings         | Semantic vector generation                               |
| Document Q&A       | Retrieval-grounded answers over uploaded content         |
| Files              | File upload and document lifecycle operations            |
| Vector Stores      | Managed retrieval backends and semantic search           |
| Prompt Engineering | Prompt templates and instruction workflows               |
| Data Management    | Local storage and structured document handling           |

## 📄 Document and Retrieval Workflows

Gipity includes infrastructure for document-aware workflows rather than simple chat alone. The
current
project configuration and dependencies indicate support for:

* PDF and document ingestion
* semantic embeddings
* vector-based retrieval with `sqlite-vec`
* file-backed grounding workflows
* document question answering

This makes Gipity suitable for structured analysis over user-provided materials in addition to
general
multimodal prompting.

## 🧪 Local Inference Notes

The default local fallback configuration uses a **4096-token context window** and a small GGUF
model.
This is a practical local option, but it should be understood as a fallback or companion path rather
than the sole centerpiece of the application.

## 🔒 Privacy and Deployment Notes

Gipity is not accurately described as a fully local application in its current form. The current
source
shows explicit support for cloud-backed provider workflows, especially OpenAI, alongside local
components.

In practice, privacy characteristics depend on which features you use:

* **OpenAI-backed workflows** involve remote API usage
* **local GGUF fallback workflows** can run on-machine
* **SQLite-backed data handling** keeps parts of application state local

## 📦 Dependency Table

| Package                 | Version / Constraint | Primary Role                                     |
| ----------------------- | -------------------- | ------------------------------------------------ |
| `openai`                | `==2.21.0`           | Primary OpenAI GPT API client                    |
| `numpy`                 | `==1.26.4`           | Numerical computing foundation                   |
| `openpyxl`              | not pinned           | Excel workbook read and write support            |
| `pydantic`              | `>=2.10,<2.12`       | Data validation and structured models            |
| `typing_extensions`     | `>=4.12,<5`          | Extended typing support                          |
| `pymupdf`               | not pinned           | PDF parsing and document processing              |
| `httpx`                 | `>=0.28.1`           | HTTP client support                              |
| `reportlab`             | `>=4.0.8`            | PDF generation and export                        |
| `sentence-transformers` | `==2.7.0`            | Embeddings and semantic similarity               |
| `transformers`          | `==4.41.2`           | Hugging Face transformer model support           |
| `huggingface_hub`       | `==0.23.4`           | Hugging Face repository and download integration |
| `tokenizers`            | `==0.19.1`           | Fast tokenizer back end                          |
| `safetensors`           | `==0.4.3`            | Safe tensor serialization support                |
| `tiktoken`              | not pinned           | Token counting and tokenization utilities        |
| `pillow`                | `>=10.0.0`           | Image processing support                         |
| `requests`              | `==2.32.4`           | HTTP utility requests                            |
| `sqlite-vec`            | not pinned           | SQLite vector search extension                   |
| `llama-cpp-python`      | not pinned           | Local GGUF inference runtime                     |
| `torch`                 | `==2.3.1`            | PyTorch runtime                                  |
| `torchvision`           | `==0.18.1`           | Vision utilities for PyTorch workflows           |
| `scikit-learn`          | `==1.5.1`            | Classical machine learning utilities             |
| `scipy`                 | `==1.13.1`           | Scientific computing support                     |
| `streamlit`             | `==1.55.0`           | Web application UI framework                     |
| `regex`                 | `==2024.5.15`        | Enhanced regular expressions                     |
| `sentencepiece`         | `==0.2.0`            | SentencePiece tokenizer support                  |
| `tqdm`                  | `==4.66.4`           | Progress bars                                    |
| `filelock`              | `==3.15.4`           | File-based locking support                       |
| `fsspec`                | `==2024.6.1`         | Filesystem abstraction layer                     |
| `packaging`             | `==24.1`             | Version and package utility helpers              |
| `pyyaml`                | `==6.0.1`            | YAML parsing and configuration support           |

## 📜 License

Refer to the repository license for application code and to any upstream model or provider terms
for:

* OpenAI model usage
* Hugging Face-hosted artifacts
* GGUF-converted local model files
* any third-party assets or dependencies

## 🙌 Acknowledgements

* OpenAI
* Hugging Face
* `llama.cpp` and `llama-cpp-python`
* Streamlit
* `sentence-transformers`
* `sqlite-vec`
* the broader open-source Python and ML tooling ecosystem

