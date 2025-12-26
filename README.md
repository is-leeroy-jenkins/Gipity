
###### Gipity
![](https://github.com/is-leeroy-jenkins/Gipity/blob/main/resources/images/gipity_project.png)

**A Local-First, Reasoning-Heavy LLM Application Powered by Bubba (20B)**

Gipity is a **local-only, privacy-preserving Streamlit application** designed for deep reasoning, structured analysis, and long-context workflows. It is powered by **Bubba**, a fine-tuned 20-billion-parameter GGUF model optimized for advanced reasoning, instruction following, and domain-specific analysis.

Unlike cloud-hosted LLM tools, Gipity runs **entirely on your machine**, giving you full control over data, models, and execution.



## ✨ Key Features

* 🧠 **Bubba 20B LLM (GGUF, llama.cpp)**
* 🔒 **100% local inference** — no APIs, no telemetry
* 💬 **Persistent chat history** (SQLite)
* 📄 **Document-based RAG**
* 🔍 **Semantic search with embeddings**
* 📊 **Live token usage & context monitoring**
* ⚙️ **Advanced parameter controls**
* 📝 **Export conversations to Markdown or PDF**
* 🖥️ **CPU-only compatible (GPU optional via llama.cpp)**

![](https://github.com/is-leeroy-jenkins/Gipity/blob/main/resources/images/Gipity-streamlit.gif)


## 🦣 About Bubba 

Gipity runs on **Bubba**, a fine-tuned reasoning model based on ChatGPT hosted on Hugging Face:

[![HuggingFace](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/leeroy-jankins/bubba)

**Bubba highlights:**

* 20B parameter transformer
* Quantized GGUF format (`Q4_K_XL`)
* Optimized for:

  * Deep reasoning
  * Long-form analysis
  * Policy, finance, and technical domains
* Designed for **llama.cpp**-based local inference




## 🗂 Repository Structure

```
gipity/
├─ app.py                     # Main Streamlit application
├─ config.py                  # Environment variable access
├─ requirements.txt           # Python dependencies
├─ resources/
│  └─ images/
│     └─ gipity_logo.png
├─ stores/
│  └─ sqlite/
│     └─ gipity.db            # Chat + embedding storage
└─ README.md
```



## ⚙️ System Requirements

### Minimum (Recommended)

* **Windows 10/11 (64-bit)**
* **Python 3.10 or 3.11**
* **16 GB RAM (32 GB strongly recommended)**
* Modern CPU with AVX2 support
* ~15–20 GB free disk space (model + runtime)

> ⚠️ Bubba is a **20B model**. Expect higher memory and CPU usage than Leeroy or Bro.


## 🚀 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-org-or-username>/gipity.git
cd gipity
```

---

### 2️⃣ Create and Activate a Virtual Environment

#### Windows (Git Bash / PowerShell)

```bash
python -m venv .venv
source .venv/Scripts/activate
```

You should see:

```
(.venv)
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```



## 📥 Download the Bubba Model

1. Visit the Bubba Hugging Face page:
   👉 
[![HuggingFace](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/leeroy-jankins/bubba)


2. Download the GGUF file:

   ```
   bubba-20b-Q4_K_XL.gguf
   ```

3. Place it anywhere on your system, for example:

   ```
   C:\Users\<you>\source\random\file\path\leeroy-jankins\bubba-20b-Q4_K_XL.gguf
   ```



## 🔑 Environment Variable Configuration

Gipity **does not auto-download models**. You must explicitly point it to the local GGUF file.

### Set the Environment Variable

#### Windows (System Environment Variables)

| Variable          | Value                                 |
| ----------------- | ------------------------------------- |
| `GIPITY_LLM_PATH` | Full path to `bubba-20b-Q4_K_XL.gguf` |

Example:

```
GIPITY_LLM_PATH=C:\Users\terry\source\llm\lmstudio\lmstudio-community\leeroy-jankins\bubba-20b-Q4_K_XL.gguf
```

⚠️ **Restart your terminal / IDE (PyCharm) after setting this.**


## ▶️ Running Gipity

Always run Streamlit **through Python** to ensure the correct virtual environment is used:

```bash
python -m streamlit run app.py
```

If everything is configured correctly, you’ll see the Gipity UI load in your browser.



## 🧭 Application Tabs Overview

| Tab                    | Purpose                                      |
| ---------------------- | -------------------------------------------- |
| System Instructions    | Define the system-level persona and behavior |
| Text Generation        | Primary chat interface                       |
| Retrieval Augmentation | Upload documents for basic RAG               |
| Semantic Search        | Build & query vector embeddings              |
| Export                 | Download chat history (Markdown / PDF)       |
| Image                  | Not supported (placeholder)                  |
| Audio                  | Not supported (placeholder)                  |


## 📊 Token & Context Monitoring

The footer displays:

* Prompt tokens
* Response tokens
* Percentage of context window used
* Active generation parameters



## 🧪 Notes on Performance

* Large context windows increase latency
* Consider lowering:

  * Context window
  * Top-k
  * Max generation tokens
    if performance becomes an issue

---

## 🔒 Privacy & Security

* No external API calls
* No telemetry
* No cloud dependencies
* All data remains local (SQLite + filesystem)



---

## 📜 License

This project is released for **personal and research use**.
Refer to the Bubba Hugging Face page for model-specific licensing terms.

---

## 🙌 Acknowledgements

* llama.cpp community
* Hugging Face
* LM Studio ecosystem
* Open-source ML tooling

