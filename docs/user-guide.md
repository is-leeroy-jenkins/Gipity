# Gipity User Guide

Gipity is a multi-modal AI Streamlit application for text generation, image generation and analysis,
audio workflows, embeddings, document question answering, file management, vector-store operations,
prompt engineering, and local data management.

The application is organized around a Streamlit user interface in `app.py`, provider wrapper classes
in `gpt.py`, and runtime constants in `config.py`. The UI exposes modes through the sidebar, while
the provider wrappers perform the OpenAI-facing operations for chat, images, audio, embeddings,
files, and vector stores.

## 🧭 Overview

Gipity provides nine primary working modes:

| Mode               | Purpose                                                                                            |
| ------------------ | -------------------------------------------------------------------------------------------------- |
| Text               | Generate, structure, summarize, reason, and retrieve text responses.                               |
| Images             | Generate, analyze, and edit image content.                                                         |
| Audio              | Create speech, transcribe audio, and translate spoken content.                                     |
| Embeddings         | Convert text into vector embeddings for search, retrieval, and analysis.                           |
| Document Q&A       | Ask questions against uploaded or indexed documents.                                               |
| Files              | Upload, list, retrieve, extract, delete, summarize, search, and survey files.                      |
| Vector Stores      | Create, manage, search, and query vector stores.                                                   |
| Prompt Engineering | Store, reuse, convert, and manage prompt templates.                                                |
| Data Management    | Manage local SQLite tables, imports, exports, browsing, filtering, and CRUD-style data operations. |

Use the sidebar to choose the active mode. Each mode exposes controls for model selection, inputs,
settings, and outputs.

## 🚀 Getting Started

Start from the project root.

```powershell
.\.venv\Scripts\activate
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

Before using provider-backed modes, confirm that the OpenAI API key is available. You can set it in
the environment before launch:

```powershell
$env:OPENAI_API_KEY="sk-your-key"
streamlit run app.py
```

You can also enter API keys in the sidebar under **API Settings → Keys**. Sidebar keys override
environment values for the current Streamlit session.

## ⚙️ Configuration

Gipity reads application settings from `config.py`.

Common configuration values include:

```python
APP_TITLE = "Gipity"
APP_SUBTITLE = "Multi-Modal AI"
DB_PATH = "stores/sqlite/gipity.db"
LOG_PATH = "logging/Exceptions.db"
LOG_FILE = "Exceptions"
```

The application modes are configured as a list:

```python
GPT_MODES = [
    "Text",
    "Images",
    "Audio",
    "Embeddings",
    "Document Q&A",
    "Files",
    "Vector Stores",
    "Prompt Engineering",
    "Data Management",
]
```

The mode-to-provider map identifies which wrapper classes support each mode:

```python
MODE_CLASS_MAP = {
    "Text": ["Chat"],
    "Images": ["Images"],
    "Audio": ["TTS", "Translation", "Transcription"],
    "Embeddings": ["Embeddings"],
    "Documents": ["Files"],
    "Files": ["Files"],
    "Vector Stores": ["VectorStores"],
}
```

## 🖥️ User Interface Workflow

The standard UI workflow is:

1. Select a mode in the sidebar.
2. Enter or confirm API keys if the selected mode requires provider access.
3. Choose a model and mode-specific options.
4. Provide input text, files, images, audio, or database selections.
5. Run the operation.
6. Review generated output, rendered previews, metrics, sources, or saved artifacts.

Most modes use `st.session_state` to preserve inputs, outputs, and settings during the session. Use
reset and clear controls when you need a clean state.

## 💬 Text Mode

Text mode uses the `Chat` provider wrapper. It supports normal text generation, structured output,
conversation context, tool selection, web search, file search, reasoning settings, and response
rendering.

Use Text mode for:

* General questions.
* Drafting and rewriting.
* Structured JSON generation.
* Grounded answers using tools.
* Multi-turn context.
* Web-search-assisted responses.
* File-search-assisted responses.

### Example 1: Basic Text Generation

```python
from gpt import Chat

chat = Chat()

answer = chat.generate_text(
    prompt="Explain the purpose of budget authority in federal appropriations.",
    model="gpt-5-nano",
    max_tokens=600,
)

print(answer)
```

### Example 2: Text Generation With System Instructions

```python
from gpt import Chat

chat = Chat()

instructions = """
You are a federal budget analyst.
Answer in plain English.
Use short paragraphs.
"""

answer = chat.generate_text(
    prompt="Summarize the difference between obligations and outlays.",
    model="gpt-5-nano",
    instruct=instructions,
    max_tokens=700,
)

print(answer)
```

### Example 3: Structured JSON Output

```python
from gpt import Chat

chat = Chat()

response_format = {
    "format": {
        "type": "json_schema",
        "name": "budget_concept_summary",
        "schema": {
            "type": "object",
            "properties": {
                "concept": {"type": "string"},
                "definition": {"type": "string"},
                "example": {"type": "string"},
                "related_terms": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["concept", "definition", "example", "related_terms"],
            "additionalProperties": False,
        },
        "strict": True,
    }
}

answer = chat.generate_text(
    prompt="Define apportionment for a new budget analyst.",
    model="gpt-5-nano",
    format=response_format,
    max_tokens=800,
)

print(answer)
```

### Example 4: Text Generation With Web Search Tool

```python
from gpt import Chat

chat = Chat()

tools = [
    {"type": "web_search"},
]

answer = chat.generate_text(
    prompt="Find recent public guidance on federal AI governance and summarize the major themes.",
    model="gpt-5-nano",
    tools=tools,
    allowed_domains=["whitehouse.gov", "nist.gov", "cio.gov"],
    tool_choice="auto",
    max_tools=3,
    max_tokens=1000,
)

print(answer)
```

### Example 5: Text Generation With Prior Context

```python
from gpt import Chat

chat = Chat()

context = [
    {
        "role": "user",
        "content": "We are preparing documentation for a Streamlit AI application.",
    },
    {
        "role": "assistant",
        "content": "The documentation should cover setup, architecture, modes, and API references.",
    },
]

answer = chat.generate_text(
    prompt="Draft a concise overview paragraph for the user guide.",
    model="gpt-5-nano",
    context=context,
    max_tokens=500,
)

print(answer)
```

## 🖼️ Images Mode

Images mode uses the `Images` provider wrapper. It supports image generation, image analysis, and
image editing.

Use Images mode for:

* Creating diagrams and conceptual visuals.
* Generating repo badges or visual assets.
* Analyzing uploaded images.
* Editing source images with a prompt.
* Producing documentation graphics.

### Example 1: Generate an Image

```python
from gpt import Images

images = Images()

result = images.generate(
    prompt="A dark blue software architecture diagram for a multi-modal AI application.",
    model="gpt-image-1-mini",
    size="1024x1024",
    quality="auto",
    fmt="png",
    number=1,
)

if isinstance(result, bytes):
    with open("generated_architecture.png", "wb") as output:
        output.write(result)
else:
    print(result)
```

### Example 2: Generate Multiple Images

```python
from gpt import Images

images = Images()

outputs = images.generate(
    prompt="A clean documentation badge for a GitHub repository named Gipity.",
    model="gpt-image-1-mini",
    size="1024x1024",
    quality="auto",
    fmt="png",
    number=3,
)

if isinstance(outputs, list):
    for index, item in enumerate(outputs, start=1):
        if isinstance(item, bytes):
            with open(f"gipity_badge_{index}.png", "wb") as output:
                output.write(item)
        else:
            print(item)
```

### Example 3: Analyze a Local Image

```python
from gpt import Images

images = Images()

description = images.analyze(
    text="Describe the architecture diagram and identify the major layers.",
    path="docs/images/gipity-architecture.png",
    model="gpt-4o-mini",
    detail="auto",
    max_tokens=700,
)

print(description)
```

### Example 4: Edit an Existing Image

```python
from gpt import Images

images = Images()

result = images.edit(
    prompt="Make this diagram darker, use blue highlights, and improve text contrast.",
    path="docs/images/gipity-architecture.png",
    model="gpt-image-1-mini",
    size="1024x1024",
    quality="auto",
    fmt="png",
)

if isinstance(result, bytes):
    with open("docs/images/gipity-architecture-edited.png", "wb") as output:
        output.write(result)
```

### Example 5: Use Image Mode From the UI

1. Select **Images** in the sidebar.
2. Choose the image operation tab: generate, analyze, or edit.
3. Select the model.
4. Enter the prompt or upload a file.
5. Configure size, quality, MIME/output format, and detail level.
6. Run the operation.
7. Review the rendered image or analysis output.

## 🔊 Audio Mode

Audio mode uses three provider wrappers:

* `TTS` for text-to-speech.
* `Transcription` for speech-to-text.
* `Translation` for audio translation.

Use Audio mode for:

* Creating spoken output from text.
* Transcribing audio files.
* Translating spoken content.
* Preparing transcripts for downstream summarization.
* Converting recorded material into Document Q&A input.

### Example 1: Text-to-Speech

```python
from gpt import TTS

tts = TTS()

audio_bytes = tts.create_speech(
    text="Gipity is a multi-modal AI Streamlit application.",
    model="gpt-4o-mini-tts",
    voice="alloy",
    response_format="mp3",
    speed=1.0,
)

if audio_bytes:
    with open("gipity_intro.mp3", "wb") as output:
        output.write(audio_bytes)
```

### Example 2: Transcribe an Audio File

```python
from gpt import Transcription

transcription = Transcription()

result = transcription.transcribe(
    file_path="resources/audio/conditions.mp3",
    model="gpt-4o-mini-transcribe",
    language="en",
    response_format="text",
)

print(result)
```

### Example 3: Transcribe With Prompting Context

```python
from gpt import Transcription

transcription = Transcription()

result = transcription.transcribe(
    file_path="meeting_audio.mp3",
    model="gpt-4o-mini-transcribe",
    language="en",
    response_format="text",
    instructions="This is a technical project meeting. Preserve acronyms and product names.",
)

print(result)
```

### Example 4: Translate Spoken Audio

```python
from gpt import Translation

translation = Translation()

result = translation.translate(
    file_path="spanish_audio.mp3",
    model="gpt-4o-mini-transcribe",
    response_format="text",
)

print(result)
```

### Example 5: Use Audio Output in Text Mode

```python
from gpt import Transcription, Chat

transcription = Transcription()
chat = Chat()

transcript = transcription.transcribe(
    file_path="meeting_audio.mp3",
    model="gpt-4o-mini-transcribe",
    language="en",
    response_format="text",
)

summary = chat.generate_text(
    prompt=f"Summarize this meeting transcript and list action items:\n\n{transcript}",
    model="gpt-5-nano",
    max_tokens=1000,
)

print(summary)
```

## 🧬 Embeddings Mode

Embeddings mode uses the `Embeddings` provider wrapper. It converts text into numeric vector
representations for similarity search, clustering, retrieval, and downstream Document Q&A workflows.

Use Embeddings mode for:

* Turning text into vectors.
* Building semantic search indexes.
* Inspecting vector dimensions.
* Exporting embeddings to a DataFrame.
* Feeding retrieval workflows.

### Example 1: Create an Embedding for One String

```python
from gpt import Embeddings

embeddings = Embeddings()

response = embeddings.create(
    input_text="Gipity supports text, images, audio, files, embeddings, and vector stores.",
    model="text-embedding-3-small",
    encoding_format="float",
)

print(response)
```

### Example 2: Create Embeddings for Multiple Chunks

```python
from gpt import Embeddings

embeddings = Embeddings()

chunks = [
    "Gipity uses Streamlit for the user interface.",
    "Gipity uses gpt.py for provider wrappers.",
    "Gipity uses SQLite for local application data.",
]

response = embeddings.create(
    input_text=chunks,
    model="text-embedding-3-small",
    encoding_format="float",
)

print(response)
```

### Example 3: Use Dimensions With Supported Models

```python
from gpt import Embeddings

embeddings = Embeddings()

response = embeddings.create(
    input_text="Create a compact embedding vector for this sentence.",
    model="text-embedding-3-small",
    dimensions=512,
    encoding_format="float",
)

print(response)
```

### Example 4: Build a Simple Similarity Search

```python
from gpt import Embeddings
import numpy as np

def cosine_similarity(left: list[float], right: list[float]) -> float:
    left_vector = np.asarray(left, dtype=np.float32)
    right_vector = np.asarray(right, dtype=np.float32)
    denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)

    if denominator == 0:
        return 0.0

    return float(np.dot(left_vector, right_vector) / denominator)

embeddings = Embeddings()

documents = [
    "Text mode generates natural language responses.",
    "Image mode creates and analyzes visual content.",
    "Audio mode transcribes and translates speech.",
]

query = "Which mode works with spoken language?"

document_response = embeddings.create(
    input_text=documents,
    model="text-embedding-3-small",
    encoding_format="float",
)

query_response = embeddings.create(
    input_text=query,
    model="text-embedding-3-small",
    encoding_format="float",
)

document_vectors = document_response
query_vector = query_response[0] if isinstance(query_response, list) else query_response

scores = [
    (documents[index], cosine_similarity(vector, query_vector))
    for index, vector in enumerate(document_vectors)
]

scores.sort(key=lambda item: item[1], reverse=True)

print(scores[0])
```

### Example 5: Use Embeddings Mode From the UI

1. Select **Embeddings** in the sidebar.
2. Choose the embedding model.
3. Enter text or paste multiple passages.
4. Configure dimensions, encoding format, chunk size, and overlap.
5. Generate embeddings.
6. Review metrics, chunks, and generated vectors.
7. Export or reuse the vectors in retrieval workflows.

## 📄 Document Q&A Mode

Document Q&A mode supports retrieval-augmented answering against uploaded or referenced documents.
It can use local document ingestion, text extraction, chunking, embeddings, SQLite storage, and
vector search.

Use Document Q&A mode for:

* Asking questions about PDFs, DOCX files, text files, CSV files, JSON files, XML files, code files,
  and Markdown files.
* Summarizing uploaded documents.
* Finding facts inside documents.
* Comparing passages.
* Building local retrieval context.
* Answering questions using OpenAI file or vector-store IDs.

### Example 1: Basic UI Workflow

1. Select **Document Q&A** in the sidebar.
2. Choose **Local Upload** as the source.
3. Upload one or more documents.
4. Review the document preview.
5. Confirm chunking and retrieval settings.
6. Ask a question.
7. Review the answer and retrieved source context.

### Example 2: Ask a Question About an Uploaded PDF

```python
from pathlib import Path
from gpt import Chat

pdf_text = Path("docs/sample-policy.pdf").read_bytes()

# In the Streamlit UI, the application extracts text from the uploaded bytes.
# This direct example assumes text extraction has already produced `document_text`.

document_text = """
Paste extracted text here or retrieve it from the application extraction helper.
"""

question = "What are the key policy requirements in this document?"

chat = Chat()

answer = chat.generate_text(
    prompt=f"""
Use the document excerpt below to answer the question.

Document:
{document_text[:12000]}

Question:
{question}
""",
    model="gpt-5-nano",
    max_tokens=1000,
)

print(answer)
```

### Example 3: Summarize an Active Document

```python
from gpt import Chat

chat = Chat()

document_text = """
Paste extracted document text here.
"""

summary = chat.generate_text(
    prompt=f"""
Provide a structured summary of the document.

Include:
- Purpose
- Major topics
- Key findings
- Important dates or entities
- Open questions

Document:
{document_text[:14000]}
""",
    model="gpt-5-nano",
    max_tokens=1200,
)

print(summary)
```

### Example 4: Ask With Retrieved Context

```python
from gpt import Chat

chat = Chat()

retrieved_chunks = [
    "[Document: policy.pdf]\nThe program requires quarterly reporting...",
    "[Document: policy.pdf]\nThe responsible office must certify the final submission...",
]

question = "Who is responsible for certifying the final submission?"

context = "\n\n".join(retrieved_chunks)

answer = chat.generate_text(
    prompt=f"""
Use only the retrieved document excerpts to answer the question.
If the answer is not in the excerpts, say that the document excerpts do not contain enough information.

Retrieved excerpts:
{context}

Question:
{question}
""",
    model="gpt-5-nano",
    max_tokens=700,
)

print(answer)
```

### Example 5: Use an OpenAI Vector Store for Document Q&A

```python
from gpt import Chat

chat = Chat()

tools = [
    {"type": "file_search"},
]

answer = chat.generate_text(
    prompt="Summarize the main requirements across the indexed documents.",
    model="gpt-5-nano",
    tools=tools,
    vector_store_ids=["vs_your_vector_store_id"],
    tool_choice="auto",
    max_tokens=1200,
)

print(answer)
```

## 📁 Files Mode

Files mode uses the `Files` provider wrapper. It supports file lifecycle operations such as upload,
list, retrieve, extract, delete, summarize, search, and survey.

Use Files mode for:

* Uploading files to provider storage.
* Listing available provider files.
* Retrieving file metadata.
* Extracting content.
* Deleting files.
* Summarizing file content.
* Searching and surveying uploaded material.

### Example 1: Upload a File

```python
from gpt import Files

files = Files()

result = files.upload(
    filepath="docs/user-guide.md",
    purpose="assistants",
)

print(result)
```

### Example 2: List Files

```python
from gpt import Files

files = Files()

available_files = files.list()

for item in available_files:
    print(item)
```

### Example 3: Retrieve File Metadata

```python
from gpt import Files

files = Files()

metadata = files.retrieve(
    file_id="file_your_file_id",
)

print(metadata)
```

### Example 4: Extract File Content

```python
from gpt import Files

files = Files()

content = files.extract(
    file_id="file_your_file_id",
)

print(content)
```

### Example 5: Summarize a File

```python
from gpt import Files

files = Files()

summary = files.summarize(
    file_id="file_your_file_id",
    prompt="Summarize the file in five bullets and identify action items.",
    model="gpt-5-nano",
)

print(summary)
```

### Example 6: Search Files

```python
from gpt import Files

files = Files()

results = files.search(
    query="budget authority",
    model="gpt-5-nano",
)

print(results)
```

### Example 7: Delete a File

```python
from gpt import Files

files = Files()

deleted = files.delete(
    file_id="file_your_file_id",
)

print(deleted)
```

## 🗂️ Vector Stores Mode

Vector Stores mode uses the `VectorStores` provider wrapper. It manages semantic search storage and
file-search workflows.

Use Vector Stores mode for:

* Creating vector stores.
* Listing vector stores.
* Retrieving vector-store metadata.
* Updating vector-store names or descriptions.
* Deleting vector stores.
* Attaching files.
* Creating file batches.
* Searching indexed content.
* Asking questions with file search.

### Example 1: Create a Vector Store

```python
from gpt import VectorStores

stores = VectorStores()

store = stores.create(
    name="Gipity Documentation",
    description="Vector store for Gipity project documentation.",
)

print(store)
```

### Example 2: List Vector Stores

```python
from gpt import VectorStores

stores = VectorStores()

items = stores.list_stores()

for item in items:
    print(item)
```

### Example 3: Retrieve a Vector Store

```python
from gpt import VectorStores

stores = VectorStores()

store = stores.retrieve(
    store_id="vs_your_vector_store_id",
)

print(store)
```

### Example 4: Attach a File to a Vector Store

```python
from gpt import VectorStores

stores = VectorStores()

result = stores.attach_file(
    store_id="vs_your_vector_store_id",
    file_id="file_your_file_id",
)

print(result)
```

### Example 5: List Files in a Vector Store

```python
from gpt import VectorStores

stores = VectorStores()

files = stores.list_files(
    store_id="vs_your_vector_store_id",
)

for item in files:
    print(item)
```

### Example 6: Search a Vector Store

```python
from gpt import VectorStores

stores = VectorStores()

results = stores.search(
    store_id="vs_your_vector_store_id",
    query="What does the documentation say about Document Q&A?",
    max_num_results=5,
)

print(results)
```

### Example 7: Answer With File Search

```python
from gpt import VectorStores

stores = VectorStores()

answer = stores.answer_with_file_search(
    store_ids=["vs_your_vector_store_id"],
    prompt="Explain how Gipity handles user-uploaded documents.",
    model="gpt-5-nano",
    max_num_results=5,
    instructions="Answer using only indexed project documentation.",
)

print(answer)
```

### Example 8: Delete a Vector Store

```python
from gpt import VectorStores

stores = VectorStores()

result = stores.delete(
    store_id="vs_your_vector_store_id",
)

print(result)
```

## 🧠 Prompt Engineering Mode

Prompt Engineering mode manages prompt templates and prompt transformations. It supports storing
reusable instructions, loading templates into other modes, and converting between XML-style prompt
blocks and Markdown-style prompt text.

Use Prompt Engineering mode for:

* Creating reusable system instructions.
* Saving prompt templates.
* Loading templates into Text, Images, or Document Q&A modes.
* Converting XML-style prompt blocks into Markdown.
* Converting Markdown headings into HTML heading tags.
* Managing prompt records in the local SQLite database.

### Example 1: XML-Style Prompt Template

```xml
<role>
You are a federal budget analyst.
</role>

<task>
Summarize the provided document for a senior executive.
</task>

<format>
Use short sections with bullets.
</format>

<constraints>
Do not invent facts.
Use only the supplied context.
</constraints>
```

### Example 2: Markdown Prompt Template

```markdown
## Role

You are a federal budget analyst.

## Task

Summarize the provided document for a senior executive.

## Format

Use short sections with bullets.

## Constraints

Do not invent facts. Use only the supplied context.
```

### Example 3: Prompt Template for Text Mode

```markdown
## Role

You are a technical documentation reviewer.

## Task

Review the supplied source documentation and identify missing user-facing sections.

## Output

Return:
- Summary
- Missing sections
- Recommended additions
- Priority order
```

### Example 4: Prompt Template for Document Q&A

```markdown
## Role

You are a document-grounded analyst.

## Instructions

Answer only from the retrieved document excerpts. When the answer is not present, say the excerpts do not contain enough information.

## Output

Use:
- Direct answer
- Supporting excerpt summary
- Confidence note
```

### Example 5: Prompt Template for Image Generation

```markdown
## Role

You are a software documentation illustrator.

## Task

Create a clean, dark-themed architecture diagram for a multi-modal AI application.

## Style

Use rounded boxes, blue accents, readable labels, and clear directional arrows.

## Constraints

Avoid clutter. Keep all text legible.
```

### Example 6: Save a Prompt Template in SQLite

```python
import sqlite3
import config as cfg

prompt_name = "Document Q&A Analyst"
prompt_text = """
## Role

You are a document-grounded analyst.

## Instructions

Answer only from retrieved document context.
"""

with sqlite3.connect(cfg.DB_PATH) as conn:
    conn.execute(
        """
        INSERT INTO Prompts (Caption, Name, Text, Version, ID)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "Document analysis prompt",
            prompt_name,
            prompt_text,
            "1.0",
            "document_qna_analyst",
        ),
    )
    conn.commit()
```

### Example 7: Load Prompt Templates From SQLite

```python
import sqlite3
import config as cfg

with sqlite3.connect(cfg.DB_PATH) as conn:
    rows = conn.execute(
        """
        SELECT Name, Caption, Version
        FROM Prompts
        ORDER BY Name
        """
    ).fetchall()

for name, caption, version in rows:
    print(name, caption, version)
```

## 🗃️ Data Management Mode

Data Management mode provides local SQLite data workflows. It supports importing, browsing,
creating, reading, updating, deleting, exploring, filtering, exporting, and managing local data
tables.

Use Data Management mode for:

* Inspecting the local application database.
* Reviewing chat history.
* Reviewing embedding records.
* Managing prompt templates.
* Importing CSV-like tabular data.
* Exporting tables.
* Filtering records.
* Performing lightweight CRUD operations.

### Example 1: Connect to the Local Database

```python
import sqlite3
import config as cfg

conn = sqlite3.connect(cfg.DB_PATH)

try:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
    ).fetchall()

    for row in rows:
        print(row[0])
finally:
    conn.close()
```

### Example 2: Read a Table Into pandas

```python
import sqlite3
import pandas as pd
import config as cfg

with sqlite3.connect(cfg.DB_PATH) as conn:
    df_chat_history = pd.read_sql_query(
        "SELECT * FROM chat_history ORDER BY id DESC LIMIT 50;",
        conn,
    )

print(df_chat_history.head())
```

### Example 3: Export a Table to CSV

```python
import sqlite3
import pandas as pd
import config as cfg

table_name = "Prompts"

with sqlite3.connect(cfg.DB_PATH) as conn:
    df_prompts = pd.read_sql_query(
        f'SELECT * FROM "{table_name}";',
        conn,
    )

df_prompts.to_csv("prompts_export.csv", index=False)
```

### Example 4: Import a CSV Into SQLite

```python
import sqlite3
import pandas as pd
import config as cfg

csv_path = "sample_data.csv"
table_name = "ImportedData"

df_imported = pd.read_csv(csv_path)

with sqlite3.connect(cfg.DB_PATH) as conn:
    df_imported.to_sql(
        table_name,
        conn,
        if_exists="replace",
        index=False,
    )
```

### Example 5: Filter a Table

```python
import sqlite3
import pandas as pd
import config as cfg

with sqlite3.connect(cfg.DB_PATH) as conn:
    df_filtered = pd.read_sql_query(
        """
        SELECT *
        FROM Prompts
        WHERE Name LIKE ?
        ORDER BY Name
        """,
        conn,
        params=("%document%",),
    )

print(df_filtered)
```

### Example 6: Insert a Record

```python
import sqlite3
import config as cfg

with sqlite3.connect(cfg.DB_PATH) as conn:
    conn.execute(
        """
        INSERT INTO chat_history (role, content)
        VALUES (?, ?)
        """,
        (
            "user",
            "Explain how Gipity uses vector stores.",
        ),
    )
    conn.commit()
```

### Example 7: Delete Records Safely

```python
import sqlite3
import config as cfg

with sqlite3.connect(cfg.DB_PATH) as conn:
    conn.execute(
        """
        DELETE FROM chat_history
        WHERE id IN (
            SELECT id
            FROM chat_history
            ORDER BY id ASC
            LIMIT 10
        )
        """
    )
    conn.commit()
```

### Example 8: Inspect Table Schema

```python
import sqlite3
import config as cfg

table_name = "Prompts"

with sqlite3.connect(cfg.DB_PATH) as conn:
    schema = conn.execute(
        f'PRAGMA table_info("{table_name}");'
    ).fetchall()

for column in schema:
    print(column)
```

## 🔁 Cross-Mode Workflows

Gipity modes are designed to work together. A typical advanced workflow can combine several modes.

### Example 1: Audio to Text to Summary

1. Use **Audio** mode to transcribe a meeting.
2. Send the transcript to **Text** mode.
3. Generate a summary, action items, and decisions.
4. Save the result to local data or a file.

```python
from gpt import Transcription, Chat

transcription = Transcription()
chat = Chat()

transcript = transcription.transcribe(
    file_path="meeting.mp3",
    model="gpt-4o-mini-transcribe",
    language="en",
    response_format="text",
)

summary = chat.generate_text(
    prompt=f"""
Summarize the transcript.

Return:
- Executive summary
- Decisions
- Action items
- Risks

Transcript:
{transcript}
""",
    model="gpt-5-nano",
    max_tokens=1200,
)

print(summary)
```

### Example 2: File Upload to Vector Store to Question Answering

1. Use **Files** mode to upload a document.
2. Use **Vector Stores** mode to attach the file to a vector store.
3. Use **Text** mode with file search, or use **Vector Stores** mode to answer questions.

```python
from gpt import Files, VectorStores

files = Files()
stores = VectorStores()

uploaded = files.upload(
    filepath="docs/user-guide.md",
    purpose="assistants",
)

file_id = uploaded.id if hasattr(uploaded, "id") else "file_your_file_id"

store = stores.create(
    name="Gipity User Guide Store",
    description="Documentation vector store for the Gipity user guide.",
)

store_id = store.id if hasattr(store, "id") else "vs_your_vector_store_id"

stores.attach_file(
    store_id=store_id,
    file_id=file_id,
)

answer = stores.answer_with_file_search(
    store_ids=[store_id],
    prompt="What modes does Gipity support?",
    model="gpt-5-nano",
)

print(answer)
```

### Example 3: Document Q&A to Embeddings to Data Export

1. Upload a document in **Document Q&A**.
2. Extract text and chunk it.
3. Generate embeddings in **Embeddings** mode.
4. Store or export the embedding data through **Data Management**.

```python
from gpt import Embeddings
import pandas as pd

chunks = [
    "Gipity supports Text mode.",
    "Gipity supports Images mode.",
    "Gipity supports Audio mode.",
]

embeddings = Embeddings()

vectors = embeddings.create(
    input_text=chunks,
    model="text-embedding-3-small",
    encoding_format="float",
)

df_embeddings = pd.DataFrame(
    {
        "chunk": chunks,
        "vector": vectors,
    }
)

df_embeddings.to_json("gipity_embeddings.json", orient="records")
```

### Example 4: Prompt Engineering to Text Mode

1. Create a reusable prompt template in **Prompt Engineering**.
2. Load it into **Text** mode as system instructions.
3. Ask a question or submit a task using the selected template.

```python
from gpt import Chat

template = """
You are a documentation reviewer.
Review the supplied content for clarity, completeness, and missing usage examples.
"""

chat = Chat()

review = chat.generate_text(
    prompt="Review the Gipity user guide and recommend improvements.",
    model="gpt-5-nano",
    instruct=template,
    max_tokens=1000,
)

print(review)
```

## 🧪 Validation Checklist

Use this checklist before publishing documentation or committing workflow changes.

| Check                              | Expected Result                                                     |
| ---------------------------------- | ------------------------------------------------------------------- |
| Application launches               | `streamlit run app.py` starts without syntax errors.                |
| Sidebar modes render               | All configured modes appear in the sidebar selector.                |
| API key handling works             | Sidebar keys override environment keys for the session.             |
| Text mode runs                     | A simple prompt returns a response.                                 |
| Images mode runs                   | A generation or analysis request completes.                         |
| Audio mode runs                    | TTS, transcription, or translation completes with a supported file. |
| Embeddings mode runs               | Input text produces vectors or vector-like output.                  |
| Document Q&A loads files           | Uploaded documents appear in the active document list.              |
| Files mode lists files             | Provider files can be listed or retrieved.                          |
| Vector Stores mode lists stores    | Stores can be listed, searched, or queried.                         |
| Prompt Engineering saves templates | Prompt records appear in the local database.                        |
| Data Management reads tables       | SQLite tables can be browsed and exported.                          |

## 🛠️ Troubleshooting

| Problem                                | Likely Cause                                 | Fix                                                                             |
| -------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------- |
| API request fails                      | Missing or invalid API key                   | Enter the key in the sidebar or set `OPENAI_API_KEY`.                           |
| Mode does not show expected options    | Configuration mismatch                       | Confirm `GPT_MODES` and mode-specific option lists in `config.py` and `gpt.py`. |
| Document Q&A returns weak answers      | Document text was not extracted or indexed   | Preview the extracted text, rebuild index, and check chunk count.               |
| File search returns no results         | Vector store has no indexed files            | Attach files and wait for indexing before querying.                             |
| Embeddings fail                        | Empty input or unsupported dimension setting | Use non-empty text and valid model-specific dimensions.                         |
| Audio transcription fails              | Unsupported file path or audio type          | Confirm the file exists and use a supported audio format.                       |
| SQLite operation fails                 | Missing table or locked database             | Confirm database initialization and close other database connections.           |
| Generated docs do not show API content | MkDocs or mkdocstrings configuration issue   | Confirm `mkdocs.yml` paths and `::: module_name` entries.                       |

## 📚 Recommended Documentation Structure

Use this guide with the following documentation pages:

```text
docs/
├── index.md
├── architecture.md
├── user-guide.md
├── development.md
├── api/
│   ├── app.md
│   ├── config.md
│   └── gpt.md
└── images/
    ├── gipity_application_architecture_diagram.png
    ├── gipity_class_map_diagram.png
    └── gipity_class_map_overview.png
```

The `api` pages should use `mkdocstrings` directives:

```markdown
# Application UI API

::: app
```

```markdown
# Configuration API

::: config
```

```markdown
# GPT Provider API

::: gpt
```

## ✅ Summary

Gipity provides a single Streamlit interface for multi-modal AI workflows. The application supports
text generation, image operations, audio processing, embeddings, document question answering,
provider file management, vector-store search, prompt engineering, and local data management.

For most users, the recommended workflow is:

1. Start in the sidebar.
2. Select the appropriate mode.
3. Configure the model and mode settings.
4. Provide the input.
5. Run the operation.
6. Review and export the result.
7. Move the output into another mode when a multi-step workflow is needed.
   | Confirm the uploaded file type, model selection, and provider credentials.                 |