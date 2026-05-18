'''
  ******************************************************************************************
      Assembly:                Gipity
      Filename:                name.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="app.py" company="Terry D. Eppler">

	     app.py
	     Copyright ©  2024  Terry Eppler

     Permission is hereby granted, free of charge, to any person obtaining a copy
     of this software and associated documentation files (the “Software”),
     to deal in the Software without restriction,
     including without limitation the rights to use,
     copy, modify, merge, publish, distribute, sublicense,
     and/or sell copies of the Software,
     and to permit persons to whom the Software is furnished to do so,
     subject to the following conditions:

     The above copyright notice and this permission notice shall be included in all
     copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
     FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
     ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
     DEALINGS IN THE SOFTWARE.

     You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov

  </copyright>
  <summary>
    app.py
  </summary>
  ******************************************************************************************
'''
import config as cfg
import os
import multiprocessing
import base64
import hashlib
import io
from boogr import Error

from openai import OpenAI
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import re
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
import sqlite3
import sqlite_vec
import streamlit as st
from streamlit.components.v1 import html
from sentence_transformers import SentenceTransformer
import time
import tiktoken
import tempfile
from typing import List, Dict, Any, Optional, Tuple

try:
	import fitz
except Exception:
	fitz = None
	
from gpt import (
	Chat,
	Images,
	Embeddings,
	Transcription,
	Translation,
	TTS,
	Files,
	VectorStores )

# ======================================================================================
# SESSION STATE INITIALIZATION
# ======================================================================================

if 'openai_api_key' not in st.session_state:
	st.session_state[ 'openai_api_key' ] = ''

if 'google_api_key' not in st.session_state:
	st.session_state[ 'google_api_key' ] = ''

if 'google_cse_id' not in st.session_state:
	st.session_state[ 'google_cse_id' ] = ''

if 'googlemaps_api_key' not in st.session_state:
	st.session_state[ 'googlemaps_api_key' ] = ''

if 'geocoding_api_key' not in st.session_state:
	st.session_state[ 'geocoding_api_key' ] = ''

if st.session_state.openai_api_key == '':
	default = cfg.OPENAI_API_KEY
	if default:
		st.session_state.openai_api_key = default
		os.environ[ 'OPENAI_API_KEY' ] = default

if st.session_state.google_api_key == '':
	default = cfg.GOOGLE_API_KEY
	if default:
		st.session_state.google_api_key = default
		os.environ[ 'GOOGLE_API_KEY' ] = default

if st.session_state.google_cse_id == '':
	default = cfg.GOOGLE_CSE_ID
	if default:
		st.session_state.google_cse_id = default
		os.environ[ 'GOOGLE_CSE_ID' ] = default

if st.session_state.googlemaps_api_key == '':
	default = cfg.GOOGLEMAPS_API_KEY
	if default:
		st.session_state.googlemaps_api_key = default
		os.environ[ 'GOOGLEMAPS_API_KEY' ] = default

if st.session_state.geocoding_api_key == '':
	default = cfg.GEOCODING_API_KEY
	if default:
		st.session_state.geocoding_api_key = default
		os.environ[ 'GEOCODING_API_KEY' ] = default

# --------CHAT-GENERATION PARAMETERS--------------------

if 'max_tools' not in st.session_state:
	st.session_state[ 'max_tools' ] = 0

if 'max_tokens' not in st.session_state:
	st.session_state[ 'max_tokens' ] = 0

if 'temperature' not in st.session_state:
	st.session_state[ 'temperature' ] = 0.0

if 'top_percent' not in st.session_state:
	st.session_state[ 'top_percent' ] = 0.0

if 'frequency_penalty' not in st.session_state:
	st.session_state[ 'frequency_penalty' ] = 0.0

if 'presense_penalty' not in st.session_state:
	st.session_state[ 'presense_penalty' ] = 0.0

if 'background' not in st.session_state:
	st.session_state[ 'background' ] = False

if 'parallel_tools' not in st.session_state:
	st.session_state[ 'parallel_tools' ] = False

if 'store' not in st.session_state:
	st.session_state[ 'store' ] = False

if 'stream' not in st.session_state:
	st.session_state[ 'stream' ] = False

if 'execution_mode' not in st.session_state:
	st.session_state[ 'execution_mode' ] = ''

if 'response_format' not in st.session_state:
	st.session_state[ 'response_format' ] = ''

if 'tool_choice' not in st.session_state:
	st.session_state[ 'tool_choice' ] = ''

if 'reasoning' not in st.session_state:
	st.session_state[ 'reasoning' ] = ''

if 'stops' not in st.session_state:
	st.session_state[ 'stops' ] = [ ]

if 'include' not in st.session_state:
	st.session_state[ 'include' ] = [ ]

if 'input' not in st.session_state:
	st.session_state[ 'input' ] = [ ]

if 'tools' not in st.session_state:
	st.session_state[ 'tools' ] = [ ]

if 'messages' not in st.session_state:
	st.session_state[ 'messages' ] = [ ]

if 'last_sources' not in st.session_state:
	st.session_state[ 'last_sources' ] = [ ]
	
if 'provider' not in st.session_state or st.session_state[ 'provider' ] is None:
	st.session_state[ 'provider' ] = 'GPT'

if 'mode' not in st.session_state or st.session_state[ 'mode' ] is None:
	st.session_state[ 'mode' ] = 'Chat'

if 'messages' not in st.session_state:
	st.session_state.messages = [ ]

if 'chat_history' not in st.session_state:
	st.session_state[ 'chat_history' ] = [ ]

if 'last_call_usage' not in st.session_state:
	st.session_state.last_call_usage = { 'prompt_tokens': 0, 'completion_tokens': 0,
			'total_tokens': 0 }

if 'token_usage' not in st.session_state:
	st.session_state.token_usage = { 'prompt_tokens': 0, 'completion_tokens': 0,
			'total_tokens': 0 }

if 'files' not in st.session_state:
	st.session_state.files = [ ]

if 'use_semantic' not in st.session_state:
	st.session_state[ 'use_semantic' ] = False

if 'is_grounded' not in st.session_state:
	st.session_state[ 'is_grounded' ] = False

if 'selected_prompt_id' not in st.session_state:
	st.session_state[ 'selected_prompt_id' ] = ''

if 'pending_system_prompt_name' not in st.session_state:
	st.session_state[ 'pending_system_prompt_name' ] = ''


# -------- INSTRUCTION VARIABLES ----------------------

if 'instructions' not in st.session_state:
	st.session_state[ 'instructions' ] = ''

if 'text_system_instructions' not in st.session_state:
	st.session_state[ 'text_system_instructions' ] = ''

if 'image_system_instructions' not in st.session_state:
	st.session_state[ 'image_system_instructions' ] = ''

if 'audio_system_instructions' not in st.session_state:
	st.session_state[ 'audio_system_instructions' ] = ''

if 'docqna_system_instructions' not in st.session_state:
	st.session_state[ 'docqna_systems_instructions' ] = ''

if 'docqna_system_instructions' not in st.session_state:
	st.session_state[ 'docqna_system_instructions' ] = ''

if 'stores_system_instructions' not in st.session_state:
	st.session_state[ 'stores_system_instructions' ] = ''

# ----------MODEL PARAMETERS --------------------------------

if 'chat_model' not in st.session_state:
	st.session_state[ 'chat_model' ] = ''

if 'text_model' not in st.session_state:
	st.session_state[ 'text_model' ] = ''

if 'image_model' not in st.session_state:
	st.session_state[ 'image_model' ] = ''

if 'audio_model' not in st.session_state:
	st.session_state[ 'audio_model' ] = ''

if 'embedding_model' not in st.session_state:
	st.session_state[ 'embedding_model' ] = ''

if 'docqna_model' not in st.session_state:
	st.session_state[ 'docqna_model' ] = ''

if 'files_model' not in st.session_state:
	st.session_state[ 'files_model' ] = ''

if 'stores_model' not in st.session_state:
	st.session_state[ 'stores_model' ] = ''

if 'tts_model' not in st.session_state:
	st.session_state[ 'tts_model' ] = ''

if 'transcription_model' not in st.session_state:
	st.session_state[ 'transcription_model' ] = ''

if 'translation_model' not in st.session_state:
	st.session_state[ 'translation_model' ] = ''

# --------CHAT-GENERATION PARAMETERS--------------------

if 'max_tools' not in st.session_state:
	st.session_state[ 'max_tools' ] = 0

if 'max_tokens' not in st.session_state:
	st.session_state[ 'max_tokens' ] = 0

if 'temperature' not in st.session_state:
	st.session_state[ 'temperature' ] = 0.0

if 'top_percent' not in st.session_state:
	st.session_state[ 'top_percent' ] = 0.0

if 'frequency_penalty' not in st.session_state:
	st.session_state[ 'frequency_penalty' ] = 0.0

if 'presense_penalty' not in st.session_state:
	st.session_state[ 'presense_penalty' ] = 0.0

if 'background' not in st.session_state:
	st.session_state[ 'background' ] = False

if 'parallel_tools' not in st.session_state:
	st.session_state[ 'parallel_tools' ] = False

if 'store' not in st.session_state:
	st.session_state[ 'store' ] = False

if 'stream' not in st.session_state:
	st.session_state[ 'stream' ] = False

if 'execution_mode' not in st.session_state:
	st.session_state[ 'execution_mode' ] = ''

if 'response_format' not in st.session_state:
	st.session_state[ 'response_format' ] = ''

if 'tool_choice' not in st.session_state:
	st.session_state[ 'tool_choice' ] = ''

if 'reasoning' not in st.session_state:
	st.session_state[ 'reasoning' ] = ''

if 'stops' not in st.session_state:
	st.session_state[ 'stops' ] = [ ]

if 'include' not in st.session_state:
	st.session_state[ 'include' ] = [ ]

if 'input' not in st.session_state:
	st.session_state[ 'input' ] = [ ]

if 'tools' not in st.session_state:
	st.session_state[ 'tools' ] = [ ]

if 'messages' not in st.session_state:
	st.session_state[ 'messages' ] = [ ]

if 'last_sources' not in st.session_state:
	st.session_state[ 'last_sources' ] = [ ]

# --------TEXT-GENERATION PARAMETERS--------------------

if 'text_max_calls' not in st.session_state:
	st.session_state[ 'text_max_calls' ] = 0

if 'text_max_tokens' not in st.session_state:
	st.session_state[ 'text_max_tokens' ] = 0

if 'text_temperature' not in st.session_state:
	st.session_state[ 'text_temperature' ] = 0.0

if 'text_top_percent' not in st.session_state:
	st.session_state[ 'text_top_percent' ] = 0.0

if 'text_frequency_penalty' not in st.session_state:
	st.session_state[ 'text_frequency_penalty' ] = 0.0

if 'text_presence_penalty' not in st.session_state:
	st.session_state[ 'text_presence_penalty' ] = 0.0

if 'text_parallel_calls' not in st.session_state:
	st.session_state[ 'text_parallel_calls' ] = False

if 'text_background' not in st.session_state:
	st.session_state[ 'text_background' ] = False

if 'text_store' not in st.session_state:
	st.session_state[ 'text_store' ] = False

if 'text_stream' not in st.session_state:
	st.session_state[ 'text_stream' ] = False

if 'text_response_format' not in st.session_state:
	st.session_state[ 'text_response_format' ] = ''

if 'text_tool_choice' not in st.session_state:
	st.session_state[ 'text_tool_choice' ] = ''

if 'text_reasoning' not in st.session_state:
	st.session_state[ 'text_reasoning' ] = ''

if 'text_input' not in st.session_state:
	st.session_state[ 'text_input' ] = ''

if 'text_previous_response_id' not in st.session_state:
	st.session_state[ 'text_previous_response_id' ] = ''

if 'text_include' not in st.session_state:
	st.session_state[ 'text_include' ] = [ ]

if 'text_domains' not in st.session_state:
	st.session_state[ 'text_domains' ] = [ ]

if 'text_tools' not in st.session_state:
	st.session_state[ 'text_tools' ] = [ ]

if 'text_context' not in st.session_state:
	st.session_state[ 'text_context' ] = [ ]

if 'text_content' not in st.session_state:
	st.session_state[ 'text_content' ] = [ ]
	
if 'text_messages' not in st.session_state:
	st.session_state.text_messages = [ ]

# --------IMAGE-GENERATION PARAMETERS--------------------

if 'image_analysis_model' not in st.session_state:
	st.session_state[ 'image_analysis_model' ] = ''

if 'image_analysis_detail' not in st.session_state:
	st.session_state[ 'image_analysis_detail' ] = 'auto'
	
if 'image_max_tokens' not in st.session_state:
	st.session_state[ 'image_max_tokens' ] = 0

if 'image_max_calls' not in st.session_state:
	st.session_state[ 'image_max_calls' ] = 0

if 'image_max_searches' not in st.session_state:
	st.session_state[ 'image_max_searches' ] = 0

if 'image_number' not in st.session_state:
	st.session_state[ 'image_number' ] = 0

if 'image_compression' not in st.session_state:
	st.session_state[ 'image_compression' ] = 0.0

if 'image_temperature' not in st.session_state:
	st.session_state[ 'image_temperature' ] = 0.0

if 'image_top_percent' not in st.session_state:
	st.session_state[ 'image_top_percent' ] = 0.0

if 'image_frequency_penalty' not in st.session_state:
	st.session_state[ 'image_frequency_penalty' ] = 0.0

if 'image_presence_penalty' not in st.session_state:
	st.session_state[ 'image_presence_penalty' ] = 0.0
	
if 'image_parallel_calls' not in st.session_state:
	st.session_state[ 'image_parallel_calls' ] = False

if 'image_background' not in st.session_state:
	st.session_state[ 'image_background' ] = False

if 'image_store' not in st.session_state:
	st.session_state[ 'image_store' ] = False

if 'image_stream' not in st.session_state:
	st.session_state[ 'image_stream' ] = False

if 'image_tool_choice' not in st.session_state:
	st.session_state[ 'image_tool_choice' ] = ''

if 'image_reasoning' not in st.session_state:
	st.session_state[ 'image_reasoning' ] = ''

if 'image_mime_type' not in st.session_state:
	st.session_state[ 'image_mime_type' ] = ''

if 'image_response_format' not in st.session_state:
	st.session_state[ 'image_response_format' ] = ''

if 'image_previous_response_id' not in st.session_state:
	st.session_state[ 'image_previous_response_id' ] = ''

if 'image_input' not in st.session_state:
	st.session_state[ 'image_input' ] = [ ]

if 'image_include' not in st.session_state:
	st.session_state[ 'image_include' ] = [ ]

if 'image_tools' not in st.session_state:
	st.session_state[ 'image_tools' ]: List[ Dict[ str, Any ] ] = [ ]

if 'image_modalities' not in st.session_state:
	st.session_state[ 'image_modalities' ] = [ ]

if 'image_messages' not in st.session_state:
	st.session_state[ 'image_messages' ] = [ ]

if 'image_context' not in st.session_state:
	st.session_state[ 'image_context' ]: List[ Dict[ str, Any ] ] = [ ]

if 'image_domains' not in st.session_state:
	st.session_state[ 'image_domains' ] = [ ]

if 'image_content' not in st.session_state:
	st.session_state[ 'image_content' ] = [ ]

# ------- IMAGE-SPECIFIC PARAMETER---------------

if 'image_analysis_model' not in st.session_state:
	st.session_state[ 'image_analysis_model' ] = ''

if 'image_output_bytes' not in st.session_state:
	st.session_state[ 'image_output_bytes' ] = None

if 'image_messages' not in st.session_state:
	st.session_state[ 'image_messages' ] = [ ]

if 'image_input' not in st.session_state:
	st.session_state[ 'image_input' ] = [ ]

# --------AUDIO-GENERATION PARAMETERS--------------------

if 'audio_max_tokens' not in st.session_state:
	st.session_state[ 'audio_max_tokens' ] = 0

if 'audio_temperature' not in st.session_state:
	st.session_state[ 'audio_temperature' ] = 0.0

if 'audio_top_percent' not in st.session_state:
	st.session_state[ 'audio_top_percent' ] = 0.0

if 'audio_frequency_penalty' not in st.session_state:
	st.session_state[ 'audio_frequency_penalty' ] = 0.0

if 'audio_presence_penalty' not in st.session_state:
	st.session_state[ 'audio_presence_penalty' ] = 0.0

if 'audio_background' not in st.session_state:
	st.session_state[ 'audio_background' ] = False

if 'audio_store' not in st.session_state:
	st.session_state[ 'audio_store' ] = False

if 'audio_stream' not in st.session_state:
	st.session_state[ 'audio_stream' ] = False

if 'audio_tool_choice' not in st.session_state:
	st.session_state[ 'audio_tool_choice' ] = ''

if 'audio_reasoning' not in st.session_state:
	st.session_state[ 'audio_reasoning' ] = ''

if 'audio_response_format' not in st.session_state:
	st.session_state[ 'audio_response_format' ] = ''

if 'audio_input' not in st.session_state:
	st.session_state[ 'audio_input' ] = ''

if 'audio_mime_type' not in st.session_state:
	st.session_state[ 'audio_mime_type' ] = ''

if 'audio_stops' not in st.session_state:
	st.session_state[ 'audio_stops' ] = [ ]

if 'audio_includes' not in st.session_state:
	st.session_state[ 'audio_includes' ] = [ ]

if 'audio_tools' not in st.session_state:
	st.session_state.audio_tools: List[ Dict[ str, Any ] ] = [ ]

if 'audio_context' not in st.session_state:
	st.session_state.audio_context: List[ Dict[ str, Any ] ] = [ ]

if 'audio_modalities' not in st.session_state:
	st.session_state[ 'audio_modalities' ] = [ ]

if 'audio_messages' not in st.session_state:
	st.session_state.audio_messages = [ ]

# -------AUDIO-SECIFIC PARAMETERS--------------

if 'audio_task' not in st.session_state:
	st.session_state[ 'audio_task' ] = ''

if 'audio_file' not in st.session_state:
	st.session_state[ 'audio_file' ] = ''

if 'audio_rate' not in st.session_state:
	st.session_state[ 'audio_rate' ] = int( cfg.SAMPLE_RATES[ 0 ] ) if cfg.SAMPLE_RATES else 44100

if 'audio_language' not in st.session_state:
	st.session_state[ 'audio_language' ] = ''

if 'audio_voice' not in st.session_state:
	st.session_state[ 'audio_voice' ] = ''

if 'audio_start_time' not in st.session_state:
	st.session_state[ 'audio_start_time' ] = 0.0

if 'audio_end_time' not in st.session_state:
	st.session_state[ 'audio_end_time' ] = 0.0

if 'audio_loop' not in st.session_state:
	st.session_state[ 'audio_loop' ] = False

if 'audio_autoplay' not in st.session_state:
	st.session_state[ 'audio_autoplay' ] = False

if 'audio_output' not in st.session_state:
	st.session_state[ 'audio_output' ] = ''

# ------ DOCQNA GENERATION PARAMETERS ------------

if 'docqna_max_tools' not in st.session_state:
	st.session_state[ 'docqna_max_tools' ] = 0

if 'docqna_max_tokens' not in st.session_state:
	st.session_state[ 'docqna_max_tokens' ] = 0

if 'docqna_max_calls' not in st.session_state:
	st.session_state[ 'docqna_max_calls' ] = 0

if 'docqna_temperature' not in st.session_state:
	st.session_state[ 'docqna_temperature' ] = 0.0

if 'docqna_top_percent' not in st.session_state:
	st.session_state[ 'docqna_top_percent' ] = 0.0

if 'docqna_frequency_penalty' not in st.session_state:
	st.session_state[ 'docqna_frequency_penalty' ] = 0.0

if 'docqna_presence_penalty' not in st.session_state:
	st.session_state[ 'docqna_presence_penalty' ] = 0.0

if 'docqna_number' not in st.session_state:
	st.session_state[ 'docqna_number' ] = 0

if 'docqna_top_k' not in st.session_state:
	st.session_state[ 'docqna_top_k' ] = 0

if 'docqna_max_searches' not in st.session_state:
	st.session_state[ 'docqna_max_searches' ] = 0

if 'docqna_parallel_tools' not in st.session_state:
	st.session_state[ 'docqna_parallel_tools' ] = False

if 'docqna_background' not in st.session_state:
	st.session_state[ 'docqna_background' ] = False

if 'docqna_store' not in st.session_state:
	st.session_state[ 'docqna_store' ] = False

if 'docqna_stream' not in st.session_state:
	st.session_state[ 'docqna_stream' ] = False

if 'docqna_response_format' not in st.session_state:
	st.session_state[ 'docqna_response_format' ] = ''

if 'docqna_tool_choice' not in st.session_state:
	st.session_state[ 'docqna_tool_choice' ] = ''

if 'docqna_resolution' not in st.session_state:
	st.session_state[ 'docqna_resolution' ] = ''

if 'docqna_media_resolution' not in st.session_state:
	st.session_state[ 'docqna_media_resolution' ] = ''

if 'docqna_reasoning' not in st.session_state:
	st.session_state[ 'docqna_reasoning' ] = ''

if 'docqna_input' not in st.session_state:
	st.session_state[ 'docqna_input' ] = ''

if 'docqna_stops' not in st.session_state:
	st.session_state[ 'docqna_stops' ] = [ ]

if 'docqna_modalities' not in st.session_state:
	st.session_state[ 'docqna_modalities' ] = [ ]

if 'docqna_include' not in st.session_state:
	st.session_state[ 'docqna_include' ] = [ ]

if 'docqna_domains' not in st.session_state:
	st.session_state[ 'docqna_domains' ] = [ ]

if 'docqna_tools' not in st.session_state:
	st.session_state[ 'docqna_tools' ] = [ ]

if 'docqna_context' not in st.session_state:
	st.session_state[ 'docqna_context' ] = [ ]

if 'docqna_content' not in st.session_state:
	st.session_state[ 'docqna_content' ] = [ ]

# ------- DOCQA-SPECIFIC PARAMATERS  ---------------------------

if 'docqna_files' not in st.session_state:
	st.session_state[ 'docqna_files' ] = [ ]

if 'docqna_uploaded' not in st.session_state:
	st.session_state[ 'docqna_uploaded' ] = ''

if 'docqna_messages' not in st.session_state:
	st.session_state.docqna_messages = [ ]

if 'docqna_active_docs' not in st.session_state:
	st.session_state.docqna_active_docs = [ ]

if 'docqna_source' not in st.session_state:
	st.session_state.docqna_source = ''

if 'docqna_multi_mode' not in st.session_state:
	st.session_state.docqna_multi_mode = False

if 'uploaded' not in st.session_state:
	st.session_state[ 'uploaded' ] = [ ]

if 'docqna_bytes' not in st.session_state:
	st.session_state[ 'docqna_bytes' ] = { }

if 'docqna_source' not in st.session_state:
	st.session_state[ 'docqna_source' ] = 'uploadlocal'

if 'docqna_vec_ready' not in st.session_state:
	st.session_state[ 'docqna_vec_ready' ] = False

if 'docqna_fingerprint' not in st.session_state:
	st.session_state[ 'docqna_fingerprint' ] = ''

if 'docqna_chunk_count' not in st.session_state:
	st.session_state[ 'docqna_chunk_count' ] = 0

if 'docqna_fallback_rows' not in st.session_state:
	st.session_state[ 'docqna_fallback_rows' ] = [ ]

# ------- EMBEDDING-SPECIFIC PARAMETERS ----------------------

if 'embedding_model' not in st.session_state:
	st.session_state[ 'embedding_model' ] = ''

if 'embeddings_dimensions' not in st.session_state:
	st.session_state[ 'embeddings_dimensions' ] = 0

if 'embeddings_chunk_size' not in st.session_state:
	st.session_state[ 'embeddings_chunk_size' ] = 0

if 'embeddings_overlap_amount' not in st.session_state:
	st.session_state[ 'embeddings_overlap_amount' ] = 0

if 'embeddings_input_text' not in st.session_state:
	st.session_state[ 'embeddings_input_text' ] = ''

if 'embeddings_encoding_format' not in st.session_state:
	st.session_state[ 'embeddings_encoding_format' ] = ''

if 'embeddings_method' not in st.session_state:
	st.session_state[ 'embeddings_method' ] = ''

# --------FILES-GENERATION PARAMETERS--------------------

if 'files_max_tokens' not in st.session_state:
	st.session_state[ 'files_max_tokens' ] = 0

if 'files_temperature' not in st.session_state:
	st.session_state[ 'files_temperature' ] = 0.0

if 'files_top_percent' not in st.session_state:
	st.session_state[ 'files_top_percent' ] = 0.0

if 'files_frequency_penalty' not in st.session_state:
	st.session_state[ 'files_frequency_penalty' ] = 0.0

if 'files_presence_penalty' not in st.session_state:
	st.session_state[ 'files_presence_penalty' ] = 0.0

if 'files_background' not in st.session_state:
	st.session_state[ 'files_background' ] = False

if 'files_store' not in st.session_state:
	st.session_state[ 'files_store' ] = False

if 'files_stream' not in st.session_state:
	st.session_state[ 'files_stream' ] = False

if 'files_tool_choice' not in st.session_state:
	st.session_state[ 'files_tool_choice' ] = ''

if 'files_reasoning' not in st.session_state:
	st.session_state[ 'files_reasoning' ] = ''

if 'files_response_format' not in st.session_state:
	st.session_state[ 'files_response_format' ] = ''

if 'files_input' not in st.session_state:
	st.session_state[ 'files_input' ] = ''

if 'files_media_resolution' not in st.session_state:
	st.session_state[ 'files_media_resolution' ] = ''

if 'files_stops' not in st.session_state:
	st.session_state[ 'files_stops' ] = [ ]

if 'files_includes' not in st.session_state:
	st.session_state[ 'files_includes' ] = [ ]

if 'files_tools' not in st.session_state:
	st.session_state.files_tools: List[ Dict[ str, Any ] ] = [ ]

if 'files_context' not in st.session_state:
	st.session_state.files_context: List[ Dict[ str, Any ] ] = [ ]

# ------- FILES-SPECIFIC PARAMETERS --------------------------

if 'files_purpose' not in st.session_state:
	st.session_state[ 'files_purpose' ] = ''

if 'files_type' not in st.session_state:
	st.session_state[ 'files_type' ] = ''

if 'files_id' not in st.session_state:
	st.session_state[ 'files_id' ] = ''

if 'files_url' not in st.session_state:
	st.session_state[ 'files_url' ] = ''

if 'files_table' not in st.session_state:
	st.session_state[ 'files_table' ] = ''

if 'files_messages' not in st.session_state:
	st.session_state.files_messages: List[ Dict[ str, Any ] ] = [ ]

# -------- VECTORSTORES-GENERATION PARAMETERS --------------------

if 'stores_temperature' not in st.session_state:
	st.session_state[ 'stores_temperature' ] = 0.0

if 'stores_top_percent' not in st.session_state:
	st.session_state[ 'stores_top_percent' ] = 0.0

if 'stores_max_tokens' not in st.session_state:
	st.session_state[ 'stores_max_tokens' ] = 0

if 'stores_frequency_penalty' not in st.session_state:
	st.session_state[ 'stores_frequency_penalty' ] = 0.0

if 'stores_presence_penalty' not in st.session_state:
	st.session_state[ 'stores_presence_penalty' ] = 0.0

if 'stores_max_calls' not in st.session_state:
	st.session_state[ 'stores_max_calls' ] = 0

if 'stores_tool_choice' not in st.session_state:
	st.session_state[ 'stores_tool_choice' ] = ''

if 'stores_response_format' not in st.session_state:
	st.session_state[ 'stores_response_format' ] = ''

if 'stores_reasoning' not in st.session_state:
	st.session_state[ 'stores_reasoning' ] = ''

if 'stores_resolution' not in st.session_state:
	st.session_state[ 'stores_resolution' ] = ''

if 'stores_media_resolution' not in st.session_state:
	st.session_state[ 'stores_media_resolution' ] = ''

if 'stores_parallel_tools' not in st.session_state:
	st.session_state[ 'stores_parallel_tools' ] = False

if 'stores_background' not in st.session_state:
	st.session_state[ 'stores_background' ] = False

if 'stores_store' not in st.session_state:
	st.session_state[ 'stores_store' ] = False

if 'stores_stream' not in st.session_state:
	st.session_state[ 'stores_stream' ] = False

if 'stores_input' not in st.session_state:
	st.session_state[ 'stores_input' ] = [ ]

if 'stores_tools' not in st.session_state:
	st.session_state[ 'stores_tools' ] = [ ]

if 'stores_messages' not in st.session_state:
	st.session_state[ 'stores_messages' ] = [ ]

if 'stores_stops' not in st.session_state:
	st.session_state[ 'stores_stops' ] = [ ]

if 'stores_include' not in st.session_state:
	st.session_state[ 'stores_include' ] = [ ]

# ------- VECTORSTORES-SPECIFIC PARAMETERS --------

if 'stores_id' not in st.session_state:
	st.session_state[ 'stores_id' ] = ''

# ------- TOKEN PARAMATERS  ---------------------------
if 'last_answer' not in st.session_state:
	st.session_state.last_answer = ''

if 'last_sources' not in st.session_state:
	st.session_state.last_sources = [ ]

if 'last_analysis' not in st.session_state:
	st.session_state.last_analysis = {
			'tables': [ ],
			'docqna_files': [ ],
			'text': [ ],
	}

if 'last_call_usage' not in st.session_state:
	st.session_state.last_call_usage = {
			'prompt_tokens': 0,
			'completion_tokens': 0,
			'total_tokens': 0, }

if 'token_usage' not in st.session_state:
	st.session_state.token_usage = { 'prompt_tokens': 0,
	                                 'completion_tokens': 0,
	                                 'total_tokens': 0, }
	
# -------------- LLM  UTILITIES -------------------

@st.cache_resource
def load_embedder( ) -> SentenceTransformer:
	"""
	
		Purpose:
		--------
		Load the sentence-transformers model used for embedding and retrieval workflows.
	
		Parameters:
		-----------
		None
	
		Returns:
		--------
		SentenceTransformer
			Loaded embedder instance.
			
	"""
	return SentenceTransformer( 'all-MiniLM-L6-v2' )

# ======================================================================================
# Utilities Section
# ======================================================================================

def extract_usage( resp: Any ) -> Dict[ str, int ]:
	"""
	
		Purpose:
		_________
		Extract token usage from a response object/dict.
		Returns dict with prompt_tokens, completion_tokens, total_tokens.
		Defensive: returns zeros if not present.
		
	"""
	usage = { 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, }
	if not resp:
		return usage
	
	raw = None
	try:
		raw = getattr( resp, 'usage', None )
	except Exception:
		raw = None
	
	if not raw and isinstance( resp, dict ):
		raw = resp.get( 'usage' )
	
	if not raw:
		return usage
	
	try:
		if isinstance( raw, dict ):
			usage[ 'prompt_tokens' ] = int( raw.get( 'prompt_tokens', 0 ) )
			usage[ 'completion_tokens' ] = int(
				raw.get( 'completion_tokens', raw.get( 'output_tokens', 0 ) )
			)
			usage[ 'total_tokens' ] = int(
				raw.get( 'total_tokens', usage[ 'prompt_tokens' ] + usage[ 'completion_tokens' ] ) )
		else:
			usage[ 'prompt_tokens' ] = int( getattr( raw, 'prompt_tokens', 0 ) )
			usage[ 'completion_tokens' ] = int(
				getattr( raw, 'completion_tokens', getattr( raw, 'output_tokens', 0 ) ) )
			usage[ 'total_tokens' ] = int(
				getattr( raw, 'total_tokens',
					usage[ 'prompt_tokens' ] + usage[ 'completion_tokens' ], ) )
	except Exception:
		usage[ 'total_tokens' ] = (usage[ 'prompt_tokens' ] + usage[ 'completion_tokens' ])
	
	return usage

def update_token_counters( resp: Any ) -> None:
	"""
		Update session_state.last_call_usage and accumulate into session_state.token_usage.
	"""
	usage = extract_usage( resp )
	st.session_state.last_call_usage = usage
	st.session_state.token_usage[ 'prompt_tokens' ] += usage.get( 'prompt_tokens', 0 )
	st.session_state.token_usage[ 'completion_tokens' ] += usage.get( 'completion_tokens', 0 )
	st.session_state.token_usage[ 'total_tokens' ] += usage.get( 'total_tokens', 0 )

def display_value( val: Any ) -> str:
	"""
	
		Render a friendly display string for header values.
		None -> em dash; otherwise str(value).
		
	"""
	if val is None:
		return '—'
	try:
		return str( val )
	except Exception:
		return '—'

# ----------- RESPONSE/CHAT API UTILITIES ------------

def extract_response_text( response: object ) -> str:
	"""
		
		Purpose:
		--------
		Safely extract assistant text from a Responses API object.
	
		Parameters:
		-----------
		response (object): The response returned from the OpenAI client.
	
		Returns:
		--------
		str: Concatenated assistant text output. Empty string if none found.
		
	"""
	if response is None:
		return ""
	
	output = getattr( response, 'output', None )
	if not output or not isinstance( output, list ):
		return ''
	
	text_chunks: list[ str ] = [ ]
	
	for item in output:
		if not hasattr( item, 'type' ):
			continue
		
		if item.type == 'message':
			content = getattr( item, 'content', None )
			if not content or not isinstance( content, list ):
				continue
			
			for part in content:
				if getattr( part, 'type', None ) == 'output_text':
					text = getattr( part, 'text', "" )
					if text:
						text_chunks.append( text )
	
	return "".join( text_chunks ).strip( )

def encode_image( path: str ) -> str:
	"""
	
		Purpose:
		_________
		
		Parametes:
		----------
		
		
		Returns:
		--------
		
		
	"""
	data = Path( path ).read_bytes( )
	return base64.b64encode( data ).decode( "utf-8" )

def normalize_text( text: str ) -> str:
	"""
		
		Purpose
		-------
		Normalize text by:
			• Converting to lowercase
			• Removing punctuation except sentence delimiters (. ! ?)
			• Ensuring clean sentence boundary spacing
			• Collapsing whitespace
	
		Parameters
		----------
		text: str
	
		Returns
		-------
		str
		
	"""
	if not text:
		return ""
	
	# Lowercase
	text = text.lower( )
	
	# Remove punctuation except . ! ?
	text = re.sub( r"[^\w\s\.\!\?]", "", text )
	
	# Ensure single space after sentence delimiters
	text = re.sub( r"([.!?])\s*", r"\1 ", text )
	
	# Normalize whitespace
	text = re.sub( r"\s+", " ", text ).strip( )
	
	return text

def chunk_text( text: str, max_tokens: int=400 ) -> list[ str ]:
	"""
		
		Purpose
		-------
		Segment normalized text into chunks by:
			1. Sentence boundaries
			2. Fallback to token windowing if needed
	
		Parameters
		----------
		text: str
		max_tokens: int
	
		Returns
		-------
		list[str]
		
	"""
	if not text:
		return [ ]
	
	# Sentence-based segmentation
	sentences = re.split( r"(?<=[.!?])\s+", text )
	sentences = [ s.strip( ) for s in sentences if s.strip( ) ]
	
	if len( sentences ) > 1:
		return sentences
	
	# Fallback: token window segmentation
	words = text.split( )
	chunks = [ ]
	current_chunk = [ ]
	token_count = 0
	
	for word in words:
		current_chunk.append( word )
		token_count += 1
		
		if token_count >= max_tokens:
			chunks.append( " ".join( current_chunk ) )
			current_chunk = [ ]
			token_count = 0
	
	if current_chunk:
		chunks.append( " ".join( current_chunk ) )
	
	return chunks

def cosine_sim( a: np.ndarray, b: np.ndarray ) -> float:
	denom = np.linalg.norm( a ) * np.linalg.norm( b )
	return float( np.dot( a, b ) / denom ) if denom else 0.0

def sanitize_markdown( text: str ) -> str:
	"""
	
		Purpose:
		_________
		
		
	"""
	# Remove bold markers
	text = re.sub( r"\*\*(.*?)\*\*", r"\1", text )
	# Optional: remove italics
	text = re.sub( r"\*(.*?)\*", r"\1", text )
	return text

def init_state( ) -> None:
	"""
	
		Purpose:
		_________
		Initializes all session state variables.
		
		
	"""
	for k in ('audio_system_instructions',
	          'image_system_instructions',
	          'docqna_system_instructions',
	          'text_system_instructions'):
		st.session_state.setdefault( k, '' )

def reset_state( ) -> None:
	"""
	
		Purpose:
		_________
		Resets the session state to default values
		
	"""
	st.session_state.chat_history = [ ]
	st.session_state.messages = [ ]
	st.session_state.last_answer = ''
	st.session_state.last_sources = [ ]
	
def normalize( obj ):
	if obj is None or isinstance( obj, (str, int, float, bool) ):
		return obj
	
	if isinstance( obj, dict ):
		return { k: normalize( v ) for k, v in obj.items( ) }
	
	if isinstance( obj, (list, tuple, set) ):
		return [ normalize( v ) for v in obj ]
	if hasattr( obj, 'model_dump' ):
		try:
			return obj.model_dump( )
		except Exception:
			return str( obj )
	return str( obj )

def extract_sources( response: Any ) -> List[ Dict[ str, Any ] ]:
	"""
	
		Purpose:
		_________
		Parses-out sources from structured response object.
		
		Parameters:
		------------
		response: Any
			Structured API response.
		
		Returns:
		---------
		List[ Dict[ str, Any ] ]
			List of normalized source dictionaries.
	
	"""
	sources: List[ Dict[ str, Any ] ] = [ ]
	
	if response is None:
		return sources
	
	output = getattr( response, 'output', None )
	if not isinstance( output, list ):
		return sources
	
	for item in output:
		if item is None:
			continue
		
		t = getattr( item, 'type', None )
		
		# ------------------------------------------------
		# Web search
		# ------------------------------------------------
		if t == 'web_search_call':
			action = getattr( item, 'action', None )
			raw = getattr( action, 'sources', None ) if action else None
			
			if not isinstance( raw, (list, tuple) ):
				continue
			
			for src in raw:
				s = normalize( src )
				if not isinstance( s, dict ):
					continue
				
				sources.append( { 'title': s.get( 'title' ), 'snippet': s.get( 'snippet' ),
				                  'url': s.get( 'url' ), 'files_id': None, } )
		
		# ------------------------------------------------
		# File search (vector store)
		# ------------------------------------------------
		elif t == 'file_search_call':
			raw = getattr( item, 'results', None )
			if not isinstance( raw, (list, tuple) ):
				continue
			
			for r in raw:
				s = normalize( r )
				if not isinstance( s, dict ):
					continue
				
				sources.append( { 'title': s.get( 'file_name' ) or s.get( 'title' ),
				                  'snippet': s.get( 'text' ), 'url': None,
				                  'files_id': s.get( 'files_id' ), } )
	
	return sources

def save_temp( upload ) -> str | None:
	"""
	
		Purpose:
		--------
		Save a Streamlit UploadedFile object to a temporary file on disk
		and return the filesystem path.
	
		Parameters:
		-----------
		upload : streamlit.runtime.uploaded_file_manager.UploadedFile
			Uploaded file object from st.file_uploader.
	
		Returns:
		--------
		str | None
			Path to the temporary file, or None if invalid input.
			
	"""
	if upload is None:
		return None
	
	try:
		_, ext = os.path.splitext( upload.name )
		ext = ext or ""
		with tempfile.NamedTemporaryFile( delete=False, suffix=ext ) as tmp:
			tmp.write( upload.getbuffer( ) )
			tmp_path = tmp.name
		
		return tmp_path
	except Exception:
		return None

def display_value( val: Any ) -> str:
	"""
	
		Render a friendly display string for header values.
		None -> em dash; otherwise str(value).
		
	"""
	if val is None:
		return '—'
	try:
		return str( val )
	except Exception:
		return '—'

def format_results( results ):
	formatted_results = ''
	for result in results.data:
		formatted_result = f'<li> "{result.name}" '
		formatted_results += formatted_result + '</li>'
	return f'<p>{formatted_results}</p>'

def count_tokens( text: str ) -> int:
	"""
		
		Purpose
		----------
		Returns the number of tokens in a text string.
		
		Parmeters
		-----------
		string : str
		encoding_name : str
		
		Return
		------------
		int
		
	"""
	encoding = tiktoken.get_encoding( 'cl100k_base' )
	num_tokens = len( encoding.encode( text ) )
	return num_tokens

# ------------ TEXT UTILITIES -----------------

def convert_xml( text: str ) -> str:
	"""
		
			Purpose:
			_________
			Convert XML-delimited prompt text into Markdown by treating XML-like
			tags as section delimiters, not as strict XML.
	
			Parameters:
			-----------
			text (str) - Prompt text containing XML-like opening and closing tags.
	
			Returns:
			---------
			Markdown-formatted text using level-2 headings (##).
	"""
	markdown_blocks: List[ str ] = [ ]
	for match in cfg.XML_BLOCK_PATTERN.finditer( text ):
		raw_tag: str = match.group( 'tag' )
		body: str = match.group( 'body' ).strip( )
		
		# Humanize tag name for Markdown heading
		heading: str = raw_tag.replace( '_', ' ' ).replace( '-', ' ' ).title( )
		markdown_blocks.append( f'## {heading}' )
		if body:
			markdown_blocks.append( body )
	return "\n\n".join( markdown_blocks )

def convert_markdown( text: Any ) -> str:
	"""
	
		Purpose:
		--------
		Convert between Markdown headings and simple XML-like heading tags.
	
		Behavior:
		---------
		Auto-detects direction:
		  - If <h1>...</h1> / <h2>...</h2> ... exist, converts to Markdown (# / ## / ###).
		  - Otherwise converts Markdown headings (# / ## / ###) to <hN>...</hN> tags.
	
		Parameters:
		-----------
		text : Any
			Source text. Non-string values return "".
	
		Returns:
		--------
		str
			Converted text.
			
	"""
	if not isinstance( text, str ) or not text.strip( ):
		return ""
	
	# Normalize newlines
	src = text.replace( '\r\n', '\n' ).replace( '\r', '\n' )
	htag_pattern = re.compile( r"<h([1-6])>(.*?)</h\1>", flags=re.IGNORECASE | re.DOTALL )
	md_heading_pattern = re.compile( r"^(#{1,6})[ \t]+(.+?)[ \t]*$", flags=re.MULTILINE )
	
	# ------------------------------------------------------------------
	# Direction detection
	# ------------------------------------------------------------------
	contains_htags = bool( htag_pattern.search( src ) )
	
	# ------------------------------------------------------------------
	# XML-like heading tags -> Markdown headings
	# ------------------------------------------------------------------
	if contains_htags:
		def _htag_to_md( match: re.Match ) -> str:
			level = int( match.group( 1 ) )
			content = match.group( 2 ).strip( )
			
			# Preserve inner newlines safely by collapsing interior whitespace
			# while keeping content readable.
			content = re.sub( r"[ \t]+\n", "\n", content )
			content = re.sub( r"\n[ \t]+", "\n", content )
			
			return f"{'#' * level} {content}"
		
		out = htag_pattern.sub( _htag_to_md, src )
		return out.strip( )
	
	# ------------------------------------------------------------------
	# Markdown headings -> XML-like heading tags
	# ------------------------------------------------------------------
	def _md_to_htag( match: re.Match ) -> str:
		hashes = match.group( 1 )
		content = match.group( 2 ).strip( )
		level = len( hashes )
		return f"<h{level}>{content}</h{level}>"
	
	out = md_heading_pattern.sub( _md_to_htag, src )
	return out.strip( )

def inject_response_css( ) -> None:
	"""
	
		Purpose:
		_________
		Set the the format via css.
		
	"""
	st.markdown(
		"""
		<style>
		/* Chat message text */
		.stChatMessage p {
			color: rgb(220, 220, 220);
			font-size: 1rem;
			line-height: 1.6;
		}

		/* Headings inside chat responses */
		.stChatMessage h1 {
			color: rgb(0, 120, 252); /* DoD Blue */
			font-size: 1.6rem;
		}

		.stChatMessage h2 {
			color: rgb(0, 120, 252);
			font-size: 1.35rem;
		}

		.stChatMessage h3 {
			color: rgb(0, 120, 252);
			font-size: 1.15rem;
		}
		
		.stChatMessage a {
			color: rgb(0, 120, 252); /* DoD Blue */
			text-decoration: underline;
		}
		
		.stChatMessage a:hover {
			color: rgb(80, 160, 255);
		}

		</style>
		""", unsafe_allow_html=True )

def style_subheaders( ) -> None:
	"""
	
		Purpose:
		_________
		Sets the style of subheaders in the main UI
		
	"""
	st.markdown(
		"""
		<style>
		div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stMarkdownContainer"] h3,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h3 {
			color: rgb(0, 120, 252) !important;
		}
		</style>
		""",
		unsafe_allow_html=True, )

def save_message( role: str, content: str ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( 'INSERT INTO chat_history (role, content) VALUES (?, ?)', (role, content) )

def load_history( ) -> List[ Tuple[ str, str ] ]:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		return conn.execute( 'SELECT role, content FROM chat_history ORDER BY id' ).fetchall( )

def clear_history( ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( "DELETE FROM chat_history" )

def ensure_text_mode_state( ) -> None:
	"""
	
		Purpose:
		--------
		Ensure Text mode session-state keys used by API request helpers exist before
		widget instantiation.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	if 'text_vector_store_ids' not in st.session_state:
		st.session_state[ 'text_vector_store_ids' ] = ''
	
	if 'text_json_schema_name' not in st.session_state:
		st.session_state[ 'text_json_schema_name' ] = 'structured_response'
	
	if 'text_json_schema' not in st.session_state:
		st.session_state[ 'text_json_schema' ] = ''
	
	if 'text_json_schema_strict' not in st.session_state:
		st.session_state[ 'text_json_schema_strict' ] = True
	
	if 'text_conversation_id' not in st.session_state:
		st.session_state[ 'text_conversation_id' ] = ''
	
	if 'text_stream' not in st.session_state:
		st.session_state[ 'text_stream' ] = False
	
	if 'text_background' not in st.session_state:
		st.session_state[ 'text_background' ] = False

def parse_text_vector_store_ids( value: str | list[ str ] | None ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Parse comma-delimited or list-based vector store identifiers for OpenAI file_search.
		
		Parameters:
		-----------
		value: str | list[str] | None
			Comma-delimited vector store IDs or an existing list of IDs.
		
		Returns:
		--------
		list[str]
			Clean vector store IDs.
		
	"""
	if value is None:
		return [ ]
	
	if isinstance( value, list ):
		return [ str( item ).strip( ) for item in value if str( item ).strip( ) ]
	
	if not isinstance( value, str ) or not value.strip( ):
		return [ ]
	
	return [ item.strip( ) for item in value.split( ',' ) if item.strip( ) ]

def build_text_response_format( response_format: str | None, schema_name: str | None = None,
		schema_text: str | None = None, strict: bool = True ) -> dict[ str, Any ] | None:
	"""
	
		Purpose:
		--------
		Build the Responses API text.format object from Text mode response-format
		controls.
		
		Parameters:
		-----------
		response_format: str | None
			Selected response format: text, json_object, or json_schema.
		
		schema_name: str | None
			Schema name used when response_format is json_schema.
		
		schema_text: str | None
			JSON Schema text used when response_format is json_schema.
		
		strict: bool
			Whether schema adherence should be strict for json_schema output.
		
		Returns:
		--------
		dict[str, Any] | None
			Responses API text-format object, or None when no format should be sent.
		
	"""
	if not isinstance( response_format, str ) or not response_format.strip( ):
		return None
	
	format_name = response_format.strip( )
	
	if format_name == 'text':
		return {
				'format': {
						'type': 'text',
				}
		}
	
	if format_name == 'json_object':
		return {
				'format': {
						'type': 'json_object',
				}
		}
	
	if format_name == 'json_schema':
		if not isinstance( schema_text, str ) or not schema_text.strip( ):
			st.warning( 'JSON Schema output requires a schema. Falling back to plain text.' )
			return {
					'format': {
							'type': 'text',
					}
			}
		
		try:
			schema = json.loads( schema_text )
		except Exception as exc:
			st.warning( f'JSON Schema could not be parsed. Falling back to plain text: {exc}' )
			return {
					'format': {
							'type': 'text',
					}
			}
		
		name = schema_name if isinstance( schema_name, str ) and schema_name.strip( ) else \
			'structured_response'
		
		return {
				'format': {
						'type': 'json_schema',
						'name': name.strip( ),
						'schema': schema,
						'strict': bool( strict ),
				}
		}
	
	return None

def build_text_tools( selected_tools: list[ str ] | None,
		vector_store_ids: list[ str ] | None = None ) -> list[ dict[ str, Any ] ]:
	"""
	
		Purpose:
		--------
		Build a safe list of Text mode tool dictionaries for the Chat wrapper.
		
		Parameters:
		-----------
		selected_tools: list[str] | None
			Tool names selected in the Text mode UI.
		
		vector_store_ids: list[str] | None
			Vector store IDs used by the file_search tool.
		
		Returns:
		--------
		list[dict[str, Any]]
			Tool dictionaries suitable for Chat.generate_text().
		
	"""
	tools: list[ dict[ str, Any ] ] = [ ]
	vector_ids = vector_store_ids if vector_store_ids is not None else [ ]
	
	if selected_tools is None or len( selected_tools ) == 0:
		return tools
	
	for name in selected_tools:
		if not isinstance( name, str ) or not name.strip( ):
			continue
		
		tool_name = name.strip( )
		
		if tool_name == 'web_search':
			tools.append( { 'type': 'web_search' } )
			continue
		
		if tool_name == 'file_search':
			if len( vector_ids ) == 0:
				st.warning( 'File Search was selected, but no vector store IDs were provided.' )
				continue
			
			tools.append( { 'type': 'file_search' } )
			continue
	
	return tools

def build_text_include( selected_include: list[ str ] | None,
		selected_tools: list[ dict[ str, Any ] ] | None = None ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Filter Text mode include fields so they correspond to the selected tool context.
		
		Parameters:
		-----------
		selected_include: list[str] | None
			Include values selected in the UI.
		
		selected_tools: list[dict[str, Any]] | None
			Normalized tool dictionaries selected for the request.
		
		Returns:
		--------
		list[str]
			Filtered include values.
		
	"""
	if selected_include is None or len( selected_include ) == 0:
		return [ ]
	
	tool_types: list[ str ] = [ ]
	if isinstance( selected_tools, list ):
		for tool in selected_tools:
			if isinstance( tool, dict ) and tool.get( 'type' ):
				tool_types.append( str( tool.get( 'type' ) ) )
	
	include_values: list[ str ] = [ ]
	
	for value in selected_include:
		if not isinstance( value, str ) or not value.strip( ):
			continue
		
		include_name = value.strip( )
		
		if include_name == 'reasoning.encrypted_content':
			include_values.append( include_name )
			continue
		
		if include_name == 'message.output_text.logprobs':
			include_values.append( include_name )
			continue
		
		if include_name.startswith( 'web_search_call.' ) and 'web_search' in tool_types:
			include_values.append( include_name )
			continue
		
		if include_name == 'file_search_call.results' and 'file_search' in tool_types:
			include_values.append( include_name )
			continue
	
	return include_values

def build_text_tool_choice( tool_choice: str | None,
		selected_tools: list[ dict[ str, Any ] ] | None = None ) -> str | None:
	"""
	
		Purpose:
		--------
		Safely pass tool_choice only when it is compatible with the final tool list.
		
		Parameters:
		-----------
		tool_choice: str | None
			Selected tool-choice option.
		
		selected_tools: list[dict[str, Any]] | None
			Final tool dictionaries selected for the request.
		
		Returns:
		--------
		str | None
			Tool-choice value or None.
		
	"""
	if not isinstance( tool_choice, str ) or not tool_choice.strip( ):
		return None
	
	choice = tool_choice.strip( )
	if choice not in [ 'auto', 'required', 'none' ]:
		return None
	
	if choice == 'none':
		return 'none'
	
	if selected_tools is None or len( selected_tools ) == 0:
		return None
	
	return choice

def build_text_context( messages: list[ dict[ str, Any ] ] | None,
		include_last_message: bool = False ) -> list[ dict[ str, str ] ]:
	"""
	
		Purpose:
		--------
		Build clean Text mode conversation context from Streamlit message state.
		
		Parameters:
		-----------
		messages: list[dict[str, Any]] | None
			Text mode messages stored in session state.
		
		include_last_message: bool
			Whether the final message should be included in context.
		
		Returns:
		--------
		list[dict[str, str]]
			Clean message context for the Responses API wrapper.
		
	"""
	if messages is None or not isinstance( messages, list ):
		return [ ]
	
	items = messages if include_last_message else messages[ :-1 ]
	context: list[ dict[ str, str ] ] = [ ]
	
	for item in items:
		if not isinstance( item, dict ):
			continue
		
		role = str( item.get( 'role', '' ) or '' ).strip( )
		content = item.get( 'content', '' )
		
		if role not in [ 'user', 'assistant', 'system', 'developer' ]:
			continue
		
		if not isinstance( content, str ) or not content.strip( ):
			continue
		
		context.append(
			{
					'role': role,
					'content': content.strip( ),
			} )
	
	return context

def get_text_conversation_id( input_mode: str | None, conversation_id: str | None ) -> str | None:
	"""
	
		Purpose:
		--------
		Return a conversation identifier only when Text mode explicitly selects API
		conversation state.
		
		Parameters:
		-----------
		input_mode: str | None
			Text mode input behavior selection.
		
		conversation_id: str | None
			OpenAI conversation identifier.
		
		Returns:
		--------
		str | None
			Conversation identifier or None.
		
	"""
	if input_mode != 'conversation':
		return None
	
	if not isinstance( conversation_id, str ) or not conversation_id.strip( ):
		return None
	
	return conversation_id.strip( )

def get_text_previous_response_id( input_mode: str | None, previous_id: str | None ) -> str | None:
	"""
	
		Purpose:
		--------
		Return previous_response_id only when the current input mode allows response
		chaining.
		
		Parameters:
		-----------
		input_mode: str | None
			Text mode input behavior selection.
		
		previous_id: str | None
			Prior Responses API response identifier.
		
		Returns:
		--------
		str | None
			Previous response identifier or None.
		
	"""
	if input_mode == 'single_turn':
		return None
	
	if input_mode == 'conversation':
		return None
	
	if not isinstance( previous_id, str ) or not previous_id.strip( ):
		return None
	
	return previous_id.strip( )

def get_text_stream_value( stream_value: bool | None ) -> None:
	"""
	
		Purpose:
		--------
		Suppress streaming for the current non-streaming Text mode execution path.
		
		Parameters:
		-----------
		stream_value: bool | None
			Stream toggle value.
		
		Returns:
		--------
		None
		
	"""
	if bool( stream_value ):
		st.info(
			'Streaming is not sent in this Text mode path until stream-event rendering is added.' )
	
	return None

def get_text_background_value( background_value: bool | None ) -> None:
	"""
	
		Purpose:
		--------
		Suppress background execution for the current immediate-response Text mode path.
		
		Parameters:
		-----------
		background_value: bool | None
			Background toggle value.
		
		Returns:
		--------
		None
		
	"""
	if bool( background_value ):
		st.info( 'Background mode is not sent in this Text mode path until polling is added.' )
	
	return None

def reset_text_structured_output_controls( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Text mode structured-output controls through a widget-safe callback.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	for key in [ 'text_json_schema_name', 'text_json_schema',
	             'text_json_schema_strict' ]:
		if key in st.session_state:
			del st.session_state[ key ]

def reset_text_api_state_controls( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Text mode API state controls through a widget-safe callback.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	for key in [ 'text_previous_response_id', 'text_conversation_id' ]:
		if key in st.session_state:
			del st.session_state[ key ]

# ------------ IMAGE API UTILITIES -----------

def clear_image_messages( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Image Mode message and output state.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'image_input' ] = [ ]
	st.session_state[ 'image_messages' ] = [ ]
	st.session_state[ 'image_output_bytes' ] = None

def clear_image_instructions( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Image Mode system instructions and the selected prompt template.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'image_system_instructions' ] = ''
	st.session_state[ 'instructions' ] = ''

def append_image_message( role: str, content: str ) -> None:
	"""
	
		Purpose:
		--------
		Append a message to Image Mode state.
		
		Parameters:
		-----------
		role: str
			Message role.
		
		content: str
			Message content.
		
		Returns:
		--------
		None
		
	"""
	if 'image_input' not in st.session_state or not isinstance(
			st.session_state[ 'image_input' ], list ):
		st.session_state[ 'image_input' ] = [ ]
	
	if 'image_messages' not in st.session_state or not isinstance(
			st.session_state[ 'image_messages' ], list ):
		st.session_state[ 'image_messages' ] = [ ]
	
	message = {
			'role': role,
			'content': content,
	}
	
	st.session_state[ 'image_input' ].append( message )
	st.session_state[ 'image_messages' ].append( message )

def load_image_instruction_template( ) -> None:
	"""
	
		Purpose:
		--------
		Load the selected prompt template into Image Mode system instructions.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	name = st.session_state.get( 'instructions' )
	if name and name != 'No Templates Found':
		text = fetch_prompt_text( cfg.DB_PATH, name )
		if text is not None:
			st.session_state[ 'image_system_instructions' ] = text

def convert_image_system_instructions( ) -> None:
	"""
	
		Purpose:
		--------
		Convert Image Mode system instructions between XML-like delimiters and Markdown
		headings.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	text = st.session_state.get( 'image_system_instructions', '' )
	if not isinstance( text, str ) or not text.strip( ):
		return
	
	source = text.strip( )
	if cfg.XML_BLOCK_PATTERN.search( source ):
		converted = convert_xml( source )
	else:
		converted = convert_markdown( source )
	
	st.session_state[ 'image_system_instructions' ] = converted

def reset_image_llm_settings( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Image Mode model-selection controls through a widget-safe callback.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	for key in [ 'image_mode', 'image_model', 'image_analysis_model', 'image_number' ]:
		if key in st.session_state:
			del st.session_state[ key ]

def reset_image_visual_settings( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Image Mode visual controls through a widget-safe callback.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	for key in [ 'image_mime_type', 'image_size', 'image_quality', 'image_backcolor',
	             'image_compression' ]:
		if key in st.session_state:
			del st.session_state[ key ]

def get_image_models( image: Images ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Return GPT Image model options for generation and editing.
		
		Parameters:
		-----------
		image: Images
			Image wrapper instance.
		
		Returns:
		--------
		list[str]
			Image model options.
		
	"""
	options = getattr( image, 'model_options', None )
	if isinstance( options, list ) and len( options ) > 0:
		return [ '' ] + options
	
	return [
			'',
			'gpt-image-2',
			'gpt-image-1.5',
			'gpt-image-1',
			'gpt-image-1-mini',
	]

def get_image_analysis_models( image: Images = None ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Return vision-capable Responses API model options for image analysis.
		
		Parameters:
		-----------
		image: Images
			Optional Image wrapper instance.
		
		Returns:
		--------
		list[str]
			Image-analysis model options.
		
	"""
	if image is not None:
		options = getattr( image, 'analysis_model_options', None )
		if isinstance( options, list ) and len( options ) > 0:
			return [ '' ] + options
	
	return [
			'',
			'gpt-4o-mini',
			'gpt-4o',
			'gpt-4.1-mini',
			'gpt-4.1',
			'gpt-5-mini',
			'gpt-5',
	]

def get_image_size_options( image: Images ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Return image size options from the wrapper when available.
		
		Parameters:
		-----------
		image: Images
			Image wrapper instance.
		
		Returns:
		--------
		list[str]
			Image size options.
		
	"""
	options = getattr( image, 'size_options', None )
	if isinstance( options, list ) and len( options ) > 0:
		return [ '' ] + options
	
	return [ '', 'auto', '1024x1024', '1024x1536', '1536x1024' ]

def get_image_quality_options( image: Images ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Return image quality options from the wrapper when available.
		
		Parameters:
		-----------
		image: Images
			Image wrapper instance.
		
		Returns:
		--------
		list[str]
			Image quality options.
		
	"""
	options = getattr( image, 'quality_options', None )
	if isinstance( options, list ) and len( options ) > 0:
		return [ '' ] + options
	
	return [ '', 'auto', 'low', 'medium', 'high' ]

def get_image_mime_options( image: Images ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Return image output format options from the wrapper when available.
		
		Parameters:
		-----------
		image: Images
			Image wrapper instance.
		
		Returns:
		--------
		list[str]
			Image output format options.
		
	"""
	options = getattr( image, 'mime_options', None )
	if isinstance( options, list ) and len( options ) > 0:
		return [ '' ] + options
	
	return [ '', 'png', 'jpeg', 'webp' ]

def get_image_background_options( image: Images ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Return image background options from the wrapper when available.
		
		Parameters:
		-----------
		image: Images
			Image wrapper instance.
		
		Returns:
		--------
		list[str]
			Image background options.
		
	"""
	options = getattr( image, 'backcolor_options', None )
	if isinstance( options, list ) and len( options ) > 0:
		return [ '' ] + options
	
	return [ '', 'auto', 'transparent', 'opaque' ]

def get_image_detail_options( image: Images ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Return image-analysis detail options from the wrapper when available.
		
		Parameters:
		-----------
		image: Images
			Image wrapper instance.
		
		Returns:
		--------
		list[str]
			Image-analysis detail options.
		
	"""
	options = getattr( image, 'detail_options', None )
	if isinstance( options, list ) and len( options ) > 0:
		return [ '' ] + options
	
	return [ '', 'auto', 'low', 'high', 'original' ]

def render_image_output( image_result: str | bytes | list[ str | bytes ] | None,
		caption: str = 'Image output' ) -> bool:
	"""
	
		Purpose:
		--------
		Render image output returned from the Images wrapper, including single outputs
		and multi-image output lists.
		
		Parameters:
		-----------
		image_result: str | bytes | list[ str | bytes ] | None
			Image bytes, image URL, list of image outputs, or None.
		
		caption: str
			Base caption used when rendering one or more images.
		
		Returns:
		--------
		bool
			True when at least one image output is rendered; otherwise False.
		
	"""
	if image_result is None:
		return False
	
	outputs: list[ str | bytes ] = image_result if isinstance( image_result, list ) else [
			image_result ]
	rendered = False
	
	for index, item in enumerate( outputs, start=1 ):
		if item is None:
			continue
		
		if isinstance( item, bytes ) and len( item ) > 0:
			if len( outputs ) > 1:
				st.image( item, caption=f'{caption} {index}', use_column_width=True )
			else:
				st.image( item, caption=caption, use_column_width=True )
			
			rendered = True
			continue
		
		if isinstance( item, str ) and item.strip( ):
			if item.strip( ).lower( ).startswith( ('http://', 'https://') ):
				if len( outputs ) > 1:
					st.image( item.strip( ), caption=f'{caption} {index}', use_column_width=True )
				else:
					st.image( item.strip( ), caption=caption, use_column_width=True )
				
				rendered = True
			else:
				st.markdown( item.strip( ) )
				rendered = True
	
	return rendered

# ------------- EMBEDDINGS API UTILITIES ---------------------

def ensure_embeddings_mode_state( ) -> None:
	"""
	
		Purpose:
		--------
		Ensure Embeddings mode session-state keys exist before widget instantiation.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	if 'embedding_model' not in st.session_state:
		st.session_state[ 'embedding_model' ] = ''
	
	if 'embeddings_dimensions' not in st.session_state:
		st.session_state[ 'embeddings_dimensions' ] = 0
	
	if 'embeddings_chunk_size' not in st.session_state:
		st.session_state[ 'embeddings_chunk_size' ] = 800
	
	if 'embeddings_overlap_amount' not in st.session_state:
		st.session_state[ 'embeddings_overlap_amount' ] = 0
	
	if 'embeddings_input_text' not in st.session_state:
		st.session_state[ 'embeddings_input_text' ] = ''
	
	if 'embeddings_encoding_format' not in st.session_state:
		st.session_state[ 'embeddings_encoding_format' ] = 'float'
	
	if 'embeddings_user' not in st.session_state:
		st.session_state[ 'embeddings_user' ] = ''
	
	if 'embeddings' not in st.session_state:
		st.session_state[ 'embeddings' ] = [ ]
	
	if 'embeddings_chunks' not in st.session_state:
		st.session_state[ 'embeddings_chunks' ] = [ ]
	
	if 'embeddings_df' not in st.session_state:
		st.session_state[ 'embeddings_df' ] = pd.DataFrame( )
	
	if 'embedding_metrics' not in st.session_state:
		st.session_state[ 'embedding_metrics' ] = { }
	
	if 'embedding_usage' not in st.session_state:
		st.session_state[ 'embedding_usage' ] = { }

def get_embedding_model_options( embedding: Embeddings ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Return OpenAI embedding model options from the wrapper when available.
		
		Parameters:
		-----------
		embedding: Embeddings
			Embeddings wrapper instance.
		
		Returns:
		--------
		list[str]
			Embedding model options.
		
	"""
	options = getattr( embedding, 'model_options', None )
	if isinstance( options, list ) and len( options ) > 0:
		return [ '' ] + options
	
	return [
			'',
			'text-embedding-3-small',
			'text-embedding-3-large',
			'text-embedding-ada-002',
	]

def get_embedding_encoding_options( embedding: Embeddings ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Return OpenAI embedding encoding-format options from the wrapper when available.
		
		Parameters:
		-----------
		embedding: Embeddings
			Embeddings wrapper instance.
		
		Returns:
		--------
		list[str]
			Encoding format options.
		
	"""
	options = getattr( embedding, 'encoding_options', None )
	if isinstance( options, list ) and len( options ) > 0:
		return options
	
	return [
			'float',
			'base64',
	]

def get_embedding_max_dimensions( model: str | None, embedding: Embeddings ) -> int:
	"""
	
		Purpose:
		--------
		Return the maximum supported dimensions for the selected embedding model.
		
		Parameters:
		-----------
		model: str | None
			Selected embedding model.
		
		embedding: Embeddings
			Embeddings wrapper instance.
		
		Returns:
		--------
		int
			Maximum supported dimensions.
		
	"""
	if not isinstance( model, str ) or not model.strip( ):
		return 1536
	
	try:
		return int( embedding.get_max_dimensions( model.strip( ) ) )
	except Exception:
		if model == 'text-embedding-3-large':
			return 3072
		
		return 1536

def embedding_model_supports_dimensions( model: str | None, embedding: Embeddings ) -> bool:
	"""
	
		Purpose:
		--------
		Determine whether the selected embedding model supports the dimensions parameter.
		
		Parameters:
		-----------
		model: str | None
			Selected embedding model.
		
		embedding: Embeddings
			Embeddings wrapper instance.
		
		Returns:
		--------
		bool
			True when the model supports the dimensions parameter.
		
	"""
	if not isinstance( model, str ) or not model.strip( ):
		return False
	
	support = getattr( embedding, 'model_dimension_support', { } )
	if isinstance( support, dict ):
		return bool( support.get( model.strip( ), False ) )
	
	return model.strip( ) in [
			'text-embedding-3-small',
			'text-embedding-3-large',
	]

def normalize_embedding_dimensions( model: str | None, dimensions: int | None,
		embedding: Embeddings ) -> int | None:
	"""
	
		Purpose:
		--------
		Normalize dimensions before calling the OpenAI Embeddings API.
		
		Parameters:
		-----------
		model: str | None
			Selected embedding model.
		
		dimensions: int | None
			Requested embedding dimensions.
		
		embedding: Embeddings
			Embeddings wrapper instance.
		
		Returns:
		--------
		int | None
			Valid dimensions value or None when the parameter should be omitted.
		
	"""
	if not isinstance( model, str ) or not model.strip( ):
		return None
	
	if dimensions is None:
		return None
	
	try:
		value = int( dimensions )
	except Exception:
		return None
	
	if value <= 0:
		return None
	
	if not embedding_model_supports_dimensions( model, embedding ):
		return None
	
	max_dimensions = get_embedding_max_dimensions( model, embedding )
	if value > max_dimensions:
		return max_dimensions
	
	return value

def normalize_embedding_chunk_settings( chunk_size: int | None,
		overlap_amount: int | None ) -> tuple[ int, int ]:
	"""
	
		Purpose:
		--------
		Normalize chunk size and overlap settings for tokenizer-aware chunking.
		
		Parameters:
		-----------
		chunk_size: int | None
			Requested chunk size in tokens.
		
		overlap_amount: int | None
			Requested overlap amount in tokens.
		
		Returns:
		--------
		tuple[int, int]
			Normalized chunk size and overlap amount.
		
	"""
	try:
		chunk_value = int( chunk_size )
	except Exception:
		chunk_value = 800
	
	try:
		overlap_value = int( overlap_amount )
	except Exception:
		overlap_value = 0
	
	if chunk_value <= 0:
		chunk_value = 800
	
	if chunk_value > 8192:
		chunk_value = 8192
	
	if overlap_value < 0:
		overlap_value = 0
	
	if overlap_value >= chunk_value:
		overlap_value = max( 0, chunk_value // 5 )
	
	return chunk_value, overlap_value

def chunk_text_for_embeddings( text: str, chunk_size: int = 800,
		overlap_amount: int = 0, encoding_name: str = 'cl100k_base' ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Split text into tokenizer-aware chunks for the OpenAI Embeddings API.
		
		Parameters:
		-----------
		text: str
			Input text to chunk.
		
		chunk_size: int
			Maximum chunk size in tokens.
		
		overlap_amount: int
			Token overlap between adjacent chunks.
		
		encoding_name: str
			Tiktoken encoding name.
		
		Returns:
		--------
		list[str]
			Text chunks.
		
	"""
	if not isinstance( text, str ) or not text.strip( ):
		return [ ]
	
	chunk_value, overlap_value = normalize_embedding_chunk_settings(
		chunk_size=chunk_size,
		overlap_amount=overlap_amount )
	
	encoding = tiktoken.get_encoding( encoding_name )
	tokens = encoding.encode( text )
	
	if len( tokens ) == 0:
		return [ ]
	
	chunks: list[ str ] = [ ]
	start = 0
	step = max( 1, chunk_value - overlap_value )
	
	while start < len( tokens ):
		end = min( start + chunk_value, len( tokens ) )
		chunk_tokens = tokens[ start:end ]
		chunk_text_value = encoding.decode( chunk_tokens ).strip( )
		
		if chunk_text_value:
			chunks.append( chunk_text_value )
		
		if end >= len( tokens ):
			break
		
		start += step
	
	return chunks

def normalize_embedding_vectors( vectors: Any ) -> list[ Any ]:
	"""
	
		Purpose:
		--------
		Normalize wrapper embedding output into a list of vectors or base64 strings.
		
		Parameters:
		-----------
		vectors: Any
			Embedding output returned by the wrapper.
		
		Returns:
		--------
		list[Any]
			Normalized embedding outputs.
		
	"""
	if vectors is None:
		return [ ]
	
	if isinstance( vectors, str ):
		return [ vectors ]
	
	if isinstance( vectors, list ):
		if len( vectors ) == 0:
			return [ ]
		
		if all( isinstance( value, (int, float) ) for value in vectors ):
			return [ vectors ]
		
		return vectors
	
	return [ vectors ]

def build_embeddings_dataframe( chunks: list[ str ], vectors: Any,
		encoding_format: str = 'float' ) -> pd.DataFrame:
	"""
	
		Purpose:
		--------
		Build a display DataFrame from embedding chunks and embedding outputs.
		
		Parameters:
		-----------
		chunks: list[str]
			Text chunks submitted to the Embeddings API.
		
		vectors: Any
			Embedding vectors or base64 strings returned by the wrapper.
		
		encoding_format: str
			Embedding encoding format: float or base64.
		
		Returns:
		--------
		pd.DataFrame
			Display-ready embeddings DataFrame.
		
	"""
	outputs = normalize_embedding_vectors( vectors )
	
	if len( outputs ) == 0:
		return pd.DataFrame( )
	
	rows: list[ dict[ str, Any ] ] = [ ]
	format_value = encoding_format if isinstance( encoding_format, str ) else 'float'
	
	if format_value == 'base64':
		for index, item in enumerate( outputs ):
			chunk = chunks[ index ] if index < len( chunks ) else ''
			rows.append(
				{
						'ChunkIndex': index + 1,
						'Chunk': chunk,
						'EmbeddingBase64': item if isinstance( item, str ) else str( item ),
				} )
		
		return pd.DataFrame( rows )
	
	for index, vector in enumerate( outputs ):
		chunk = chunks[ index ] if index < len( chunks ) else ''
		
		if not isinstance( vector, list ):
			rows.append(
				{
						'ChunkIndex': index + 1,
						'Chunk': chunk,
						'Embedding': str( vector ),
				} )
			continue
		
		row: dict[ str, Any ] = {
				'ChunkIndex': index + 1,
				'Chunk': chunk,
		}
		
		for dim_index, value in enumerate( vector ):
			row[ f'dim_{dim_index}' ] = value
		
		rows.append( row )
	
	return pd.DataFrame( rows )

def get_embedding_vector_dimension( vectors: Any ) -> int:
	"""
	
		Purpose:
		--------
		Return the numeric embedding vector dimension when available.
		
		Parameters:
		-----------
		vectors: Any
			Embedding output returned by the wrapper.
		
		Returns:
		--------
		int
			Vector dimension count, or zero for non-numeric/base64 output.
		
	"""
	outputs = normalize_embedding_vectors( vectors )
	if len( outputs ) == 0:
		return 0
	
	first = outputs[ 0 ]
	if isinstance( first, list ):
		return len( first )
	
	return 0

def extract_embedding_usage( response: Any ) -> dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Extract usage metadata from an OpenAI Embeddings API response object.
		
		Parameters:
		-----------
		response: Any
			Embeddings API response object.
		
		Returns:
		--------
		dict[str, Any]
			Normalized usage metadata.
		
	"""
	if response is None:
		return { }
	
	try:
		raw = getattr( response, 'usage', None )
	except Exception:
		raw = None
	
	if raw is None:
		return { }
	
	if isinstance( raw, dict ):
		return raw
	
	if hasattr( raw, 'model_dump' ):
		try:
			return raw.model_dump( )
		except Exception:
			return { 'raw': str( raw ) }
	
	return { 'raw': str( raw ) }

def build_embedding_metrics( source_text: str, normalized_text: str, chunks: list[ str ],
		vectors: Any, usage: dict[ str, Any ] | None = None ) -> dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Build Embeddings mode metrics for display and session-state storage.
		
		Parameters:
		-----------
		source_text: str
			Original input text.
		
		normalized_text: str
			Normalized text submitted to chunking.
		
		chunks: list[str]
			Embedding chunks.
		
		vectors: Any
			Embedding output returned by the wrapper.
		
		usage: dict[str, Any] | None
			Optional usage metadata from the API.
		
		Returns:
		--------
		dict[str, Any]
			Embedding metrics.
		
	"""
	source_value = source_text if isinstance( source_text, str ) else ''
	normalized_value = normalized_text if isinstance( normalized_text, str ) else ''
	outputs = normalize_embedding_vectors( vectors )
	
	words = normalized_value.split( )
	unique_words = set( words )
	token_total = count_tokens( normalized_value ) if normalized_value else 0
	vector_dimension = get_embedding_vector_dimension( outputs )
	
	metrics: dict[ str, Any ] = {
			'characters': len( source_value ),
			'normalized_characters': len( normalized_value ),
			'words': len( words ),
			'unique_words': len( unique_words ),
			'type_token_ratio': round( len( unique_words ) / len( words ), 4 ) if len(
				words ) else 0.0,
			'tokens': token_total,
			'chunks': len( chunks ),
			'embeddings': len( outputs ),
			'vector_dimension': vector_dimension,
			'encoding_format': st.session_state.get( 'embeddings_encoding_format', 'float' ),
			'usage': usage if isinstance( usage, dict ) else { },
	}
	
	return metrics

def render_embedding_metrics( metrics: dict[ str, Any ] | None ) -> None:
	"""
	
		Purpose:
		--------
		Render Embeddings mode metrics using Streamlit metric controls.
		
		Parameters:
		-----------
		metrics: dict[str, Any] | None
			Embedding metrics dictionary.
		
		Returns:
		--------
		None
		
	"""
	if not isinstance( metrics, dict ) or len( metrics ) == 0:
		return
	
	metric_c1, metric_c2, metric_c3, metric_c4, metric_c5 = st.columns(
		[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
	
	with metric_c1:
		st.metric( 'Tokens', metrics.get( 'tokens', 0 ) )
	
	with metric_c2:
		st.metric( 'Chunks', metrics.get( 'chunks', 0 ) )
	
	with metric_c3:
		st.metric( 'Embeddings', metrics.get( 'embeddings', 0 ) )
	
	with metric_c4:
		st.metric( 'Dimensions', metrics.get( 'vector_dimension', 0 ) )
	
	with metric_c5:
		st.metric( 'Words', metrics.get( 'words', 0 ) )

def render_embeddings_dataframe( df_embeddings: pd.DataFrame ) -> None:
	"""
	
		Purpose:
		--------
		Render Embeddings mode output DataFrame safely.
		
		Parameters:
		-----------
		df_embeddings: pd.DataFrame
			Embeddings output DataFrame.
		
		Returns:
		--------
		None
		
	"""
	if df_embeddings is None or df_embeddings.empty:
		st.info( 'No embeddings available.' )
		return
	
	st.data_editor( df_embeddings, use_container_width=True, hide_index=True )

def reset_embeddings_controls( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Embeddings mode configuration controls through a widget-safe callback.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	for key in [ 'embedding_model', 'embeddings_dimensions',
	             'embeddings_chunk_size', 'embeddings_overlap_amount',
	             'embeddings_encoding_format', 'embeddings_user' ]:
		if key in st.session_state:
			del st.session_state[ key ]

def clear_embeddings_output( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Embeddings mode outputs while preserving configuration controls.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'embeddings' ] = [ ]
	st.session_state[ 'embeddings_chunks' ] = [ ]
	st.session_state[ 'embeddings_df' ] = pd.DataFrame( )
	st.session_state[ 'embedding_metrics' ] = { }
	st.session_state[ 'embedding_usage' ] = { }

def reset_embeddings_all( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Embeddings mode configuration, input, and output state.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	reset_embeddings_controls( )
	clear_embeddings_output( )
	
	if 'embeddings_input_text' in st.session_state:
		del st.session_state[ 'embeddings_input_text' ]

# ----------  DOCQNA UTILITIES ----------

def route_document_query( prompt: str ) -> str:
	"""
	
		Purpose:
		--------
		Route a document question through the unified chat pipeline and return a
		model-generated answer.
	
		Parameters:
		-----------
		prompt : str
			The user question to answer about active documents.
	
		Returns:
		--------
		str
			The assistant answer text.
	
	"""
	prompt = str( prompt or '' ).strip( )
	if not prompt:
		return 'Please enter a question about the active document.'
	
	try:
		user_input = build_document_user_input( prompt )
	except Exception as exc:
		return f'Document retrieval failed: {exc}'
	
	if not user_input:
		user_input = prompt
		
	return str( user_input or '' ).strip( )

def summarize_active_document( ) -> str:
	"""
	
		Uses the routing layer to summarize the currently active document.
		
	"""
	system_instructions = st.session_state.get( 'system_instructions', '' )
	summary_prompt = """
		Provide a clear, structured summary of this document.
		Include:
		- Purpose
		- Key themes
		- Major conclusions
		- Important data points (if any)
		- Policy implications (if applicable)
		
		Be precise and concise.
		"""
	if system_instructions:
		summary_prompt = f"{system_instructions}\n\n{summary_prompt}"
	
	return route_document_query( summary_prompt.strip( ) )

def compute_fingerprint( active_docs: List[ str ], doc_bytes: Dict[ str, bytes ] ) -> str:
	'''
		
		Purpose:
		--------
		Computes a stable fingerprint for the currently selected active documents and their byte contents.
	
		Parameters:
		-----------
		active_docs:
			A List[ str ] of active document names.
		doc_bytes:
			A Dict[ str, bytes ] mapping document name to file bytes.
	
		Returns:
		--------
		A str fingerprint suitable for cache invalidation.
	
	'''
	h = hashlib.sha256( )
	for name in sorted( active_docs ):
		b = doc_bytes.get( name, b'' )
		h.update( name.encode( 'utf-8', errors='ignore' ) )
		h.update( len( b ).to_bytes( 8, 'little', signed=False ) )
		h.update( hashlib.sha256( b ).digest( ) )
	return h.hexdigest( )

def extract_text_from_pdf( file_bytes: bytes ) -> str:
	'''
	
		Purpose:
		--------
		Extracts text from a PDF byte stream using PyMuPDF.
	
		Parameters:
		-----------
		file_bytes:
			The PDF bytes.
	
		Returns:
		--------
		A str containing extracted text.
	
	'''
	if not file_bytes:
		return ''
	
	try:
		doc = fitz.open( stream=file_bytes, filetype='pdf' )
		parts: List[ str ] = [ ]
		for page in doc:
			parts.append( page.get_text( 'text' ) or '' )
		return '\n'.join( parts ).strip( )
	except Exception:
		return ''

def load_sqlite_vec( conn: sqlite3.Connection ) -> bool:
	'''
		
		Purpose:
		--------
		Attempts to load sqlite-vec into the provided SQLite connection.
	
		Parameters:
		-----------
		conn:
			The sqlite3.Connection.
	
		Returns:
		--------
		True if sqlite-vec loaded successfully; otherwise False.
		
	'''
	try:
		import sqlite_vec
		
		sqlite_vec.load( conn )
		return True
	except Exception:
		return False

def ensure_vec_schema( dim: int ) -> bool:
	'''
	
		Purpose:
		--------
		Creates the sqlite-vec virtual table used for Document Q&A embeddings if possible.
	
		Parameters:
		-----------
		dim:
			The embedding dimension (e.g., 384 for all-MiniLM-L6-v2).
	
		Returns:
		--------
		True if the schema exists and is usable; otherwise False.
	
	'''
	conn = create_connection( )
	try:
		ok = load_sqlite_vec( conn )
		if not ok:
			return False
		
		cur = conn.cursor( )
		cur.execute(
			f'''
			CREATE VIRTUAL TABLE IF NOT EXISTS docqna_vec
			USING vec0(
				embedding float[{int( dim )}],
				doc_name TEXT,
				chunk TEXT
			);
			'''
		)
		conn.commit( )
		return True
	except Exception:
		return False
	finally:
		conn.close( )

def rebuild_index( embedder: SentenceTransformer ) -> None:
	'''
		
		Purpose:
		--------
		Builds or refreshes the Document Q&A vector index when active documents change.
	
		Parameters:
		-----------
		embedder:
			The SentenceTransformer used to generate embeddings.
	
		Returns:
		--------
		None
		
	'''
	active_docs: List[ str ] = st.session_state.get( 'docqna_active_docs', [ ] )
	doc_bytes: Dict[ str, bytes ] = st.session_state.get( 'docqna_bytes', { } )
	fp = compute_fingerprint( active_docs, doc_bytes )
	if fp and fp == st.session_state.get( 'docqna_fingerprint', '' ):
		return
	
	st.session_state[ 'docqna_fingerprint' ] = fp
	st.session_state[ 'docqna_chunk_count' ] = 0
	st.session_state[ 'docqna_fallback_rows' ] = [ ]
	dim_value = getattr( embedder, 'get_sentence_embedding_dimension', lambda: 384 )( )
	dim = int( dim_value ) if dim_value else 384
	vec_ready = ensure_vec_schema( dim )
	st.session_state[ 'docqna_vec_ready' ] = bool( vec_ready )
	conn = create_connection( )
	try:
		cur = conn.cursor( )
		if vec_ready:
			try:
				cur.execute( 'DELETE FROM docqna_vec;' )
				conn.commit( )
			except Exception:
				st.session_state[ 'docqna_vec_ready' ] = False
				vec_ready = False
		
		total_chunks = 0
		fallback_rows: List[ Tuple[ str, str, bytes ] ] = [ ]
		for name in active_docs:
			b = doc_bytes.get( name )
			if not b:
				continue
			
			text = extract_text_from_pdf( b )
			if not text:
				continue
			
			chunks = chunk_text( text )
			if not chunks:
				continue
			
			vecs = embedder.encode( chunks, show_progress_bar=False )
			vecs = np.asarray( vecs, dtype=np.float32 )
			
			if vec_ready:
				for chunk_text_value, v in zip( chunks, vecs ):
					cur.execute(
						'INSERT INTO docqna_vec ( embedding, doc_name, chunk ) VALUES ( ?, ?, ? );',
						(v.tobytes( ), name, chunk_text_value) )
			else:
				for chunk_text_value, v in zip( chunks, vecs ):
					fallback_rows.append( (name, chunk_text_value, v.tobytes( )) )
			
			total_chunks += int( len( chunks ) )
		
		conn.commit( )
		st.session_state[ 'docqna_chunk_count' ] = total_chunks
		if not vec_ready:
			st.session_state[ 'docqna_fallback_rows' ] = fallback_rows
	
	except Exception:
		st.session_state[ 'docqna_vec_ready' ] = False
		st.session_state[ 'docqna_fallback_rows' ] = [ ]
		st.session_state[ 'docqna_chunk_count' ] = 0
	finally:
		conn.close( )

def retrieve_top_doc_chunks( query: str, k: int = 6 ) -> List[ Tuple[ str, str, float ] ]:
	'''
	
		Purpose:
		--------
		Retrieves top-k document chunks relevant to the query, using sqlite-vec when available, and falling
		back to in-memory cosine similarity when not.
	
		Parameters:
		-----------
		query:
			The user query string.
		k:
			The number of chunks to return.
	
		Returns:
		--------
		A List[ Tuple[ str, str, float ] ] of (doc_name, chunk, score_or_distance).
	
	'''
	if not query or not query.strip( ):
		return [ ]
	
	embedder: SentenceTransformer = load_embedder( )
	rebuild_index( embedder )
	qv = embedder.encode( [ query ], show_progress_bar=False )
	qv = np.asarray( qv, dtype=np.float32 )[ 0 ]
	if st.session_state.get( 'docqna_vec_ready', False ):
		conn = create_connection( )
		try:
			load_sqlite_vec( conn )
			cur = conn.cursor( )
			cur.execute(
				'''
                SELECT doc_name, chunk, distance
                FROM docqna_vec
                WHERE embedding MATCH ?
                ORDER BY distance ASC LIMIT ?;
				''',
				(qv.tobytes( ), int( k ))
			)
			rows = cur.fetchall( )
			return [ (r[ 0 ], r[ 1 ], float( r[ 2 ] )) for r in rows ]
		except Exception:
			st.session_state[ 'docqna_vec_ready' ] = False
		finally:
			conn.close( )
	
	fallback_rows: List[
		Tuple[ str, str, bytes ] ] = st.session_state.get( 'docqna_fallback_rows', [ ] )
	results: List[ Tuple[ str, str, float ] ] = [ ]
	
	for doc_name, chunk_text_value, vec_blob in fallback_rows:
		if not vec_blob:
			continue
		
		v = np.frombuffer( vec_blob, dtype=np.float32 )
		if v.size == 0:
			continue
		
		score = cosine_sim( qv, v )
		results.append( (doc_name, chunk_text_value, float( score )) )
	
	results.sort( key=lambda r: r[ 2 ], reverse=True )
	return results[ : int( k ) ]

def build_document_user_input( user_query: str, k: int = 6 ) -> str:
	'''
	
		Purpose:
		--------
		Builds a Document Q&A prompt that injects retrieved chunks (RAG) instead of stuffing full documents.
	
		Parameters:
		-----------
		user_query:
			The user question.
		k:
			The number of retrieved chunks to include.
	
		Returns:
		--------
		A str prompt suitable for llama.cpp completion.
	
	'''
	system = str( st.session_state.get( 'system_instructions', '' ) or '' ).strip( )
	hits = retrieve_top_doc_chunks( user_query, k=int( k ) )
	
	context_blocks: List[ str ] = [ ]
	for doc_name, chunk, score in hits:
		context_blocks.append( f'[Document: {doc_name}]\n{chunk}'.strip( ) )
	
	context = '\n\n'.join( context_blocks ).strip( )
	
	prompt_parts: List[ str ] = [ ]
	
	if system:
		prompt_parts.append( system )
	
	if context:
		prompt_parts.append(
			'Use the following document excerpts to answer the question. If the excerpts do not contain '
			'the answer, say you do not have enough information.\n\n'
			f'{context}'
		)
	
	prompt_parts.append( f'Question:\n{user_query}\n\nAnswer:' )
	
	return '\n\n'.join( prompt_parts ).strip( )

# ------------ DOCUMENT Q&A MODE UTILITIES -----------------

def ensure_docqna_mode_state( ) -> None:
	"""
	
		Purpose:
		--------
		Ensure Document Q&A mode session-state keys exist before widget instantiation.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	if 'docqna_model' not in st.session_state:
		st.session_state[ 'docqna_model' ] = ''
	
	if 'docqna_source' not in st.session_state:
		st.session_state[ 'docqna_source' ] = 'Local Upload'
	
	if 'docqna_uploaded' not in st.session_state:
		st.session_state[ 'docqna_uploaded' ] = None
	
	if 'docqna_files' not in st.session_state:
		st.session_state[ 'docqna_files' ] = [ ]
	
	if 'docqna_active_docs' not in st.session_state:
		st.session_state[ 'docqna_active_docs' ] = [ ]
	
	if 'docqna_bytes' not in st.session_state:
		st.session_state[ 'docqna_bytes' ] = None
	
	if 'docqna_texts' not in st.session_state:
		st.session_state[ 'docqna_texts' ] = { }
	
	if 'docqna_chunks' not in st.session_state:
		st.session_state[ 'docqna_chunks' ] = [ ]
	
	if 'docqna_last_hits' not in st.session_state:
		st.session_state[ 'docqna_last_hits' ] = [ ]
	
	if 'docqna_last_sources' not in st.session_state:
		st.session_state[ 'docqna_last_sources' ] = [ ]
	
	if 'docqna_last_answer' not in st.session_state:
		st.session_state[ 'docqna_last_answer' ] = ''
	
	if 'docqna_context' not in st.session_state:
		st.session_state[ 'docqna_context' ] = ''
	
	if 'docqna_messages' not in st.session_state:
		st.session_state.docqna_messages = [ ]
	
	if not isinstance( st.session_state.get( 'docqna_messages' ), list ):
		st.session_state.docqna_messages = [ ]
	
	if 'docqna_system_instructions' not in st.session_state:
		st.session_state[ 'docqna_system_instructions' ] = ''
	
	if 'docqna_file_id' not in st.session_state:
		st.session_state[ 'docqna_file_id' ] = ''
	
	if 'docqna_vector_store_id' not in st.session_state:
		st.session_state[ 'docqna_vector_store_id' ] = ''
	
	if 'docqna_multi_mode' not in st.session_state:
		st.session_state[ 'docqna_multi_mode' ] = False
	
	if 'docqna_top_k' not in st.session_state:
		st.session_state[ 'docqna_top_k' ] = 6
	
	if 'docqna_chunk_size' not in st.session_state:
		st.session_state[ 'docqna_chunk_size' ] = 900
	
	if 'docqna_chunk_overlap' not in st.session_state:
		st.session_state[ 'docqna_chunk_overlap' ] = 150
	
	if 'docqna_vec_ready' not in st.session_state:
		st.session_state[ 'docqna_vec_ready' ] = False
	
	if 'docqna_fingerprint' not in st.session_state:
		st.session_state[ 'docqna_fingerprint' ] = ''
	
	if 'docqna_chunk_count' not in st.session_state:
		st.session_state[ 'docqna_chunk_count' ] = 0
	
	if 'docqna_index_status' not in st.session_state:
		st.session_state[ 'docqna_index_status' ] = 'Not indexed'
	
	if 'docqna_backend' not in st.session_state:
		st.session_state[ 'docqna_backend' ] = 'local'
	
	if 'docqna_show_diagnostics' not in st.session_state:
		st.session_state[ 'docqna_show_diagnostics' ] = True
	
	if 'last_answer' not in st.session_state:
		st.session_state[ 'last_answer' ] = ''
	
	if 'last_sources' not in st.session_state:
		st.session_state[ 'last_sources' ] = [ ]

def clear_docqna_messages( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Document Q&A chat messages without clearing loaded documents or index state.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state.docqna_messages = [ ]

def clear_docqna_outputs( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Document Q&A answer, context, source, and retrieval output state.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'docqna_last_answer' ] = ''
	st.session_state[ 'docqna_last_hits' ] = [ ]
	st.session_state[ 'docqna_last_sources' ] = [ ]
	st.session_state[ 'docqna_context' ] = ''
	st.session_state[ 'last_answer' ] = ''
	st.session_state[ 'last_sources' ] = [ ]

def unload_docqna_documents( ) -> None:
	"""
	
		Purpose:
		--------
		Unload active Document Q&A files, extracted text, chunks, and index state.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'docqna_uploaded' ] = None
	st.session_state[ 'docqna_files' ] = [ ]
	st.session_state[ 'docqna_active_docs' ] = [ ]
	st.session_state[ 'docqna_bytes' ] = None
	st.session_state[ 'docqna_texts' ] = { }
	st.session_state[ 'docqna_chunks' ] = [ ]
	st.session_state[ 'docqna_vec_ready' ] = False
	st.session_state[ 'docqna_fingerprint' ] = ''
	st.session_state[ 'docqna_chunk_count' ] = 0
	st.session_state[ 'docqna_index_status' ] = 'Not indexed'
	clear_docqna_outputs( )

def reset_docqna_controls( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Document Q&A controls through a widget-safe callback.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	for key in [ 'docqna_model', 'docqna_source', 'docqna_file_id',
	             'docqna_vector_store_id', 'docqna_multi_mode', 'docqna_top_k',
	             'docqna_chunk_size', 'docqna_chunk_overlap',
	             'docqna_show_diagnostics' ]:
		if key in st.session_state:
			del st.session_state[ key ]

def reset_docqna_all( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Document Q&A controls, loaded documents, index state, outputs, and messages.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	reset_docqna_controls( )
	unload_docqna_documents( )
	clear_docqna_messages( )

def clear_docqna_instructions( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Document Q&A system instructions and selected prompt template.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'docqna_system_instructions' ] = ''
	st.session_state[ 'instructions' ] = ''

def load_docqna_instruction_template( ) -> None:
	"""
	
		Purpose:
		--------
		Load the selected prompt template into Document Q&A system instructions.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	name = st.session_state.get( 'instructions' )
	if name and name != 'No Templates Found':
		text = fetch_prompt_text( cfg.DB_PATH, name )
		if text is not None:
			st.session_state[ 'docqna_system_instructions' ] = text

def convert_docqna_system_instructions( ) -> None:
	"""
	
		Purpose:
		--------
		Convert Document Q&A system instructions between XML-like delimiters and Markdown
		headings.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	text = st.session_state.get( 'docqna_system_instructions', '' )
	if not isinstance( text, str ) or not text.strip( ):
		return
	
	source = text.strip( )
	if cfg.XML_BLOCK_PATTERN.search( source ):
		converted = convert_xml( source )
	else:
		converted = convert_markdown( source )
	
	st.session_state[ 'docqna_system_instructions' ] = converted

def get_docqna_source_options( ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Return supported Document Q&A source options.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		list[str]
			Document Q&A source option names.
		
	"""
	return [
			'Local Upload',
			'OpenAI File ID',
			'OpenAI Vector Store ID',
	]

def get_docqna_file_extension( filename: str | None ) -> str:
	"""
	
		Purpose:
		--------
		Return a lowercase file extension for Document Q&A extraction and preview routing.
		
		Parameters:
		-----------
		filename: str | None
			File name to inspect.
		
		Returns:
		--------
		str
			Lowercase file extension including the leading period.
		
	"""
	if not isinstance( filename, str ) or not filename.strip( ):
		return ''
	
	return Path( filename ).suffix.lower( )

def compute_docqna_fingerprint( documents: list[ dict[ str, Any ] ] ) -> str:
	"""
	
		Purpose:
		--------
		Compute a stable fingerprint for active Document Q&A files.
		
		Parameters:
		-----------
		documents: list[dict[str, Any]]
			Active document metadata and byte payloads.
		
		Returns:
		--------
		str
			SHA-256 fingerprint for the active document set.
		
	"""
	hasher = hashlib.sha256( )
	
	if not isinstance( documents, list ):
		return ''
	
	for doc in documents:
		if not isinstance( doc, dict ):
			continue
		
		name = str( doc.get( 'name', '' ) )
		content = doc.get( 'bytes', b'' )
		hasher.update( name.encode( 'utf-8', errors='ignore' ) )
		
		if isinstance( content, bytes ):
			hasher.update( content )
	
	return hasher.hexdigest( )

def compute_fingerprint( file_bytes: bytes | None ) -> str:
	"""
	
		Purpose:
		--------
		Backward-compatible fingerprint helper for a single byte payload.
		
		Parameters:
		-----------
		file_bytes: bytes | None
			File byte payload.
		
		Returns:
		--------
		str
			SHA-256 fingerprint.
		
	"""
	if not isinstance( file_bytes, bytes ):
		return ''
	
	return hashlib.sha256( file_bytes ).hexdigest( )

def extract_docqna_pdf_text( file_bytes: bytes ) -> str:
	"""
	
		Purpose:
		--------
		Extract text from PDF bytes using PyMuPDF when available.
		
		Parameters:
		-----------
		file_bytes: bytes
			PDF byte payload.
		
		Returns:
		--------
		str
			Extracted PDF text.
		
	"""
	if not isinstance( file_bytes, bytes ) or len( file_bytes ) == 0:
		return ''
	
	try:
		import fitz
		
		pages: list[ str ] = [ ]
		with fitz.open( stream=file_bytes, filetype='pdf' ) as doc:
			for page in doc:
				pages.append( page.get_text( 'text' ) )
		
		return '\n\n'.join( pages ).strip( )
	except Exception:
		try:
			return extract_text_from_bytes( file_bytes )
		except Exception:
			return ''

def extract_docqna_text_file( file_bytes: bytes ) -> str:
	"""
	
		Purpose:
		--------
		Decode plain-text-like document bytes.
		
		Parameters:
		-----------
		file_bytes: bytes
			Text byte payload.
		
		Returns:
		--------
		str
			Decoded text.
		
	"""
	if not isinstance( file_bytes, bytes ) or len( file_bytes ) == 0:
		return ''
	
	for encoding in [ 'utf-8', 'utf-8-sig', 'cp1252', 'latin-1' ]:
		try:
			return file_bytes.decode( encoding ).strip( )
		except Exception:
			continue
	
	return ''

def extract_docqna_docx_text( file_bytes: bytes ) -> str:
	"""
	
		Purpose:
		--------
		Extract text from DOCX bytes using the zipped WordprocessingML document body.
		
		Parameters:
		-----------
		file_bytes: bytes
			DOCX byte payload.
		
		Returns:
		--------
		str
			Extracted DOCX text.
		
	"""
	if not isinstance( file_bytes, bytes ) or len( file_bytes ) == 0:
		return ''
	
	try:
		with zipfile.ZipFile( io.BytesIO( file_bytes ) ) as archive:
			xml_bytes = archive.read( 'word/document.xml' )
		
		root = ET.fromstring( xml_bytes )
		namespace = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
		paragraphs: list[ str ] = [ ]
		
		for paragraph in root.iter( f'{namespace}p' ):
			parts: list[ str ] = [ ]
			
			for node in paragraph.iter( f'{namespace}t' ):
				if node.text:
					parts.append( node.text )
			
			text = ''.join( parts ).strip( )
			if text:
				paragraphs.append( text )
		
		return '\n\n'.join( paragraphs ).strip( )
	except Exception:
		return ''

def extract_docqna_text( filename: str, file_bytes: bytes ) -> str:
	"""
	
		Purpose:
		--------
		Extract document text from supported Document Q&A file types.
		
		Parameters:
		-----------
		filename: str
			Uploaded file name.
		
		file_bytes: bytes
			Uploaded file bytes.
		
		Returns:
		--------
		str
			Extracted text.
		
	"""
	extension = get_docqna_file_extension( filename )
	
	if extension == '.pdf':
		return extract_docqna_pdf_text( file_bytes )
	
	if extension == '.docx':
		return extract_docqna_docx_text( file_bytes )
	
	if extension in [ '.txt', '.md', '.csv', '.json', '.xml', '.py', '.cs', '.sql',
	                  '.yaml', '.yml', '.html', '.css', '.js', '.ts' ]:
		return extract_docqna_text_file( file_bytes )
	
	return extract_docqna_text_file( file_bytes )

def load_docqna_uploaded_files( uploaded: Any ) -> list[ dict[ str, Any ] ]:
	"""
	
		Purpose:
		--------
		Load one or more Streamlit uploaded files into Document Q&A state.
		
		Parameters:
		-----------
		uploaded: Any
			Streamlit uploaded file object or list of uploaded file objects.
		
		Returns:
		--------
		list[dict[str, Any]]
			Loaded active document records.
		
	"""
	if uploaded is None:
		return [ ]
	
	files = uploaded if isinstance( uploaded, list ) else [ uploaded ]
	active_docs: list[ dict[ str, Any ] ] = [ ]
	texts: dict[ str, str ] = { }
	
	for item in files:
		if item is None:
			continue
		
		name = getattr( item, 'name', 'uploaded_document' )
		
		try:
			content = item.getvalue( ) if hasattr( item, 'getvalue' ) else item.read( )
		except Exception:
			content = None
		
		if not isinstance( content, bytes ) or len( content ) == 0:
			continue
		
		text = extract_docqna_text( filename=name, file_bytes=content )
		
		active_docs.append(
			{
					'name': name,
					'extension': get_docqna_file_extension( name ),
					'bytes': content,
					'text': text,
					'size': len( content ),
			} )
		
		texts[ name ] = text
	
	st.session_state[ 'docqna_uploaded' ] = uploaded
	st.session_state[ 'docqna_files' ] = active_docs
	st.session_state[ 'docqna_active_docs' ] = active_docs
	st.session_state[ 'docqna_texts' ] = texts
	
	if len( active_docs ) == 1:
		st.session_state[ 'docqna_bytes' ] = active_docs[ 0 ].get( 'bytes' )
		st.session_state[ 'doc_bytes' ] = active_docs[ 0 ].get( 'bytes' )
	elif len( active_docs ) > 1:
		st.session_state[ 'docqna_bytes' ] = active_docs[ 0 ].get( 'bytes' )
		st.session_state[ 'doc_bytes' ] = active_docs[ 0 ].get( 'bytes' )
	
	fingerprint = compute_docqna_fingerprint( active_docs )
	if fingerprint != st.session_state.get( 'docqna_fingerprint', '' ):
		st.session_state[ 'docqna_vec_ready' ] = False
		st.session_state[ 'docqna_fingerprint' ] = fingerprint
		st.session_state[ 'docqna_index_status' ] = 'Loaded; not indexed'
	
	return active_docs

def get_docqna_active_document_names( ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Return active Document Q&A document names.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		list[str]
			Active document names.
		
	"""
	docs = st.session_state.get( 'docqna_active_docs', [ ] )
	
	if not isinstance( docs, list ):
		return [ ]
	
	return [
			doc.get( 'name', '' ) for doc in docs
			if isinstance( doc, dict ) and doc.get( 'name' )
	]

def get_docqna_active_bytes( ) -> bytes | None:
	"""
	
		Purpose:
		--------
		Return active Document Q&A bytes using the canonical key with legacy fallback.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		bytes | None
			Active document bytes.
		
	"""
	value = st.session_state.get( 'docqna_bytes', None )
	
	if isinstance( value, bytes ):
		return value
	
	legacy = st.session_state.get( 'doc_bytes', None )
	if isinstance( legacy, bytes ):
		st.session_state[ 'docqna_bytes' ] = legacy
		return legacy
	
	return None

def render_docqna_document_preview( ) -> None:
	"""
	
		Purpose:
		--------
		Render active Document Q&A document previews by file extension.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	docs = st.session_state.get( 'docqna_active_docs', [ ] )
	
	if not isinstance( docs, list ) or len( docs ) == 0:
		st.info( 'No active document loaded.' )
		return
	
	for doc in docs:
		if not isinstance( doc, dict ):
			continue
		
		name = doc.get( 'name', 'Document' )
		extension = doc.get( 'extension', '' )
		content = doc.get( 'bytes', b'' )
		text = doc.get( 'text', '' )
		
		with st.expander( label=f'Preview: {name}', icon='📄',
				expanded=False, width='stretch' ):
			st.caption(
				f'File type: {extension or "unknown"} | Size: {doc.get( "size", 0 )} bytes' )
			
			if extension == '.pdf' and isinstance( content, bytes ):
				try:
					st.pdf( content, height=420 )
				except Exception:
					st.text_area( label='Extracted Text Preview',
						value=text[ :12000 ] if isinstance( text, str ) else '',
						height=300, width='stretch', disabled=True )
			elif extension == '.md' and isinstance( text, str ):
				st.markdown( text[ :12000 ] )
			elif isinstance( text, str ) and text.strip( ):
				st.text_area( label='Extracted Text Preview',
					value=text[ :12000 ],
					height=300, width='stretch', disabled=True )
			else:
				st.warning( 'No readable text preview is available for this file.' )

def normalize_docqna_text( text: str ) -> str:
	"""
	
		Purpose:
		--------
		Normalize extracted document text for local Document Q&A retrieval.
		
		Parameters:
		-----------
		text: str
			Extracted document text.
		
		Returns:
		--------
		str
			Normalized document text.
		
	"""
	if not isinstance( text, str ):
		return ''
	
	value = text.replace( '\x00', ' ' )
	value = re.sub( r'[ \t]+', ' ', value )
	value = re.sub( r'\n{3,}', '\n\n', value )
	return value.strip( )

def chunk_docqna_text( text: str, chunk_size: int = 900,
		chunk_overlap: int = 150 ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Split document text into overlapping word-based chunks for local retrieval.
		
		Parameters:
		-----------
		text: str
			Document text.
		
		chunk_size: int
			Maximum words per chunk.
		
		chunk_overlap: int
			Words overlapping between adjacent chunks.
		
		Returns:
		--------
		list[str]
			Document text chunks.
		
	"""
	if not isinstance( text, str ) or not text.strip( ):
		return [ ]
	
	try:
		size = int( chunk_size )
	except Exception:
		size = 900
	
	try:
		overlap = int( chunk_overlap )
	except Exception:
		overlap = 150
	
	if size <= 0:
		size = 900
	
	if overlap < 0:
		overlap = 0
	
	if overlap >= size:
		overlap = max( 0, size // 5 )
	
	words = text.split( )
	if len( words ) == 0:
		return [ ]
	
	chunks: list[ str ] = [ ]
	step = max( 1, size - overlap )
	start = 0
	
	while start < len( words ):
		end = min( start + size, len( words ) )
		chunk = ' '.join( words[ start:end ] ).strip( )
		
		if chunk:
			chunks.append( chunk )
		
		if end >= len( words ):
			break
		
		start += step
	
	return chunks

def rebuild_docqna_index( ) -> list[ dict[ str, Any ] ]:
	"""
	
		Purpose:
		--------
		Rebuild the local Document Q&A retrieval index from active documents.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		list[dict[str, Any]]
			Indexed chunk records.
		
	"""
	docs = st.session_state.get( 'docqna_active_docs', [ ] )
	
	if not isinstance( docs, list ) or len( docs ) == 0:
		st.session_state[ 'docqna_chunks' ] = [ ]
		st.session_state[ 'docqna_vec_ready' ] = False
		st.session_state[ 'docqna_chunk_count' ] = 0
		st.session_state[ 'docqna_index_status' ] = 'No documents loaded'
		return [ ]
	
	chunk_records: list[ dict[ str, Any ] ] = [ ]
	chunk_size = st.session_state.get( 'docqna_chunk_size', 900 )
	chunk_overlap = st.session_state.get( 'docqna_chunk_overlap', 150 )
	
	for doc in docs:
		if not isinstance( doc, dict ):
			continue
		
		name = doc.get( 'name', 'Document' )
		text = normalize_docqna_text( doc.get( 'text', '' ) )
		
		for index, chunk in enumerate( chunk_docqna_text(
				text=text, chunk_size=chunk_size, chunk_overlap=chunk_overlap ) ):
			chunk_records.append(
				{
						'document': name,
						'chunk_index': index + 1,
						'text': chunk,
				} )
	
	st.session_state[ 'docqna_chunks' ] = chunk_records
	st.session_state[ 'docqna_chunk_count' ] = len( chunk_records )
	st.session_state[ 'docqna_vec_ready' ] = len( chunk_records ) > 0
	st.session_state[ 'docqna_index_status' ] = 'Ready' if len(
		chunk_records ) > 0 else 'No text extracted'
	return chunk_records

def tokenize_docqna_query( text: str ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Tokenize text for lightweight local retrieval scoring.
		
		Parameters:
		-----------
		text: str
			Text to tokenize.
		
		Returns:
		--------
		list[str]
			Lowercase alphanumeric tokens.
		
	"""
	if not isinstance( text, str ):
		return [ ]
	
	return re.findall( r'[A-Za-z0-9_]+', text.lower( ) )

def score_docqna_chunk( query_tokens: list[ str ], chunk_text: str ) -> float:
	"""
	
		Purpose:
		--------
		Score a document chunk against query tokens using a lightweight cosine-like score.
		
		Parameters:
		-----------
		query_tokens: list[str]
			Query tokens.
		
		chunk_text: str
			Chunk text.
		
		Returns:
		--------
		float
			Similarity score.
		
	"""
	if not isinstance( query_tokens, list ) or len( query_tokens ) == 0:
		return 0.0
	
	chunk_tokens = tokenize_docqna_query( chunk_text )
	if len( chunk_tokens ) == 0:
		return 0.0
	
	query_counts: dict[ str, int ] = { }
	chunk_counts: dict[ str, int ] = { }
	
	for token in query_tokens:
		query_counts[ token ] = query_counts.get( token, 0 ) + 1
	
	for token in chunk_tokens:
		chunk_counts[ token ] = chunk_counts.get( token, 0 ) + 1
	
	dot = sum( query_counts.get( token, 0 ) * chunk_counts.get( token, 0 )
	           for token in query_counts )
	query_norm = math.sqrt( sum( value * value for value in query_counts.values( ) ) )
	chunk_norm = math.sqrt( sum( value * value for value in chunk_counts.values( ) ) )
	
	if query_norm == 0 or chunk_norm == 0:
		return 0.0
	
	return dot / (query_norm * chunk_norm)

def retrieve_docqna_chunks( query: str, top_k: int | None = None ) -> list[ dict[ str, Any ] ]:
	"""
	
		Purpose:
		--------
		Retrieve the most relevant local Document Q&A chunks for a user query.
		
		Parameters:
		-----------
		query: str
			User query.
		
		top_k: int | None
			Maximum number of chunks to return.
		
		Returns:
		--------
		list[dict[str, Any]]
			Ranked retrieval hit records.
		
	"""
	if not st.session_state.get( 'docqna_vec_ready', False ):
		rebuild_docqna_index( )
	
	chunks = st.session_state.get( 'docqna_chunks', [ ] )
	if not isinstance( chunks, list ) or len( chunks ) == 0:
		return [ ]
	
	try:
		k = int( top_k if top_k is not None else st.session_state.get( 'docqna_top_k', 6 ) )
	except Exception:
		k = 6
	
	if k <= 0:
		k = 6
	
	query_tokens = tokenize_docqna_query( query )
	hits: list[ dict[ str, Any ] ] = [ ]
	
	for chunk in chunks:
		if not isinstance( chunk, dict ):
			continue
		
		score = score_docqna_chunk( query_tokens, chunk.get( 'text', '' ) )
		hits.append(
			{
					'rank': 0,
					'document': chunk.get( 'document', '' ),
					'chunk_index': chunk.get( 'chunk_index', 0 ),
					'score': round( score, 6 ),
					'text': chunk.get( 'text', '' ),
			} )
	
	hits = sorted( hits, key=lambda item: item.get( 'score', 0.0 ), reverse=True )[ :k ]
	
	for index, hit in enumerate( hits ):
		hit[ 'rank' ] = index + 1
	
	st.session_state[ 'docqna_last_hits' ] = hits
	st.session_state[ 'docqna_last_sources' ] = [
			{
					'document': hit.get( 'document', '' ),
					'chunk_index': hit.get( 'chunk_index', 0 ),
					'score': hit.get( 'score', 0.0 ),
			}
			for hit in hits
	]
	st.session_state[ 'last_sources' ] = st.session_state[ 'docqna_last_sources' ]
	return hits

def build_docqna_local_prompt( query: str, hits: list[ dict[ str, Any ] ] ) -> str:
	"""
	
		Purpose:
		--------
		Build a grounded local Document Q&A prompt from retrieved chunks.
		
		Parameters:
		-----------
		query: str
			User query.
		
		hits: list[dict[str, Any]]
			Retrieved chunk records.
		
		Returns:
		--------
		str
			Prompt for the language model.
		
	"""
	context_blocks: list[ str ] = [ ]
	
	for hit in hits:
		if not isinstance( hit, dict ):
			continue
		
		context_blocks.append(
			f"[Source: {hit.get( 'document', '' )}, Chunk: {hit.get( 'chunk_index', 0 )}, "
			f"Score: {hit.get( 'score', 0.0 )}]\n{hit.get( 'text', '' )}" )
	
	context = '\n\n---\n\n'.join( context_blocks )
	st.session_state[ 'docqna_context' ] = context
	
	instructions = st.session_state.get( 'docqna_system_instructions', '' )
	
	return (
			f"{instructions.strip( )}\n\n" if isinstance( instructions, str )
			                                  and instructions.strip( ) else ''
	) + (
			'Answer the user question using only the document context below. '
			'If the answer is not supported by the context, say that the document context '
			'does not contain enough information.\n\n'
			f'Document Context:\n{context}\n\n'
			f'User Question:\n{query}'
	)

def docqna_call_openai_text_model( prompt: str ) -> str:
	"""
	
		Purpose:
		--------
		Call the available OpenAI chat/text wrapper for a Document Q&A prompt.
		
		Parameters:
		-----------
		prompt: str
			Prompt to submit.
		
		Returns:
		--------
		str
			Model response text.
		
	"""
	if not isinstance( prompt, str ) or not prompt.strip( ):
		return ''
	
	model = st.session_state.get( 'docqna_model' ) or 'gpt-4o-mini'
	
	try:
		run_turn = globals( ).get( 'run_llm_turn' )
		if callable( run_turn ):
			return str( run_turn(
				prompt=prompt,
				model=model,
				temperature=st.session_state.get( 'docqna_temperature', 0.2 ),
				top_p=st.session_state.get( 'docqna_top_percent', 1.0 ),
				max_tokens=st.session_state.get( 'docqna_max_tokens', 2000 ) ) )
	except Exception:
		pass
	
	try:
		chat = Chat( )
		
		for method_name in [ 'generate_text', 'create', 'ask', 'complete' ]:
			method = getattr( chat, method_name, None )
			if callable( method ):
				try:
					result = method( text=prompt, model=model )
				except TypeError:
					try:
						result = method( prompt=prompt, model=model )
					except TypeError:
						result = method( prompt )
				
				return str( getattr( result, 'output_text', result ) )
	except Exception as exc:
		return f'Document Q&A model call failed: {exc}'
	
	return prompt

def run_docqna_local_query( query: str ) -> str:
	"""
	
		Purpose:
		--------
		Run a Document Q&A query against locally uploaded and indexed documents.
		
		Parameters:
		-----------
		query: str
			User query.
		
		Returns:
		--------
		str
			Generated answer.
		
	"""
	if not isinstance( query, str ) or not query.strip( ):
		return ''
	
	if not st.session_state.get( 'docqna_vec_ready', False ):
		rebuild_docqna_index( )
	
	hits = retrieve_docqna_chunks(
		query=query,
		top_k=st.session_state.get( 'docqna_top_k', 6 ) )
	
	if len( hits ) == 0:
		answer = 'No readable or retrievable document context is available.'
		st.session_state[ 'docqna_last_answer' ] = answer
		st.session_state[ 'last_answer' ] = answer
		return answer
	
	prompt = build_docqna_local_prompt( query=query, hits=hits )
	answer = docqna_call_openai_text_model( prompt )
	
	st.session_state[ 'docqna_last_answer' ] = answer
	st.session_state[ 'last_answer' ] = answer
	return answer

def run_docqna_file_query( query: str ) -> str:
	"""
	
		Purpose:
		--------
		Run a Document Q&A query against an OpenAI File ID using the Files wrapper.
		
		Parameters:
		-----------
		query: str
			User query.
		
		Returns:
		--------
		str
			Generated answer.
		
	"""
	file_id = st.session_state.get( 'docqna_file_id', '' )
	
	if not isinstance( file_id, str ) or not file_id.strip( ):
		return 'No OpenAI file ID is selected.'
	
	try:
		files = Files( )
		answer = files.search(
			id=file_id.strip( ),
			query=query,
			model=st.session_state.get( 'docqna_model' ) or 'gpt-4o-mini' )
		
		answer = answer if isinstance( answer, str ) else str( answer )
		st.session_state[ 'docqna_last_answer' ] = answer
		st.session_state[ 'last_answer' ] = answer
		st.session_state[ 'docqna_last_sources' ] = [ { 'file_id': file_id.strip( ) } ]
		st.session_state[ 'last_sources' ] = st.session_state[ 'docqna_last_sources' ]
		return answer
	except Exception as exc:
		return f'OpenAI file query failed: {exc}'

def run_docqna_vector_store_query( query: str ) -> str:
	"""
	
		Purpose:
		--------
		Run a Document Q&A query against an OpenAI Vector Store ID.
		
		Parameters:
		-----------
		query: str
			User query.
		
		Returns:
		--------
		str
			Generated answer.
		
	"""
	store_id = st.session_state.get( 'docqna_vector_store_id', '' )
	
	if not isinstance( store_id, str ) or not store_id.strip( ):
		return 'No OpenAI vector store ID is selected.'
	
	try:
		vector = VectorStores( )
		answer = vector.answer_with_file_search(
			store_ids=[ store_id.strip( ) ],
			prompt=query,
			model=st.session_state.get( 'docqna_model' ) or 'gpt-4o-mini',
			max_num_results=st.session_state.get( 'docqna_top_k', 6 ),
			instructions=st.session_state.get( 'docqna_system_instructions', '' ) or None )
		
		answer = answer if isinstance( answer, str ) else str( answer )
		st.session_state[ 'docqna_last_answer' ] = answer
		st.session_state[ 'last_answer' ] = answer
		st.session_state[ 'docqna_last_sources' ] = [ { 'vector_store_id': store_id.strip( ) } ]
		st.session_state[ 'last_sources' ] = st.session_state[ 'docqna_last_sources' ]
		return answer
	except Exception as exc:
		return f'OpenAI vector store query failed: {exc}'

def route_document_query( prompt: str ) -> str:
	"""
	
		Purpose:
		--------
		Route a Document Q&A prompt to the selected backend.
		
		Parameters:
		-----------
		prompt: str
			User question or instruction.
		
		Returns:
		--------
		str
			Backend answer text.
		
	"""
	if not isinstance( prompt, str ) or not prompt.strip( ):
		return ''
	
	source = st.session_state.get( 'docqna_source', 'Local Upload' )
	
	if source == 'OpenAI File ID':
		return run_docqna_file_query( prompt )
	
	if source == 'OpenAI Vector Store ID':
		return run_docqna_vector_store_query( prompt )
	
	return run_docqna_local_query( prompt )

def summarize_active_document( ) -> str:
	"""
	
		Purpose:
		--------
		Summarize the active document source using the selected Document Q&A backend.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		str
			Summary text.
		
	"""
	return route_document_query(
		'Summarize the active document. Include the main topic, key sections, '
		'important findings, and any limitations visible in the source.' )

def render_docqna_retrieval_hits( ) -> None:
	"""
	
		Purpose:
		--------
		Render Document Q&A retrieval hits and source diagnostics.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	hits = st.session_state.get( 'docqna_last_hits', [ ] )
	
	if not isinstance( hits, list ) or len( hits ) == 0:
		st.info( 'No retrieval hits available.' )
		return
	
	df_hits = pd.DataFrame( hits )
	st.data_editor( df_hits, use_container_width=True, hide_index=True )

def render_docqna_status( ) -> None:
	"""
	
		Purpose:
		--------
		Render Document Q&A index and source status metrics.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	c1, c2, c3, c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
		border=True, gap='xxsmall' )
	
	with c1:
		st.metric( 'Documents', len( get_docqna_active_document_names( ) ) )
	
	with c2:
		st.metric( 'Chunks', st.session_state.get( 'docqna_chunk_count', 0 ) )
	
	with c3:
		st.metric( 'Source', st.session_state.get( 'docqna_source', 'Local Upload' ) )
	
	with c4:
		st.metric( 'Index', st.session_state.get( 'docqna_index_status', 'Not indexed' ) )
		
# ------------ FILES API UTILITIES -----------------

def ensure_files_mode_state( ) -> None:
	"""
	
		Purpose:
		--------
		Ensure Files mode session-state keys exist before widget instantiation.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	if 'files_model' not in st.session_state:
		st.session_state[ 'files_model' ] = ''
	
	if 'files_purpose' not in st.session_state:
		st.session_state[ 'files_purpose' ] = 'user_data'
	
	if 'files_filter_purpose' not in st.session_state:
		st.session_state[ 'files_filter_purpose' ] = ''
	
	if 'files_type' not in st.session_state:
		st.session_state[ 'files_type' ] = ''
	
	if 'files_id' not in st.session_state:
		st.session_state[ 'files_id' ] = ''
	
	if 'files_url' not in st.session_state:
		st.session_state[ 'files_url' ] = ''
	
	if 'files_table' not in st.session_state:
		st.session_state[ 'files_table' ] = [ ]
	
	if 'files_df' not in st.session_state:
		st.session_state[ 'files_df' ] = pd.DataFrame( )
	
	if 'files_metadata' not in st.session_state:
		st.session_state[ 'files_metadata' ] = { }
	
	if 'files_content' not in st.session_state:
		st.session_state[ 'files_content' ] = ''
	
	if 'files_content_bytes' not in st.session_state:
		st.session_state[ 'files_content_bytes' ] = None
	
	if 'files_delete_result' not in st.session_state:
		st.session_state[ 'files_delete_result' ] = { }
	
	if 'files_last_answer' not in st.session_state:
		st.session_state[ 'files_last_answer' ] = ''
	
	if 'files_system_instructions' not in st.session_state:
		st.session_state[ 'files_system_instructions' ] = ''
	
	if 'files_messages' not in st.session_state:
		st.session_state.files_messages = [ ]
	
	if not isinstance( st.session_state.get( 'files_messages' ), list ):
		st.session_state.files_messages = [ ]

def get_files_upload_purpose_options( files: Files ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Return OpenAI upload purpose options from the Files wrapper.
		
		Parameters:
		-----------
		files: Files
			Files wrapper instance.
		
		Returns:
		--------
		list[str]
			Upload purpose options.
		
	"""
	options = getattr( files, 'upload_purpose_options', None )
	if isinstance( options, list ) and len( options ) > 0:
		return options
	
	options = getattr( files, 'purpose_options', None )
	if isinstance( options, list ) and len( options ) > 0:
		return options
	
	return [
			'assistants',
			'batch',
			'fine-tune',
			'vision',
			'user_data',
			'evals',
	]

def get_files_filter_purpose_options( files: Files ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Return file purpose filter options for listing OpenAI files.
		
		Parameters:
		-----------
		files: Files
			Files wrapper instance.
		
		Returns:
		--------
		list[str]
			File purpose filter options.
		
	"""
	options = getattr( files, 'file_purpose_options', None )
	if isinstance( options, list ) and len( options ) > 0:
		return [ '' ] + options
	
	return [
			'',
			'assistants',
			'assistants_output',
			'batch',
			'batch_output',
			'fine-tune',
			'fine-tune-results',
			'vision',
			'user_data',
			'evals',
	]

def get_files_model_options( files: Files ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Return model options for optional selected-file analysis workflows.
		
		Parameters:
		-----------
		files: Files
			Files wrapper instance.
		
		Returns:
		--------
		list[str]
			Model options.
		
	"""
	options = getattr( files, 'model_options', None )
	if isinstance( options, list ) and len( options ) > 0:
		return [ '' ] + options
	
	return [
			'',
			'gpt-5-mini',
			'gpt-5-nano',
			'gpt-4.1-mini',
			'gpt-4.1-nano',
			'gpt-4o-mini',
	]

def save_files_upload( uploaded_file: Any ) -> str | None:
	"""
	
		Purpose:
		--------
		Save a Streamlit uploaded file to a temporary local path for Files API upload.
		
		Parameters:
		-----------
		uploaded_file: Any
			Streamlit uploaded file object.
		
		Returns:
		--------
		str | None
			Temporary file path or None.
		
	"""
	if uploaded_file is None:
		return None
	
	try:
		name = getattr( uploaded_file, 'name', '' )
		suffix = Path( name ).suffix if isinstance( name, str ) and name.strip( ) else ''
		
		if not suffix:
			suffix = '.tmp'
		
		with tempfile.NamedTemporaryFile( delete=False, suffix=suffix ) as tmp:
			if hasattr( uploaded_file, 'getbuffer' ):
				tmp.write( uploaded_file.getbuffer( ) )
			elif hasattr( uploaded_file, 'read' ):
				tmp.write( uploaded_file.read( ) )
			else:
				return None
			
			return tmp.name
	except Exception:
		return None

def normalize_files_table( rows: Any ) -> list[ dict[ str, Any ] ]:
	"""
	
		Purpose:
		--------
		Normalize Files API list output into table rows used by Files mode.
		
		Parameters:
		-----------
		rows: Any
			Files API rows, response data, or wrapper-normalized rows.
		
		Returns:
		--------
		list[dict[str, Any]]
			Normalized table rows.
		
	"""
	if rows is None:
		return [ ]
	
	if isinstance( rows, dict ) and isinstance( rows.get( 'data' ), list ):
		items = rows.get( 'data', [ ] )
	elif isinstance( rows, list ):
		items = rows
	else:
		items = getattr( rows, 'data', [ ] )
	
	normalized: list[ dict[ str, Any ] ] = [ ]
	for item in items:
		if isinstance( item, dict ):
			source = item
		elif hasattr( item, 'model_dump' ):
			try:
				source = item.model_dump( )
			except Exception:
				source = { }
		else:
			source = {
					'id': getattr( item, 'id', None ),
					'filename': getattr( item, 'filename', None ),
					'purpose': getattr( item, 'purpose', None ),
					'bytes': getattr( item, 'bytes', None ),
					'created_at': getattr( item, 'created_at', None ),
					'expires_at': getattr( item, 'expires_at', None ),
					'status': getattr( item, 'status', None ),
					'object': getattr( item, 'object', None ),
			}
		
		file_id = source.get( 'id' )
		if not file_id:
			continue
		
		normalized.append(
			{
					'id': file_id,
					'filename': source.get( 'filename', '' ),
					'purpose': source.get( 'purpose', '' ),
					'bytes': source.get( 'bytes', 0 ),
					'created_at': source.get( 'created_at', '' ),
					'expires_at': source.get( 'expires_at', '' ),
					'status': source.get( 'status', '' ),
					'object': source.get( 'object', '' ),
			} )
	
	return normalized

def build_files_dataframe( rows: list[ dict[ str, Any ] ] ) -> pd.DataFrame:
	"""
	
		Purpose:
		--------
		Build a display-ready DataFrame from normalized Files API rows.
		
		Parameters:
		-----------
		rows: list[dict[str, Any]]
			Normalized file metadata rows.
		
		Returns:
		--------
		pd.DataFrame
			Display-ready file metadata table.
		
	"""
	if not isinstance( rows, list ) or len( rows ) == 0:
		return pd.DataFrame( )
	
	df_files = pd.DataFrame( rows )
	preferred = [
			'id',
			'filename',
			'purpose',
			'bytes',
			'created_at',
			'expires_at',
			'status',
			'object',
	]
	
	columns = [ column for column in preferred if column in df_files.columns ]
	extras = [ column for column in df_files.columns if column not in columns ]
	return df_files[ columns + extras ]

def build_file_selection_options( rows: list[ dict[ str, Any ] ] ) -> dict[ str, str ]:
	"""
	
		Purpose:
		--------
		Build display labels mapped to OpenAI file IDs.
		
		Parameters:
		-----------
		rows: list[dict[str, Any]]
			Normalized file metadata rows.
		
		Returns:
		--------
		dict[str, str]
			Display label to file ID mapping.
		
	"""
	options: dict[ str, str ] = { }
	
	if not isinstance( rows, list ):
		return options
	
	for row in rows:
		if not isinstance( row, dict ):
			continue
		
		file_id = row.get( 'id' )
		if not isinstance( file_id, str ) or not file_id.strip( ):
			continue
		
		filename = row.get( 'filename' ) or 'Unnamed file'
		purpose = row.get( 'purpose' ) or 'unknown'
		label = f'{filename} — {file_id} — {purpose}'
		options[ label ] = file_id
	
	return options

def get_selected_file_id( selected_label: str | None,
		options: dict[ str, str ] | None = None ) -> str | None:
	"""
	
		Purpose:
		--------
		Return the selected OpenAI file ID from a display label or direct ID.
		
		Parameters:
		-----------
		selected_label: str | None
			Selected display label or raw file ID.
		
		options: dict[str, str] | None
			Display label to file ID mapping.
		
		Returns:
		--------
		str | None
			Selected OpenAI file ID.
		
	"""
	if not isinstance( selected_label, str ) or not selected_label.strip( ):
		return None
	
	value = selected_label.strip( )
	
	if isinstance( options, dict ) and value in options:
		return options[ value ]
	
	if value.startswith( 'file-' ):
		return value
	
	return None

def render_files_table( rows: list[ dict[ str, Any ] ] ) -> None:
	"""
	
		Purpose:
		--------
		Render normalized Files API rows as a Streamlit data editor.
		
		Parameters:
		-----------
		rows: list[dict[str, Any]]
			Normalized file metadata rows.
		
		Returns:
		--------
		None
		
	"""
	df_files = build_files_dataframe( rows )
	st.session_state[ 'files_df' ] = df_files
	
	if df_files.empty:
		st.info( 'No files available.' )
		return
	
	st.data_editor( df_files, use_container_width=True, hide_index=True )

def render_file_metadata( metadata: dict[ str, Any ] | None ) -> None:
	"""
	
		Purpose:
		--------
		Render selected OpenAI file metadata.
		
		Parameters:
		-----------
		metadata: dict[str, Any] | None
			Normalized file metadata.
		
		Returns:
		--------
		None
		
	"""
	if not isinstance( metadata, dict ) or len( metadata ) == 0:
		st.info( 'No file metadata available.' )
		return
	
	m1, m2, m3, m4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
		border=True, gap='xxsmall' )
	
	with m1:
		st.metric( 'Purpose', metadata.get( 'purpose', '—' ) or '—' )
	
	with m2:
		st.metric( 'Bytes', metadata.get( 'bytes', 0 ) or 0 )
	
	with m3:
		st.metric( 'Status', metadata.get( 'status', '—' ) or '—' )
	
	with m4:
		st.metric( 'Object', metadata.get( 'object', '—' ) or '—' )
	
	st.json( metadata )

def render_file_content( content: str | bytes | dict[ str, Any ] | None ) -> None:
	"""
	
		Purpose:
		--------
		Render normalized OpenAI file content safely.
		
		Parameters:
		-----------
		content: str | bytes | dict[str, Any] | None
			Normalized file content.
		
		Returns:
		--------
		None
		
	"""
	if content is None:
		st.info( 'No file content available.' )
		return
	
	if isinstance( content, bytes ):
		st.download_button( label='Download File Content',
			data=content, file_name='openai_file_content.bin',
			mime='application/octet-stream', width='stretch' )
		return
	
	if isinstance( content, dict ):
		st.json( content )
		return
	
	if isinstance( content, str ):
		if not content.strip( ):
			st.info( 'File content is empty.' )
			return
		
		st.text_area( label='File Content', value=content,
			height=300, width='stretch', disabled=True )
		return
	
	st.write( content )

def render_file_delete_result( result: dict[ str, Any ] | None ) -> None:
	"""
	
		Purpose:
		--------
		Render an OpenAI Files API deletion result.
		
		Parameters:
		-----------
		result: dict[str, Any] | None
			Normalized delete result.
		
		Returns:
		--------
		None
		
	"""
	if not isinstance( result, dict ) or len( result ) == 0:
		return
	
	deleted = result.get( 'deleted' )
	if deleted is True:
		st.success( f"Deleted file: {result.get( 'id', '' )}" )
	else:
		st.warning( 'Delete request completed, but deletion status was not confirmed.' )
	
	st.json( result )

def clear_files_messages( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Files mode chat messages.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state.files_messages = [ ]

def clear_files_outputs( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Files mode output state while preserving configuration controls.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'files_metadata' ] = { }
	st.session_state[ 'files_content' ] = ''
	st.session_state[ 'files_content_bytes' ] = None
	st.session_state[ 'files_delete_result' ] = { }
	st.session_state[ 'files_last_answer' ] = ''

def reset_files_controls( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Files mode controls through a widget-safe callback.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	for key in [ 'files_model', 'files_purpose', 'files_filter_purpose',
	             'files_type', 'files_id', 'files_url' ]:
		if key in st.session_state:
			del st.session_state[ key ]

def reset_files_all( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Files mode controls, outputs, table state, and messages.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	reset_files_controls( )
	clear_files_outputs( )
	st.session_state[ 'files_table' ] = [ ]
	st.session_state[ 'files_df' ] = pd.DataFrame( )
	st.session_state.files_messages = [ ]

def clear_files_instructions( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Files mode system instructions and selected prompt template.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'files_system_instructions' ] = ''
	st.session_state[ 'instructions' ] = ''

def load_files_instruction_template( ) -> None:
	"""
	
		Purpose:
		--------
		Load the selected prompt template into Files mode system instructions.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	name = st.session_state.get( 'instructions' )
	if name and name != 'No Templates Found':
		text = fetch_prompt_text( cfg.DB_PATH, name )
		if text is not None:
			st.session_state[ 'files_system_instructions' ] = text

def convert_files_system_instructions( ) -> None:
	"""
	
		Purpose:
		--------
		Convert Files mode system instructions between XML-like delimiters and Markdown
		headings.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	text = st.session_state.get( 'files_system_instructions', '' )
	if not isinstance( text, str ) or not text.strip( ):
		return
	
	source = text.strip( )
	if cfg.XML_BLOCK_PATTERN.search( source ):
		converted = convert_xml( source )
	else:
		converted = convert_markdown( source )
	
	st.session_state[ 'files_system_instructions' ] = converted

def run_files_upload( files: Files, uploaded_file: Any, purpose: str | None ) -> dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Save and upload a Streamlit file through the Files wrapper.
		
		Parameters:
		-----------
		files: Files
			Files wrapper instance.
		
		uploaded_file: Any
			Streamlit uploaded file object.
		
		purpose: str | None
			Selected OpenAI upload purpose.
		
		Returns:
		--------
		dict[str, Any]
			Uploaded file metadata.
		
	"""
	if uploaded_file is None:
		st.warning( 'Select a file before uploading.' )
		return { }
	
	filepath = save_files_upload( uploaded_file )
	if not filepath:
		st.warning( 'The uploaded file could not be saved locally.' )
		return { }
	
	try:
		metadata = files.upload( filepath=filepath, purpose=purpose or 'user_data' )
		metadata = metadata if isinstance( metadata, dict ) else { }
		
		if metadata.get( 'id' ):
			st.session_state[ 'files_id' ] = metadata.get( 'id' )
			st.session_state[ 'files_metadata' ] = metadata
		
		return metadata
	finally:
		try:
			if os.path.exists( filepath ):
				os.remove( filepath )
		except Exception:
			pass

def run_files_list( files: Files, purpose: str | None = None ) -> list[ dict[ str, Any ] ]:
	"""
	
		Purpose:
		--------
		List files through the Files wrapper and store normalized rows.
		
		Parameters:
		-----------
		files: Files
			Files wrapper instance.
		
		purpose: str | None
			Optional file purpose filter.
		
		Returns:
		--------
		list[dict[str, Any]]
			Normalized file metadata rows.
		
	"""
	rows = files.list(
		purpose=purpose if isinstance( purpose, str ) and purpose.strip( ) else None )
	rows = normalize_files_table( rows )
	st.session_state[ 'files_table' ] = rows
	st.session_state[ 'files_df' ] = build_files_dataframe( rows )
	return rows

def run_files_retrieve( files: Files, file_id: str | None ) -> dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Retrieve metadata for a selected OpenAI file.
		
		Parameters:
		-----------
		files: Files
			Files wrapper instance.
		
		file_id: str | None
			Selected OpenAI file ID.
		
		Returns:
		--------
		dict[str, Any]
			Normalized file metadata.
		
	"""
	if not isinstance( file_id, str ) or not file_id.strip( ):
		st.warning( 'Select or enter a file ID before retrieving metadata.' )
		return { }
	
	metadata = files.retrieve( id=file_id.strip( ) )
	metadata = metadata if isinstance( metadata, dict ) else { }
	st.session_state[ 'files_metadata' ] = metadata
	return metadata

def run_files_extract( files: Files, file_id: str | None ) -> str | bytes | dict[ str, Any ] | None:
	"""
	
		Purpose:
		--------
		Retrieve content for a selected OpenAI file.
		
		Parameters:
		-----------
		files: Files
			Files wrapper instance.
		
		file_id: str | None
			Selected OpenAI file ID.
		
		Returns:
		--------
		str | bytes | dict[str, Any] | None
			Normalized file content.
		
	"""
	if not isinstance( file_id, str ) or not file_id.strip( ):
		st.warning( 'Select or enter a file ID before retrieving content.' )
		return None
	
	content = files.extract( id=file_id.strip( ) )
	
	if isinstance( content, bytes ):
		st.session_state[ 'files_content_bytes' ] = content
		st.session_state[ 'files_content' ] = ''
	elif isinstance( content, str ):
		st.session_state[ 'files_content' ] = content
		st.session_state[ 'files_content_bytes' ] = None
	elif isinstance( content, dict ):
		st.session_state[ 'files_content' ] = str( content )
		st.session_state[ 'files_content_bytes' ] = None
	else:
		st.session_state[ 'files_content' ] = ''
		st.session_state[ 'files_content_bytes' ] = None
	
	return content

def run_files_delete( files: Files, file_id: str | None ) -> dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Delete a selected OpenAI file.
		
		Parameters:
		-----------
		files: Files
			Files wrapper instance.
		
		file_id: str | None
			Selected OpenAI file ID.
		
		Returns:
		--------
		dict[str, Any]
			Normalized deletion result.
		
	"""
	if not isinstance( file_id, str ) or not file_id.strip( ):
		st.warning( 'Select or enter a file ID before deleting a file.' )
		return { }
	
	result = files.delete( id=file_id.strip( ) )
	result = result if isinstance( result, dict ) else { }
	st.session_state[ 'files_delete_result' ] = result
	
	if result.get( 'deleted' ) is True:
		st.session_state[ 'files_id' ] = ''
	
	return result

def run_files_analysis( files: Files, file_id: str | None, prompt: str | None,
		model: str | None = None ) -> str | None:
	"""
	
		Purpose:
		--------
		Run selected-file analysis through the Files wrapper.
		
		Parameters:
		-----------
		files: Files
			Files wrapper instance.
		
		file_id: str | None
			Selected OpenAI file ID.
		
		prompt: str | None
			User analysis prompt.
		
		model: str | None
			Optional model name.
		
		Returns:
		--------
		str | None
			Analysis output text.
		
	"""
	if not isinstance( file_id, str ) or not file_id.strip( ):
		st.warning( 'Select or enter a file ID before analyzing a file.' )
		return None
	
	if not isinstance( prompt, str ) or not prompt.strip( ):
		st.warning( 'Enter a prompt before analyzing a file.' )
		return None
	
	model_value = model if isinstance( model, str ) and model.strip( ) else 'gpt-4o-mini'
	response = files.search( id=file_id.strip( ), query=prompt.strip( ), model=model_value )
	
	if isinstance( response, str ) and response.strip( ):
		st.session_state[ 'files_last_answer' ] = response.strip( )
		return response.strip( )
	
	return response

# ------------ VECTOR STORES UTILITIES -----------------

def ensure_vectorstores_mode_state( ) -> None:
	"""
	
		Purpose:
		--------
		Ensure Vector Stores mode session-state keys exist before widget instantiation.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	if 'stores_model' not in st.session_state:
		st.session_state[ 'stores_model' ] = ''
	
	if 'stores_id' not in st.session_state:
		st.session_state[ 'stores_id' ] = ''
	
	if 'stores_manual_id' not in st.session_state:
		st.session_state[ 'stores_manual_id' ] = ''
	
	if 'stores_selected_label' not in st.session_state:
		st.session_state[ 'stores_selected_label' ] = ''
	
	if 'stores_name' not in st.session_state:
		st.session_state[ 'stores_name' ] = ''
	
	if 'stores_description' not in st.session_state:
		st.session_state[ 'stores_description' ] = ''
	
	if 'stores_metadata' not in st.session_state:
		st.session_state[ 'stores_metadata' ] = ''
	
	if 'stores_expires_days' not in st.session_state:
		st.session_state[ 'stores_expires_days' ] = 0
	
	if 'stores_expires_anchor' not in st.session_state:
		st.session_state[ 'stores_expires_anchor' ] = 'last_active_at'
	
	if 'stores_file_ids' not in st.session_state:
		st.session_state[ 'stores_file_ids' ] = ''
	
	if 'stores_chunking_strategy' not in st.session_state:
		st.session_state[ 'stores_chunking_strategy' ] = 'auto'
	
	if 'stores_chunk_size' not in st.session_state:
		st.session_state[ 'stores_chunk_size' ] = 800
	
	if 'stores_chunk_overlap' not in st.session_state:
		st.session_state[ 'stores_chunk_overlap' ] = 400
	
	if 'stores_table' not in st.session_state:
		st.session_state[ 'stores_table' ] = [ ]
	
	if 'stores_df' not in st.session_state:
		st.session_state[ 'stores_df' ] = pd.DataFrame( )
	
	if 'stores_store_metadata' not in st.session_state:
		st.session_state[ 'stores_store_metadata' ] = { }
	
	if 'stores_files_table' not in st.session_state:
		st.session_state[ 'stores_files_table' ] = [ ]
	
	if 'stores_files_df' not in st.session_state:
		st.session_state[ 'stores_files_df' ] = pd.DataFrame( )
	
	if 'stores_file_id' not in st.session_state:
		st.session_state[ 'stores_file_id' ] = ''
	
	if 'stores_file_selected_label' not in st.session_state:
		st.session_state[ 'stores_file_selected_label' ] = ''
	
	if 'stores_file_attributes' not in st.session_state:
		st.session_state[ 'stores_file_attributes' ] = ''
	
	if 'stores_batch_id' not in st.session_state:
		st.session_state[ 'stores_batch_id' ] = ''
	
	if 'stores_batch_result' not in st.session_state:
		st.session_state[ 'stores_batch_result' ] = { }
	
	if 'stores_search_query' not in st.session_state:
		st.session_state[ 'stores_search_query' ] = ''
	
	if 'stores_search_results' not in st.session_state:
		st.session_state[ 'stores_search_results' ] = [ ]
	
	if 'stores_max_results' not in st.session_state:
		st.session_state[ 'stores_max_results' ] = 10
	
	if 'stores_ranker' not in st.session_state:
		st.session_state[ 'stores_ranker' ] = 'auto'
	
	if 'stores_score_threshold' not in st.session_state:
		st.session_state[ 'stores_score_threshold' ] = 0.0
	
	if 'stores_rewrite_query' not in st.session_state:
		st.session_state[ 'stores_rewrite_query' ] = False
	
	if 'stores_last_answer' not in st.session_state:
		st.session_state[ 'stores_last_answer' ] = ''
	
	if 'stores_system_instructions' not in st.session_state:
		st.session_state[ 'stores_system_instructions' ] = ''
	
	if 'stores_messages' not in st.session_state:
		st.session_state.stores_messages = [ ]
	
	if not isinstance( st.session_state.get( 'stores_messages' ), list ):
		st.session_state.stores_messages = [ ]

def get_vector_store_model_options( vector: VectorStores ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Return model options for Vector Stores answer workflows.
		
		Parameters:
		-----------
		vector: VectorStores
			VectorStores wrapper instance.
		
		Returns:
		--------
		list[str]
			Model option names.
		
	"""
	options = getattr( vector, 'model_options', None )
	if isinstance( options, list ) and len( options ) > 0:
		return [ '' ] + options
	
	return [
			'',
			'gpt-5-mini',
			'gpt-5-nano',
			'gpt-4.1-mini',
			'gpt-4.1-nano',
			'gpt-4o-mini',
	]

def get_vector_store_ranker_options( vector: VectorStores ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Return native vector store search ranker options.
		
		Parameters:
		-----------
		vector: VectorStores
			VectorStores wrapper instance.
		
		Returns:
		--------
		list[str]
			Ranker option names.
		
	"""
	options = getattr( vector, 'ranker_options', None )
	if isinstance( options, list ) and len( options ) > 0:
		return options
	
	return [
			'auto',
			'default-2024-11-15',
	]

def get_vector_store_chunking_options( vector: VectorStores ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Return vector store chunking strategy options.
		
		Parameters:
		-----------
		vector: VectorStores
			VectorStores wrapper instance.
		
		Returns:
		--------
		list[str]
			Chunking strategy option names.
		
	"""
	options = getattr( vector, 'chunking_strategy_options', None )
	if isinstance( options, list ) and len( options ) > 0:
		return options
	
	return [
			'auto',
			'static',
	]

def parse_vector_store_file_ids( value: str | list[ str ] | None ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Parse comma-delimited or list-based OpenAI file IDs for vector store workflows.
		
		Parameters:
		-----------
		value: str | list[str] | None
			Comma-delimited file IDs or a list of file IDs.
		
		Returns:
		--------
		list[str]
			Clean file IDs.
		
	"""
	if value is None:
		return [ ]
	
	if isinstance( value, list ):
		return [ str( item ).strip( ) for item in value if str( item ).strip( ) ]
	
	if not isinstance( value, str ) or not value.strip( ):
		return [ ]
	
	return [ item.strip( ) for item in value.split( ',' ) if item.strip( ) ]

def parse_vector_store_json( value: str | None, label: str = 'JSON' ) -> dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Parse optional JSON text into a dictionary for metadata, attributes, or filters.
		
		Parameters:
		-----------
		value: str | None
			Optional JSON object text.
		
		label: str
			Display label used in warning messages.
		
		Returns:
		--------
		dict[str, Any]
			Parsed dictionary or empty dictionary.
		
	"""
	if not isinstance( value, str ) or not value.strip( ):
		return { }
	
	try:
		parsed = json.loads( value )
	except Exception as exc:
		st.warning( f'{label} could not be parsed and will be omitted: {exc}' )
		return { }
	
	if not isinstance( parsed, dict ):
		st.warning( f'{label} must be a JSON object. It will be omitted.' )
		return { }
	
	return parsed

def build_vector_store_expires_after_from_state( vector: VectorStores ) -> dict[ str, Any ] | None:
	"""
	
		Purpose:
		--------
		Build a vector store expiration policy from Streamlit session state.
		
		Parameters:
		-----------
		vector: VectorStores
			VectorStores wrapper instance.
		
		Returns:
		--------
		dict[str, Any] | None
			Expiration policy or None.
		
	"""
	days = st.session_state.get( 'stores_expires_days', 0 )
	anchor = st.session_state.get( 'stores_expires_anchor', 'last_active_at' )
	
	try:
		day_value = int( days )
	except Exception:
		day_value = 0
	
	if day_value <= 0:
		return None
	
	return vector.build_expires_after( anchor=anchor, days=day_value )

def build_vector_store_chunking_strategy_from_state( vector: VectorStores ) -> Dict[  str, Any ] | None:
	"""
	
		Purpose:
		--------
		Build a vector store chunking strategy from Streamlit session state.
		
		Parameters:
		-----------
		vector: VectorStores
			VectorStores wrapper instance.
		
		Returns:
		--------
		dict[str, Any] | None
			Chunking strategy or None.
		
	"""
	strategy = st.session_state.get( 'stores_chunking_strategy', 'auto' )
	
	if not isinstance( strategy, str ) or not strategy.strip( ):
		return None
	
	if strategy == 'auto':
		return vector.build_chunking_strategy( strategy='auto' )
	
	return vector.build_chunking_strategy(
		strategy='static',
		max_chunk_size_tokens=st.session_state.get( 'stores_chunk_size', 800 ),
		chunk_overlap_tokens=st.session_state.get( 'stores_chunk_overlap', 400 ) )

def build_vector_store_ranking_options_from_state( ) -> dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Build native vector store search ranking options from Streamlit session state.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		dict[str, Any]
			Ranking options dictionary.
		
	"""
	ranker = st.session_state.get( 'stores_ranker', 'auto' )
	score_threshold = st.session_state.get( 'stores_score_threshold', 0.0 )
	options: dict[ str, Any ] = { }
	
	if isinstance( ranker, str ) and ranker.strip( ):
		options[ 'ranker' ] = ranker.strip( )
	
	try:
		threshold = float( score_threshold )
	except Exception:
		threshold = 0.0
	
	if threshold > 0.0:
		options[ 'score_threshold' ] = threshold
	
	return options

def normalize_vector_store_rows( rows: Any ) -> list[ dict[ str, Any ] ]:
	"""
	
		Purpose:
		--------
		Normalize vector store rows returned from the wrapper or API.
		
		Parameters:
		-----------
		rows: Any
			Vector store rows or response object.
		
		Returns:
		--------
		list[dict[str, Any]]
			Normalized vector store rows.
		
	"""
	if rows is None:
		return [ ]
	
	if isinstance( rows, dict ) and isinstance( rows.get( 'data' ), list ):
		items = rows.get( 'data', [ ] )
	elif isinstance( rows, list ):
		items = rows
	else:
		items = getattr( rows, 'data', [ ] )
	
	normalized: list[ dict[ str, Any ] ] = [ ]
	for item in items:
		if isinstance( item, dict ):
			source = item
		elif hasattr( item, 'model_dump' ):
			try:
				source = item.model_dump( )
			except Exception:
				source = { }
		else:
			source = {
					'id': getattr( item, 'id', None ),
					'name': getattr( item, 'name', None ),
					'description': getattr( item, 'description', None ),
					'created_at': getattr( item, 'created_at', None ),
					'object': getattr( item, 'object', None ),
					'usage_bytes': getattr( item, 'usage_bytes', None ),
					'file_counts': getattr( item, 'file_counts', None ),
					'status': getattr( item, 'status', None ),
					'expires_at': getattr( item, 'expires_at', None ),
					'last_active_at': getattr( item, 'last_active_at', None ),
			}
		
		store_id = source.get( 'id' )
		if not store_id:
			continue
		
		normalized.append(
			{
					'id': store_id,
					'name': source.get( 'name', '' ),
					'description': source.get( 'description', '' ),
					'status': source.get( 'status', '' ),
					'usage_bytes': source.get( 'usage_bytes', 0 ),
					'file_counts': source.get( 'file_counts', { } ),
					'created_at': source.get( 'created_at', '' ),
					'expires_at': source.get( 'expires_at', '' ),
					'last_active_at': source.get( 'last_active_at', '' ),
					'object': source.get( 'object', '' ),
					'metadata': source.get( 'metadata', { } ),
			} )
	
	return normalized

def normalize_vector_store_file_rows( rows: Any ) -> list[ dict[ str, Any ] ]:
	"""
	
		Purpose:
		--------
		Normalize vector store file rows returned from the wrapper or API.
		
		Parameters:
		-----------
		rows: Any
			Vector store file rows or response object.
		
		Returns:
		--------
		list[dict[str, Any]]
			Normalized vector store file rows.
		
	"""
	if rows is None:
		return [ ]
	
	if isinstance( rows, dict ) and isinstance( rows.get( 'data' ), list ):
		items = rows.get( 'data', [ ] )
	elif isinstance( rows, list ):
		items = rows
	else:
		items = getattr( rows, 'data', [ ] )
	
	normalized: list[ dict[ str, Any ] ] = [ ]
	for item in items:
		if isinstance( item, dict ):
			source = item
		elif hasattr( item, 'model_dump' ):
			try:
				source = item.model_dump( )
			except Exception:
				source = { }
		else:
			source = {
					'id': getattr( item, 'id', None ),
					'object': getattr( item, 'object', None ),
					'created_at': getattr( item, 'created_at', None ),
					'vector_store_id': getattr( item, 'vector_store_id', None ),
					'status': getattr( item, 'status', None ),
					'last_error': getattr( item, 'last_error', None ),
					'usage_bytes': getattr( item, 'usage_bytes', None ),
					'attributes': getattr( item, 'attributes', None ),
			}
		
		file_id = source.get( 'id' )
		if not file_id:
			continue
		
		normalized.append(
			{
					'id': file_id,
					'vector_store_id': source.get( 'vector_store_id', '' ),
					'status': source.get( 'status', '' ),
					'usage_bytes': source.get( 'usage_bytes', 0 ),
					'created_at': source.get( 'created_at', '' ),
					'last_error': source.get( 'last_error', '' ),
					'attributes': source.get( 'attributes', { } ),
					'object': source.get( 'object', '' ),
					'chunking_strategy': source.get( 'chunking_strategy', { } ),
			} )
	
	return normalized

def build_vector_stores_dataframe( rows: list[ dict[ str, Any ] ] ) -> pd.DataFrame:
	"""
	
		Purpose:
		--------
		Build a display-ready DataFrame from normalized vector store rows.
		
		Parameters:
		-----------
		rows: list[dict[str, Any]]
			Normalized vector store rows.
		
		Returns:
		--------
		pd.DataFrame
			Display-ready vector stores table.
		
	"""
	if not isinstance( rows, list ) or len( rows ) == 0:
		return pd.DataFrame( )
	
	df_stores = pd.DataFrame( rows )
	preferred = [
			'id',
			'name',
			'description',
			'status',
			'usage_bytes',
			'file_counts',
			'created_at',
			'expires_at',
			'last_active_at',
			'object',
			'metadata',
	]
	
	columns = [ column for column in preferred if column in df_stores.columns ]
	extras = [ column for column in df_stores.columns if column not in columns ]
	return df_stores[ columns + extras ]

def build_vector_store_files_dataframe( rows: list[ dict[ str, Any ] ] ) -> pd.DataFrame:
	"""
	
		Purpose:
		--------
		Build a display-ready DataFrame from normalized vector store file rows.
		
		Parameters:
		-----------
		rows: list[dict[str, Any]]
			Normalized vector store file rows.
		
		Returns:
		--------
		pd.DataFrame
			Display-ready vector store files table.
		
	"""
	if not isinstance( rows, list ) or len( rows ) == 0:
		return pd.DataFrame( )
	
	df_files = pd.DataFrame( rows )
	preferred = [
			'id',
			'vector_store_id',
			'status',
			'usage_bytes',
			'created_at',
			'last_error',
			'attributes',
			'object',
			'chunking_strategy',
	]
	
	columns = [ column for column in preferred if column in df_files.columns ]
	extras = [ column for column in df_files.columns if column not in columns ]
	return df_files[ columns + extras ]

def build_vector_store_selection_options( rows: list[ dict[ str, Any ] ] ) -> dict[ str, str ]:
	"""
	
		Purpose:
		--------
		Build vector store display labels mapped to vector store IDs.
		
		Parameters:
		-----------
		rows: list[dict[str, Any]]
			Normalized vector store rows.
		
		Returns:
		--------
		dict[str, str]
			Display label to vector store ID mapping.
		
	"""
	options: dict[ str, str ] = { }
	
	if not isinstance( rows, list ):
		return options
	
	for row in rows:
		if not isinstance( row, dict ):
			continue
		
		store_id = row.get( 'id' )
		if not isinstance( store_id, str ) or not store_id.strip( ):
			continue
		
		name = row.get( 'name' ) or 'Unnamed store'
		status = row.get( 'status' ) or 'unknown'
		label = f'{name} — {store_id} — {status}'
		options[ label ] = store_id
	
	return options

def get_selected_vector_store_id( selected_label: str | None,
		options: dict[ str, str ] | None = None ) -> str | None:
	"""
	
		Purpose:
		--------
		Return the selected vector store ID from a display label or direct ID.
		
		Parameters:
		-----------
		selected_label: str | None
			Selected display label or raw vector store ID.
		
		options: dict[str, str] | None
			Display label to vector store ID mapping.
		
		Returns:
		--------
		str | None
			Selected vector store ID.
		
	"""
	if not isinstance( selected_label, str ) or not selected_label.strip( ):
		return None
	
	value = selected_label.strip( )
	
	if isinstance( options, dict ) and value in options:
		return options[ value ]
	
	if value.startswith( 'vs_' ):
		return value
	
	return None

def build_vector_store_file_selection_options( rows: list[ dict[ str, Any ] ] ) -> dict[ str, str ]:
	"""
	
		Purpose:
		--------
		Build vector store file display labels mapped to OpenAI file IDs.
		
		Parameters:
		-----------
		rows: list[dict[str, Any]]
			Normalized vector store file rows.
		
		Returns:
		--------
		dict[str, str]
			Display label to file ID mapping.
		
	"""
	options: dict[ str, str ] = { }
	
	if not isinstance( rows, list ):
		return options
	
	for row in rows:
		if not isinstance( row, dict ):
			continue
		
		file_id = row.get( 'id' )
		if not isinstance( file_id, str ) or not file_id.strip( ):
			continue
		
		status = row.get( 'status' ) or 'unknown'
		label = f'{file_id} — {status}'
		options[ label ] = file_id
	
	return options

def get_selected_vector_store_file_id( selected_label: str | None,
		options: dict[ str, str ] | None = None ) -> str | None:
	"""
	
		Purpose:
		--------
		Return the selected vector store file ID from a display label or direct ID.
		
		Parameters:
		-----------
		selected_label: str | None
			Selected display label or raw file ID.
		
		options: dict[str, str] | None
			Display label to file ID mapping.
		
		Returns:
		--------
		str | None
			Selected OpenAI file ID.
		
	"""
	if not isinstance( selected_label, str ) or not selected_label.strip( ):
		return None
	
	value = selected_label.strip( )
	
	if isinstance( options, dict ) and value in options:
		return options[ value ]
	
	if value.startswith( 'file-' ):
		return value
	
	return None

def render_vector_stores_table( rows: list[ dict[ str, Any ] ] ) -> None:
	"""
	
		Purpose:
		--------
		Render vector stores as a Streamlit data editor.
		
		Parameters:
		-----------
		rows: list[dict[str, Any]]
			Normalized vector store rows.
		
		Returns:
		--------
		None
		
	"""
	df_stores = build_vector_stores_dataframe( rows )
	st.session_state[ 'stores_df' ] = df_stores
	
	if df_stores.empty:
		st.info( 'No vector stores available.' )
		return
	
	st.data_editor( df_stores, use_container_width=True, hide_index=True )

def render_vector_store_metadata( metadata: dict[ str, Any ] | None ) -> None:
	"""
	
		Purpose:
		--------
		Render selected vector store metadata.
		
		Parameters:
		-----------
		metadata: dict[str, Any] | None
			Normalized vector store metadata.
		
		Returns:
		--------
		None
		
	"""
	if not isinstance( metadata, dict ) or len( metadata ) == 0:
		st.info( 'No vector store metadata available.' )
		return
	
	m1, m2, m3, m4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
		border=True, gap='xxsmall' )
	
	with m1:
		st.metric( 'Status', metadata.get( 'status', '—' ) or '—' )
	
	with m2:
		st.metric( 'Usage Bytes', metadata.get( 'usage_bytes', 0 ) or 0 )
	
	with m3:
		st.metric( 'Object', metadata.get( 'object', '—' ) or '—' )
	
	with m4:
		file_counts = metadata.get( 'file_counts', { } )
		if isinstance( file_counts, dict ):
			st.metric( 'Files', file_counts.get( 'total', 0 ) or 0 )
		else:
			st.metric( 'Files', '—' )
	
	st.json( metadata )

def render_vector_store_files_table( rows: list[ dict[ str, Any ] ] ) -> None:
	"""
	
		Purpose:
		--------
		Render vector store files as a Streamlit data editor.
		
		Parameters:
		-----------
		rows: list[dict[str, Any]]
			Normalized vector store file rows.
		
		Returns:
		--------
		None
		
	"""
	df_files = build_vector_store_files_dataframe( rows )
	st.session_state[ 'stores_files_df' ] = df_files
	
	if df_files.empty:
		st.info( 'No vector store files available.' )
		return
	
	st.data_editor( df_files, use_container_width=True, hide_index=True )

def render_vector_store_search_results( results: list[ dict[ str, Any ] ] ) -> None:
	"""
	
		Purpose:
		--------
		Render native vector store search results.
		
		Parameters:
		-----------
		results: list[dict[str, Any]]
			Normalized search result rows.
		
		Returns:
		--------
		None
		
	"""
	if not isinstance( results, list ) or len( results ) == 0:
		st.info( 'No search results available.' )
		return
	
	df_results = pd.DataFrame( results )
	st.data_editor( df_results, use_container_width=True, hide_index=True )

def render_vector_store_batch_result( result: dict[ str, Any ] | None ) -> None:
	"""
	
		Purpose:
		--------
		Render vector store file batch metadata.
		
		Parameters:
		-----------
		result: dict[str, Any] | None
			Normalized file batch metadata.
		
		Returns:
		--------
		None
		
	"""
	if not isinstance( result, dict ) or len( result ) == 0:
		st.info( 'No batch result available.' )
		return
	
	st.json( result )

def clear_vector_store_outputs( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Vector Stores mode output state while preserving controls.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'stores_store_metadata' ] = { }
	st.session_state[ 'stores_files_table' ] = [ ]
	st.session_state[ 'stores_files_df' ] = pd.DataFrame( )
	st.session_state[ 'stores_batch_result' ] = { }
	st.session_state[ 'stores_search_results' ] = [ ]
	st.session_state[ 'stores_last_answer' ] = ''

def clear_vector_store_messages( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Vector Stores mode chat messages.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state.stores_messages = [ ]

def reset_vector_store_controls( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Vector Stores mode controls through a widget-safe callback.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	for key in [ 'stores_model', 'stores_id', 'stores_manual_id',
	             'stores_selected_label', 'stores_name', 'stores_description',
	             'stores_metadata', 'stores_expires_days', 'stores_expires_anchor',
	             'stores_file_ids', 'stores_chunking_strategy', 'stores_chunk_size',
	             'stores_chunk_overlap', 'stores_file_id', 'stores_file_selected_label',
	             'stores_file_attributes', 'stores_batch_id', 'stores_search_query',
	             'stores_max_results', 'stores_ranker', 'stores_score_threshold',
	             'stores_rewrite_query' ]:
		if key in st.session_state:
			del st.session_state[ key ]

def reset_vector_store_all( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Vector Stores mode controls, outputs, tables, and messages.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	reset_vector_store_controls( )
	clear_vector_store_outputs( )
	st.session_state[ 'stores_table' ] = [ ]
	st.session_state[ 'stores_df' ] = pd.DataFrame( )
	st.session_state.stores_messages = [ ]

def clear_vector_store_instructions( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Vector Stores mode system instructions and selected prompt template.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'stores_system_instructions' ] = ''
	st.session_state[ 'instructions' ] = ''

def load_vector_store_instruction_template( ) -> None:
	"""
	
		Purpose:
		--------
		Load the selected prompt template into Vector Stores mode system instructions.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	name = st.session_state.get( 'instructions' )
	if name and name != 'No Templates Found':
		text = fetch_prompt_text( cfg.DB_PATH, name )
		if text is not None:
			st.session_state[ 'stores_system_instructions' ] = text

def convert_vector_store_system_instructions( ) -> None:
	"""
	
		Purpose:
		--------
		Convert Vector Stores mode system instructions between XML-like delimiters and
		Markdown headings.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	text = st.session_state.get( 'stores_system_instructions', '' )
	if not isinstance( text, str ) or not text.strip( ):
		return
	
	source = text.strip( )
	if cfg.XML_BLOCK_PATTERN.search( source ):
		converted = convert_xml( source )
	else:
		converted = convert_markdown( source )
	
	st.session_state[ 'stores_system_instructions' ] = converted

def run_vector_store_create( vector: VectorStores ) -> dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Create a vector store through the VectorStores wrapper.
		
		Parameters:
		-----------
		vector: VectorStores
			VectorStores wrapper instance.
		
		Returns:
		--------
		dict[str, Any]
			Normalized vector store metadata.
		
	"""
	name = st.session_state.get( 'stores_name', '' )
	if not isinstance( name, str ) or not name.strip( ):
		st.warning( 'Enter a vector store name before creating a store.' )
		return { }
	
	metadata = parse_vector_store_json(
		st.session_state.get( 'stores_metadata', '' ),
		label='Vector store metadata' )
	
	file_ids = parse_vector_store_file_ids(
		st.session_state.get( 'stores_file_ids', '' ) )
	
	result = vector.create(
		name=name.strip( ),
		description=st.session_state.get( 'stores_description', '' ) or None,
		metadata=metadata,
		expires_after=build_vector_store_expires_after_from_state( vector ),
		file_ids=file_ids,
		chunking_strategy=build_vector_store_chunking_strategy_from_state( vector ) )
	
	result = result if isinstance( result, dict ) else { }
	
	if result.get( 'id' ):
		st.session_state[ 'stores_id' ] = result.get( 'id' )
		st.session_state[ 'stores_store_metadata' ] = result
	
	return result

def run_vector_store_list( vector: VectorStores ) -> list[ dict[ str, Any ] ]:
	"""
	
		Purpose:
		--------
		List vector stores through the VectorStores wrapper and store normalized rows.
		
		Parameters:
		-----------
		vector: VectorStores
			VectorStores wrapper instance.
		
		Returns:
		--------
		list[dict[str, Any]]
			Normalized vector store rows.
		
	"""
	rows = vector.list_stores( limit=100, order='desc' )
	rows = normalize_vector_store_rows( rows )
	st.session_state[ 'stores_table' ] = rows
	st.session_state[ 'stores_df' ] = build_vector_stores_dataframe( rows )
	return rows

def run_vector_store_retrieve( vector: VectorStores, store_id: str | None ) -> dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Retrieve vector store metadata through the VectorStores wrapper.
		
		Parameters:
		-----------
		vector: VectorStores
			VectorStores wrapper instance.
		
		store_id: str | None
			Selected vector store ID.
		
		Returns:
		--------
		dict[str, Any]
			Normalized vector store metadata.
		
	"""
	if not isinstance( store_id, str ) or not store_id.strip( ):
		st.warning( 'Select or enter a vector store ID before retrieving metadata.' )
		return { }
	
	result = vector.retrieve( store_id=store_id.strip( ) )
	result = result if isinstance( result, dict ) else { }
	st.session_state[ 'stores_store_metadata' ] = result
	st.session_state[ 'stores_id' ] = store_id.strip( )
	return result

def run_vector_store_update( vector: VectorStores, store_id: str | None ) -> dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Update vector store metadata through the VectorStores wrapper.
		
		Parameters:
		-----------
		vector: VectorStores
			VectorStores wrapper instance.
		
		store_id: str | None
			Selected vector store ID.
		
		Returns:
		--------
		dict[str, Any]
			Normalized vector store metadata.
		
	"""
	if not isinstance( store_id, str ) or not store_id.strip( ):
		st.warning( 'Select or enter a vector store ID before updating.' )
		return { }
	
	metadata = parse_vector_store_json(
		st.session_state.get( 'stores_metadata', '' ),
		label='Vector store metadata' )
	
	result = vector.update(
		store_id=store_id.strip( ),
		name=st.session_state.get( 'stores_name', '' ) or None,
		description=st.session_state.get( 'stores_description', '' ) or None,
		metadata=metadata,
		expires_after=build_vector_store_expires_after_from_state( vector ) )
	
	result = result if isinstance( result, dict ) else { }
	st.session_state[ 'stores_store_metadata' ] = result
	return result

def run_vector_store_delete( vector: VectorStores, store_id: str | None ) -> dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Delete a vector store through the VectorStores wrapper.
		
		Parameters:
		-----------
		vector: VectorStores
			VectorStores wrapper instance.
		
		store_id: str | None
			Selected vector store ID.
		
		Returns:
		--------
		dict[str, Any]
			Normalized delete result.
		
	"""
	if not isinstance( store_id, str ) or not store_id.strip( ):
		st.warning( 'Select or enter a vector store ID before deleting.' )
		return { }
	
	result = vector.delete( store_id=store_id.strip( ) )
	result = result if isinstance( result, dict ) else { }
	
	if result.get( 'deleted' ) is True:
		st.session_state[ 'stores_id' ] = ''
		st.session_state[ 'stores_store_metadata' ] = { }
	
	return result

def run_vector_store_attach_file( vector: VectorStores,
		store_id: str | None ) -> dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Attach a file to a vector store through the VectorStores wrapper.
		
		Parameters:
		-----------
		vector: VectorStores
			VectorStores wrapper instance.
		
		store_id: str | None
			Selected vector store ID.
		
		Returns:
		--------
		dict[str, Any]
			Normalized vector store file metadata.
		
	"""
	if not isinstance( store_id, str ) or not store_id.strip( ):
		st.warning( 'Select or enter a vector store ID before attaching a file.' )
		return { }
	
	file_id = st.session_state.get( 'stores_file_id', '' )
	if not isinstance( file_id, str ) or not file_id.strip( ):
		st.warning( 'Enter an OpenAI file ID before attaching a file.' )
		return { }
	
	attributes = parse_vector_store_json(
		st.session_state.get( 'stores_file_attributes', '' ),
		label='Vector store file attributes' )
	
	result = vector.attach_file(
		store_id=store_id.strip( ),
		file_id=file_id.strip( ),
		attributes=attributes,
		chunking_strategy=build_vector_store_chunking_strategy_from_state( vector ) )
	
	return result if isinstance( result, dict ) else { }

def run_vector_store_list_files( vector: VectorStores,
		store_id: str | None ) -> list[ dict[ str, Any ] ]:
	"""
	
		Purpose:
		--------
		List files attached to a vector store.
		
		Parameters:
		-----------
		vector: VectorStores
			VectorStores wrapper instance.
		
		store_id: str | None
			Selected vector store ID.
		
		Returns:
		--------
		list[dict[str, Any]]
			Normalized vector store file rows.
		
	"""
	if not isinstance( store_id, str ) or not store_id.strip( ):
		st.warning( 'Select or enter a vector store ID before listing files.' )
		return [ ]
	
	rows = vector.list_files( store_id=store_id.strip( ), limit=100, order='desc' )
	rows = normalize_vector_store_file_rows( rows )
	st.session_state[ 'stores_files_table' ] = rows
	st.session_state[ 'stores_files_df' ] = build_vector_store_files_dataframe( rows )
	return rows

def run_vector_store_delete_file( vector: VectorStores, store_id: str | None,
		file_id: str | None ) -> dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Delete a file from a vector store.
		
		Parameters:
		-----------
		vector: VectorStores
			VectorStores wrapper instance.
		
		store_id: str | None
			Selected vector store ID.
		
		file_id: str | None
			Selected OpenAI file ID.
		
		Returns:
		--------
		dict[str, Any]
			Normalized delete result.
		
	"""
	if not isinstance( store_id, str ) or not store_id.strip( ):
		st.warning( 'Select or enter a vector store ID before deleting a file.' )
		return { }
	
	if not isinstance( file_id, str ) or not file_id.strip( ):
		st.warning( 'Select or enter a file ID before deleting it from the vector store.' )
		return { }
	
	result = vector.delete_file( store_id=store_id.strip( ), file_id=file_id.strip( ) )
	return result if isinstance( result, dict ) else { }

def run_vector_store_create_batch( vector: VectorStores,
		store_id: str | None ) -> dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Create a vector store file batch from comma-delimited file IDs.
		
		Parameters:
		-----------
		vector: VectorStores
			VectorStores wrapper instance.
		
		store_id: str | None
			Selected vector store ID.
		
		Returns:
		--------
		dict[str, Any]
			Normalized file batch metadata.
		
	"""
	if not isinstance( store_id, str ) or not store_id.strip( ):
		st.warning( 'Select or enter a vector store ID before creating a batch.' )
		return { }
	
	file_ids = parse_vector_store_file_ids( st.session_state.get( 'stores_file_ids', '' ) )
	if len( file_ids ) == 0:
		st.warning( 'Enter one or more OpenAI file IDs before creating a batch.' )
		return { }
	
	attributes = parse_vector_store_json(
		st.session_state.get( 'stores_file_attributes', '' ),
		label='Vector store file attributes' )
	
	result = vector.create_file_batch(
		store_id=store_id.strip( ),
		file_ids=file_ids,
		attributes=attributes,
		chunking_strategy=build_vector_store_chunking_strategy_from_state( vector ) )
	
	result = result if isinstance( result, dict ) else { }
	st.session_state[ 'stores_batch_result' ] = result
	
	if result.get( 'id' ):
		st.session_state[ 'stores_batch_id' ] = result.get( 'id' )
	
	return result

def run_vector_store_retrieve_batch( vector: VectorStores,
		store_id: str | None ) -> dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Retrieve vector store file batch metadata.
		
		Parameters:
		-----------
		vector: VectorStores
			VectorStores wrapper instance.
		
		store_id: str | None
			Selected vector store ID.
		
		Returns:
		--------
		dict[str, Any]
			Normalized file batch metadata.
		
	"""
	if not isinstance( store_id, str ) or not store_id.strip( ):
		st.warning( 'Select or enter a vector store ID before retrieving a batch.' )
		return { }
	
	batch_id = st.session_state.get( 'stores_batch_id', '' )
	if not isinstance( batch_id, str ) or not batch_id.strip( ):
		st.warning( 'Enter a file batch ID before retrieving batch metadata.' )
		return { }
	
	result = vector.retrieve_file_batch(
		store_id=store_id.strip( ),
		batch_id=batch_id.strip( ) )
	
	result = result if isinstance( result, dict ) else { }
	st.session_state[ 'stores_batch_result' ] = result
	return result

def run_vector_store_cancel_batch( vector: VectorStores,
		store_id: str | None ) -> dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Cancel a vector store file batch.
		
		Parameters:
		-----------
		vector: VectorStores
			VectorStores wrapper instance.
		
		store_id: str | None
			Selected vector store ID.
		
		Returns:
		--------
		dict[str, Any]
			Normalized file batch metadata.
		
	"""
	if not isinstance( store_id, str ) or not store_id.strip( ):
		st.warning( 'Select or enter a vector store ID before cancelling a batch.' )
		return { }
	
	batch_id = st.session_state.get( 'stores_batch_id', '' )
	if not isinstance( batch_id, str ) or not batch_id.strip( ):
		st.warning( 'Enter a file batch ID before cancelling a batch.' )
		return { }
	
	result = vector.cancel_file_batch(
		store_id=store_id.strip( ),
		batch_id=batch_id.strip( ) )
	
	result = result if isinstance( result, dict ) else { }
	st.session_state[ 'stores_batch_result' ] = result
	return result

def run_vector_store_search( vector: VectorStores,
		store_id: str | None ) -> list[ dict[ str, Any ] ]:
	"""
	
		Purpose:
		--------
		Run native vector store search.
		
		Parameters:
		-----------
		vector: VectorStores
			VectorStores wrapper instance.
		
		store_id: str | None
			Selected vector store ID.
		
		Returns:
		--------
		list[dict[str, Any]]
			Normalized search result rows.
		
	"""
	if not isinstance( store_id, str ) or not store_id.strip( ):
		st.warning( 'Select or enter a vector store ID before searching.' )
		return [ ]
	
	query = st.session_state.get( 'stores_search_query', '' )
	if not isinstance( query, str ) or not query.strip( ):
		st.warning( 'Enter a search query before searching a vector store.' )
		return [ ]
	
	results = vector.search_store(
		store_id=store_id.strip( ),
		query=query.strip( ),
		max_num_results=st.session_state.get( 'stores_max_results', 10 ),
		ranking_options=build_vector_store_ranking_options_from_state( ),
		rewrite_query=st.session_state.get( 'stores_rewrite_query', False ) )
	
	results = results if isinstance( results, list ) else [ ]
	st.session_state[ 'stores_search_results' ] = results
	return results

def run_vector_store_answer( vector: VectorStores, store_id: str | None,
		prompt: str | None ) -> str | None:
	"""
	
		Purpose:
		--------
		Answer a prompt using Responses API file_search over the selected vector store.
		
		Parameters:
		-----------
		vector: VectorStores
			VectorStores wrapper instance.
		
		store_id: str | None
			Selected vector store ID.
		
		prompt: str | None
			User prompt.
		
		Returns:
		--------
		str | None
			Generated answer text.
		
	"""
	if not isinstance( store_id, str ) or not store_id.strip( ):
		st.warning( 'Select or enter a vector store ID before asking a question.' )
		return None
	
	if not isinstance( prompt, str ) or not prompt.strip( ):
		st.warning( 'Enter a prompt before querying the vector store.' )
		return None
	
	model = st.session_state.get( 'stores_model' ) or 'gpt-4o-mini'
	instructions = st.session_state.get( 'stores_system_instructions', '' )
	
	answer = vector.answer_with_file_search(
		store_ids=[ store_id.strip( ) ],
		prompt=prompt.strip( ),
		model=model,
		max_num_results=st.session_state.get( 'stores_max_results', 10 ),
		instructions=instructions if isinstance( instructions, str ) and instructions.strip( )
		else None )
	
	if isinstance( answer, str ) and answer.strip( ):
		st.session_state[ 'stores_last_answer' ] = answer.strip( )
		return answer.strip( )
	
	return answer

# ----------  DATABASE UTILITIES ----------

def initialize_database( ) -> None:
	"""
		Purpose:
		--------
		Ensure required SQLite tables exist and that the Prompts table contains the
		columns required by the prompt utilities and Prompt Engineering mode.

		Parameters:
		-----------
		None

		Returns:
		--------
		None
	"""
	Path( 'stores/sqlite' ).mkdir( parents=True, exist_ok=True )
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute(
			"""
            CREATE TABLE IF NOT EXISTS chat_history
            (
                id
                INTEGER
                PRIMARY
                KEY
                AUTOINCREMENT,
                role
                TEXT,
                content
                TEXT
            )
			"""
		)
		
		conn.execute(
			"""
            CREATE TABLE IF NOT EXISTS embeddings
            (
                id
                INTEGER
                PRIMARY
                KEY
                AUTOINCREMENT,
                chunk
                TEXT,
                vector
                BLOB
            )
			"""
		)
		
		conn.execute(
			"""
            CREATE TABLE IF NOT EXISTS Prompts
            (
                PromptsId
                INTEGER
                NOT
                NULL
                PRIMARY
                KEY
                AUTOINCREMENT,
                Caption
                TEXT,
                Name
                TEXT
            (
                80
            ),
                Text TEXT,
                Version TEXT
            (
                80
            ),
                ID TEXT
            (
                80
            )
                )
			"""
		)
		
		prompt_columns = [ row[ 1 ] for row in
		                   conn.execute( 'PRAGMA table_info("Prompts");' ).fetchall( ) ]
		
		if 'Caption' not in prompt_columns:
			conn.execute( 'ALTER TABLE "Prompts" ADD COLUMN "Caption" TEXT;' )
		
		conn.commit( )

def create_connection( ) -> sqlite3.Connection:
	return sqlite3.connect( cfg.DB_PATH )

def list_tables( ) -> List[ str ]:
	with create_connection( ) as conn:
		_query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
		rows = conn.execute( _query ).fetchall( )
		return [ r[ 0 ] for r in rows ]

def create_schema( table: str ) -> List[ Tuple ]:
	with create_connection( ) as conn:
		return conn.execute( f'PRAGMA table_info("{table}");' ).fetchall( )

def read_table( table: str, limit: int = None, offset: int = 0 ) -> pd.DataFrame:
	"""
	
		Purpose:
		--------
		Read a SQLite table into a pandas DataFrame using a normalized scalar-only path.
	
		Parameters:
		-----------
		table : str
			Table name.
		limit : int = None
			Optional row limit.
		offset : int = 0
			Optional row offset.
	
		Returns:
		--------
		pd.DataFrame
			DataFrame of plain Python scalar values.
	
	"""
	if not table:
		return pd.DataFrame( )
	
	query = f'SELECT * FROM "{table}"'
	if limit:
		query += f' LIMIT {int( limit )} OFFSET {int( offset )}'
	
	with create_connection( ) as conn:
		cur = conn.cursor( )
		cur.execute( query )
		
		raw_columns = [ d[ 0 ] for d in (cur.description or [ ]) ]
		rows = cur.fetchall( )
	
	seen: Dict[ str, int ] = { }
	columns: List[ str ] = [ ]
	
	for col in raw_columns:
		name = str( col )
		if name not in seen:
			seen[ name ] = 0
			columns.append( name )
		else:
			seen[ name ] += 1
			columns.append( f'{name}_{seen[ name ]}' )
	
	def _scalarize( value: Any ) -> Any:
		if value is None or isinstance( value, (str, int, float, bool) ):
			return value
		
		if isinstance( value, bytes ):
			try:
				return value.decode( 'utf-8' )
			except Exception:
				return value.hex( )
		
		if isinstance( value, (list, tuple, set, dict) ):
			try:
				return str( normalize( value ) )
			except Exception:
				return str( value )
		
		if hasattr( value, 'model_dump' ):
			try:
				return str( value.model_dump( ) )
			except Exception:
				return str( value )
		
		return str( value )
	
	normalized_rows: List[ Dict[ str, Any ] ] = [ ]
	for row in rows:
		record: Dict[ str, Any ] = { }
		for idx, col in enumerate( columns ):
			record[ col ] = _scalarize( row[ idx ] )
		normalized_rows.append( record )
	
	return pd.DataFrame( normalized_rows, columns=columns )

def render_table( df: pd.DataFrame ) -> None:
	"""
	
		Purpose:
		--------
		Render a DataFrame safely in Streamlit. Use the normal interactive dataframe
		first, and fall back to HTML rendering if Streamlit/PyArrow serialization fails.
	
		Parameters:
		-----------
		df : pd.DataFrame
			The DataFrame to render.
	
		Returns:
		--------
		None
	
	"""
	if df is None:
		st.info( 'No data available.' )
		return
	
	try:
		st.data_editor( df, use_container_width=True )
		return
	except Exception:
		pass
	
	fallback_df = df.copy( )
	fallback_df = fallback_df.where( pd.notnull( fallback_df ), '' )
	
	for col in fallback_df.columns:
		fallback_df[ col ] = fallback_df[ col ].map(
			lambda x: x if isinstance( x, (str, int, float, bool) ) or x == '' else str( x ) )
	
	st.markdown( fallback_df.to_html( index=False, escape=True ), unsafe_allow_html=True )

def make_display_safe( df: pd.DataFrame ) -> pd.DataFrame:
	display_df = df.copy( )
	
	for col in display_df.columns:
		display_df[ col ] = display_df[ col ].map(
			lambda x: '' if x is None else str( x )
		)
	
	return display_df

def drop_table( table: str ) -> None:
	"""
		Purpose:
		--------
		Safely drop a table if it exists.
	
		Parameters:
		-----------
		table : str
			Table name.
	"""
	if not table:
		return
	
	with create_connection( ) as conn:
		conn.execute( f'DROP TABLE IF EXISTS "{table}";' )
		conn.commit( )

def create_index( table: str, column: str ) -> None:
	"""
		Purpose:
		--------
		Create a safe SQLite index on a specified table column.
	
		Handles:
			- Spaces in column names
			- Special characters
			- Reserved words
			- Duplicate index names
			- Validation against actual table schema
	
		Parameters:
		-----------
		table : str
			Table name.
		column : str
			Column name to index.
	"""
	if not table or not column:
		return
	
	# ------------------------------------------------------------------
	# Validate table exists
	# ------------------------------------------------------------------
	tables = list_tables( )
	if table not in tables:
		raise ValueError( 'Invalid table name.' )
	
	# ------------------------------------------------------------------
	# Validate column exists
	# ------------------------------------------------------------------
	schema = create_schema( table )
	valid_columns = [ col[ 1 ] for col in schema ]
	
	if column not in valid_columns:
		raise ValueError( 'Invalid column name.' )
	
	# ------------------------------------------------------------------
	# Sanitize index name (identifier only)
	# ------------------------------------------------------------------
	safe_index_name = re.sub( r"[^0-9a-zA-Z_]+", "_", f"idx_{table}_{column}" )
	
	# ------------------------------------------------------------------
	# Create index safely (quote identifiers)
	# ------------------------------------------------------------------
	sql = f'CREATE INDEX IF NOT EXISTS "{safe_index_name}" ON "{table}"("{column}");'
	
	with create_connection( ) as conn:
		conn.execute( sql )
		conn.commit( )

def apply_filters( df: pd.DataFrame ) -> pd.DataFrame:
	st.subheader( 'Advanced Filters' )
	conditions = [ ]
	col1, col2, col3 = st.columns( 3 )
	column = col1.selectbox( 'Column', df.columns )
	operator = col2.selectbox( 'Operator', [ '=', '!=', '>', '<', '>=', '<=', 'contains' ] )
	value = col3.text_input( 'Value' )
	if value:
		if operator == '=':
			df = df[ df[ column ] == value ]
		elif operator == '!=':
			df = df[ df[ column ] != value ]
		elif operator == '>':
			df = df[ df[ column ].astype( float ) > float( value ) ]
		elif operator == '<':
			df = df[ df[ column ].astype( float ) < float( value ) ]
		elif operator == '>=':
			df = df[ df[ column ].astype( float ) >= float( value ) ]
		elif operator == '<=':
			df = df[ df[ column ].astype( float ) <= float( value ) ]
		elif operator == 'contains':
			df = df[ df[ column ].astype( str ).str.contains( value ) ]
	
	return df

def create_aggregation( df: pd.DataFrame ):
	st.subheader( 'Aggregation Engine' )
	
	numeric_cols = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
	
	if not numeric_cols:
		st.info( 'No numeric columns available.' )
		return
	
	col = st.selectbox( 'Column', numeric_cols )
	agg = st.selectbox( 'Aggregation', [ 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'MEDIAN' ] )
	
	if agg == 'COUNT':
		result = df[ col ].count( )
	elif agg == 'SUM':
		result = df[ col ].sum( )
	elif agg == 'AVG':
		result = df[ col ].mean( )
	elif agg == 'MIN':
		result = df[ col ].min( )
	elif agg == 'MAX':
		result = df[ col ].max( )
	elif agg == 'MEDIAN':
		result = df[ col ].median( )
	
	st.metric( 'Result', result )

def create_visualization( df: pd.DataFrame ) -> None:
	"""
	
		Purpose:
		--------
		Render data visualizations without passing pandas objects directly into
		Plotly/Narwhals.
		
		Parameters:
		-----------
		df : pd.DataFrame
			The input DataFrame.
		
		Returns:
		--------
		None
		
	"""
	st.subheader( 'Visualization Engine' )
	
	if df is None or df.empty:
		st.info( 'No data available.' )
		return
	
	df_plot = df.copy( )
	
	for col in df_plot.columns:
		if df_plot[ col ].dtype == object:
			df_plot[ col ] = df_plot[ col ].map(
				lambda x: '' if x is None else str( x )
			)
	
	numeric_cols: List[ str ] = [ ]
	for col in df_plot.columns:
		series_num = pd.to_numeric( df_plot[ col ], errors='coerce' )
		if series_num.notna( ).any( ):
			numeric_cols.append( col )
	
	categorical_cols: List[ str ] = [ col for col in df_plot.columns if col not in numeric_cols ]
	
	chart = st.selectbox(
		'Chart Type',
		[ 'Histogram', 'Bar', 'Line', 'Scatter', 'Box', 'Pie', 'Correlation' ] )
	
	if chart == 'Histogram':
		if not numeric_cols:
			st.info( 'No numeric columns available.' )
			return
		
		col = st.selectbox( 'Column', numeric_cols )
		values = pd.to_numeric( df_plot[ col ], errors='coerce' ).dropna( ).tolist( )
		
		fig = go.Figure( data=[ go.Histogram( x=values ) ] )
		fig.update_layout( xaxis_title=col, yaxis_title='Count' )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Bar':
		if not numeric_cols:
			st.info( 'No numeric columns available.' )
			return
		
		x = st.selectbox( 'X', df_plot.columns )
		y = st.selectbox( 'Y', numeric_cols )
		
		x_values = df_plot[ x ].astype( str ).tolist( )
		y_values = pd.to_numeric( df_plot[ y ], errors='coerce' ).fillna( 0 ).tolist( )
		
		fig = go.Figure( data=[ go.Bar( x=x_values, y=y_values ) ] )
		fig.update_layout( xaxis_title=x, yaxis_title=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Line':
		if not numeric_cols:
			st.info( 'No numeric columns available.' )
			return
		
		x = st.selectbox( 'X', df_plot.columns )
		y = st.selectbox( 'Y', numeric_cols )
		
		x_values = df_plot[ x ].astype( str ).tolist( )
		y_values = pd.to_numeric( df_plot[ y ], errors='coerce' ).fillna( 0 ).tolist( )
		
		fig = go.Figure( data=[ go.Scatter( x=x_values, y=y_values, mode='lines' ) ] )
		fig.update_layout( xaxis_title=x, yaxis_title=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Scatter':
		if len( numeric_cols ) < 2:
			st.info( 'At least two numeric columns are required.' )
			return
		
		x = st.selectbox( 'X', numeric_cols, key='viz_scatter_x' )
		y = st.selectbox( 'Y', numeric_cols, key='viz_scatter_y' )
		
		x_series = pd.to_numeric( df_plot[ x ], errors='coerce' )
		y_series = pd.to_numeric( df_plot[ y ], errors='coerce' )
		mask = x_series.notna( ) & y_series.notna( )
		
		x_values = x_series[ mask ].tolist( )
		y_values = y_series[ mask ].tolist( )
		
		fig = go.Figure( data=[ go.Scatter( x=x_values, y=y_values, mode='markers' ) ] )
		fig.update_layout( xaxis_title=x, yaxis_title=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Box':
		if not numeric_cols:
			st.info( 'No numeric columns available.' )
			return
		
		col = st.selectbox( 'Column', numeric_cols, key='viz_box_col' )
		values = pd.to_numeric( df_plot[ col ], errors='coerce' ).dropna( ).tolist( )
		
		fig = go.Figure( data=[ go.Box( y=values, name=col ) ] )
		fig.update_layout( yaxis_title=col )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Pie':
		if not categorical_cols:
			st.info( 'No categorical columns available.' )
			return
		
		col = st.selectbox( 'Category Column', categorical_cols )
		counts = df_plot[ col ].astype( str ).value_counts( )
		
		fig = go.Figure(
			data=[ go.Pie( labels=counts.index.tolist( ), values=counts.values.tolist( ) ) ] )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Correlation':
		if len( numeric_cols ) < 2:
			st.info( 'At least two numeric columns are required.' )
			return
		
		corr_df = pd.DataFrame( )
		for col in numeric_cols:
			corr_df[ col ] = pd.to_numeric( df_plot[ col ], errors='coerce' )
		
		corr = corr_df.corr( )
		
		fig = go.Figure(
			data=[ go.Heatmap(
				z=corr.values.tolist( ),
				x=corr.columns.tolist( ),
				y=corr.index.tolist( ) ) ] )
		st.plotly_chart( fig, use_container_width=True )

def convert_dataframe( table_name: str, df: pd.DataFrame ):
	columns = [ ]
	for col in df.columns:
		sql_type = get_sqlite_type( df[ col ].dtype )
		safe_col = col.replace( ' ', '_' )
		columns.append( f'{safe_col} {sql_type}' )
	
	create_stmt = f'CREATE TABLE IF NOT EXISTS {table_name} ({", ".join( columns )});'
	
	with create_connection( ) as conn:
		conn.execute( create_stmt )
		conn.commit( )

def insert_data( table_name: str, df: pd.DataFrame ):
	df = df.copy( )
	df.columns = [ c.replace( ' ', '_' ) for c in df.columns ]
	
	placeholders = ', '.join( [ '?' ] * len( df.columns ) )
	stmt = f'INSERT INTO {table_name} VALUES ({placeholders});'
	
	with create_connection( ) as conn:
		conn.executemany( stmt, df.values.tolist( ) )
		conn.commit( )

def get_sqlite_type( dtype ) -> str:
	"""
		Purpose:
		--------
		Map a pandas dtype to an appropriate SQLite column type.
	
		Parameters:
		-----------
		dtype : pandas dtype
			The dtype of a pandas Series.
	
		Returns:
		--------
		str
			SQLite column type.
	"""
	dtype_str = str( dtype ).lower( )
	
	# ------------------------------------------------------------------
	# Integer Types (including nullable Int64)
	# ------------------------------------------------------------------
	if 'int' in dtype_str:
		return 'INTEGER'
	
	# ------------------------------------------------------------------
	# Float Types
	# ------------------------------------------------------------------
	if 'float' in dtype_str:
		return 'REAL'
	
	# ------------------------------------------------------------------
	# Boolean
	# ------------------------------------------------------------------
	if 'bool' in dtype_str:
		return 'INTEGER'
	
	# ------------------------------------------------------------------
	# Datetime
	# ------------------------------------------------------------------
	if 'datetime' in dtype_str:
		return 'TEXT'
	
	# ------------------------------------------------------------------
	# Categorical
	# ------------------------------------------------------------------
	if 'category' in dtype_str:
		return 'TEXT'
	
	# ------------------------------------------------------------------
	# Default fallback
	# ------------------------------------------------------------------
	return 'TEXT'

def create_custom_table( table_name: str, columns: list ) -> None:
	"""
		Purpose:
		--------
		Create a custom SQLite table from column definitions.
	
		Parameters:
		-----------
		table_name : str
			Name of table.
	
		columns : list of dict
			[
				{
					"name": str,
					"type": str,
					"not_null": bool,
					"primary_key": bool,
					"auto_increment": bool
				}
			]
	"""
	if not table_name:
		raise ValueError( 'Table name required.' )
	
	# Validate identifier
	if not re.match( r"^[A-Za-z_][A-Za-z0-9_]*$", table_name ):
		raise ValueError( 'Invalid table name.' )
	
	col_defs = [ ]
	
	for col in columns:
		col_name = col[ 'name' ]
		col_type = col[ 'type' ].upper( )
		
		if not re.match( r"^[A-Za-z_][A-Za-z0-9_]*$", col_name ):
			raise ValueError( f"Invalid column name: {col_name}" )
		
		definition = f'"{col_name}" {col_type}'
		
		if col[ 'primary_key' ]:
			definition += ' PRIMARY KEY'
			if col[ 'auto_increment' ] and col_type == 'INTEGER':
				definition += ' AUTOINCREMENT'
		
		if col[ "not_null" ]:
			definition += " NOT NULL"
		
		col_defs.append( definition )
	
	sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join( col_defs )});'
	
	with create_connection( ) as conn:
		conn.execute( sql )
		conn.commit( )

def is_safe_query( query: str ) -> bool:
	"""
	
		Purpose:
		--------
		Determine whether a SQL query is read-only and safe to execute.
	
		Allows:
			SELECT
			WITH (CTE returning SELECT)
			EXPLAIN SELECT
			PRAGMA (read-only)
	
		Blocks:
			INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH,
			DETACH, VACUUM, REPLACE, TRIGGER, and multiple statements.
			
	"""
	if not query or not isinstance( query, str ):
		return False
	
	q = query.strip( ).lower( )
	
	# ------------------------------------------------------------------
	# Block multiple statements
	# ------------------------------------------------------------------
	if ';' in q[ :-1 ]:
		return False
	
	# ------------------------------------------------------------------
	# Remove SQL comments
	# ------------------------------------------------------------------
	q = re.sub( r"--.*?$", "", q, flags=re.MULTILINE )
	q = re.sub( r"/\*.*?\*/", "", q, flags=re.DOTALL )
	q = q.strip( )
	
	# ------------------------------------------------------------------
	# Allowed starting keywords
	# ------------------------------------------------------------------
	allowed_starts = ('select', 'with', 'explain', 'pragma')
	if not q.startswith( allowed_starts ):
		return False
	
	# ------------------------------------------------------------------
	# Block dangerous keywords anywhere
	# ------------------------------------------------------------------
	blocked_keywords = ('insert ', 'update ', 'delete ', 'drop ', 'alter ',
	                    'create ', 'attach ', 'detach ', 'vacuum ', 'replace ', 'trigger ')
	
	for keyword in blocked_keywords:
		if keyword in q:
			return False
	
	return True

def create_identifier( name: str ) -> str:
	"""
	
		Purpose:
		--------
		Sanitize a string into a safe SQLite identifier.
	
		- Replaces invalid characters with underscores
		- Ensures it starts with a letter or underscore
		- Prevents empty names
		
	"""
	if not name or not isinstance( name, str ):
		raise ValueError( 'Invalid Identifier.' )
	
	safe = re.sub( r'[^0-9a-zA-Z_]', '_', name.strip( ) )
	if not re.match( r'^[A-Za-z_]', safe ):
		safe = f'_{safe}'
	
	if not safe:
		raise ValueError( 'Invalid identifier after sanitization.' )
	
	return safe

def get_indexes( table: str ):
	with create_connection( ) as conn:
		rows = conn.execute( f'PRAGMA index_list("{table}");' ).fetchall( )
		return rows

def add_column( table: str, column: str, col_type: str ):
	column = create_identifier( column )
	col_type = col_type.upper( )
	
	with create_connection( ) as conn:
		conn.execute(
			f'ALTER TABLE "{table}" ADD COLUMN "{column}" {col_type};' )
		conn.commit( )

def rename_column( table_name: str, old_name: str, new_name: str ) -> None:
	"""
	
		Purpose:
		--------
		Rename a column within an existing SQLite table. Attempts native ALTER TABLE rename
		first; if it fails, falls back to a schema-safe rebuild preserving column order, data,
		and indexes.

		Parameters:
		-----------
		table_name : str
			Table containing the column.

		old_name : str
			Existing column name.

		new_name : str
			New column name.

		Returns:
		--------
		None
		
	"""
	if not table_name or not old_name or not new_name:
		return
	
	with create_connection( ) as conn:
		try:
			conn.execute(
				f'ALTER TABLE "{table_name}" RENAME COLUMN "{old_name}" TO "{new_name}";'
			)
			conn.commit( )
			return
		except Exception:
			pass
		
		row = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='table' AND name =?
			""",
			(table_name,)
		).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( "Table definition not found." )
		
		create_sql = row[ 0 ]
		
		indexes = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
			""",
			(table_name,)
		).fetchall( )
		
		schema = conn.execute( f'PRAGMA table_info("{table_name}");' ).fetchall( )
		cols = [ r[ 1 ] for r in schema ]
		if old_name not in cols:
			raise ValueError( "Column not found." )
		
		mapped_cols = [ (new_name if c == old_name else c) for c in cols ]
		
		temp_table = f"{table_name}__rebuild_temp"
		
		col_defs: List[ str ] = [ ]
		pk_cols = [ r for r in schema if int( r[ 5 ] or 0 ) > 0 ]
		single_pk = len( pk_cols ) == 1
		
		for row in schema:
			col_name = row[ 1 ]
			col_type = row[ 2 ] or ''
			not_null = int( row[ 3 ] or 0 )
			default_value = row[ 4 ]
			pk = int( row[ 5 ] or 0 )
			
			out_name = new_name if col_name == old_name else col_name
			col_def = f'"{out_name}" {col_type}'.strip( )
			
			if not_null:
				col_def += ' NOT NULL'
			
			if default_value is not None:
				col_def += f' DEFAULT {default_value}'
			
			if single_pk and pk == 1:
				col_def += ' PRIMARY KEY'
			
			col_defs.append( col_def )
		
		new_create_sql = f'CREATE TABLE "{temp_table}" ({", ".join( col_defs )});'
		
		old_select = ", ".join( [ f'"{c}"' for c in cols ] )
		new_insert = ", ".join( [ f'"{c}"' for c in mapped_cols ] )
		
		conn.execute( "BEGIN" )
		conn.execute( new_create_sql )
		conn.execute(
			f'INSERT INTO "{temp_table}" ({new_insert}) SELECT {old_select} FROM "{table_name}";'
		)
		
		conn.execute( f'DROP TABLE "{table_name}";' )
		conn.execute( f'ALTER TABLE "{temp_table}" RENAME TO "{table_name}";' )
		
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if idx_sql:
				idx_sql = idx_sql.replace( f'"{old_name}"', f'"{new_name}"' )
				conn.execute( idx_sql )
		
		conn.commit( )

def create_profile_table( table: str ):
	df = read_table( table )
	profile_rows = [ ]
	total_rows = len( df )
	for col in df.columns:
		series = df[ col ]
		null_count = series.isna( ).sum( )
		distinct_count = series.nunique( dropna=True )
		row = \
			{
					'column': col, 'dtype': str( series.dtype ),
					'null_%': round( (null_count / total_rows) * 100, 2 ) if total_rows else 0,
					'distinct_%': round( (
							                     distinct_count / total_rows) * 100, 2 ) if total_rows else 0,
			}
		
		if pd.api.types.is_numeric_dtype( series ):
			row[ 'min' ] = series.min( )
			row[ 'max' ] = series.max( )
			row[ 'mean' ] = series.mean( )
		else:
			row[ 'min' ] = None
			row[ 'max' ] = None
			row[ 'mean' ] = None
		
		profile_rows.append( row )
	
	return pd.DataFrame( profile_rows )

def drop_column( table: str, column: str ):
	if not table or not column:
		raise ValueError( 'Table and column required.' )
	
	with create_connection( ) as conn:
		# ------------------------------------------------------------
		# Fetch original CREATE TABLE statement
		# ------------------------------------------------------------
		row = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='table' AND name =?
			""",
			(table,)
		).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( 'Table definition not found.' )
		
		create_sql = row[ 0 ]
		
		# ------------------------------------------------------------
		# Extract column definitions
		# ------------------------------------------------------------
		open_paren = create_sql.find( "(" )
		close_paren = create_sql.rfind( ")" )
		
		if open_paren == -1 or close_paren == -1:
			raise ValueError( "Malformed CREATE TABLE statement." )
		
		inner = create_sql[ open_paren + 1: close_paren ]
		
		column_defs = [ c.strip( ) for c in inner.split( "," ) ]
		
		# Remove target column
		new_defs = [ ]
		for col_def in column_defs:
			col_name = col_def.split( )[ 0 ].strip( '"' )
			if col_name != column:
				new_defs.append( col_def )
		
		if len( new_defs ) == len( column_defs ):
			raise ValueError( "Column not found." )
		
		# ------------------------------------------------------------
		# Build new CREATE TABLE statement
		# ------------------------------------------------------------
		temp_table = f"{table}_rebuild_temp"
		
		new_create_sql = (
				f'CREATE TABLE "{temp_table}" ('
				+ ", ".join( new_defs )
				+ ");"
		)
		
		# ------------------------------------------------------------
		# Begin transaction
		# ------------------------------------------------------------
		conn.execute( "BEGIN" )
		
		conn.execute( new_create_sql )
		
		remaining_cols = [
				c.split( )[ 0 ].strip( '"' )
				for c in new_defs
		]
		
		col_list = ", ".join( [ f'"{c}"' for c in remaining_cols ] )
		
		conn.execute(
			f'INSERT INTO "{temp_table}" ({col_list}) '
			f'SELECT {col_list} FROM "{table}";'
		)
		
		# Preserve indexes
		indexes = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
			""",
			(table,)
		).fetchall( )
		
		conn.execute( f'DROP TABLE "{table}";' )
		conn.execute(
			f'ALTER TABLE "{temp_table}" RENAME TO "{table}";'
		)
		
		# Recreate indexes
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if column not in idx_sql:
				conn.execute( idx_sql )
		
		conn.commit( )

def rename_table( old_name: str, new_name: str ) -> None:
	"""
	
		Purpose:
		--------
		Rename an existing SQLite table. Attempts native ALTER TABLE rename first; if it fails,
		falls back to a schema-safe rebuild using the original CREATE TABLE statement and
		preserves indexes.

		Parameters:
		-----------
		old_name : str
			Existing table name.

		new_name : str
			New table name.

		Returns:
		--------
		None
		
	"""
	if not old_name or not new_name:
		return
	
	with create_connection( ) as conn:
		try:
			conn.execute( f'ALTER TABLE "{old_name}" RENAME TO "{new_name}";' )
			conn.commit( )
			return
		except Exception:
			pass
		
		row = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='table' AND name =?
			""",
			(old_name,)
		).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( "Table definition not found." )
		
		create_sql = row[ 0 ]
		
		indexes = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
			""",
			(old_name,)
		).fetchall( )
		
		open_paren = create_sql.find( "(" )
		if open_paren == -1:
			raise ValueError( "Malformed CREATE TABLE statement." )
		
		temp_name = f"{new_name}__rebuild_temp"
		
		conn.execute( "BEGIN" )
		conn.execute( f'CREATE TABLE "{temp_name}" {create_sql[ open_paren: ]}' )
		
		cols = [ r[ 1 ] for r in conn.execute( f'PRAGMA table_info("{old_name}");' ).fetchall( ) ]
		col_list = ", ".join( [ f'"{c}"' for c in cols ] )
		
		conn.execute(
			f'INSERT INTO "{temp_name}" ({col_list}) SELECT {col_list} FROM "{old_name}";'
		)
		
		conn.execute( f'DROP TABLE "{old_name}";' )
		conn.execute( f'ALTER TABLE "{temp_name}" RENAME TO "{new_name}";' )
		
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if idx_sql:
				idx_sql = idx_sql.replace( f'ON "{old_name}"', f'ON "{new_name}"' )
				conn.execute( idx_sql )
		
		conn.commit( )

# ---------- PROMPT ENGINEERING UTILITIES ---------------

def fetch_prompt_names( db_path: str ) -> list[ str ]:
	"""
		Purpose:
		--------
		Retrieve template names from Prompts table.
	
		Parameters:
		-----------
		db_path : str
			SQLite database path.
	
		Returns:
		--------
		list[str]
			Sorted prompt names.
	"""
	try:
		conn = sqlite3.connect( db_path )
		cur = conn.cursor( )
		cur.execute( "SELECT Caption FROM Prompts ORDER BY PromptsId;" )
		rows = cur.fetchall( )
		conn.close( )
		return [ r[ 0 ] for r in rows if r and r[ 0 ] is not None ]
	except Exception:
		return [ ]

def fetch_prompt_text( db_path: str, name: str ) -> str | None:
	"""
		Purpose:
		--------
		Retrieve template text by name.
	
		Parameters:
		-----------
		db_path : str
			SQLite database path.
		name : str
			Template name.
	
		Returns:
		--------
		str | None
			Prompt text if found.
	"""
	try:
		conn = sqlite3.connect( db_path )
		cur = conn.cursor( )
		cur.execute( "SELECT Text FROM Prompts WHERE Caption = ?;", (name,) )
		row = cur.fetchone( )
		conn.close( )
		return str( row[ 0 ] ) if row and row[ 0 ] is not None else None
	except Exception:
		return None

def fetch_prompts_df( ) -> pd.DataFrame:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		df = pd.read_sql_query(
			"SELECT PromptsId, Caption,  Name, Version, ID FROM Prompts ORDER BY PromptsId DESC",
			conn )
	df.insert( 0, "Selected", False )
	return df

def fetch_prompt_by_id( pid: int ) -> Dict[ str, Any ] | None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		cur = conn.execute(
			"SELECT PromptsId, Caption, Name, Text, Version, ID FROM Prompts WHERE PromptsId=?",
			(pid,)
		)
		row = cur.fetchone( )
		return dict( zip( [ c[ 0 ] for c in cur.description ], row ) ) if row else None

def fetch_prompt_by_name( name: str ) -> Dict[ str, Any ] | None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		cur = conn.execute(
			"SELECT PromptsId, Caption, Name, Text, Version, ID FROM Prompts WHERE Caption=?",
			(name,)
		)
		row = cur.fetchone( )
		return dict( zip( [ c[ 0 ] for c in cur.description ], row ) ) if row else None

def insert_prompt( data: Dict[ str, Any ] ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( 'INSERT INTO Prompts (Caption, Name, Text, Version, ID) VALUES (?, ?, ?, ?, ?)',
			(data[ 'Caption' ], data[ 'Name' ], data[ 'Text' ], data[ 'Version' ], data[ 'ID' ]) )

def update_prompt( pid: int, data: Dict[ str, Any ] ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute(
			"UPDATE Prompts SET Caption=?, Name=?, Text=?, Version=?, ID=? WHERE PromptsId=?",
			(data[ 'Caption' ], data[ 'Name' ], data[ 'Text' ], data[ 'Version' ], data[ 'ID' ],
			 pid)
		)

def delete_prompt( pid: int ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( "DELETE FROM Prompts WHERE PromptsId=?", (pid,) )

def build_prompt( user_input: str ) -> str:
	"""
		Purpose:
		--------
		Build a prompt using the application's system instructions, optional
		retrieval context (semantic + basic RAG), and the current in-memory chat history.

		Parameters:
		-----------
		user_input : str
			The current user turn to append to the prompt.

		Returns:
		--------
		str
			A fully constructed prompt in chat template format.
	"""
	system_instructions = st.session_state.get( 'docqna_system_instructions', '' )
	use_semantic = bool( st.session_state.get( 'use_semantic', False ) )
	basic_docs = st.session_state.get( 'basic_docs', [ ] )
	messages = st.session_state.get( 'messages', [ ] )
	
	top_k_value = int( st.session_state.get( 'top_k', 0 ) )
	if top_k_value <= 0:
		top_k_value = 4
	
	prompt = f"<|system|>\n{system_instructions}\n</s>\n"
	
	if use_semantic:
		with sqlite3.connect( cfg.DB_PATH ) as conn:
			rows = conn.execute( "SELECT chunk, vector FROM embeddings" ).fetchall( )
		
		if rows:
			q = embedder.encode( [ user_input ] )[ 0 ]
			scored = [ (c, cosine_sim( q, np.frombuffer( v ) )) for c, v in rows ]
			for c, _ in sorted( scored, key=lambda x: x[ 1 ], reverse=True )[ :top_k_value ]:
				prompt += f"<|system|>\n{c}\n</s>\n"
	
	for d in basic_docs[ :6 ]:
		prompt += f"<|system|>\n{d}\n</s>\n"
	
	if isinstance( messages, list ):
		for msg in messages:
			role = ''
			content = ''
			
			if isinstance( msg, tuple ) or isinstance( msg, list ):
				if len( msg ) == 2:
					role = str( msg[ 0 ] or '' ).strip( )
					content = str( msg[ 1 ] or '' )
			elif isinstance( msg, dict ):
				role = str( msg.get( 'role', '' ) or '' ).strip( )
				content = str( msg.get( 'content', '' ) or '' )
			
			if role:
				prompt += f"<|{role}|>\n{content}\n</s>\n"
	
	prompt += f"<|user|>\n{user_input}\n</s>\n<|assistant|>\n"
	return prompt

def reset_audio_model_controls( ) -> None:
	'''
	
		Purpose:
		--------
		Resets widget-backed Audio model controls using a callback-safe path.
	
	
		Parameters:
		-----------
		None
	
	
		Returns:
		--------
		None

	'''
	st.session_state[ 'audio_task' ] = None
	st.session_state[ 'audio_model' ] = None
	st.session_state[ 'audio_language' ] = None
	st.session_state[ 'audio_background' ] = False
	st.session_state[ 'audio_reasoning' ] = None
	st.session_state[ 'audio_rate' ] = int( cfg.SAMPLE_RATES[ 0 ] ) if cfg.SAMPLE_RATES else 44100
	st.session_state[ 'audio_mime_type' ] = None

def reset_audio_inference_controls( ) -> None:
	'''
	
		Purpose:
		--------
		Resets widget-backed Audio inference controls using a callback-safe path.
	
	
		Parameters:
		-----------
		None
	
	
		Returns:
		--------
		None

	'''
	st.session_state[ 'audio_top_percent' ] = 0.0
	st.session_state[ 'audio_temperature' ] = 0.0
	st.session_state[ 'audio_presence_penalty' ] = 0.0
	st.session_state[ 'audio_frequency_penalty' ] = 0.0
	st.session_state[ 'audio_modalities' ] = [ ]
	st.session_state[ 'audio_response_format' ] = None

def reset_audio_sound_controls( ) -> None:
	'''
	
		Purpose:
		--------
		Resets widget-backed Audio playback controls using a callback-safe path.
	
	
		Parameters:
		-----------
		None
	
	
		Returns:
		--------
		None

	'''
	st.session_state[ 'audio_language' ] = None
	st.session_state[ 'audio_voice' ] = None
	st.session_state[ 'audio_loop' ] = False
	st.session_state[ 'audio_autoplay' ] = False
	st.session_state[ 'audio_start_time' ] = 0.0
	st.session_state[ 'audio_end_time' ] = 0.0

# ------------ AUDIO MODE UTILITIES -----------------

def ensure_audio_mode_state( ) -> None:
	"""
	
		Purpose:
		--------
		Ensure Audio mode session-state keys used by API request helpers exist before
		widget instantiation.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	if 'audio_task' not in st.session_state:
		st.session_state[ 'audio_task' ] = ''
	
	if 'audio_model' not in st.session_state:
		st.session_state[ 'audio_model' ] = ''
	
	if 'audio_language' not in st.session_state:
		st.session_state[ 'audio_language' ] = ''
	
	if 'audio_response_format' not in st.session_state:
		st.session_state[ 'audio_response_format' ] = ''
	
	if 'audio_mime_type' not in st.session_state:
		st.session_state[ 'audio_mime_type' ] = ''
	
	if 'audio_include' not in st.session_state:
		st.session_state[ 'audio_include' ] = [ ]
	
	if 'audio_speed' not in st.session_state:
		st.session_state[ 'audio_speed' ] = 1.0
	
	if 'audio_voice' not in st.session_state:
		st.session_state[ 'audio_voice' ] = ''
	
	if 'audio_temperature' not in st.session_state:
		st.session_state[ 'audio_temperature' ] = 0.0
	
	if 'audio_system_instructions' not in st.session_state:
		st.session_state[ 'audio_system_instructions' ] = ''
	
	if 'audio_output' not in st.session_state:
		st.session_state[ 'audio_output' ] = ''
	
	if 'audio_output_bytes' not in st.session_state:
		st.session_state[ 'audio_output_bytes' ] = None
	
	if 'audio_last_result' not in st.session_state:
		st.session_state[ 'audio_last_result' ] = { }
	
	if 'audio_last_usage' not in st.session_state:
		st.session_state[ 'audio_last_usage' ] = { }
	
	if 'audio_messages' not in st.session_state:
		st.session_state.audio_messages = [ ]

def get_audio_task_options( ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Return Audio mode task options.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		list[str]
			Audio task names.
		
	"""
	return [
			'',
			'Transcribe',
			'Translate',
			'Text-to-Speech',
	]

def get_audio_model_options( task: str | None, transcriber: Transcription,
		translator: Translation, tts: TTS ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Return task-aware Audio API model options.
		
		Parameters:
		-----------
		task: str | None
			Selected Audio mode task.
		
		transcriber: Transcription
			Transcription wrapper instance.
		
		translator: Translation
			Translation wrapper instance.
		
		tts: TTS
			Text-to-speech wrapper instance.
		
		Returns:
		--------
		list[str]
			Model option names.
		
	"""
	if task == 'Transcribe':
		options = getattr( transcriber, 'model_options', [ ] )
	elif task == 'Translate':
		options = getattr( translator, 'model_options', [ ] )
	elif task == 'Text-to-Speech':
		options = getattr( tts, 'model_options', [ ] )
	else:
		options = [ ]
	
	if isinstance( options, list ) and len( options ) > 0:
		return [ '' ] + options
	
	return [ '' ]

def get_audio_language_options( transcriber: Transcription ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Return language options for transcription source-language hints.
		
		Parameters:
		-----------
		transcriber: Transcription
			Transcription wrapper instance.
		
		Returns:
		--------
		list[str]
			Language code options.
		
	"""
	options = getattr( transcriber, 'language_options', [ ] )
	if isinstance( options, list ) and len( options ) > 0:
		return [ '' ] + options
	
	return [ '' ]

def get_audio_voice_options( tts: TTS ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Return text-to-speech voice options.
		
		Parameters:
		-----------
		tts: TTS
			Text-to-speech wrapper instance.
		
		Returns:
		--------
		list[str]
			Voice option names.
		
	"""
	options = getattr( tts, 'voice_options', [ ] )
	if isinstance( options, list ) and len( options ) > 0:
		return [ '' ] + options
	
	return [ '' ]

def get_audio_speed_options( tts: TTS ) -> list[ float ]:
	"""
	
		Purpose:
		--------
		Return text-to-speech speed options.
		
		Parameters:
		-----------
		tts: TTS
			Text-to-speech wrapper instance.
		
		Returns:
		--------
		list[float]
			Speech speed options.
		
	"""
	options = getattr( tts, 'speed_options', [ ] )
	if isinstance( options, list ) and len( options ) > 0:
		return options
	
	return [
			0.25,
			0.50,
			0.75,
			1.0,
			1.25,
			1.50,
			2.0,
			3.0,
			4.0,
	]

def get_audio_response_format_options( task: str | None, model: str | None,
		transcriber: Transcription, translator: Translation, tts: TTS ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Return task-aware response/output format options for Audio mode.
		
		Parameters:
		-----------
		task: str | None
			Selected Audio mode task.
		
		model: str | None
			Selected Audio API model.
		
		transcriber: Transcription
			Transcription wrapper instance.
		
		translator: Translation
			Translation wrapper instance.
		
		tts: TTS
			Text-to-speech wrapper instance.
		
		Returns:
		--------
		list[str]
			Response/output format options.
		
	"""
	if task == 'Transcribe':
		format_map = getattr( transcriber, 'response_format_options', { } )
		if isinstance( format_map, dict ):
			options = format_map.get( model, [ 'json' ] )
			return [ '' ] + options
		
		return [ '', 'json' ]
	
	if task == 'Translate':
		options = getattr( translator, 'response_format_options', [ ] )
		if isinstance( options, list ) and len( options ) > 0:
			return [ '' ] + options
		
		return [ '', 'json', 'text', 'srt', 'verbose_json', 'vtt' ]
	
	if task == 'Text-to-Speech':
		options = getattr( tts, 'mime_options', [ ] )
		if isinstance( options, list ) and len( options ) > 0:
			return [ '' ] + options
		
		return [ '', 'mp3', 'opus', 'aac', 'flac', 'wav', 'pcm' ]
	
	return [ '' ]

def get_audio_include_options( task: str | None, model: str | None,
		transcriber: Transcription ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Return task-aware Audio API include options.
		
		Parameters:
		-----------
		task: str | None
			Selected Audio mode task.
		
		model: str | None
			Selected Audio API model.
		
		transcriber: Transcription
			Transcription wrapper instance.
		
		Returns:
		--------
		list[str]
			Include option names.
		
	"""
	if task != 'Transcribe':
		return [ ]
	
	if model not in [
			'gpt-4o-transcribe',
			'gpt-4o-mini-transcribe',
			'gpt-4o-mini-transcribe-2025-12-15',
	]:
		return [ ]
	
	options = getattr( transcriber, 'include_options', [ ] )
	if isinstance( options, list ) and len( options ) > 0:
		return options
	
	return [ 'logprobs' ]

def get_audio_response_format_value( task: str | None, selected_format: str | None,
		selected_mime_type: str | None = None ) -> str | None:
	"""
	
		Purpose:
		--------
		Return the correct API format value for the selected Audio mode task.
		
		Parameters:
		-----------
		task: str | None
			Selected Audio mode task.
		
		selected_format: str | None
			Selected response-format value.
		
		selected_mime_type: str | None
			Legacy MIME/output format value.
		
		Returns:
		--------
		str | None
			Format value for the wrapper call.
		
	"""
	if task == 'Text-to-Speech':
		if isinstance( selected_format, str ) and selected_format.strip( ):
			return selected_format.strip( )
		
		if isinstance( selected_mime_type, str ) and selected_mime_type.strip( ):
			return selected_mime_type.strip( )
		
		return 'mp3'
	
	if isinstance( selected_format, str ) and selected_format.strip( ):
		return selected_format.strip( )
	
	return None

def get_audio_prompt_value( task: str | None, prompt: str | None ) -> str | None:
	"""
	
		Purpose:
		--------
		Return prompt/instruction text only when it is meaningful for the selected task.
		
		Parameters:
		-----------
		task: str | None
			Selected Audio mode task.
		
		prompt: str | None
			System instruction or task prompt text.
		
		Returns:
		--------
		str | None
			Clean prompt text or None.
		
	"""
	if not isinstance( prompt, str ) or not prompt.strip( ):
		return None
	
	if task in [ 'Transcribe', 'Translate', 'Text-to-Speech' ]:
		return prompt.strip( )
	
	return None

def save_audio_upload( upload: Any ) -> str | None:
	"""
	
		Purpose:
		--------
		Save a Streamlit UploadedFile or audio recording object to a temporary file.
		
		Parameters:
		-----------
		upload: Any
			Uploaded or recorded audio file object.
		
		Returns:
		--------
		str | None
			Temporary file path, or None when no valid audio object exists.
		
	"""
	if upload is None:
		return None
	
	try:
		name = getattr( upload, 'name', '' )
		_, ext = os.path.splitext( name )
		if not ext:
			ext = '.wav'
		
		with tempfile.NamedTemporaryFile( delete=False, suffix=ext ) as tmp:
			if hasattr( upload, 'getbuffer' ):
				tmp.write( upload.getbuffer( ) )
			elif hasattr( upload, 'read' ):
				tmp.write( upload.read( ) )
			else:
				return None
			
			return tmp.name
	except Exception:
		return None

def render_audio_segments( result: dict[ str, Any ] | None ) -> None:
	"""
	
		Purpose:
		--------
		Render verbose or diarized audio segments when the wrapper exposes them.
		
		Parameters:
		-----------
		result: dict[str, Any] | None
			Normalized audio result containing optional segments.
		
		Returns:
		--------
		None
		
	"""
	if not isinstance( result, dict ):
		return
	
	segments = result.get( 'segments' )
	if not isinstance( segments, list ) or len( segments ) == 0:
		return
	
	rows: list[ dict[ str, Any ] ] = [ ]
	for index, segment in enumerate( segments, start=1 ):
		if not isinstance( segment, dict ):
			continue
		
		rows.append(
			{
					'Index': index,
					'Speaker': segment.get( 'speaker' ) or segment.get( 'speaker_id' ) or '',
					'Start': segment.get( 'start' ) or segment.get( 'start_time' ) or '',
					'End': segment.get( 'end' ) or segment.get( 'end_time' ) or '',
					'Text': segment.get( 'text' ) or '',
			} )
	
	if len( rows ) == 0:
		return
	
	df_segments = pd.DataFrame( rows )
	st.caption( 'Segments' )
	st.data_editor( df_segments, use_container_width=True, hide_index=True )

def render_audio_text_result( title: str, result_text: str | None,
		result: dict[ str, Any ] | None = None ) -> None:
	"""
	
		Purpose:
		--------
		Render Audio mode text output and any available structured segment output.
		
		Parameters:
		-----------
		title: str
			Display label for the text result.
		
		result_text: str | None
			Text result to render.
		
		result: dict[str, Any] | None
			Optional normalized result dictionary.
		
		Returns:
		--------
		None
		
	"""
	text_value = result_text if isinstance( result_text, str ) else ''
	st.text_area( title, value=text_value, height=250, width='stretch' )
	
	if isinstance( result, dict ):
		language = result.get( 'language' )
		duration = result.get( 'duration' )
		
		if language or duration:
			m1, m2 = st.columns( [ 0.5, 0.5 ] )
			with m1:
				st.metric( 'Language', language or '—' )
			
			with m2:
				st.metric( 'Duration', duration or '—' )
		
		render_audio_segments( result )

def extract_audio_usage( response: Any ) -> dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Extract token or duration usage metadata from an Audio API response when
		available.
		
		Parameters:
		-----------
		response: Any
			Audio API response object.
		
		Returns:
		--------
		dict[str, Any]
			Normalized usage metadata.
		
	"""
	usage: dict[ str, Any ] = { }
	if response is None:
		return usage
	
	try:
		raw = getattr( response, 'usage', None )
	except Exception:
		raw = None
	
	if raw is None:
		return usage
	
	if isinstance( raw, dict ):
		return raw
	
	if hasattr( raw, 'model_dump' ):
		try:
			return raw.model_dump( )
		except Exception:
			return { 'raw': str( raw ) }
	
	return { 'raw': str( raw ) }

def run_audio_file_task( task: str | None, file_path: str | None,
		transcriber: Transcription, translator: Translation ) -> str | None:
	"""
	
		Purpose:
		--------
		Run the selected file-based Audio mode task against a temporary audio file.
		
		Parameters:
		-----------
		task: str | None
			Selected Audio mode task.
		
		file_path: str | None
			Temporary audio file path.
		
		transcriber: Transcription
			Transcription wrapper instance.
		
		translator: Translation
			Translation wrapper instance.
		
		Returns:
		--------
		str | None
			Text output from transcription or translation.
		
	"""
	if not isinstance( task, str ) or not task.strip( ):
		st.warning( 'Select an audio task before processing audio.' )
		return None
	
	if not isinstance( file_path, str ) or not file_path.strip( ):
		st.warning( 'Upload or record audio before processing.' )
		return None
	
	prompt = get_audio_prompt_value( task,
		st.session_state.get( 'audio_system_instructions', '' ) )
	
	response_format = get_audio_response_format_value(
		task=task,
		selected_format=st.session_state.get( 'audio_response_format' ),
		selected_mime_type=st.session_state.get( 'audio_mime_type' ) )
	
	model = st.session_state.get( 'audio_model' )
	language = st.session_state.get( 'audio_language' )
	temperature = st.session_state.get( 'audio_temperature' )
	include = st.session_state.get( 'audio_include', [ ] )
	
	if task == 'Transcribe':
		result_text = transcriber.transcribe(
			path=file_path,
			model=model or 'gpt-4o-transcribe',
			language=language or None,
			prompt=prompt,
			format=response_format,
			temperature=temperature,
			include=include )
		
		result = getattr( transcriber, 'normalized_result', { } )
		st.session_state[ 'audio_output' ] = result_text or ''
		st.session_state[ 'audio_last_result' ] = result if isinstance( result, dict ) else { }
		st.session_state[ 'audio_last_usage' ] = extract_audio_usage(
			getattr( transcriber, 'response', None ) )
		
		return result_text
	
	if task == 'Translate':
		result_text = translator.translate(
			filepath=file_path,
			model=model or 'whisper-1',
			prompt=prompt,
			format=response_format,
			temperature=temperature,
			language=language or None )
		
		result = getattr( translator, 'normalized_result', { } )
		st.session_state[ 'audio_output' ] = result_text or ''
		st.session_state[ 'audio_last_result' ] = result if isinstance( result, dict ) else { }
		st.session_state[ 'audio_last_usage' ] = extract_audio_usage(
			getattr( translator, 'response', None ) )
		
		return result_text
	
	if task == 'Text-to-Speech':
		st.info( 'Use the Text-to-Speech input area to generate speech from text.' )
		return None
	
	return None

def run_audio_tts_task( text: str | None, tts: TTS ) -> bytes | None:
	"""
	
		Purpose:
		--------
		Run the selected Text-to-Speech task and store generated audio bytes.
		
		Parameters:
		-----------
		text: str | None
			Text to synthesize.
		
		tts: TTS
			Text-to-speech wrapper instance.
		
		Returns:
		--------
		bytes | None
			Generated audio bytes.
		
	"""
	if not isinstance( text, str ) or not text.strip( ):
		st.warning( 'Enter text before generating speech.' )
		return None
	
	model = st.session_state.get( 'audio_model' ) or 'gpt-4o-mini-tts'
	voice = st.session_state.get( 'audio_voice' ) or 'alloy'
	speed = st.session_state.get( 'audio_speed', 1.0 )
	response_format = get_audio_response_format_value(
		task='Text-to-Speech',
		selected_format=st.session_state.get( 'audio_response_format' ),
		selected_mime_type=st.session_state.get( 'audio_mime_type' ) )
	
	instructions = get_audio_prompt_value(
		task='Text-to-Speech',
		prompt=st.session_state.get( 'audio_system_instructions', '' ) )
	
	audio_bytes = tts.create_speech(
		text=text.strip( ),
		model=model,
		format=response_format or 'mp3',
		speed=speed,
		voice=voice,
		instruct=instructions )
	
	st.session_state[ 'audio_output_bytes' ] = audio_bytes
	st.session_state[ 'audio_output' ] = text.strip( )
	st.session_state[ 'audio_last_result' ] = {
			'text': text.strip( ),
			'format': response_format or 'mp3',
			'voice': voice,
			'speed': speed,
	}
	st.session_state[ 'audio_last_usage' ] = extract_audio_usage(
		getattr( tts, 'response', None ) )
	
	return audio_bytes

def clear_audio_outputs( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Audio mode output state.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'audio_output' ] = ''
	st.session_state[ 'audio_output_bytes' ] = None
	st.session_state[ 'audio_last_result' ] = { }
	st.session_state[ 'audio_last_usage' ] = { }

def clear_audio_messages( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Audio mode message and output state.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state.audio_messages = [ ]
	clear_audio_outputs( )

def clear_audio_instructions( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Audio mode system instructions and selected prompt template.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'audio_system_instructions' ] = ''
	st.session_state[ 'instructions' ] = ''

def load_audio_instruction_template( ) -> None:
	"""
	
		Purpose:
		--------
		Load the selected prompt template into Audio mode system instructions.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	name = st.session_state.get( 'instructions' )
	if name and name != 'No Templates Found':
		text = fetch_prompt_text( cfg.DB_PATH, name )
		if text is not None:
			st.session_state[ 'audio_system_instructions' ] = text

def convert_audio_system_instructions( ) -> None:
	"""
	
		Purpose:
		--------
		Convert Audio mode system instructions between XML-like delimiters and Markdown
		headings.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	text = st.session_state.get( 'audio_system_instructions', '' )
	if not isinstance( text, str ) or not text.strip( ):
		return
	
	source = text.strip( )
	if cfg.XML_BLOCK_PATTERN.search( source ):
		converted = convert_xml( source )
	else:
		converted = convert_markdown( source )
	
	st.session_state[ 'audio_system_instructions' ] = converted

def reset_audio_task_controls( ) -> None:
	"""
	
		Purpose:
		--------
		Reset task/model/format Audio mode controls through a widget-safe callback.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	for key in [ 'audio_task', 'audio_model', 'audio_language',
	             'audio_response_format', 'audio_include' ]:
		if key in st.session_state:
			del st.session_state[ key ]

def reset_audio_tts_controls( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Text-to-Speech Audio mode controls through a widget-safe callback.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	for key in [ 'audio_voice', 'audio_speed', 'audio_mime_type' ]:
		if key in st.session_state:
			del st.session_state[ key ]
			
# ==============================================================================
# Init
# ==============================================================================

initialize_database( )
embedder = load_embedder( )

if not isinstance( st.session_state.get( 'messages' ), list ):
	st.session_state[ 'messages' ] = [ ]

if len( st.session_state[ 'messages' ] ) == 0:
	st.session_state[ 'messages' ] = load_history( )

if 'system_instructions' not in st.session_state:
	st.session_state[ 'system_instructions' ] = ''

# ======================================================================================
# Page Configuration
# ======================================================================================

st.set_page_config( page_title='Gipity', page_icon=cfg.FAVICON,
	layout='wide', initial_sidebar_state='collapsed' )

st.caption( cfg.APP_SUBTITLE )
inject_response_css( )
init_state( )

# ======================================================================================
# Sidebar
# ======================================================================================
with st.sidebar:
	style_subheaders( )
	st.logo( cfg.LOGO_PATH, size='large' )
	
	st.divider( )
	
	st.text( 'AI Mode' )
	mode = st.sidebar.radio( 'Select Mode', cfg.GPT_MODES, index=0, label_visibility='collapsed' )
	
	st.divider( )
	
	st.text( 'API Settings' )
	
	# -----API KEY Expander------------------------------
	with st.expander( label='Keys', icon='🔑', expanded=False ):
		openai_key = st.text_input( 'Open API Key', type='password',
			value=st.session_state.openai_api_key or '',
			help='Overrides OPENAI_API_KEY from config.py for this session only.' )
		
		if openai_key:
			st.session_state.openai_api_key = openai_key
			os.environ[ 'OPENAI_API_KEY' ] = openai_key
		
		google_key = st.text_input( 'Google API Key', type='password',
			value=st.session_state.google_api_key or '',
			help='Overrides GOOGLE_API_KEY from config.py for this session only.' )
		
		if google_key:
			st.session_state.google_api_key = google_key
			os.environ[ 'GOOGLE_API_KEY' ] = google_key
		
		googlemaps_key = st.text_input( 'Google Maps API Key', type='password',
			value=st.session_state.googlemaps_api_key or '',
			help='Overrides GOOGLEMAPS_API_KEY from config.py for this session only.' )
		
		if googlemaps_key:
			st.session_state.googlemaps_api_key = googlemaps_key
			os.environ[ 'GOOGLEMAPS_API_KEY' ] = googlemaps_key
		
		geocoding_key = st.text_input( 'Geocoding API Key', type='password',
			value=st.session_state.geocoding_api_key or '',
			help='Overrides GEOCODING_API_KEY from config.py for this session only.' )
		
		if geocoding_key:
			st.session_state.geocoding_api_key = geocoding_key
			os.environ[ 'GEOCODING_API_KEY' ] = geocoding_key
		
		google_cse_id = st.text_input( 'Google Custom Search ID', type='password',
			value=st.session_state.google_cse_id or '',
			help='Overrides GOOGLE_CSE_ID from config.py for this session only.' )
		
		if google_cse_id:
			st.session_state.google_cse_id = google_cse_id
			os.environ[ 'GOOGLE_CSE_ID' ] = google_cse_id

# ======================================================================================
# TEXT MODE
# ======================================================================================
if mode == 'Text':
	ensure_text_mode_state( )
	text = Chat( )
	
	# ------------------------------------------------------------------
	# Session State
	# ------------------------------------------------------------------
	text_model = st.session_state.get( 'text_model', '' )
	text_reasoning = st.session_state.get( 'text_reasoning', '' )
	text_response_format = st.session_state.get( 'text_response_format', '' )
	text_tool_choice = st.session_state.get( 'text_tool_choice', '' )
	text_content = st.session_state.get( 'text_content', '' )
	text_input = st.session_state.get( 'text_input', '' )
	text_previous_response_id = st.session_state.get( 'text_previous_response_id', '' )
	text_conversation_id = st.session_state.get( 'text_conversation_id', '' )
	text_max_calls = st.session_state.get( 'text_max_calls', 0 )
	text_max_tokens = st.session_state.get( 'text_max_tokens', 0 )
	text_top_percent = st.session_state.get( 'text_top_percent', 0.0 )
	text_frequency_penalty = st.session_state.get( 'text_frequency_penalty', 0.0 )
	text_presence_penalty = st.session_state.get( 'text_presence_penalty', 0.0 )
	text_temperature = st.session_state.get( 'text_temperature', 0.0 )
	text_stream = st.session_state.get( 'text_stream', False )
	text_parallel_calls = st.session_state.get( 'text_parallel_calls', False )
	text_store = st.session_state.get( 'text_store', False )
	text_background = st.session_state.get( 'text_background', False )
	text_tools = st.session_state.get( 'text_tools', [ ] )
	text_context = st.session_state.get( 'text_context', [ ] )
	text_include = st.session_state.get( 'text_include', [ ] )
	text_domains = st.session_state.get( 'text_domains', [ ] )
	text_vector_store_ids = st.session_state.get( 'text_vector_store_ids', '' )
	text_json_schema_name = st.session_state.get( 'text_json_schema_name', 'structured_response' )
	text_json_schema = st.session_state.get( 'text_json_schema', '' )
	text_json_schema_strict = st.session_state.get( 'text_json_schema_strict', True )
	
	# ------------------------------------------------------------------
	# Reset Callbacks
	# ------------------------------------------------------------------
	def reset_text_llm_controls( ) -> None:
		"""
		
			Purpose:
			--------
			Reset Text mode LLM controls through a widget-safe callback.
			
			Parameters:
			-----------
			None
			
			Returns:
			--------
			None
			
		"""
		for key in [ 'text_model', 'text_temperature', 'text_presence_penalty',
		             'text_reasoning', 'text_top_percent', 'text_frequency_penalty' ]:
			if key in st.session_state:
				del st.session_state[ key ]
	
	def reset_text_tool_controls( ) -> None:
		"""
		
			Purpose:
			--------
			Reset Text mode tool controls through a widget-safe callback.
			
			Parameters:
			-----------
			None
			
			Returns:
			--------
			None
			
		"""
		for key in [ 'text_max_calls', 'text_tool_choice', 'text_include',
		             'text_tools', 'text_domains_input', 'text_domains',
		             'text_parallel_calls', 'text_vector_store_ids' ]:
			if key in st.session_state:
				del st.session_state[ key ]
	
	def reset_text_response_controls( ) -> None:
		"""
		
			Purpose:
			--------
			Reset Text mode response controls through a widget-safe callback.
			
			Parameters:
			-----------
			None
			
			Returns:
			--------
			None
			
		"""
		for key in [ 'text_stream', 'text_store', 'text_max_tokens',
		             'text_background', 'text_response_format', 'text_input',
		             'text_previous_response_id', 'text_conversation_id' ]:
			if key in st.session_state:
				del st.session_state[ key ]
	
	def clear_text_instructions( ) -> None:
		"""
		
			Purpose:
			--------
			Clear Text mode system instructions and selected prompt template.
			
			Parameters:
			-----------
			None
			
			Returns:
			--------
			None
			
		"""
		st.session_state[ 'text_system_instructions' ] = ''
		st.session_state[ 'instructions' ] = ''
	
	def convert_text_system_instructions( ) -> None:
		"""
		
			Purpose:
			--------
			Convert Text mode system instructions between XML-like delimiters and
			Markdown headings.
			
			Parameters:
			-----------
			None
			
			Returns:
			--------
			None
			
		"""
		text_value = st.session_state.get( 'text_system_instructions', '' )
		if not isinstance( text_value, str ) or not text_value.strip( ):
			return
		
		source = text_value.strip( )
		if cfg.XML_BLOCK_PATTERN.search( source ):
			converted = convert_xml( source )
		else:
			converted = convert_markdown( source )
		
		st.session_state[ 'text_system_instructions' ] = converted
	
	def load_text_instruction_template( ) -> None:
		"""
		
			Purpose:
			--------
			Load the selected prompt template into Text mode system instructions.
			
			Parameters:
			-----------
			None
			
			Returns:
			--------
			None
			
		"""
		name = st.session_state.get( 'instructions' )
		if name and name != 'No Templates Found':
			prompt_text = fetch_prompt_text( cfg.DB_PATH, name )
			if prompt_text is not None:
				st.session_state[ 'text_system_instructions' ] = prompt_text
	
	# ------------------------------------------------------------------
	# Main Chat UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with center:
		st.subheader( "💬 Text Generation", help=cfg.TEXT_GENERATION )
		st.divider( )
		
		if st.session_state.get( 'clear_instructions' ):
			st.session_state[ 'text_system_instructions' ] = ''
			st.session_state[ 'instructions_last_loaded' ] = ''
			st.session_state[ 'clear_instructions' ] = False
		
		with st.expander( label='Mind Controls', icon='🧠', expanded=False, width='stretch' ):
			with st.expander( label='LLM Settings', icon='🧊', expanded=False, width='stretch' ):
				llm_c1, llm_c2, llm_c3, llm_c4, llm_c5, llm_c6 = st.columns(
					[ 0.16, 0.16, 0.16, 0.16, 0.16, 0.16 ], border=True, gap='xxsmall' )
				
				# ---------- Model ------------
				with llm_c1:
					model_options = list( text.model_options )
					set_text_model = st.selectbox( label='Select Model', options=model_options,
						key='text_model', placeholder='Options', index=None,
						help='REQUIRED. Text Generation model used by the AI', )
					
					text_model = st.session_state[ 'text_model' ]
				
				# ---------- Reasoning ------------
				with llm_c2:
					reasoning_options = list( text.reasoning_options )
					set_text_reasoning = st.selectbox( label='Reasoning',
						options=reasoning_options, key='text_reasoning',
						help=cfg.REASONING, index=None, placeholder='Options' )
					
					text_reasoning = st.session_state[ 'text_reasoning' ]
				
				# ---------- Top-P ------------
				with llm_c3:
					set_text_top_p = st.slider( label='Top-P', min_value=0.0, max_value=1.0,
						step=0.01, help=cfg.TOP_P, key='text_top_percent' )
					
					text_top_percent = st.session_state[ 'text_top_percent' ]
				
				# ---------- Temperature ------------
				with llm_c4:
					set_text_temperature = st.slider( label='Temperature', min_value=0.0,
						max_value=2.0, step=0.01, help=cfg.TEMPERATURE,
						key='text_temperature' )
					
					text_temperature = st.session_state[ 'text_temperature' ]
				
				# ---------- Presence ------------
				with llm_c5:
					set_text_presence = st.slider( label='Presense Penalty',
						min_value=-2.0, max_value=2.0, step=0.01,
						help=cfg.PRESENCE_PENALTY, key='text_presence_penalty' )
					
					text_presence = st.session_state[ 'text_presence_penalty' ]
				
				# ---------- Frequency ------------
				with llm_c6:
					set_text_freq = st.slider( label='Frequency Penalty',
						min_value=-2.0, max_value=2.0, step=0.01,
						help=cfg.FREQUENCY_PENALTY, key='text_frequency_penalty' )
					
					text_frequency = st.session_state[ 'text_frequency_penalty' ]
				
				# ---------- Reset Model ------------
				st.button( label='Reset', key='reset_text_model',
					width='stretch', on_click=reset_text_llm_controls )
			
			with st.expander( label='Tool Settings', icon='🛠️', expanded=False, width='stretch' ):
				tool_c1, tool_c2, tool_c3, tool_c4, tool_c5, tool_c6 = st.columns(
					[ 0.16, 0.16, 0.16, 0.16, 0.16, 0.16 ], border=True, gap='xxsmall' )
				
				# ---------- Max Calls ------------
				with tool_c1:
					set_text_calls = st.slider( label='Max Calls', min_value=0, max_value=10,
						step=1, help=cfg.MAX_TOOL_CALLS, key='text_max_calls' )
					
					text_max_calls = st.session_state[ 'text_max_calls' ]
				
				# ---------- Choice ------------
				with tool_c2:
					choice_options = list( text.choice_options )
					set_text_choice = st.selectbox( label='Choice', options=choice_options,
						key='text_tool_choice', help=cfg.CHOICE, index=None,
						placeholder='Options' )
					
					text_tool_choice = st.session_state[ 'text_tool_choice' ]
				
				# ---------- Include ------------
				with tool_c3:
					include_options = list( text.include_options )
					set_text_include = st.multiselect( label='Include', options=include_options,
						key='text_include', help=cfg.INCLUDE, placeholder='Options' )
					
					text_include = st.session_state[ 'text_include' ]
				
				# ---------- Domains ------------
				with tool_c4:
					set_text_domains = st.text_input( label='Allowed Domains',
						key='text_domains_input',
						value=','.join( st.session_state.get( 'text_domains', [ ] ) ),
						help=cfg.ALLOWED_DOMAINS, width='stretch',
						placeholder='Enter Domains' )
					
					text_domains = [ d.strip( ) for d in set_text_domains.split( ',' )
					                 if d.strip( ) ]
					
					st.session_state[ 'text_domains' ] = text_domains
				
				# ---------- Tools ------------
				with tool_c5:
					tool_options = list( text.tool_options )
					set_text_tools = st.multiselect( label='Tools', options=tool_options,
						key='text_tools', help=cfg.TOOLS, placeholder='Options' )
					
					text_tools = st.session_state[ 'text_tools' ]
				
				# ---------- Parallel ------------
				with tool_c6:
					set_text_parallel = st.toggle( label='Allow Parallel',
						key='text_parallel_calls', help=cfg.PARALLEL_TOOL_CALLS )
					
					text_parallel_calls = st.session_state[ 'text_parallel_calls' ]
				
				# ---------- Vector Store IDs ------------
				set_text_vector_store_ids = st.text_input( label='Vector Store IDs',
					key='text_vector_store_ids',
					value=st.session_state.get( 'text_vector_store_ids', '' ),
					help=('Required when the file_search tool is selected. Enter one or more '
					      'vector store IDs separated by commas.'),
					width='stretch', placeholder='vs_...' )
				
				text_vector_store_ids = st.session_state.get( 'text_vector_store_ids', '' )
				
				# ---------- Reset Tools ------------
				st.button( label='Reset', key='reset_text_tools',
					width='stretch', on_click=reset_text_tool_controls )
			
			with st.expander( label='Response Settings', icon='↔️', expanded=False,
					width='stretch' ):
				resp_c1, resp_c2, resp_c3, resp_c4, resp_c5, resp_c6, resp_c7 = st.columns(
					[ 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14 ],
					border=True, gap='xxsmall' )
				
				# ---------- Input Mode ------------
				with resp_c1:
					input_mode_options = [ '', 'conversation', 'single_turn' ]
					set_text_input = st.selectbox( label='Input Mode', options=input_mode_options,
						key='text_input',
						help=('Optional. Controls whether prior chat messages are '
						      'sent back to the Responses API as context.'),
						placeholder='Options' )
					
					text_input = st.session_state.get( 'text_input', '' )
				
				# ---------- Max Tokens ------------
				with resp_c2:
					set_text_tokens = st.slider( label='Max Tokens', min_value=0,
						max_value=100000, step=500, help=cfg.MAX_OUTPUT_TOKENS,
						key='text_max_tokens' )
					
					text_tokens = st.session_state[ 'text_max_tokens' ]
				
				# ---------- Response Format ------------
				with resp_c3:
					format_options = list( text.format_options )
					set_text_response_format = st.selectbox( label='Response Format',
						options=format_options, key='text_response_format',
						help=('Optional. Responses API text.format setting. '
						      'Use "text" for plain text responses.'),
						placeholder='Options' )
					
					text_response_format = st.session_state.get( 'text_response_format', '' )
				
				# ---------- Previous Response ID ------------
				with resp_c4:
					set_text_previous_id = st.text_input( label='Previous Response ID',
						key='text_previous_response_id',
						value=st.session_state.get( 'text_previous_response_id', '' ),
						help=('Optional. Responses API previous_response_id. '
						      'Ignored in single_turn and conversation modes.'),
						width='stretch', placeholder='Enter Previous Response ID' )
					
					text_previous_response_id = st.session_state.get(
						'text_previous_response_id', '' )
				
				# ---------- Store ------------
				with resp_c5:
					set_text_store = st.toggle( label='Store', key='text_store', help=cfg.STORE )
					
					text_store = st.session_state[ 'text_store' ]
				
				# ---------- Stream ------------
				with resp_c6:
					set_text_stream = st.toggle( label='Stream', key='text_stream',
						help=cfg.STREAM )
					
					text_stream = st.session_state[ 'text_stream' ]
				
				# ---------- Background ------------
				with resp_c7:
					set_text_background = st.toggle( label='Background',
						key='text_background', help=cfg.BACKGROUND_MODE )
					
					text_background = st.session_state[ 'text_background' ]
				
				# ---------- Conversation ID ------------
				set_text_conversation_id = st.text_input( label='Conversation ID',
					key='text_conversation_id',
					value=st.session_state.get( 'text_conversation_id', '' ),
					help=('Optional. Only used when Input Mode is conversation. '
					      'Leave blank to use local message context instead.'),
					width='stretch', placeholder='conv_...' )
				
				text_conversation_id = st.session_state.get( 'text_conversation_id', '' )
				
				# ---------- Reset Response ------------
				st.button( label='Reset', key='reset_text_response',
					width='stretch', on_click=reset_text_response_controls )
			
			with st.expander( label='Structured Output Settings', icon='🧾',
					expanded=False, width='stretch' ):
				struct_c1, struct_c2 = st.columns( [ 0.6, 0.4 ], border=True, gap='xxsmall' )
				
				# ---------- Schema Name ------------
				with struct_c1:
					set_text_json_schema_name = st.text_input( label='Schema Name',
						key='text_json_schema_name',
						value=st.session_state.get( 'text_json_schema_name',
							'structured_response' ),
						help='Used only when Response Format is json_schema.',
						width='stretch', placeholder='structured_response' )
					
					text_json_schema_name = st.session_state.get(
						'text_json_schema_name', 'structured_response' )
				
				# ---------- Strict ------------
				with struct_c2:
					set_text_json_schema_strict = st.toggle( label='Strict Schema',
						key='text_json_schema_strict',
						help='Used only when Response Format is json_schema.' )
					
					text_json_schema_strict = st.session_state.get(
						'text_json_schema_strict', True )
				
				# ---------- JSON Schema ------------
				set_text_json_schema = st.text_area( label='JSON Schema',
					height=160, width='stretch', key='text_json_schema',
					help=('Used only when Response Format is json_schema. Enter the JSON Schema '
					      'object, not a Python dictionary.'),
					placeholder='{ "type": "object", "properties": { ... }, "required": [ ... ] }' )
				
				text_json_schema = st.session_state.get( 'text_json_schema', '' )
				
				# ---------- Reset Structured Output ------------
				st.button( label='Reset', key='reset_text_structured_output',
					width='stretch', on_click=reset_text_structured_output_controls )
		
		with st.expander( label='System Instructions', icon='🖥️',
				expanded=False, width='stretch' ):
			in_left, in_right = st.columns( [ 0.8, 0.2 ] )
			
			prompt_names = fetch_prompt_names( cfg.DB_PATH )
			if not prompt_names:
				prompt_names = [ '' ]
			
			with in_left:
				st.text_area( label='Enter Text', height=50, width='stretch',
					help=cfg.SYSTEM_INSTRUCTIONS, key='text_system_instructions' )
			
			with in_right:
				st.selectbox( label='Use Template', options=prompt_names, index=None,
					key='instructions', on_change=load_text_instruction_template )
			
			btn_c1, btn_c2 = st.columns( [ 0.8, 0.2 ] )
			with btn_c1:
				st.button( label='Clear Instructions', width='stretch',
					on_click=clear_text_instructions )
			
			with btn_c2:
				st.button( label='XML <-> Markdown', width='stretch',
					on_click=convert_text_system_instructions )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ---------------------------------------------------
		#                   MESSAGES
		# ---------------------------------------------------
		if st.session_state.get( 'text_messages' ) is not None:
			for msg in st.session_state.text_messages:
				self_avatar = cfg.GIPITY if msg.get( 'role' ) == 'assistant' else ''
				with st.chat_message( msg.get( 'role', 'assistant' ), avatar=self_avatar ):
					st.markdown( msg.get( 'content', '' ) )
		
		prompt = st.chat_input( 'Gipity Generate …' )
		if prompt is not None and str( prompt ).strip( ):
			prompt = str( prompt ).strip( )
			st.session_state.text_messages.append(
				{
						'role': 'user',
						'content': prompt,
				} )
			
			with st.chat_message( 'assistant', avatar=cfg.GIPITY ):
				with st.spinner( 'Thinking…' ):
					response_text = None
					response_obj = None
					
					try:
						vector_store_ids = parse_text_vector_store_ids(
							st.session_state.get( 'text_vector_store_ids', '' ) )
						
						text_tools = build_text_tools(
							selected_tools=st.session_state.get( 'text_tools', [ ] ),
							vector_store_ids=vector_store_ids )
						
						text_include = build_text_include(
							selected_include=st.session_state.get( 'text_include', [ ] ),
							selected_tools=text_tools )
						
						text_tool_choice = build_text_tool_choice(
							tool_choice=st.session_state.get( 'text_tool_choice' ),
							selected_tools=text_tools )
						
						text_format = build_text_response_format(
							response_format=st.session_state.get( 'text_response_format' ),
							schema_name=st.session_state.get( 'text_json_schema_name' ),
							schema_text=st.session_state.get( 'text_json_schema' ),
							strict=st.session_state.get( 'text_json_schema_strict', True ) )
						
						if st.session_state.get( 'text_input' ) != 'single_turn':
							text_context = build_text_context(
								messages=st.session_state.get( 'text_messages', [ ] ),
								include_last_message=False )
						else:
							text_context = [ ]
						
						st.session_state[ 'text_context' ] = text_context
						
						text_previous_id = get_text_previous_response_id(
							input_mode=st.session_state.get( 'text_input' ),
							previous_id=st.session_state.get( 'text_previous_response_id' ) )
						
						text_conversation_id = get_text_conversation_id(
							input_mode=st.session_state.get( 'text_input' ),
							conversation_id=st.session_state.get( 'text_conversation_id' ) )
						
						text_stream_value = get_text_stream_value(
							st.session_state.get( 'text_stream' ) )
						
						text_background_value = get_text_background_value(
							st.session_state.get( 'text_background' ) )
						
						response_text = text.generate_text( prompt=prompt,
							model=st.session_state.get( 'text_model' ),
							temperature=st.session_state.get( 'text_temperature' ),
							format=text_format,
							top_p=st.session_state.get( 'text_top_percent' ),
							frequency=st.session_state.get( 'text_frequency_penalty' ),
							presence=st.session_state.get( 'text_presence_penalty' ),
							max_tools=st.session_state.get( 'text_max_calls' ),
							max_tokens=st.session_state.get( 'text_max_tokens' ),
							store=st.session_state.get( 'text_store' ),
							stream=text_stream_value,
							instruct=st.session_state.get( 'text_system_instructions' ),
							background=text_background_value,
							reasoning=st.session_state.get( 'text_reasoning' ),
							include=text_include,
							tools=text_tools,
							allowed_domains=st.session_state.get( 'text_domains', [ ] ),
							previous_id=text_previous_id,
							tool_choice=text_tool_choice,
							is_parallel=st.session_state.get( 'text_parallel_calls' ),
							context=text_context,
							vector_store_ids=vector_store_ids,
							conversation_id=text_conversation_id )
						
						response_obj = getattr( text, 'response', None )
						st.session_state[ 'text_previous_response_id' ] = (
								getattr( text, 'previous_id', None ) or '')
					
					except Exception as exc:
						err = Error( exc )
						st.error( f'Generation Failed: {err.info}' )
						response_text = None
						response_obj = getattr( text, 'response', None )
					
					if response_text is not None and str( response_text ).strip( ):
						st.markdown( response_text )
						st.session_state.text_messages.append(
							{
									'role': 'assistant',
									'content': str( response_text ).strip( ),
							} )
						
						st.session_state[ 'text_context' ] = build_text_context(
							messages=st.session_state.get( 'text_messages', [ ] ),
							include_last_message=True )
						
						st.session_state.last_answer = str( response_text ).strip( )
						st.session_state.last_sources = extract_sources( response_obj )
					else:
						st.error( 'Generation Failed!.' )
					
					try:
						update_token_counters( response_obj )
					except Exception:
						pass
		
		# --------  Reset Button
		if st.button( 'Clear Messages' ):
			st.session_state.text_messages = [ ]
			st.session_state[ 'text_previous_response_id' ] = ''
			st.session_state[ 'text_conversation_id' ] = ''
			st.session_state.last_answer = ''
			st.session_state.last_sources = [ ]
			st.rerun( )

# ======================================================================================
# IMAGES MODE
# ======================================================================================
elif mode == 'Images':
	image = Images( )
	
	# ------------------------------------------------------------------
	# Session State
	# ------------------------------------------------------------------
	if st.session_state.get( 'clear_instructions' ):
		st.session_state[ 'image_system_instructions' ] = ''
		st.session_state[ 'clear_image_instructions' ] = False
		st.session_state[ 'clear_instructions' ] = False
	
	if not isinstance( st.session_state.get( 'image_number' ), int ):
		st.session_state[ 'image_number' ] = 1
	
	if int( st.session_state.get( 'image_number', 1 ) or 1 ) < 1:
		st.session_state[ 'image_number' ] = 1
	
	if not isinstance( st.session_state.get( 'image_max_tokens' ), int ):
		st.session_state[ 'image_max_tokens' ] = 0
	
	if not isinstance( st.session_state.get( 'image_temperature' ), float ):
		st.session_state[ 'image_temperature' ] = 0.0
	
	if not isinstance( st.session_state.get( 'image_include' ), list ):
		st.session_state[ 'image_include' ] = [ ]
	
	if not isinstance( st.session_state.get( 'image_compression' ), float ):
		st.session_state[ 'image_compression' ] = 0.0
	
	if not st.session_state.get( 'image_analysis_detail' ):
		st.session_state[ 'image_analysis_detail' ] = 'auto'
	
	# ------------------------------------------------------------------
	# Main UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with center:
		st.subheader( '📷 Images API', help=cfg.IMAGES_API )
		st.divider( )
		
		with st.expander( label='Mind Controls', icon='🧠', expanded=False, width='stretch' ):
			with st.expander( label='LLM Settings', icon='🧊', expanded=False, width='stretch' ):
				llm_c1, llm_c2, llm_c3, llm_c4 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall' )
				
				# ---------- Image Mode ------------
				with llm_c1:
					image_mode = st.selectbox( label='Image Mode',
						options=[ 'Generation', 'Analysis', 'Editing' ],
						key='image_mode', help='Available OpenAI image workflows.',
						index=None, placeholder='Options' )
				
				# ---------- Image Model ------------
				with llm_c2:
					image_model = st.selectbox( label='Select Model',
						options=get_image_models( image ), key='image_model',
						help='Required for image generation and image editing.',
						index=None, placeholder='Options' )
				
				# ---------- Analysis Model ------------
				with llm_c3:
					image_analysis_model = st.selectbox( label='Analysis Model',
						options=get_image_analysis_models( image ), key='image_analysis_model',
						help='Responses API vision model used for image analysis.',
						index=None, placeholder='Options' )
				
				# ---------- Number ------------
				with llm_c4:
					image_number = st.slider( label='Number', min_value=1, max_value=10,
						step=1, help='Number of images to request.',
						key='image_number' )
				
				# ---------- Reset ------------
				st.button( label='Reset', key='reset_image_llm',
					width='stretch', on_click=reset_image_llm_settings )
			
			with st.expander( label='Visual Settings', icon='👁️', expanded=False, width='stretch' ):
				vis_c1, vis_c2, vis_c3, vis_c4, vis_c5 = st.columns(
					[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
				
				# ---------- Output Format ------------
				with vis_c1:
					image_mime_type = st.selectbox( label='Output Format',
						options=get_image_mime_options( image ), key='image_mime_type',
						help='Image output format: png, jpeg, or webp.',
						index=None, placeholder='Options' )
				
				# ---------- Image Size ------------
				with vis_c2:
					image_size = st.selectbox( label='Image Size',
						options=get_image_size_options( image ), key='image_size',
						help='Requested output image size.',
						index=None, placeholder='Options' )
				
				# ---------- Image Quality ------------
				with vis_c3:
					image_quality = st.selectbox( label='Image Quality',
						options=get_image_quality_options( image ), key='image_quality',
						help='Requested image quality.',
						index=None, placeholder='Options' )
				
				# ---------- Background ------------
				with vis_c4:
					image_backcolor = st.selectbox( label='Background',
						options=get_image_background_options( image ), key='image_backcolor',
						help='Requested background mode for image generation and editing.',
						index=None, placeholder='Options' )
				
				# ---------- Compression ------------
				with vis_c5:
					image_compression = st.slider( label='Compression',
						min_value=0.0, max_value=1.0, step=0.01,
						help=cfg.IMAGE_COMPRESSION, key='image_compression' )
				
				# ---------- Reset ------------
				st.button( label='Reset', key='reset_image_visual',
					width='stretch', on_click=reset_image_visual_settings )
			
			with st.expander( label='Analysis Settings', icon='🔎', expanded=False,
					width='stretch' ):
				ana_s1, ana_s2, ana_s3, ana_s4, ana_s5 = st.columns(
					[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
				
				# ---------- Detail ------------
				with ana_s1:
					image_analysis_detail = st.selectbox( label='Detail',
						options=get_image_detail_options( image ), key='image_analysis_detail',
						help='Vision detail level for image analysis.',
						index=None, placeholder='Options' )
				
				# ---------- Max Tokens ------------
				with ana_s2:
					image_max_tokens = st.slider( label='Max Tokens',
						min_value=0, max_value=100000, step=500,
						help=cfg.MAX_OUTPUT_TOKENS, key='image_max_tokens' )
				
				# ---------- Temperature ------------
				with ana_s3:
					image_temperature = st.slider( label='Temperature',
						min_value=0.0, max_value=2.0, step=0.01,
						help=cfg.TEMPERATURE, key='image_temperature' )
				
				# ---------- Include ------------
				with ana_s4:
					include_options = list( image.include_options )
					image_include = st.multiselect( label='Include',
						options=include_options, key='image_include',
						help=cfg.INCLUDE, placeholder='Options' )
				
				# ---------- Store ------------
				with ana_s5:
					image_store = st.toggle( label='Store',
						key='image_store', help=cfg.STORE )
				
				# ---------- Reset ------------
				if st.button( label='Reset', key='reset_image_analysis', width='stretch' ):
					for key in [ 'image_analysis_detail', 'image_max_tokens',
					             'image_temperature', 'image_include', 'image_store' ]:
						if key in st.session_state:
							del st.session_state[ key ]
					
					st.rerun( )
		
		with st.expander( label='System Instructions', icon='🖥️', expanded=False, width='stretch' ):
			in_left, in_right = st.columns( [ 0.8, 0.2 ] )
			
			prompt_names = fetch_prompt_names( cfg.DB_PATH )
			if not prompt_names:
				prompt_names = [ '' ]
			
			# ---------- System Instructions ------------
			with in_left:
				image_system_instructions = st.text_area( label='Enter Text',
					height=50, width='stretch', help=cfg.SYSTEM_INSTRUCTIONS,
					key='image_system_instructions' )
			
			# ---------- Template ------------
			with in_right:
				st.selectbox( label='Use Template', options=prompt_names,
					index=None, key='instructions',
					on_change=load_image_instruction_template )
			
			btn_c1, btn_c2 = st.columns( [ 0.8, 0.2 ] )
			
			# ---------- Clear Instructions ------------
			with btn_c1:
				st.button( label='Clear Instructions', width='stretch',
					on_click=clear_image_instructions )
			
			# ---------- XML <-> Markdown ------------
			with btn_c2:
				st.button( label='XML <-> Markdown', width='stretch',
					on_click=convert_image_system_instructions )
		
		# ------------------------------------------------------------------
		# Tab Section
		# ------------------------------------------------------------------
		tab_gen, tab_analyze, tab_edit = st.tabs( [ 'Generate', 'Analyze', 'Edit' ] )
		
		with tab_gen:
			# ---------------------------------------------------
			#                   MESSAGES
			# ---------------------------------------------------
			if st.session_state.get( 'image_input' ) is not None:
				for msg in st.session_state.get( 'image_input', [ ] ):
					if isinstance( msg, dict ):
						with st.chat_message( msg.get( 'role', 'assistant' ), avatar='' ):
							st.markdown( msg.get( 'content', '' ) )
			
			prompt = st.chat_input( 'Enter image generation prompt...',
				key='image_generate_message' )
			gen_c1, gen_c2 = st.columns( [ 0.5, 0.5 ] )
			
			# ---------- Generate Image ------------
			with gen_c1:
				if st.button( 'Generate Image', key='generate_image' ):
					with st.spinner( 'Generating…' ):
						try:
							if not isinstance( prompt, str ) or not prompt.strip( ):
								st.warning( 'Enter a prompt before generating an image.' )
							elif not isinstance( image_model, str ) or not image_model.strip( ):
								st.warning( 'Select a model before generating an image.' )
							else:
								append_image_message( 'user', prompt.strip( ) )
								
								image_result = image.generate(
									prompt=prompt.strip( ),
									number=image_number,
									model=image_model,
									size=image_size or '1024x1024',
									quality=image_quality or 'auto',
									fmt=image_mime_type or 'jpeg',
									compression=image_compression,
									background=image_backcolor or None )
								
								if image_result is None:
									st.warning( 'No image output was returned.' )
								else:
									st.session_state[ 'image_output_bytes' ] = image_result
									rendered = render_image_output( image_result,
										caption='Generated image' )
									
									if rendered:
										append_image_message(
											'assistant',
											'Generated image returned successfully.' )
									else:
										st.warning(
											'Image output was returned but could not be rendered.' )
								
								try:
									update_token_counters( getattr( image, 'response', None ) )
								except Exception:
									pass
						except Exception as exc:
							st.error( f'Image generation failed: {exc}' )
			
			# ---------- Clear Messages ------------
			with gen_c2:
				if st.button( 'Clear Messages', key='clear_image_generation',
						on_click=clear_image_messages ):
					st.rerun( )
		
		with tab_analyze:
			# ---------- Upload Image ------------
			uploaded_img = st.file_uploader( 'Upload an image for analysis',
				type=[ 'png', 'jpg', 'jpeg', 'webp' ],
				accept_multiple_files=False, key='images_analyze_uploader' )
			
			tmp_path = None
			if uploaded_img:
				tmp_path = save_temp( uploaded_img )
				st.image( uploaded_img, caption='Uploaded image preview',
					use_column_width=True )
			
			if st.session_state.get( 'image_input' ) is not None:
				for msg in st.session_state.get( 'image_input', [ ] ):
					if isinstance( msg, dict ):
						with st.chat_message( msg.get( 'role', 'assistant' ), avatar='' ):
							st.markdown( msg.get( 'content', '' ) )
			
			st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
			
			# ---------------------------------------------------
			#                   MESSAGES
			# ---------------------------------------------------
			prompt = st.chat_input( 'Enter image analysis prompt...',
				key='image_analysis_message' )
			ana_c1, ana_c2 = st.columns( [ 0.5, 0.5 ] )
			
			# ---------- Analyze Image ------------
			with ana_c1:
				if st.button( 'Analyze Image', key='analyze_image' ):
					with st.spinner( 'Analyzing image…' ):
						try:
							if not tmp_path:
								st.warning( 'Upload an image before running analysis.' )
							elif not isinstance( prompt, str ) or not prompt.strip( ):
								st.warning( 'Enter a prompt before analyzing an image.' )
							else:
								model = image_analysis_model or 'gpt-4o-mini'
								append_image_message( 'user', prompt.strip( ) )
								
								analysis_result = image.analyze(
									text=prompt.strip( ),
									path=tmp_path,
									instruct=image_system_instructions,
									model=model,
									max_tokens=image_max_tokens,
									temperature=image_temperature,
									include=image_include,
									store=image_store,
									stream=None,
									detail=image_analysis_detail or 'auto' )
								
								if analysis_result is None:
									st.warning( 'No analysis output was returned.' )
								else:
									st.markdown( '**Analysis result:**' )
									st.write( analysis_result )
									append_image_message( 'assistant', str( analysis_result ) )
								
								try:
									update_token_counters( getattr( image, 'response', None ) )
								except Exception:
									pass
						except Exception as exc:
							st.error( f'Analysis Failed: {exc}' )
			
			# ---------- Clear Messages ------------
			with ana_c2:
				if st.button( 'Clear Messages', key='clear_image_analysis',
						on_click=clear_image_messages ):
					st.rerun( )
		
		with tab_edit:
			# ---------- Upload Image ------------
			uploaded_img = st.file_uploader( 'Upload Image for Edit',
				type=[ 'png', 'jpg', 'jpeg', 'webp' ],
				accept_multiple_files=False, key='images_edit_uploader' )
			
			tmp_path = None
			if uploaded_img:
				tmp_path = save_temp( uploaded_img )
				st.image( uploaded_img, caption='Uploaded image preview',
					use_column_width=True )
			
			if st.session_state.get( 'image_input' ) is not None:
				for msg in st.session_state.get( 'image_input', [ ] ):
					if isinstance( msg, dict ):
						with st.chat_message( msg.get( 'role', 'assistant' ), avatar='' ):
							st.markdown( msg.get( 'content', '' ) )
			
			st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
			
			# ---------------------------------------------------
			#                   MESSAGES
			# ---------------------------------------------------
			prompt = st.chat_input( 'Enter image editing prompt...',
				key='image_edit_message' )
			edit_c1, edit_c2 = st.columns( [ 0.5, 0.5 ] )
			
			# ---------- Edit Image ------------
			with edit_c1:
				if st.button( 'Edit Image', key='edit_image' ):
					with st.spinner( 'Editing image…' ):
						try:
							if not tmp_path:
								st.warning( 'Upload an image before editing.' )
							elif not isinstance( prompt, str ) or not prompt.strip( ):
								st.warning( 'Enter a prompt before editing an image.' )
							elif not isinstance( image_model, str ) or not image_model.strip( ):
								st.warning( 'Select a model before editing an image.' )
							else:
								append_image_message( 'user', prompt.strip( ) )
								
								edit_result = image.edit(
									prompt=prompt.strip( ),
									path=tmp_path,
									model=image_model,
									size=image_size or '1024x1024',
									quality=image_quality or 'auto',
									fmt=image_mime_type or 'jpeg',
									compression=image_compression,
									background=image_backcolor or None,
									number=image_number )
								
								if edit_result is None:
									st.warning( 'No edited image output was returned.' )
								else:
									st.session_state[ 'image_output_bytes' ] = edit_result
									rendered = render_image_output( edit_result,
										caption='Edited image' )
									
									if rendered:
										append_image_message(
											'assistant',
											'Edited image returned successfully.' )
									else:
										st.warning(
											'Edited image output was returned but could not be rendered.' )
								
								try:
									update_token_counters( getattr( image, 'response', None ) )
								except Exception:
									pass
						except Exception as exc:
							st.error( f'Edit Failed: {exc}' )
			
			# ---------- Clear Messages ------------
			with edit_c2:
				if st.button( 'Clear Messages', key='clear_image_edit',
						on_click=clear_image_messages ):
					st.rerun( )

# ======================================================================================
# AUDIO MODE
# ======================================================================================
elif mode == 'Audio':
	ensure_audio_mode_state( )
	transcriber = Transcription( )
	translator = Translation( )
	tts = TTS( )
	
	# ------------------------------------------------------------------
	# Session State
	# ------------------------------------------------------------------
	if st.session_state.get( 'clear_instructions' ):
		st.session_state[ 'audio_system_instructions' ] = ''
		st.session_state[ 'clear_instructions' ] = False
	
	if 'audio_tts_input' not in st.session_state:
		st.session_state[ 'audio_tts_input' ] = ''
	
	if not isinstance( st.session_state.get( 'audio_messages' ), list ):
		st.session_state.audio_messages = [ ]
	
	if not isinstance( st.session_state.get( 'audio_include' ), list ):
		st.session_state[ 'audio_include' ] = [ ]
	
	if not isinstance( st.session_state.get( 'audio_speed' ), float ):
		st.session_state[ 'audio_speed' ] = 1.0
	
	if not isinstance( st.session_state.get( 'audio_output' ), str ):
		st.session_state[ 'audio_output' ] = ''
	
	if 'audio_output_bytes' not in st.session_state:
		st.session_state[ 'audio_output_bytes' ] = None
	
	if not isinstance( st.session_state.get( 'audio_last_result' ), dict ):
		st.session_state[ 'audio_last_result' ] = { }
	
	if not isinstance( st.session_state.get( 'audio_last_usage' ), dict ):
		st.session_state[ 'audio_last_usage' ] = { }
	
	# ------------------------------------------------------------------
	# Main UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with center:
		st.subheader( '🎧 Audio API', help=getattr( cfg, 'AUDIO_API',
				'OpenAI audio transcription, translation, and text-to-speech workflows.' ) )
		st.divider( )
		
		with st.expander( label='Mind Controls', icon='🧠', expanded=False, width='stretch' ):
			with st.expander( label='LLM Options', icon='🧊', expanded=False, width='stretch' ):
				llm_c1, llm_c2, llm_c3, llm_c4 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall' )
				
				# ---------- Task ------------
				with llm_c1:
					audio_task = st.selectbox( label='Task',
						options=get_audio_task_options( ), key='audio_task',
						help='Select the Audio API workflow to run.',
						index=None, placeholder='Options' )
					
					audio_task = st.session_state.get( 'audio_task', '' )
				
				# Normalize dependent widget-backed values before dependent widgets exist.
				model_options = get_audio_model_options(
					audio_task, transcriber, translator, tts )
				if st.session_state.get( 'audio_model' ) not in model_options:
					st.session_state[ 'audio_model' ] = ''
				
				format_options = get_audio_response_format_options(
					audio_task, st.session_state.get( 'audio_model' ),
					transcriber, translator, tts )
				if st.session_state.get( 'audio_response_format' ) not in format_options:
					st.session_state[ 'audio_response_format' ] = ''
				
				include_options = get_audio_include_options(
					audio_task, st.session_state.get( 'audio_model' ), transcriber )
				if len( include_options ) == 0:
					st.session_state[ 'audio_include' ] = [ ]
				else:
					st.session_state[ 'audio_include' ] = [
							value for value in st.session_state.get( 'audio_include', [ ] )
							if value in include_options
					]
				
				language_options = get_audio_language_options( transcriber )
				if st.session_state.get( 'audio_language' ) not in language_options:
					st.session_state[ 'audio_language' ] = ''
				
				voice_options = get_audio_voice_options( tts )
				if st.session_state.get( 'audio_voice' ) not in voice_options:
					st.session_state[ 'audio_voice' ] = ''
				
				speed_options = get_audio_speed_options( tts )
				if st.session_state.get( 'audio_speed' ) not in speed_options:
					st.session_state[ 'audio_speed' ] = 1.0
				
				# ---------- Model ------------
				with llm_c2:
					audio_model = st.selectbox( label='Model',
						options=model_options, key='audio_model',
						help='Task-aware OpenAI Audio API model.',
						index=None, placeholder='Options' )
					
					audio_model = st.session_state.get( 'audio_model', '' )
				
				# Recompute format/include after model selection.
				format_options = get_audio_response_format_options(
					audio_task, audio_model, transcriber, translator, tts )
				if st.session_state.get( 'audio_response_format' ) not in format_options:
					st.session_state[ 'audio_response_format' ] = ''
				
				include_options = get_audio_include_options(
					audio_task, audio_model, transcriber )
				if len( include_options ) == 0:
					st.session_state[ 'audio_include' ] = [ ]
				else:
					st.session_state[ 'audio_include' ] = [
							value for value in st.session_state.get( 'audio_include', [ ] )
							if value in include_options
					]
				
				# ---------- Language ------------
				with llm_c3:
					if audio_task == 'Transcribe':
						audio_language = st.selectbox( label='Language',
							options=language_options, key='audio_language',
							help='Optional source-language hint for transcription.',
							index=None, placeholder='Options' )
					else:
						st.caption( 'Language' )
						st.info( 'Only used as a transcription source-language hint.' )
						audio_language = st.session_state.get( 'audio_language', '' )
				
				# ---------- Format ------------
				with llm_c4:
					audio_response_format = st.selectbox( label='Format',
						options=format_options, key='audio_response_format',
						help='Task-aware response format or TTS audio output format.',
						index=None, placeholder='Options' )
					
					audio_response_format = st.session_state.get( 'audio_response_format', '' )
				
				# ---------- Reset ------------
				st.button( label='Reset', key='reset_audio_task',
					width='stretch', on_click=reset_audio_task_controls )
			
			with st.expander( label='Inference Options', icon='🎛️',
					expanded=False, width='stretch' ):
				inf_c1, inf_c2, inf_c3, inf_c4 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall' )
				
				# ---------- Temperature ------------
				with inf_c1:
					audio_temperature = st.slider( label='Temperature',
						min_value=0.0, max_value=1.0, step=0.01,
						help=('Used by Whisper transcription/translation paths where supported. '
						      'Ignored by TTS.'),
						key='audio_temperature' )
				
				# ---------- Include ------------
				with inf_c2:
					if len( include_options ) > 0:
						audio_include = st.multiselect( label='Include',
							options=include_options, key='audio_include',
							help='Optional transcription include fields.',
							placeholder='Options' )
					else:
						st.caption( 'Include' )
						st.info( 'No include options for the selected task/model.' )
						audio_include = st.session_state.get( 'audio_include', [ ] )
				
				# ---------- Stream ------------
				with inf_c3:
					audio_stream = st.toggle( label='Stream',
						key='audio_stream', help=getattr( cfg, 'STREAM',
							'Streaming is retained but not sent until stream rendering is implemented.' ) )
					
					if audio_stream:
						st.caption( 'Streaming is not sent until stream-event rendering is added.' )
				
				# ---------- Background ------------
				with inf_c4:
					audio_background = st.toggle( label='Background',
						key='audio_background', help=getattr( cfg, 'BACKGROUND_MODE',
							'Background mode is retained but not sent for Audio API calls.' ) )
					
					if audio_background:
						st.caption( 'Background mode is not sent for these Audio API calls.' )
				
				# ---------- Reset ------------
				if st.button( label='Reset', key='reset_audio_inference',
						width='stretch' ):
					for key in [ 'audio_temperature', 'audio_include',
					             'audio_stream', 'audio_background' ]:
						if key in st.session_state:
							del st.session_state[ key ]
					
					st.rerun( )
			
			with st.expander( label='Sound Options', icon='🔊',
					expanded=False, width='stretch' ):
				snd_c1, snd_c2, snd_c3, snd_c4, snd_c5 = st.columns(
					[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
				
				# ---------- Voice ------------
				with snd_c1:
					if audio_task == 'Text-to-Speech':
						audio_voice = st.selectbox( label='Voice',
							options=voice_options, key='audio_voice',
							help='Text-to-speech voice.',
							index=None, placeholder='Options' )
					else:
						st.caption( 'Voice' )
						st.info( 'Only used by Text-to-Speech.' )
						audio_voice = st.session_state.get( 'audio_voice', '' )
				
				# ---------- Speed ------------
				with snd_c2:
					if audio_task == 'Text-to-Speech':
						audio_speed = st.select_slider( label='Speed',
							options=speed_options, key='audio_speed',
							help='Text-to-speech playback speed.' )
					else:
						st.caption( 'Speed' )
						st.info( 'Only used by Text-to-Speech.' )
						audio_speed = st.session_state.get( 'audio_speed', 1.0 )
				
				# ---------- Start Time ------------
				with snd_c3:
					audio_start_time = st.slider( label='Start Time',
						min_value=0.0, max_value=600.0, step=0.5,
						help='Playback start time in seconds.',
						key='audio_start_time' )
				
				# ---------- End Time ------------
				with snd_c4:
					audio_end_time = st.slider( label='End Time',
						min_value=0.0, max_value=600.0, step=0.5,
						help='Playback end time in seconds. Zero leaves playback unconstrained.',
						key='audio_end_time' )
				
				# ---------- Loop / Autoplay ------------
				with snd_c5:
					audio_loop = st.toggle( label='Loop', key='audio_loop',
						help='Loop local audio playback when supported.' )
					
					audio_autoplay = st.toggle( label='Autoplay', key='audio_autoplay',
						help='Autoplay local audio playback when supported.' )
				
				# ---------- Reset ------------
				st.button( label='Reset', key='reset_audio_tts',
					width='stretch', on_click=reset_audio_tts_controls )
		
		with st.expander( label='System Instructions', icon='🖥️',
				expanded=False, width='stretch' ):
			in_left, in_right = st.columns( [ 0.8, 0.2 ] )
			
			prompt_names = fetch_prompt_names( cfg.DB_PATH )
			if not prompt_names:
				prompt_names = [ '' ]
			
			# ---------- System Instructions ------------
			with in_left:
				st.text_area( label='Enter Text', height=70, width='stretch',
					help=getattr( cfg, 'SYSTEM_INSTRUCTIONS',
						'Optional prompt/instructions for the selected Audio task.' ),
					key='audio_system_instructions' )
			
			# ---------- Template ------------
			with in_right:
				st.selectbox( label='Use Template', options=prompt_names,
					index=None, key='instructions',
					on_change=load_audio_instruction_template )
			
			btn_c1, btn_c2 = st.columns( [ 0.8, 0.2 ] )
			
			# ---------- Clear Instructions ------------
			with btn_c1:
				st.button( label='Clear Instructions', width='stretch',
					on_click=clear_audio_instructions )
			
			# ---------- XML <-> Markdown ------------
			with btn_c2:
				st.button( label='XML <-> Markdown', width='stretch',
					on_click=convert_audio_system_instructions )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ------------------------------------------------------------------
		# Audio Body
		# ------------------------------------------------------------------
		upload_col, record_col, playback_col = st.columns(
			[ 0.34, 0.33, 0.33 ], border=True, gap='small' )
		
		audio_input_types = sorted(
			set( transcriber.mime_options or [ ] ) | set( translator.mime_options or [ ] ) )
		
		# ------------------------------------------------------------------
		# Upload Column
		# ------------------------------------------------------------------
		with upload_col:
			st.markdown( '#### Upload Audio' )
			uploaded_audio = st.file_uploader( label='Upload Audio File',
				type=audio_input_types, accept_multiple_files=False,
				key='audio_upload_file' )
			
			if uploaded_audio is not None:
				st.audio( uploaded_audio,
					format=f'audio/{Path( uploaded_audio.name ).suffix[ 1: ]}' )
			
			if st.button( 'Process Uploaded Audio', key='process_uploaded_audio',
					width='stretch' ):
				with st.spinner( 'Processing uploaded audio…' ):
					try:
						file_path = save_audio_upload( uploaded_audio )
						result_text = run_audio_file_task(
							task=st.session_state.get( 'audio_task' ),
							file_path=file_path,
							transcriber=transcriber,
							translator=translator )
						
						if result_text:
							title = 'Transcript' if st.session_state.get(
								'audio_task' ) == 'Transcribe' else 'Translation'
							render_audio_text_result(
								title=title,
								result_text=result_text,
								result=st.session_state.get( 'audio_last_result', { } ) )
							
							st.session_state.audio_messages.append(
								{
										'role': 'assistant',
										'content': result_text,
								} )
						
						try:
							response_obj = getattr( transcriber, 'response', None )
							if st.session_state.get( 'audio_task' ) == 'Translate':
								response_obj = getattr( translator, 'response', None )
							
							update_token_counters( response_obj )
						except Exception:
							pass
					except Exception as exc:
						st.error( f'Audio processing failed: {exc}' )
		
		# ------------------------------------------------------------------
		# Recording Column
		# ------------------------------------------------------------------
		with record_col:
			st.markdown( '#### Record Audio' )
			recorded_audio = st.audio_input( label='Record Audio',
				key='audio_recording_input' )
			
			if recorded_audio is not None:
				st.audio( recorded_audio )
			
			if st.button( 'Process Recording', key='process_recorded_audio',
					width='stretch' ):
				with st.spinner( 'Processing recorded audio…' ):
					try:
						file_path = save_audio_upload( recorded_audio )
						result_text = run_audio_file_task(
							task=st.session_state.get( 'audio_task' ),
							file_path=file_path,
							transcriber=transcriber,
							translator=translator )
						
						if result_text:
							title = 'Transcript' if st.session_state.get(
								'audio_task' ) == 'Transcribe' else 'Translation'
							render_audio_text_result(
								title=title,
								result_text=result_text,
								result=st.session_state.get( 'audio_last_result', { } ) )
							
							st.session_state.audio_messages.append(
								{
										'role': 'assistant',
										'content': result_text,
								} )
						
						try:
							response_obj = getattr( transcriber, 'response', None )
							if st.session_state.get( 'audio_task' ) == 'Translate':
								response_obj = getattr( translator, 'response', None )
							
							update_token_counters( response_obj )
						except Exception:
							pass
					except Exception as exc:
						st.error( f'Recording processing failed: {exc}' )
		
		# ------------------------------------------------------------------
		# Playback / TTS Column
		# ------------------------------------------------------------------
		with playback_col:
			st.markdown( '#### Playback / Speech' )
			
			if st.session_state.get( 'audio_task' ) == 'Text-to-Speech':
				st.text_area( label='Text-to-Speech Input', height=180,
					width='stretch', key='audio_tts_input',
					help='Text that will be synthesized into speech.' )
				
				if st.button( 'Generate Speech', key='generate_audio_speech',
						width='stretch' ):
					with st.spinner( 'Generating speech…' ):
						try:
							audio_bytes = run_audio_tts_task(
								text=st.session_state.get( 'audio_tts_input' ),
								tts=tts )
							
							if audio_bytes:
								audio_format = get_audio_response_format_value(
									task='Text-to-Speech',
									selected_format=st.session_state.get(
										'audio_response_format' ),
									selected_mime_type=st.session_state.get(
										'audio_mime_type' ) )
								
								st.audio( audio_bytes, format=f'audio/{audio_format or "mp3"}' )
								st.session_state.audio_messages.append(
									{
											'role': 'assistant',
											'content': 'Generated speech returned successfully.',
									} )
							
							try:
								update_token_counters( getattr( tts, 'response', None ) )
							except Exception:
								pass
						except Exception as exc:
							st.error( f'Text-to-Speech failed: {exc}' )
			else:
				st.info( 'Select Text-to-Speech to generate speech from text.' )
			
			audio_bytes = st.session_state.get( 'audio_output_bytes' )
			if isinstance( audio_bytes, bytes ) and len( audio_bytes ) > 0:
				audio_format = get_audio_response_format_value(
					task='Text-to-Speech',
					selected_format=st.session_state.get( 'audio_response_format' ),
					selected_mime_type=st.session_state.get( 'audio_mime_type' ) )
				
				st.caption( 'Last generated audio' )
				st.audio( audio_bytes, format=f'audio/{audio_format or "mp3"}',
					start_time=float( st.session_state.get( 'audio_start_time', 0.0 ) or 0.0 ),
					loop=bool( st.session_state.get( 'audio_loop', False ) ),
					autoplay=bool( st.session_state.get( 'audio_autoplay', False ) ) )
			
			output_text = st.session_state.get( 'audio_output', '' )
			if isinstance( output_text, str ) and output_text.strip( ):
				st.caption( 'Last text output' )
				st.text_area( label='Audio Output', value=output_text,
					height=140, width='stretch', disabled=True )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ---------------------------------------------------
		#                   MESSAGES
		# ---------------------------------------------------
		if st.session_state.get( 'audio_messages' ) is not None:
			for msg in st.session_state.audio_messages:
				if not isinstance( msg, dict ):
					continue
				
				self_avatar = cfg.GIPITY if msg.get( 'role' ) == 'assistant' else ''
				with st.chat_message( msg.get( 'role', 'assistant' ), avatar=self_avatar ):
					st.markdown( msg.get( 'content', '' ) )
		
		prompt = st.chat_input( 'Enter audio generation prompt …' )
		if prompt is not None and str( prompt ).strip( ):
			prompt = str( prompt ).strip( )
			st.session_state.audio_messages.append(
				{
						'role': 'user',
						'content': prompt,
				} )
			
			if st.session_state.get( 'audio_task' ) == 'Text-to-Speech':
				with st.chat_message( 'assistant', avatar=cfg.GIPITY ):
					with st.spinner( 'Generating speech…' ):
						try:
							audio_bytes = run_audio_tts_task( text=prompt, tts=tts )
							
							if audio_bytes:
								audio_format = get_audio_response_format_value(
									task='Text-to-Speech',
									selected_format=st.session_state.get(
										'audio_response_format' ),
									selected_mime_type=st.session_state.get(
										'audio_mime_type' ) )
								
								st.audio( audio_bytes, format=f'audio/{audio_format or "mp3"}' )
								message = 'Generated speech returned successfully.'
								st.markdown( message )
								st.session_state.audio_messages.append(
									{
											'role': 'assistant',
											'content': message,
									} )
							
							try:
								update_token_counters( getattr( tts, 'response', None ) )
							except Exception:
								pass
						except Exception as exc:
							st.error( f'Text-to-Speech failed: {exc}' )
			else:
				with st.chat_message( 'assistant', avatar=cfg.GIPITY ):
					message = (
							'Audio chat input is routed to Text-to-Speech only. '
							'Use Upload Audio or Record Audio for transcription and translation.'
					)
					st.markdown( message )
					st.session_state.audio_messages.append(
						{
								'role': 'assistant',
								'content': message,
						} )
		
		# ------------------------------------------------------------------
		# Status / Usage
		# ------------------------------------------------------------------
		audio_last_usage = st.session_state.get( 'audio_last_usage', { } )
		if isinstance( audio_last_usage, dict ) and len( audio_last_usage ) > 0:
			with st.expander( label='Audio Usage', icon='📊',
					expanded=False, width='stretch' ):
				st.json( audio_last_usage )
		
		# ------------------------------------------------------------------
		# Reset Buttons
		# ------------------------------------------------------------------
		reset_c1, reset_c2 = st.columns( [ 0.5, 0.5 ] )
		with reset_c1:
			if st.button( 'Clear Messages', key='clear_audio_messages',
					width='stretch', on_click=clear_audio_messages ):
				st.rerun( )
		
		with reset_c2:
			if st.button( 'Clear Outputs', key='clear_audio_outputs',
					width='stretch', on_click=clear_audio_outputs ):
				st.rerun( )

# ======================================================================================
# DOCUMENT Q&A MODE
# ======================================================================================
elif mode == 'Document Q&A':
	ensure_docqna_mode_state( )
	
	# ------------------------------------------------------------------
	# Session State
	# ------------------------------------------------------------------
	if not isinstance( st.session_state.get( 'docqna_messages' ), list ):
		st.session_state.docqna_messages = [ ]
	
	if not isinstance( st.session_state.get( 'docqna_active_docs' ), list ):
		st.session_state[ 'docqna_active_docs' ] = [ ]
	
	if not isinstance( st.session_state.get( 'docqna_files' ), list ):
		st.session_state[ 'docqna_files' ] = [ ]
	
	if not isinstance( st.session_state.get( 'docqna_texts' ), dict ):
		st.session_state[ 'docqna_texts' ] = { }
	
	if not isinstance( st.session_state.get( 'docqna_chunks' ), list ):
		st.session_state[ 'docqna_chunks' ] = [ ]
	
	if not isinstance( st.session_state.get( 'docqna_last_hits' ), list ):
		st.session_state[ 'docqna_last_hits' ] = [ ]
	
	if not isinstance( st.session_state.get( 'docqna_last_sources' ), list ):
		st.session_state[ 'docqna_last_sources' ] = [ ]
	
	if not isinstance( st.session_state.get( 'docqna_last_answer' ), str ):
		st.session_state[ 'docqna_last_answer' ] = ''
	
	if not isinstance( st.session_state.get( 'docqna_context' ), str ):
		st.session_state[ 'docqna_context' ] = ''
	
	if st.session_state.get( 'clear_instructions' ):
		st.session_state[ 'docqna_system_instructions' ] = ''
		st.session_state[ 'clear_instructions' ] = False
	
	# ------------------------------------------------------------------
	# Main UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	with center:
		st.subheader( '📖 Document Q & A', help=getattr( cfg, 'DOCUMENT_QNA',
			'Ask questions against local uploads, OpenAI file IDs, or OpenAI vector stores.' ) )
		st.divider( )
		
		# ------------------------------------------------------------------
		# Mind Controls
		# ------------------------------------------------------------------
		with st.expander( label='Mind Controls', icon='🧠',
				expanded=False, width='stretch' ):
			# ------------------------------------------------------------------
			# Source Controls
			# ------------------------------------------------------------------
			with st.expander( label='Source Controls', icon='📚',
					expanded=False, width='stretch' ):
				source_c1, source_c2, source_c3, source_c4 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall' )
				
				# ---------- Source ------------
				with source_c1:
					source_options = get_docqna_source_options( )
					if st.session_state.get( 'docqna_source' ) not in source_options:
						st.session_state[ 'docqna_source' ] = 'Local Upload'
					
					docqna_source = st.selectbox( label='Source', options=source_options,
						key='docqna_source', help='Select the backend used for Document Q&A.',
						index=source_options.index(
							st.session_state.get( 'docqna_source', 'Local Upload' ) )
						if st.session_state.get( 'docqna_source' ) in source_options
						else None, placeholder='Options' )
				
				# ---------- Model ------------
				with source_c2:
					model_options = [ '', 'gpt-5-mini', 'gpt-5-nano',
					                  'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4o-mini' ]
					
					if st.session_state.get( 'docqna_model' ) not in model_options:
						st.session_state[ 'docqna_model' ] = ''
					
					docqna_model = st.selectbox( label='Model', options=model_options, key='docqna_model',
						help='Model used for local generated answers, file analysis, or vector stores.',
						index=None, placeholder='Options' )
				
				# ---------- Multi-Document Mode ------------
				with source_c3:
					docqna_multi_mode = st.toggle( label='Multi-Document',
						key='docqna_multi_mode',
						help='Allow multiple local uploaded documents.' )
				
				# ---------- Diagnostics ------------
				with source_c4:
					docqna_show_diagnostics = st.toggle( label='Diagnostics',
						key='docqna_show_diagnostics', help='Show retrieval and source diagnostics.' )
				
				# ---------- OpenAI File ID ------------
				st.text_input( label='OpenAI File ID', key='docqna_file_id',
					value=st.session_state.get( 'docqna_file_id', '' ),
					help='OpenAI file ID used when Source is OpenAI File ID.',
					width='stretch', placeholder='file-...' )
				
				# ---------- OpenAI Vector Store ID ------------
				st.text_input( label='OpenAI Vector Store ID', key='docqna_vector_store_id',
					value=st.session_state.get( 'docqna_vector_store_id', '' ),
					help='OpenAI vector store ID used when Source is OpenAI Vector Store ID.',
					width='stretch', placeholder='vs_...' )
				
				link_c1, link_c2 = st.columns( [ 0.50, 0.50 ] )
				
				# ---------- Use Current Files Mode ID ------------
				with link_c1:
					if st.button( 'Use Current Files ID', key='docqna_use_files_id',
							width='stretch' ):
						current_file_id = st.session_state.get( 'files_id', '' )
						if isinstance( current_file_id, str ) and current_file_id.strip( ):
							st.session_state[ 'docqna_file_id' ] = current_file_id.strip( )
							st.session_state[ 'docqna_source' ] = 'OpenAI File ID'
							st.rerun( )
						else:
							st.warning( 'No current Files mode file ID is available.' )
				
				# ---------- Use Current Vector Store ID ------------
				with link_c2:
					if st.button( 'Use Current Vector Store ID',
							key='docqna_use_vector_store_id', width='stretch' ):
						current_store_id = st.session_state.get( 'stores_id', '' )
						if isinstance( current_store_id, str ) and current_store_id.strip( ):
							st.session_state[ 'docqna_vector_store_id' ] = current_store_id.strip( )
							st.session_state[ 'docqna_source' ] = 'OpenAI Vector Store ID'
							st.rerun( )
						else:
							st.warning( 'No current Vector Stores mode store ID is available.' )
			
			# ------------------------------------------------------------------
			# Retrieval Controls
			# ------------------------------------------------------------------
			with st.expander( label='Retrieval Controls', icon='🧩', expanded=False, width='stretch' ):
				retrieval_c1, retrieval_c2, retrieval_c3 = st.columns(
					[ 0.34, 0.33, 0.33 ], border=True, gap='xxsmall' )
				
				# ---------- Top-K ------------
				with retrieval_c1:
					st.slider( label='Top-K Chunks', min_value=1, max_value=25, step=1, key='docqna_top_k',
						help='Number of local chunks or vector store results to retrieve.' )
				
				# ---------- Chunk Size ------------
				with retrieval_c2:
					try:
						current_chunk_size = int(
							st.session_state.get( 'docqna_chunk_size', 900 ) or 900 )
					except Exception:
						current_chunk_size = 900
					
					if current_chunk_size < 100:
						st.session_state[ 'docqna_chunk_size' ] = 100
					elif current_chunk_size > 5000:
						st.session_state[ 'docqna_chunk_size' ] = 5000
					
					st.slider( label='Chunk Size', min_value=100, max_value=5000, step=50,
						key='docqna_chunk_size', help='Local word-based chunk size.' )
				
				# ---------- Chunk Overlap ------------
				with retrieval_c3:
					max_overlap = max( 0, int( st.session_state.get( 'docqna_chunk_size', 900 ) or 900 ) - 1 )
					
					try:
						current_overlap = int(
							st.session_state.get( 'docqna_chunk_overlap', 150 ) or 150 )
					except Exception:
						current_overlap = 150
					
					if current_overlap < 0:
						st.session_state[ 'docqna_chunk_overlap' ] = 0
					elif current_overlap >= max_overlap:
						st.session_state[ 'docqna_chunk_overlap' ] = max( 0, max_overlap // 5 )
					
					st.slider( label='Chunk Overlap', min_value=0, max_value=max_overlap, step=25,
						key='docqna_chunk_overlap',
						help='Local word overlap between adjacent chunks.' )
				
				action_c1, action_c2, action_c3 = st.columns( [ 0.34, 0.33, 0.33 ] )
				
				# ---------- Rebuild Local Index ------------
				with action_c1:
					if st.button( 'Rebuild Local Index', key='docqna_rebuild_index',
							width='stretch' ):
						with st.spinner( 'Rebuilding local document index…' ):
							try:
								chunks = rebuild_docqna_index( )
								if len( chunks ) > 0:
									st.success( f'Rebuilt index with {len( chunks )} chunk(s).' )
								else:
									st.warning( 'No chunks were produced.' )
							except Exception as exc:
								st.error( f'Rebuild index failed: {exc}' )
				
				# ---------- Summarize Active Source ------------
				with action_c2:
					if st.button( 'Summarize Active Source', key='docqna_summarize_source',
							width='stretch' ):
						with st.spinner( 'Summarizing active source…' ):
							try:
								answer = summarize_active_document( )
								
								if isinstance( answer, str ) and answer.strip( ):
									st.session_state.docqna_messages.append(
										{
												'role': 'assistant',
												'content': answer.strip( ),
										} )
									st.success( 'Summary generated.' )
							except Exception as exc:
								st.error( f'Summary failed: {exc}' )
				
				# ---------- Clear Outputs ------------
				with action_c3:
					st.button( label='Clear Outputs', key='docqna_clear_outputs',
						width='stretch', on_click=clear_docqna_outputs )
			
			# ------------------------------------------------------------------
			# Generation Controls
			# ------------------------------------------------------------------
			with st.expander( label='Generation Controls', icon='🎛️', expanded=False, width='stretch' ):
				gen_c1, gen_c2, gen_c3, gen_c4 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall' )
				
				# ---------- Temperature ------------
				with gen_c1:
					if 'docqna_temperature' not in st.session_state:
						st.session_state[ 'docqna_temperature' ] = 0.2
					
					st.slider( label='Temperature', min_value=0.0, max_value=2.0, step=0.05,
						key='docqna_temperature', help='Sampling temperature' )
				
				# ---------- Top-P ------------
				with gen_c2:
					if 'docqna_top_percent' not in st.session_state:
						st.session_state[ 'docqna_top_percent' ] = 1.0
					
					st.slider( label='Top-P',
						min_value=0.0, max_value=1.0, step=0.05,
						key='docqna_top_percent',
						help='Nucleus sampling value used by the generated-answer path.' )
				
				# ---------- Max Tokens ------------
				with gen_c3:
					if 'docqna_max_tokens' not in st.session_state:
						st.session_state[ 'docqna_max_tokens' ] = 2000
					
					st.slider( label='Max Tokens', min_value=256, max_value=16000, step=256,
						key='docqna_max_tokens', help='Maximum output tokens.' )
				
				# ---------- Reasoning ------------
				with gen_c4:
					if 'docqna_reasoning' not in st.session_state:
						st.session_state[ 'docqna_reasoning' ] = ''
					
					st.selectbox( label='Reasoning',
						options=[ '', 'minimal', 'low', 'medium', 'high' ], key='docqna_reasoning',
						help='Reserved for compatible models and wrappers.',
						index=None, placeholder='Options' )
			
			# ------------------------------------------------------------------
			# Reset Controls
			# ------------------------------------------------------------------
			reset_controls_c1, reset_controls_c2 = st.columns( [ 0.50, 0.50 ] )
			
			with reset_controls_c1:
				st.button( label='Reset Controls',
					key='docqna_reset_controls',
					width='stretch',
					on_click=reset_docqna_controls )
			
			with reset_controls_c2:
				st.button( label='Unload Documents', key='docqna_unload_documents',
					width='stretch', on_click=unload_docqna_documents )
		
		# ------------------------------------------------------------------
		# System Instructions
		# ------------------------------------------------------------------
		with st.expander( label='System Instructions', icon='🖥️', expanded=False, width='stretch' ):
			in_left, in_right = st.columns( [ 0.8, 0.2 ] )
			
			prompt_names = fetch_prompt_names( cfg.DB_PATH )
			if not prompt_names:
				prompt_names = [ '' ]
			
			# ---------- System Instructions ------------
			with in_left:
				st.text_area( label='Enter Text', height=90, width='stretch',
					help=getattr( cfg, 'SYSTEM_INSTRUCTIONS',
						'Optional instructions used by Document Q&A answering workflows.' ),
					key='docqna_system_instructions' )
			
			# ---------- Template ------------
			with in_right:
				st.selectbox( label='Use Template', options=prompt_names, index=None,
					key='instructions',
					on_change=load_docqna_instruction_template )
			
			btn_c1, btn_c2 = st.columns( [ 0.8, 0.2 ] )
			
			# ---------- Clear Instructions ------------
			with btn_c1:
				st.button( label='Clear Instructions',
					width='stretch',
					on_click=clear_docqna_instructions )
			
			# ---------- XML <-> Markdown ------------
			with btn_c2:
				st.button( label='XML <-> Markdown', width='stretch',
					on_click=convert_docqna_system_instructions )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ------------------------------------------------------------------
		# Document Loading and Status
		# ------------------------------------------------------------------
		load_col, status_col = st.columns( [ 0.40, 0.60 ], border=True, gap='small' )
		
		# ------------------------------------------------------------------
		# Document Loading
		# ------------------------------------------------------------------
		with load_col:
			st.markdown( '#### Document Loading' )
			
			accepted_types = [ 'pdf', 'txt', 'md', 'docx', 'csv', 'json', 'xml',
			                   'py', 'cs', 'sql', 'yaml', 'yml', 'html', 'css', 'js', 'ts' ]
			
			uploaded = st.file_uploader( label='Upload Document', type=accepted_types,
				accept_multiple_files=bool( st.session_state.get( 'docqna_multi_mode', False ) ),
				key='docqna_upload_widget', help='Upload one or more local documents' )
			
			if uploaded is not None:
				try:
					active_docs = load_docqna_uploaded_files( uploaded )
					if len( active_docs ) > 0:
						st.success( f'Loaded {len( active_docs )} document(s).' )
						if st.session_state.get( 'docqna_source' ) == 'Local Upload':
							try:
								if not st.session_state.get( 'docqna_vec_ready', False ):
									rebuild_docqna_index( )
							except Exception:
								pass
					else:
						st.warning( 'No readable document bytes were loaded.' )
				except Exception as exc:
					st.error( f'Document loading failed: {exc}' )
			
			names = get_docqna_active_document_names( )
			if len( names ) > 0:
				st.caption( 'Active documents: ' + ', '.join( names ) )
			else:
				st.info( 'No local document is currently loaded.' )
			
			preview_c1, preview_c2 = st.columns( [ 0.50, 0.50 ] )
			
			# ---------- Preview Documents ------------
			with preview_c1:
				if st.button( 'Preview Documents', key='docqna_preview_documents',
						width='stretch' ):
					st.session_state[ 'docqna_show_preview' ] = True
			
			# ---------- Hide Preview ------------
			with preview_c2:
				if st.button( 'Hide Preview', key='docqna_hide_preview',
						width='stretch' ):
					st.session_state[ 'docqna_show_preview' ] = False
			
			if st.session_state.get( 'docqna_show_preview', True ):
				render_docqna_document_preview( )
		
		# ------------------------------------------------------------------
		# Status / Diagnostics
		# ------------------------------------------------------------------
		with status_col:
			st.markdown( '#### Document Status' )
			render_docqna_status( )
			
			st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
			
			source = st.session_state.get( 'docqna_source', 'Local Upload' )
			
			if source == 'OpenAI File ID':
				file_id = st.session_state.get( 'docqna_file_id', '' )
				if isinstance( file_id, str ) and file_id.strip( ):
					st.success( f'OpenAI File ID selected: {file_id.strip( )}' )
				else:
					st.warning( 'OpenAI File ID source selected, but no file ID is set.' )
			
			elif source == 'OpenAI Vector Store ID':
				store_id = st.session_state.get( 'docqna_vector_store_id', '' )
				if isinstance( store_id, str ) and store_id.strip( ):
					st.success( f'OpenAI Vector Store ID selected: {store_id.strip( )}' )
				else:
					st.warning(
						'OpenAI Vector Store ID source selected, but no vector store ID is set.' )
			
			else:
				if st.session_state.get( 'docqna_vec_ready', False ):
					st.success( 'Local document index is ready.' )
				else:
					st.warning( 'Local document index is not ready.' )
			
			if st.session_state.get( 'docqna_show_diagnostics', True ):
				with st.expander( label='Retrieval Diagnostics', icon='🔎',
						expanded=False, width='stretch' ):
					st.write(
						{
								'source': st.session_state.get( 'docqna_source', 'Local Upload' ),
								'index_status': st.session_state.get( 'docqna_index_status',
									'Not indexed' ),
								'chunk_count': st.session_state.get( 'docqna_chunk_count', 0 ),
								'fingerprint': st.session_state.get( 'docqna_fingerprint', '' ),
								'active_documents': get_docqna_active_document_names( ),
								'file_id': st.session_state.get( 'docqna_file_id', '' ),
								'vector_store_id': st.session_state.get( 'docqna_vector_store_id',
									'' ),
						} )
					
					render_docqna_retrieval_hits( )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ------------------------------------------------------------------
		# Messages
		# ------------------------------------------------------------------
		if st.session_state.get( 'docqna_messages' ) is not None:
			for msg in st.session_state.docqna_messages:
				if not isinstance( msg, dict ):
					continue
				
				self_avatar = cfg.GIPITY if msg.get( 'role' ) == 'assistant' else ''
				with st.chat_message( msg.get( 'role', 'assistant' ), avatar=self_avatar ):
					st.markdown( msg.get( 'content', '' ) )
		
		prompt = st.chat_input( 'Ask a question about the active document source …' )
		if prompt is not None and str( prompt ).strip( ):
			prompt = str( prompt ).strip( )
			
			st.session_state.docqna_messages.append(
				{
						'role': 'user',
						'content': prompt,
				} )
			
			with st.chat_message( 'assistant', avatar=cfg.GIPITY ):
				with st.spinner( 'Answering from the active document source…' ):
					try:
						answer = route_document_query( prompt )
						
						if isinstance( answer, str ) and answer.strip( ):
							st.markdown( answer )
							st.session_state.docqna_messages.append(
								{
										'role': 'assistant',
										'content': answer.strip( ),
								} )
							st.session_state[ 'docqna_last_answer' ] = answer.strip( )
							st.session_state[ 'last_answer' ] = answer.strip( )
						else:
							message = 'No Document Q&A answer was returned.'
							st.warning( message )
							st.session_state.docqna_messages.append(
								{
										'role': 'assistant',
										'content': message,
								} )
					except Exception as exc:
						st.error( f'Document Q&A failed: {exc}' )
		
		# ------------------------------------------------------------------
		# Last Answer and Sources
		# ------------------------------------------------------------------
		last_answer = st.session_state.get( 'docqna_last_answer', '' )
		if isinstance( last_answer, str ) and last_answer.strip( ):
			with st.expander( label='Last Document Answer', icon='🧠',
					expanded=False, width='stretch' ):
				st.markdown( last_answer )
		
		last_sources = st.session_state.get( 'docqna_last_sources', [ ] )
		if isinstance( last_sources, list ) and len( last_sources ) > 0:
			with st.expander( label='Last Document Sources', icon='📌',
					expanded=False, width='stretch' ):
				df_sources = pd.DataFrame( last_sources )
				st.data_editor( df_sources, use_container_width=True, hide_index=True )
		
		# ------------------------------------------------------------------
		# Reset Buttons
		# ------------------------------------------------------------------
		reset_c1, reset_c2, reset_c3 = st.columns( [ 0.34, 0.33, 0.33 ] )
		
		with reset_c1:
			if st.button( 'Clear Messages', key='docqna_clear_messages',
					width='stretch', on_click=clear_docqna_messages ):
				st.rerun( )
		
		with reset_c2:
			if st.button( 'Clear Outputs', key='docqna_clear_mode_outputs',
					width='stretch', on_click=clear_docqna_outputs ):
				st.rerun( )
		
		with reset_c3:
			if st.button( 'Reset All', key='docqna_reset_all',
					width='stretch', on_click=reset_docqna_all ):
				st.rerun( )

# ======================================================================================
# EMBEDDINGS MODE
# ======================================================================================
elif mode == 'Embeddings':
	ensure_embeddings_mode_state( )
	embedding = Embeddings( )
	
	# ------------------------------------------------------------------
	# Session State
	# ------------------------------------------------------------------
	if not isinstance( st.session_state.get( 'embeddings_input_text' ), str ):
		st.session_state[ 'embeddings_input_text' ] = ''
	
	if not isinstance( st.session_state.get( 'embeddings_encoding_format' ), str ):
		st.session_state[ 'embeddings_encoding_format' ] = 'float'
	
	if not isinstance( st.session_state.get( 'embeddings_chunks' ), list ):
		st.session_state[ 'embeddings_chunks' ] = [ ]
	
	if not isinstance( st.session_state.get( 'embedding_metrics' ), dict ):
		st.session_state[ 'embedding_metrics' ] = { }
	
	if not isinstance( st.session_state.get( 'embedding_usage' ), dict ):
		st.session_state[ 'embedding_usage' ] = { }
	
	if 'embeddings_df' not in st.session_state or not isinstance(
			st.session_state.get( 'embeddings_df' ), pd.DataFrame ):
		st.session_state[ 'embeddings_df' ] = pd.DataFrame( )
	
	# ------------------------------------------------------------------
	# Main UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with center:
		st.subheader( '🧬 Embeddings API', help=getattr( cfg, 'EMBEDDINGS_API',
				'Create vector embeddings from text using the OpenAI Embeddings API.' ) )
		st.divider( )
		
		with st.expander( label='Configuration', icon='🧊', expanded=False, width='stretch' ):
			cfg_c1, cfg_c2, cfg_c3, cfg_c4, cfg_c5 = st.columns(
				[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
			
			# ---------- Model ------------
			with cfg_c1:
				model_options = get_embedding_model_options( embedding )
				if st.session_state.get( 'embedding_model' ) not in model_options:
					st.session_state[ 'embedding_model' ] = ''
				
				embedding_model = st.selectbox( label='Model',
					options=model_options, key='embedding_model',
					help='OpenAI embedding model.',
					index=None, placeholder='Options' )
				
				embedding_model = st.session_state.get( 'embedding_model', '' )
			
			# Normalize dimension state before the dimension slider is instantiated.
			max_dimensions = get_embedding_max_dimensions( embedding_model, embedding )
			supports_dimensions = embedding_model_supports_dimensions(
				embedding_model, embedding )
			
			try:
				current_dimensions = int( st.session_state.get( 'embeddings_dimensions', 0 ) or 0 )
			except Exception:
				current_dimensions = 0
			
			if not supports_dimensions:
				st.session_state[ 'embeddings_dimensions' ] = 0
			elif current_dimensions > max_dimensions:
				st.session_state[ 'embeddings_dimensions' ] = max_dimensions
			elif current_dimensions < 0:
				st.session_state[ 'embeddings_dimensions' ] = 0
			
			# ---------- Encoding Format ------------
			with cfg_c2:
				encoding_options = get_embedding_encoding_options( embedding )
				if st.session_state.get( 'embeddings_encoding_format' ) not in encoding_options:
					st.session_state[ 'embeddings_encoding_format' ] = 'float'
				
				embeddings_encoding_format = st.selectbox( label='Encoding Format',
					options=encoding_options, key='embeddings_encoding_format',
					help='Embedding encoding format returned by the API.',
					index=None, placeholder='Options' )
				
				embeddings_encoding_format = st.session_state.get(
					'embeddings_encoding_format', 'float' )
			
			# ---------- Dimensions ------------
			with cfg_c3:
				embeddings_dimensions = st.slider( label='Dimensions',
					min_value=0, max_value=max_dimensions, step=1,
					help=('Optional reduced dimensions for text-embedding-3 models. '
					      'Zero omits the dimensions parameter.'),
					key='embeddings_dimensions',
					disabled=not supports_dimensions )
				
				embeddings_dimensions = st.session_state.get( 'embeddings_dimensions', 0 )
				
				if not supports_dimensions:
					st.caption( 'Dimensions are omitted for this model.' )
			
			# ---------- Chunk Size ------------
			with cfg_c4:
				try:
					current_chunk_size = int( st.session_state.get( 'embeddings_chunk_size', 800 ) )
				except Exception:
					current_chunk_size = 800
				
				if current_chunk_size <= 0:
					st.session_state[ 'embeddings_chunk_size' ] = 800
				elif current_chunk_size > 8192:
					st.session_state[ 'embeddings_chunk_size' ] = 8192
				
				embeddings_chunk_size = st.slider( label='Chunk Size', min_value=1, max_value=8192,
					step=50, help='Maximum chunk size in tokenizer tokens.',
					key='embeddings_chunk_size' )
				
				embeddings_chunk_size = st.session_state.get( 'embeddings_chunk_size', 800 )
			
			# ---------- Overlap Amount ------------
			with cfg_c5:
				try:
					current_overlap = int(
						st.session_state.get( 'embeddings_overlap_amount', 0 ) or 0 )
				except Exception:
					current_overlap = 0
				
				if current_overlap < 0:
					st.session_state[ 'embeddings_overlap_amount' ] = 0
				elif current_overlap >= int( st.session_state.get( 'embeddings_chunk_size', 800 ) ):
					st.session_state[ 'embeddings_overlap_amount' ] = max(
						0, int( st.session_state.get( 'embeddings_chunk_size', 800 ) ) // 5 )
				
				embeddings_overlap_amount = st.slider( label='Overlap Amount', min_value=0,
					max_value=max( 10, int( st.session_state.get( 'embeddings_chunk_size', 800 ) ) - 1 ),
					step=10, help='Token overlap between adjacent embedding chunks.',
					key='embeddings_overlap_amount' )
				
				embeddings_overlap_amount = st.session_state.get(
					'embeddings_overlap_amount', 0 )
			
			# ---------- Optional User ------------
			st.text_input( label='User Identifier', key='embeddings_user',
				value=st.session_state.get( 'embeddings_user', '' ),
				help='Optional OpenAI user identifier for abuse monitoring.',
				width='stretch', placeholder='Optional user identifier' )
			
			btn_cfg1, btn_cfg2 = st.columns( [ 0.5, 0.5 ] )
			with btn_cfg1:
				st.button( label='Reset Configuration', key='reset_embeddings_config',
					width='stretch', on_click=reset_embeddings_controls )
			
			with btn_cfg2:
				st.button( label='Clear Output', key='clear_embeddings_output',
					width='stretch', on_click=clear_embeddings_output )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ------------------------------------------------------------------
		# Input
		# ------------------------------------------------------------------
		st.text_area( label='Input Text', height=260, width='stretch',
			key='embeddings_input_text',
			help='Text to normalize, chunk, and submit to the OpenAI Embeddings API.',
			placeholder='Enter text to embed...' )
		
		action_c1, action_c2 = st.columns( [ 0.5, 0.5 ] )
		
		# ------------------------------------------------------------------
		# Create Embeddings
		# ------------------------------------------------------------------
		with action_c1:
			if st.button( 'Create Embeddings', key='create_embeddings',
					width='stretch' ):
				with st.spinner( 'Creating embeddings…' ):
					try:
						source_text = st.session_state.get( 'embeddings_input_text', '' )
						model = st.session_state.get( 'embedding_model' ) or \
						        'text-embedding-3-small'
						encoding_format = st.session_state.get(
							'embeddings_encoding_format' ) or 'float'
						
						if not isinstance( source_text, str ) or not source_text.strip( ):
							st.warning( 'Enter text before creating embeddings.' )
						else:
							normalized_text = normalize_text( source_text )
							chunk_size, overlap_amount = normalize_embedding_chunk_settings(
								chunk_size=st.session_state.get( 'embeddings_chunk_size', 800 ),
								overlap_amount=st.session_state.get(
									'embeddings_overlap_amount', 0 ) )
							
							chunks = chunk_text_for_embeddings( text=normalized_text,
								chunk_size=chunk_size, overlap_amount=overlap_amount )
							
							if len( chunks ) == 0:
								st.warning( 'No valid chunks were produced from the input text.' )
							else:
								dimensions = normalize_embedding_dimensions(
									model=model,
									dimensions=st.session_state.get( 'embeddings_dimensions', 0 ),
									embedding=embedding )
								
								user_value = st.session_state.get( 'embeddings_user', '' )
								user_value = user_value.strip( ) if isinstance(
									user_value, str ) and user_value.strip( ) else None
								
								vectors = embedding.create(
									text=chunks,
									model=model,
									format=encoding_format,
									dimensions=dimensions,
									user=user_value )
								
								usage = extract_embedding_usage(
									getattr( embedding, 'response', None ) )
								
								df_embeddings = build_embeddings_dataframe( chunks=chunks,
									vectors=vectors, encoding_format=encoding_format )
								
								metrics = build_embedding_metrics( source_text=source_text,
									normalized_text=normalized_text, chunks=chunks,
									vectors=vectors, usage=usage )
								
								st.session_state[ 'embeddings' ] = normalize_embedding_vectors(
									vectors )
								st.session_state[ 'embeddings_chunks' ] = chunks
								st.session_state[ 'embeddings_df' ] = df_embeddings
								st.session_state[ 'embedding_metrics' ] = metrics
								st.session_state[ 'embedding_usage' ] = usage
								
								try:
									update_token_counters( getattr( embedding, 'response', None ) )
								except Exception:
									pass
								
								st.success( 'Embeddings created successfully.' )
					except Exception as exc:
						err = Error( exc )
						st.error( f'Embedding creation failed: {err.info}' )
		
		# ------------------------------------------------------------------
		# Reset All
		# ------------------------------------------------------------------
		with action_c2:
			if st.button( 'Reset All', key='reset_embeddings_all',
					width='stretch', on_click=reset_embeddings_all ):
				st.rerun( )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ------------------------------------------------------------------
		# Metrics
		# ------------------------------------------------------------------
		metrics = st.session_state.get( 'embedding_metrics', { } )
		if isinstance( metrics, dict ) and len( metrics ) > 0:
			render_embedding_metrics( metrics )
		
		# ------------------------------------------------------------------
		# Embeddings Output
		# ------------------------------------------------------------------
		df_embeddings = st.session_state.get( 'embeddings_df', pd.DataFrame( ) )
		if isinstance( df_embeddings, pd.DataFrame ) and not df_embeddings.empty:
			st.markdown( '#### Embedding Output' )
			render_embeddings_dataframe( df_embeddings )
		
		# ------------------------------------------------------------------
		# Chunks
		# ------------------------------------------------------------------
		chunks = st.session_state.get( 'embeddings_chunks', [ ] )
		if isinstance( chunks, list ) and len( chunks ) > 0:
			with st.expander( label='Chunks', icon='🧩',
					expanded=False, width='stretch' ):
				df_chunks = pd.DataFrame(
					[
							{
									'ChunkIndex': index + 1,
									'Text': chunk,
									'Tokens': count_tokens( chunk ),
							}
							for index, chunk in enumerate( chunks )
					] )
				st.data_editor( df_chunks, use_container_width=True, hide_index=True )
		
		# ------------------------------------------------------------------
		# Usage
		# ------------------------------------------------------------------
		usage = st.session_state.get( 'embedding_usage', { } )
		if isinstance( usage, dict ) and len( usage ) > 0:
			with st.expander( label='Embedding Usage', icon='📊',
					expanded=False, width='stretch' ):
				st.json( usage )

# ======================================================================================
# FILES MODE
# ======================================================================================
elif mode == 'Files':
	ensure_files_mode_state( )
	files = Files( )
	
	# ------------------------------------------------------------------
	# Session State
	# ------------------------------------------------------------------
	if 'files_manual_id' not in st.session_state:
		st.session_state[ 'files_manual_id' ] = ''
	
	if 'files_selected_label' not in st.session_state:
		st.session_state[ 'files_selected_label' ] = ''
	
	if not isinstance( st.session_state.get( 'files_table' ), list ):
		st.session_state[ 'files_table' ] = [ ]
	
	if not isinstance( st.session_state.get( 'files_metadata' ), dict ):
		st.session_state[ 'files_metadata' ] = { }
	
	if not isinstance( st.session_state.get( 'files_delete_result' ), dict ):
		st.session_state[ 'files_delete_result' ] = { }
	
	if not isinstance( st.session_state.get( 'files_last_answer' ), str ):
		st.session_state[ 'files_last_answer' ] = ''
	
	if not isinstance( st.session_state.get( 'files_messages' ), list ):
		st.session_state.files_messages = [ ]
	
	if st.session_state.get( 'clear_instructions' ):
		st.session_state[ 'files_system_instructions' ] = ''
		st.session_state[ 'clear_instructions' ] = False
	
	# ------------------------------------------------------------------
	# Main UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	with center:
		st.subheader( '📁 Files API', help=getattr( cfg, 'FILES_API',
			'Upload, list, retrieve, inspect, and delete OpenAI Files API files.' ) )
		st.divider( )
		
		with st.expander( label='Mind Controls', icon='🧠', expanded=False, width='stretch' ):
			# ------------------------------------------------------------------
			# File Management Controls
			# ------------------------------------------------------------------
			with st.expander( label='File Management', icon='📂',
					expanded=False, width='stretch' ):
				mgmt_c1, mgmt_c2, mgmt_c3, mgmt_c4 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall' )
				
				# ---------- Upload Purpose ------------
				with mgmt_c1:
					upload_purposes = get_files_upload_purpose_options( files )
					if st.session_state.get( 'files_purpose' ) not in upload_purposes:
						st.session_state[ 'files_purpose' ] = 'user_data'
					
					files_purpose = st.selectbox( label='Upload Purpose',
						options=upload_purposes, key='files_purpose',
						help='Required OpenAI Files API upload purpose.',
						index=upload_purposes.index(
							st.session_state.get( 'files_purpose', 'user_data' ) )
						if st.session_state.get( 'files_purpose' ) in upload_purposes else None,
						placeholder='Options' )
				
				# ---------- List Purpose Filter ------------
				with mgmt_c2:
					filter_purposes = get_files_filter_purpose_options( files )
					if st.session_state.get( 'files_filter_purpose' ) not in filter_purposes:
						st.session_state[ 'files_filter_purpose' ] = ''
					
					files_filter_purpose = st.selectbox( label='List Purpose Filter',
						options=filter_purposes, key='files_filter_purpose',
						help='Optional purpose filter used when listing files.',
						index=filter_purposes.index(
							st.session_state.get( 'files_filter_purpose', '' ) )
						if st.session_state.get( 'files_filter_purpose' ) in filter_purposes
						else None,
						placeholder='Options' )
				
				# ---------- Analysis Model ------------
				with mgmt_c3:
					model_options = get_files_model_options( files )
					if st.session_state.get( 'files_model' ) not in model_options:
						st.session_state[ 'files_model' ] = ''
					
					files_model = st.selectbox( label='Analysis Model',
						options=model_options, key='files_model',
						help='Optional model used for selected-file analysis.',
						index=None, placeholder='Options' )
				
				# ---------- File Type ------------
				with mgmt_c4:
					files_type = st.selectbox( label='File Type',
						options=[ '', 'metadata', 'content', 'analysis' ],
						key='files_type',
						help='Optional local UI classification for the selected file workflow.',
						index=None, placeholder='Options' )
				
				# ---------- Manual File ID ------------
				st.text_input( label='Manual File ID',
					key='files_manual_id',
					value=st.session_state.get( 'files_manual_id', '' ),
					help='Optional direct OpenAI file ID. Use this if the file is not in the current table.',
					width='stretch', placeholder='file-...' )
				
				# ---------- Reset ------------
				st.button( label='Reset Controls', key='reset_files_controls',
					width='stretch', on_click=reset_files_controls )
			
			# ------------------------------------------------------------------
			# Current File Selection
			# ------------------------------------------------------------------
			with st.expander( label='Current File', icon='🧾',
					expanded=False, width='stretch' ):
				file_rows = st.session_state.get( 'files_table', [ ] )
				selection_options = build_file_selection_options( file_rows )
				selection_labels = [ '' ] + list( selection_options.keys( ) )
				
				if st.session_state.get( 'files_selected_label' ) not in selection_labels:
					st.session_state[ 'files_selected_label' ] = ''
				
				selected_label = st.selectbox( label='Selected File',
					options=selection_labels, key='files_selected_label',
					help='Select a file from the latest file list. The displayed label maps to the file ID.',
					index=selection_labels.index(
						st.session_state.get( 'files_selected_label', '' ) )
					if st.session_state.get( 'files_selected_label' ) in selection_labels else None,
					placeholder='Options' )
				
				selected_from_table = get_selected_file_id(
					selected_label=selected_label,
					options=selection_options )
				
				manual_id = st.session_state.get( 'files_manual_id', '' )
				selected_file_id = selected_from_table or (
						manual_id.strip( ) if isinstance( manual_id, str ) and manual_id.strip( )
						else st.session_state.get( 'files_id', '' ))
				
				if selected_file_id:
					st.caption( f'Selected File ID: `{selected_file_id}`' )
				else:
					st.caption( 'No file selected.' )
				
				st.text_input( label='Selected File ID',
					value=selected_file_id or '',
					disabled=True,
					help='Resolved file ID used by Retrieve, Content, Delete, and Analyze actions.',
					key='files_selected_id_display',
					width='stretch' )
			
			# ------------------------------------------------------------------
			# File Actions
			# ------------------------------------------------------------------
			with st.expander( label='File Actions', icon='⚙️',
					expanded=False, width='stretch' ):
				action_c1, action_c2, action_c3, action_c4 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall' )
				
				# ---------- List Files ------------
				with action_c1:
					if st.button( 'List Files', key='list_openai_files',
							width='stretch' ):
						with st.spinner( 'Listing files…' ):
							try:
								filter_value = st.session_state.get( 'files_filter_purpose', '' )
								rows = run_files_list(
									files=files,
									purpose=filter_value if filter_value else None )
								
								if len( rows ) > 0:
									st.success( f'Listed {len( rows )} file(s).' )
								else:
									st.info( 'No files were returned.' )
							except Exception as exc:
								st.error( f'List files failed: {exc}' )
				
				# ---------- Retrieve Metadata ------------
				with action_c2:
					if st.button( 'Retrieve Metadata', key='retrieve_openai_file',
							width='stretch' ):
						with st.spinner( 'Retrieving file metadata…' ):
							try:
								metadata = run_files_retrieve(
									files=files,
									file_id=selected_file_id )
								
								if metadata:
									st.success( 'File metadata retrieved.' )
							except Exception as exc:
								st.error( f'Retrieve metadata failed: {exc}' )
				
				# ---------- Retrieve Content ------------
				with action_c3:
					if st.button( 'Retrieve Content', key='retrieve_openai_file_content',
							width='stretch' ):
						with st.spinner( 'Retrieving file content…' ):
							try:
								content = run_files_extract(
									files=files,
									file_id=selected_file_id )
								
								if content is not None:
									st.success( 'File content retrieved.' )
							except Exception as exc:
								st.error( f'Retrieve content failed: {exc}' )
				
				# ---------- Delete File ------------
				with action_c4:
					if st.button( 'Delete File', key='delete_openai_file',
							width='stretch' ):
						with st.spinner( 'Deleting file…' ):
							try:
								result = run_files_delete(
									files=files,
									file_id=selected_file_id )
								
								if result.get( 'deleted' ) is True:
									st.success( 'File deleted.' )
									try:
										rows = run_files_list(
											files=files,
											purpose=st.session_state.get(
												'files_filter_purpose' ) or None )
										st.session_state[ 'files_table' ] = rows
									except Exception:
										pass
							except Exception as exc:
								st.error( f'Delete file failed: {exc}' )
				
				# ---------- Clear Outputs ------------
				st.button( label='Clear Outputs', key='clear_files_outputs',
					width='stretch', on_click=clear_files_outputs )
		
		# ------------------------------------------------------------------
		# System Instructions
		# ------------------------------------------------------------------
		with st.expander( label='System Instructions', icon='🖥️',
				expanded=False, width='stretch' ):
			in_left, in_right = st.columns( [ 0.8, 0.2 ] )
			
			prompt_names = fetch_prompt_names( cfg.DB_PATH )
			if not prompt_names:
				prompt_names = [ '' ]
			
			# ---------- System Instructions ------------
			with in_left:
				st.text_area( label='Enter Text',
					height=70, width='stretch',
					help=getattr( cfg, 'SYSTEM_INSTRUCTIONS',
						'Optional instructions used for selected-file analysis.' ),
					key='files_system_instructions' )
			
			# ---------- Template ------------
			with in_right:
				st.selectbox( label='Use Template',
					options=prompt_names, index=None,
					key='instructions',
					on_change=load_files_instruction_template )
			
			btn_c1, btn_c2 = st.columns( [ 0.8, 0.2 ] )
			
			# ---------- Clear Instructions ------------
			with btn_c1:
				st.button( label='Clear Instructions',
					width='stretch', on_click=clear_files_instructions )
			
			# ---------- XML <-> Markdown ------------
			with btn_c2:
				st.button( label='XML <-> Markdown',
					width='stretch', on_click=convert_files_system_instructions )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ------------------------------------------------------------------
		# Body: Upload, Table, Details
		# ------------------------------------------------------------------
		upload_col, table_col, detail_col = st.columns(
			[ 0.30, 0.40, 0.30 ], border=True, gap='small' )
		
		# ------------------------------------------------------------------
		# Upload Column
		# ------------------------------------------------------------------
		with upload_col:
			st.markdown( '#### Upload File' )
			
			uploaded_file = st.file_uploader( label='Select File',
				accept_multiple_files=False,
				key='files_upload_file',
				help='Select a local file to upload to the OpenAI Files API.' )
			
			if uploaded_file is not None:
				st.caption( f"Selected: {getattr( uploaded_file, 'name', 'uploaded file' )}" )
				st.caption( f"Size: {getattr( uploaded_file, 'size', 0 )} bytes" )
			
			if st.button( 'Upload File', key='upload_openai_file',
					width='stretch' ):
				with st.spinner( 'Uploading file…' ):
					try:
						metadata = run_files_upload(
							files=files,
							uploaded_file=uploaded_file,
							purpose=st.session_state.get( 'files_purpose', 'user_data' ) )
						
						if metadata.get( 'id' ):
							st.success( f"Uploaded file: {metadata.get( 'id' )}" )
							
							try:
								rows = run_files_list(
									files=files,
									purpose=st.session_state.get(
										'files_filter_purpose' ) or None )
								st.session_state[ 'files_table' ] = rows
							except Exception:
								pass
					except Exception as exc:
						st.error( f'Upload failed: {exc}' )
		
		# ------------------------------------------------------------------
		# Table Column
		# ------------------------------------------------------------------
		with table_col:
			st.markdown( '#### Files' )
			
			rows = st.session_state.get( 'files_table', [ ] )
			render_files_table( rows )
		
		# ------------------------------------------------------------------
		# Details Column
		# ------------------------------------------------------------------
		with detail_col:
			st.markdown( '#### Selected File Details' )
			
			metadata = st.session_state.get( 'files_metadata', { } )
			if isinstance( metadata, dict ) and len( metadata ) > 0:
				render_file_metadata( metadata )
			else:
				st.info( 'Retrieve metadata to inspect a selected file.' )
			
			delete_result = st.session_state.get( 'files_delete_result', { } )
			if isinstance( delete_result, dict ) and len( delete_result ) > 0:
				render_file_delete_result( delete_result )
		
		# ------------------------------------------------------------------
		# Content Output
		# ------------------------------------------------------------------
		content_value = st.session_state.get( 'files_content', '' )
		content_bytes = st.session_state.get( 'files_content_bytes', None )
		
		if isinstance( content_bytes, bytes ) and len( content_bytes ) > 0:
			with st.expander( label='File Content', icon='📄',
					expanded=False, width='stretch' ):
				render_file_content( content_bytes )
		elif isinstance( content_value, str ) and content_value.strip( ):
			with st.expander( label='File Content', icon='📄',
					expanded=False, width='stretch' ):
				render_file_content( content_value )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ------------------------------------------------------------------
		# Messages
		# ------------------------------------------------------------------
		if st.session_state.get( 'files_messages' ) is not None:
			for msg in st.session_state.files_messages:
				if not isinstance( msg, dict ):
					continue
				
				self_avatar = cfg.GIPITY if msg.get( 'role' ) == 'assistant' else ''
				with st.chat_message( msg.get( 'role', 'assistant' ), avatar=self_avatar ):
					st.markdown( msg.get( 'content', '' ) )
		
		prompt = st.chat_input( 'Ask a question about the selected file …' )
		if prompt is not None and str( prompt ).strip( ):
			prompt = str( prompt ).strip( )
			st.session_state.files_messages.append(
				{
						'role': 'user',
						'content': prompt,
				} )
			
			with st.chat_message( 'assistant', avatar=cfg.GIPITY ):
				with st.spinner( 'Analyzing selected file…' ):
					try:
						# Re-resolve the selected file ID at execution time to avoid stale locals.
						current_rows = st.session_state.get( 'files_table', [ ] )
						current_options = build_file_selection_options( current_rows )
						current_label = st.session_state.get( 'files_selected_label', '' )
						current_selected = get_selected_file_id(
							selected_label=current_label,
							options=current_options )
						
						current_manual = st.session_state.get( 'files_manual_id', '' )
						current_file_id = current_selected or (
								current_manual.strip( ) if isinstance( current_manual, str )
								                           and current_manual.strip( ) else st.session_state.get(
									'files_id', '' ))
						
						instruction_text = st.session_state.get( 'files_system_instructions', '' )
						if isinstance( instruction_text, str ) and instruction_text.strip( ):
							analysis_prompt = f'{instruction_text.strip( )}\n\nUser Question: {prompt}'
						else:
							analysis_prompt = prompt
						
						answer = run_files_analysis(
							files=files,
							file_id=current_file_id,
							prompt=analysis_prompt,
							model=st.session_state.get( 'files_model' ) or 'gpt-4o-mini' )
						
						if isinstance( answer, str ) and answer.strip( ):
							st.markdown( answer )
							st.session_state.files_messages.append(
								{
										'role': 'assistant',
										'content': answer.strip( ),
								} )
							st.session_state[ 'files_last_answer' ] = answer.strip( )
							
							try:
								update_token_counters( getattr( files, 'response', None ) )
							except Exception:
								pass
						else:
							message = 'No file analysis response was returned.'
							st.warning( message )
							st.session_state.files_messages.append(
								{
										'role': 'assistant',
										'content': message,
								} )
					except Exception as exc:
						st.error( f'File analysis failed: {exc}' )
		
		# ------------------------------------------------------------------
		# Last Analysis Output
		# ------------------------------------------------------------------
		last_answer = st.session_state.get( 'files_last_answer', '' )
		if isinstance( last_answer, str ) and last_answer.strip( ):
			with st.expander( label='Last File Analysis', icon='🧠',
					expanded=False, width='stretch' ):
				st.markdown( last_answer )
		
		# ------------------------------------------------------------------
		# Reset Buttons
		# ------------------------------------------------------------------
		reset_c1, reset_c2, reset_c3 = st.columns( [ 0.34, 0.33, 0.33 ] )
		
		with reset_c1:
			if st.button( 'Clear Messages', key='clear_files_messages',
					width='stretch', on_click=clear_files_messages ):
				st.rerun( )
		
		with reset_c2:
			if st.button( 'Clear Outputs', key='clear_files_mode_outputs',
					width='stretch', on_click=clear_files_outputs ):
				st.rerun( )
		
		with reset_c3:
			if st.button( 'Reset All', key='reset_files_all',
					width='stretch', on_click=reset_files_all ):
				st.rerun( )

# ======================================================================================
# VECTOR STORES MODE
# ======================================================================================
elif mode == 'Vector Stores':
	ensure_vectorstores_mode_state( )
	vector = VectorStores( )
	
	# ------------------------------------------------------------------
	# Session State
	# ------------------------------------------------------------------
	if not isinstance( st.session_state.get( 'stores_table' ), list ):
		st.session_state[ 'stores_table' ] = [ ]
	
	if not isinstance( st.session_state.get( 'stores_files_table' ), list ):
		st.session_state[ 'stores_files_table' ] = [ ]
	
	if not isinstance( st.session_state.get( 'stores_store_metadata' ), dict ):
		st.session_state[ 'stores_store_metadata' ] = { }
	
	if not isinstance( st.session_state.get( 'stores_batch_result' ), dict ):
		st.session_state[ 'stores_batch_result' ] = { }
	
	if not isinstance( st.session_state.get( 'stores_search_results' ), list ):
		st.session_state[ 'stores_search_results' ] = [ ]
	
	if not isinstance( st.session_state.get( 'stores_messages' ), list ):
		st.session_state.stores_messages = [ ]
	
	if st.session_state.get( 'clear_instructions' ):
		st.session_state[ 'stores_system_instructions' ] = ''
		st.session_state[ 'clear_instructions' ] = False
	
	# ------------------------------------------------------------------
	# Main UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( '🧊 Vector Stores', help=getattr( cfg, 'VECTORSTORES_API',
			'Create, manage, search, and query OpenAI vector stores.' ) )
		st.divider( )
		
		with st.expander( label='Mind Controls', icon='🧠', expanded=False, width='stretch' ):
			# ------------------------------------------------------------------
			# Store Controls
			# ------------------------------------------------------------------
			with st.expander( label='Store Controls', icon='🗄️',
					expanded=False, width='stretch' ):
				ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall' )
				
				# ---------- Store Name ------------
				with ctrl_c1:
					st.text_input( label='Store Name',
						key='stores_name',
						value=st.session_state.get( 'stores_name', '' ),
						help='Name used when creating or updating a vector store.',
						width='stretch', placeholder='Enter store name' )
				
				# ---------- Model ------------
				with ctrl_c2:
					model_options = get_vector_store_model_options( vector )
					if st.session_state.get( 'stores_model' ) not in model_options:
						st.session_state[ 'stores_model' ] = ''
					
					st.selectbox( label='Answer Model',
						options=model_options,
						key='stores_model',
						help='Model used only for Responses API file_search answers.',
						index=None, placeholder='Options' )
				
				# ---------- Expiration Anchor ------------
				with ctrl_c3:
					st.selectbox( label='Expiration Anchor',
						options=[ 'last_active_at' ],
						key='stores_expires_anchor',
						help='Expiration anchor for vector store expiration policy.',
						index=0, placeholder='Options' )
				
				# ---------- Expiration Days ------------
				with ctrl_c4:
					st.slider( label='Expiration Days',
						min_value=0, max_value=365, step=1,
						key='stores_expires_days',
						help='Optional expiration days. Zero omits expires_after.' )
				
				# ---------- Description ------------
				st.text_area( label='Store Description',
					key='stores_description',
					value=st.session_state.get( 'stores_description', '' ),
					height=80, width='stretch',
					help='Optional vector store description.',
					placeholder='Optional description' )
				
				# ---------- Metadata ------------
				st.text_area( label='Store Metadata JSON',
					key='stores_metadata',
					value=st.session_state.get( 'stores_metadata', '' ),
					height=100, width='stretch',
					help='Optional JSON object used as vector store metadata.',
					placeholder='{ "project": "example" }' )
			
			# ------------------------------------------------------------------
			# Chunking Controls
			# ------------------------------------------------------------------
			with st.expander( label='Chunking Controls', icon='🧩',
					expanded=False, width='stretch' ):
				chunk_c1, chunk_c2, chunk_c3 = st.columns(
					[ 0.34, 0.33, 0.33 ], border=True, gap='xxsmall' )
				
				# ---------- Chunking Strategy ------------
				with chunk_c1:
					chunking_options = get_vector_store_chunking_options( vector )
					if st.session_state.get( 'stores_chunking_strategy' ) not in chunking_options:
						st.session_state[ 'stores_chunking_strategy' ] = 'auto'
					
					st.selectbox( label='Chunking Strategy',
						options=chunking_options,
						key='stores_chunking_strategy',
						help='Chunking strategy used when creating stores or attaching files.',
						index=chunking_options.index(
							st.session_state.get( 'stores_chunking_strategy', 'auto' ) )
						if st.session_state.get( 'stores_chunking_strategy' ) in chunking_options
						else None,
						placeholder='Options' )
				
				# ---------- Max Chunk Size ------------
				with chunk_c2:
					try:
						current_chunk_size = int( st.session_state.get(
							'stores_chunk_size', 800 ) or 800 )
					except Exception:
						current_chunk_size = 800
					
					if current_chunk_size < 100:
						st.session_state[ 'stores_chunk_size' ] = 100
					elif current_chunk_size > 4096:
						st.session_state[ 'stores_chunk_size' ] = 4096
					
					st.slider( label='Max Chunk Tokens',
						min_value=100, max_value=4096, step=50,
						key='stores_chunk_size',
						help='Static chunking max chunk size in tokens.' )
				
				# ---------- Chunk Overlap ------------
				with chunk_c3:
					max_overlap = max( 0, int( st.session_state.get(
						'stores_chunk_size', 800 ) or 800 ) // 2 )
					
					try:
						current_overlap = int( st.session_state.get(
							'stores_chunk_overlap', 400 ) or 400 )
					except Exception:
						current_overlap = 400
					
					if current_overlap < 0:
						st.session_state[ 'stores_chunk_overlap' ] = 0
					elif current_overlap > max_overlap:
						st.session_state[ 'stores_chunk_overlap' ] = max_overlap
					
					st.slider( label='Chunk Overlap',
						min_value=0, max_value=max_overlap, step=25,
						key='stores_chunk_overlap',
						help='Static chunking overlap in tokens. Cannot exceed half the chunk size.' )
			
			# ------------------------------------------------------------------
			# File Controls
			# ------------------------------------------------------------------
			with st.expander( label='File Controls', icon='📎',
					expanded=False, width='stretch' ):
				file_c1, file_c2 = st.columns(
					[ 0.50, 0.50 ], border=True, gap='xxsmall' )
				
				# ---------- Single File ID ------------
				with file_c1:
					st.text_input( label='File ID',
						key='stores_file_id',
						value=st.session_state.get( 'stores_file_id', '' ),
						help='OpenAI file ID used when attaching or managing one vector store file.',
						width='stretch', placeholder='file-...' )
				
				# ---------- Batch File IDs ------------
				with file_c2:
					st.text_input( label='File IDs',
						key='stores_file_ids',
						value=st.session_state.get( 'stores_file_ids', '' ),
						help='Comma-delimited OpenAI file IDs used for create/store file batch workflows.',
						width='stretch', placeholder='file-..., file-...' )
				
				# ---------- File Attributes ------------
				st.text_area( label='File Attributes JSON',
					key='stores_file_attributes',
					value=st.session_state.get( 'stores_file_attributes', '' ),
					height=90, width='stretch',
					help='Optional JSON object used as vector store file attributes.',
					placeholder='{ "source": "manual-upload" }' )
				
				# ---------- Batch ID ------------
				st.text_input( label='Batch ID',
					key='stores_batch_id',
					value=st.session_state.get( 'stores_batch_id', '' ),
					help='Vector store file batch ID used for retrieve/cancel workflows.',
					width='stretch', placeholder='vsfb_...' )
			
			# ------------------------------------------------------------------
			# Search Controls
			# ------------------------------------------------------------------
			with st.expander( label='Search Controls', icon='🔎',
					expanded=False, width='stretch' ):
				search_c1, search_c2, search_c3, search_c4 = st.columns(
					[ 0.40, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
				
				# ---------- Search Query ------------
				with search_c1:
					st.text_input( label='Search Query',
						key='stores_search_query',
						value=st.session_state.get( 'stores_search_query', '' ),
						help='Native vector store search query.',
						width='stretch', placeholder='Enter search query' )
				
				# ---------- Max Results ------------
				with search_c2:
					st.slider( label='Max Results',
						min_value=1, max_value=50, step=1,
						key='stores_max_results',
						help='Maximum number of native search or file_search results.' )
				
				# ---------- Ranker ------------
				with search_c3:
					ranker_options = get_vector_store_ranker_options( vector )
					if st.session_state.get( 'stores_ranker' ) not in ranker_options:
						st.session_state[ 'stores_ranker' ] = 'auto'
					
					st.selectbox( label='Ranker',
						options=ranker_options,
						key='stores_ranker',
						help='Native vector store search ranker.',
						index=ranker_options.index(
							st.session_state.get( 'stores_ranker', 'auto' ) )
						if st.session_state.get( 'stores_ranker' ) in ranker_options
						else None,
						placeholder='Options' )
				
				# ---------- Score Threshold / Rewrite ------------
				with search_c4:
					st.slider( label='Score Threshold',
						min_value=0.0, max_value=1.0, step=0.01,
						key='stores_score_threshold',
						help='Optional native search score threshold.' )
					
					st.toggle( label='Rewrite Query',
						key='stores_rewrite_query',
						help='Optional native vector store query rewriting.' )
			
			# ------------------------------------------------------------------
			# Current Store Selection
			# ------------------------------------------------------------------
			with st.expander( label='Current Store', icon='🎯',
					expanded=False, width='stretch' ):
				store_rows = st.session_state.get( 'stores_table', [ ] )
				store_options = build_vector_store_selection_options( store_rows )
				store_labels = [ '' ] + list( store_options.keys( ) )
				
				if st.session_state.get( 'stores_selected_label' ) not in store_labels:
					st.session_state[ 'stores_selected_label' ] = ''
				
				selected_label = st.selectbox( label='Selected Vector Store',
					options=store_labels,
					key='stores_selected_label',
					help='Select a vector store from the latest list.',
					index=store_labels.index(
						st.session_state.get( 'stores_selected_label', '' ) )
					if st.session_state.get( 'stores_selected_label' ) in store_labels
					else None,
					placeholder='Options' )
				
				selected_from_table = get_selected_vector_store_id(
					selected_label=selected_label,
					options=store_options )
				
				manual_id = st.session_state.get( 'stores_manual_id', '' )
				selected_store_id = selected_from_table or (
						manual_id.strip( ) if isinstance( manual_id, str )
						                      and manual_id.strip( ) else st.session_state.get(
							'stores_id', '' ))
				
				st.text_input( label='Manual Vector Store ID',
					key='stores_manual_id',
					value=st.session_state.get( 'stores_manual_id', '' ),
					help='Optional direct vector store ID. Use this if the store is not in the current table.',
					width='stretch', placeholder='vs_...' )
				
				if selected_store_id:
					st.caption( f'Selected Vector Store ID: `{selected_store_id}`' )
				else:
					st.caption( 'No vector store selected.' )
				
				st.text_input( label='Resolved Store ID',
					value=selected_store_id or '',
					disabled=True,
					key='stores_selected_id_display',
					help='Resolved vector store ID used by store, file, batch, search, and answer actions.',
					width='stretch' )
		
		# ------------------------------------------------------------------
		# System Instructions
		# ------------------------------------------------------------------
		with st.expander( label='System Instructions', icon='🖥️',
				expanded=False, width='stretch' ):
			in_left, in_right = st.columns( [ 0.8, 0.2 ] )
			
			prompt_names = fetch_prompt_names( cfg.DB_PATH )
			if not prompt_names:
				prompt_names = [ '' ]
			
			# ---------- System Instructions ------------
			with in_left:
				st.text_area( label='Enter Text',
					height=70, width='stretch',
					help=getattr( cfg, 'SYSTEM_INSTRUCTIONS',
						'Optional instructions used for Responses API file_search answers.' ),
					key='stores_system_instructions' )
			
			# ---------- Template ------------
			with in_right:
				st.selectbox( label='Use Template',
					options=prompt_names, index=None,
					key='instructions',
					on_change=load_vector_store_instruction_template )
			
			btn_c1, btn_c2 = st.columns( [ 0.8, 0.2 ] )
			
			# ---------- Clear Instructions ------------
			with btn_c1:
				st.button( label='Clear Instructions',
					width='stretch', on_click=clear_vector_store_instructions )
			
			# ---------- XML <-> Markdown ------------
			with btn_c2:
				st.button( label='XML <-> Markdown',
					width='stretch', on_click=convert_vector_store_system_instructions )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ------------------------------------------------------------------
		# Primary Management Columns
		# ------------------------------------------------------------------
		left_col, right_col = st.columns( [ 0.50, 0.50 ], border=True, gap='small' )
		
		# ------------------------------------------------------------------
		# Left Column: Store Management
		# ------------------------------------------------------------------
		with left_col:
			st.markdown( '#### Vector Store Management' )
			
			create_c1, create_c2 = st.columns( [ 0.50, 0.50 ] )
			
			# ---------- Create Store ------------
			with create_c1:
				if st.button( 'Create Store', key='create_vector_store',
						width='stretch' ):
					with st.spinner( 'Creating vector store…' ):
						try:
							result = run_vector_store_create( vector )
							
							if result.get( 'id' ):
								st.success( f"Created vector store: {result.get( 'id' )}" )
								
								try:
									rows = run_vector_store_list( vector )
									st.session_state[ 'stores_table' ] = rows
								except Exception:
									pass
						except Exception as exc:
							st.error( f'Create vector store failed: {exc}' )
			
			# ---------- List Stores ------------
			with create_c2:
				if st.button( 'List Stores', key='list_vector_stores',
						width='stretch' ):
					with st.spinner( 'Listing vector stores…' ):
						try:
							rows = run_vector_store_list( vector )
							
							if len( rows ) > 0:
								st.success( f'Listed {len( rows )} vector store(s).' )
							else:
								st.info( 'No vector stores were returned.' )
						except Exception as exc:
							st.error( f'List vector stores failed: {exc}' )
			
			retrieve_c1, retrieve_c2, retrieve_c3 = st.columns(
				[ 0.34, 0.33, 0.33 ] )
			
			# ---------- Retrieve Store ------------
			with retrieve_c1:
				if st.button( 'Retrieve Store', key='retrieve_vector_store',
						width='stretch' ):
					with st.spinner( 'Retrieving vector store…' ):
						try:
							result = run_vector_store_retrieve(
								vector=vector,
								store_id=selected_store_id )
							
							if result:
								st.success( 'Vector store metadata retrieved.' )
						except Exception as exc:
							st.error( f'Retrieve vector store failed: {exc}' )
			
			# ---------- Update Store ------------
			with retrieve_c2:
				if st.button( 'Update Store', key='update_vector_store',
						width='stretch' ):
					with st.spinner( 'Updating vector store…' ):
						try:
							result = run_vector_store_update(
								vector=vector,
								store_id=selected_store_id )
							
							if result:
								st.success( 'Vector store updated.' )
								
								try:
									rows = run_vector_store_list( vector )
									st.session_state[ 'stores_table' ] = rows
								except Exception:
									pass
						except Exception as exc:
							st.error( f'Update vector store failed: {exc}' )
			
			# ---------- Delete Store ------------
			with retrieve_c3:
				if st.button( 'Delete Store', key='delete_vector_store',
						width='stretch' ):
					with st.spinner( 'Deleting vector store…' ):
						try:
							result = run_vector_store_delete(
								vector=vector,
								store_id=selected_store_id )
							
							if result.get( 'deleted' ) is True:
								st.success( 'Vector store deleted.' )
								
								try:
									rows = run_vector_store_list( vector )
									st.session_state[ 'stores_table' ] = rows
								except Exception:
									pass
							elif result:
								st.warning( 'Delete request completed without confirmed deletion.' )
						except Exception as exc:
							st.error( f'Delete vector store failed: {exc}' )
			
			st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
			
			# ---------- Vector Stores Table ------------
			st.markdown( '#### Vector Stores' )
			render_vector_stores_table( st.session_state.get( 'stores_table', [ ] ) )
		
		# ------------------------------------------------------------------
		# Right Column: Store Details / Search
		# ------------------------------------------------------------------
		with right_col:
			st.markdown( '#### Selected Store Details' )
			
			metadata = st.session_state.get( 'stores_store_metadata', { } )
			if isinstance( metadata, dict ) and len( metadata ) > 0:
				render_vector_store_metadata( metadata )
			else:
				st.info( 'Retrieve a vector store to inspect metadata.' )
			
			st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
			
			search_c1, search_c2 = st.columns( [ 0.50, 0.50 ] )
			
			# ---------- Native Search ------------
			with search_c1:
				if st.button( 'Search Store', key='search_vector_store',
						width='stretch' ):
					with st.spinner( 'Searching vector store…' ):
						try:
							results = run_vector_store_search(
								vector=vector,
								store_id=selected_store_id )
							
							if len( results ) > 0:
								st.success( f'Returned {len( results )} search result(s).' )
							else:
								st.info( 'No vector store search results were returned.' )
						except Exception as exc:
							st.error( f'Vector store search failed: {exc}' )
			
			# ---------- Clear Outputs ------------
			with search_c2:
				st.button( label='Clear Outputs',
					key='clear_vector_store_outputs',
					width='stretch',
					on_click=clear_vector_store_outputs )
			
			# ---------- Search Results ------------
			results = st.session_state.get( 'stores_search_results', [ ] )
			if isinstance( results, list ) and len( results ) > 0:
				with st.expander( label='Search Results', icon='🔎',
						expanded=False, width='stretch' ):
					render_vector_store_search_results( results )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ------------------------------------------------------------------
		# File and Batch Operations
		# ------------------------------------------------------------------
		file_col, batch_col = st.columns( [ 0.50, 0.50 ], border=True, gap='small' )
		
		# ------------------------------------------------------------------
		# File Column
		# ------------------------------------------------------------------
		with file_col:
			st.markdown( '#### Vector Store Files' )
			
			file_action_c1, file_action_c2, file_action_c3 = st.columns(
				[ 0.34, 0.33, 0.33 ] )
			
			# ---------- Attach File ------------
			with file_action_c1:
				if st.button( 'Attach File', key='attach_vector_store_file',
						width='stretch' ):
					with st.spinner( 'Attaching file…' ):
						try:
							result = run_vector_store_attach_file(
								vector=vector,
								store_id=selected_store_id )
							
							if result.get( 'id' ):
								st.success( f"Attached file: {result.get( 'id' )}" )
								
								try:
									rows = run_vector_store_list_files(
										vector=vector,
										store_id=selected_store_id )
									st.session_state[ 'stores_files_table' ] = rows
								except Exception:
									pass
						except Exception as exc:
							st.error( f'Attach file failed: {exc}' )
			
			# ---------- List Files ------------
			with file_action_c2:
				if st.button( 'List Files', key='list_vector_store_files',
						width='stretch' ):
					with st.spinner( 'Listing vector store files…' ):
						try:
							rows = run_vector_store_list_files(
								vector=vector,
								store_id=selected_store_id )
							
							if len( rows ) > 0:
								st.success( f'Listed {len( rows )} vector store file(s).' )
							else:
								st.info( 'No files are attached to this vector store.' )
						except Exception as exc:
							st.error( f'List vector store files failed: {exc}' )
			
			# ---------- Delete Selected File ------------
			with file_action_c3:
				file_rows = st.session_state.get( 'stores_files_table', [ ] )
				file_options = build_vector_store_file_selection_options( file_rows )
				file_labels = [ '' ] + list( file_options.keys( ) )
				
				if st.session_state.get( 'stores_file_selected_label' ) not in file_labels:
					st.session_state[ 'stores_file_selected_label' ] = ''
				
				selected_file_label = st.selectbox( label='Selected File',
					options=file_labels,
					key='stores_file_selected_label',
					help='Select a vector store file from the latest file list.',
					index=file_labels.index(
						st.session_state.get( 'stores_file_selected_label', '' ) )
					if st.session_state.get( 'stores_file_selected_label' ) in file_labels
					else None,
					placeholder='Options' )
				
				selected_file_id = get_selected_vector_store_file_id(
					selected_label=selected_file_label,
					options=file_options ) or st.session_state.get( 'stores_file_id', '' )
				
				if st.button( 'Delete File', key='delete_vector_store_file',
						width='stretch' ):
					with st.spinner( 'Deleting vector store file…' ):
						try:
							result = run_vector_store_delete_file(
								vector=vector,
								store_id=selected_store_id,
								file_id=selected_file_id )
							
							if result.get( 'deleted' ) is True:
								st.success( 'Vector store file deleted.' )
								
								try:
									rows = run_vector_store_list_files(
										vector=vector,
										store_id=selected_store_id )
									st.session_state[ 'stores_files_table' ] = rows
								except Exception:
									pass
							elif result:
								st.warning(
									'Delete file request completed without confirmed deletion.' )
						except Exception as exc:
							st.error( f'Delete vector store file failed: {exc}' )
			
			# ---------- Files Table ------------
			render_vector_store_files_table(
				st.session_state.get( 'stores_files_table', [ ] ) )
		
		# ------------------------------------------------------------------
		# Batch Column
		# ------------------------------------------------------------------
		with batch_col:
			st.markdown( '#### File Batches' )
			
			batch_c1, batch_c2, batch_c3 = st.columns(
				[ 0.34, 0.33, 0.33 ] )
			
			# ---------- Create Batch ------------
			with batch_c1:
				if st.button( 'Create Batch', key='create_vector_store_batch',
						width='stretch' ):
					with st.spinner( 'Creating vector store file batch…' ):
						try:
							result = run_vector_store_create_batch(
								vector=vector,
								store_id=selected_store_id )
							
							if result.get( 'id' ):
								st.success( f"Created batch: {result.get( 'id' )}" )
						except Exception as exc:
							st.error( f'Create batch failed: {exc}' )
			
			# ---------- Retrieve Batch ------------
			with batch_c2:
				if st.button( 'Retrieve Batch', key='retrieve_vector_store_batch',
						width='stretch' ):
					with st.spinner( 'Retrieving vector store file batch…' ):
						try:
							result = run_vector_store_retrieve_batch(
								vector=vector,
								store_id=selected_store_id )
							
							if result:
								st.success( 'Batch metadata retrieved.' )
						except Exception as exc:
							st.error( f'Retrieve batch failed: {exc}' )
			
			# ---------- Cancel Batch ------------
			with batch_c3:
				if st.button( 'Cancel Batch', key='cancel_vector_store_batch',
						width='stretch' ):
					with st.spinner( 'Cancelling vector store file batch…' ):
						try:
							result = run_vector_store_cancel_batch(
								vector=vector,
								store_id=selected_store_id )
							
							if result:
								st.success( 'Batch cancellation requested.' )
						except Exception as exc:
							st.error( f'Cancel batch failed: {exc}' )
			
			# ---------- Batch Result ------------
			batch_result = st.session_state.get( 'stores_batch_result', { } )
			if isinstance( batch_result, dict ) and len( batch_result ) > 0:
				render_vector_store_batch_result( batch_result )
			else:
				st.info( 'No batch result available.' )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ------------------------------------------------------------------
		# Messages
		# ------------------------------------------------------------------
		if st.session_state.get( 'stores_messages' ) is not None:
			for msg in st.session_state.stores_messages:
				if not isinstance( msg, dict ):
					continue
				
				self_avatar = cfg.GIPITY if msg.get( 'role' ) == 'assistant' else ''
				with st.chat_message( msg.get( 'role', 'assistant' ), avatar=self_avatar ):
					st.markdown( msg.get( 'content', '' ) )
		
		prompt = st.chat_input( 'Ask a question using the selected vector store …' )
		if prompt is not None and str( prompt ).strip( ):
			prompt = str( prompt ).strip( )
			st.session_state.stores_messages.append(
				{
						'role': 'user',
						'content': prompt,
				} )
			
			with st.chat_message( 'assistant', avatar=cfg.GIPITY ):
				with st.spinner( 'Querying vector store…' ):
					try:
						# Re-resolve the selected store ID at execution time to avoid stale locals.
						current_rows = st.session_state.get( 'stores_table', [ ] )
						current_options = build_vector_store_selection_options( current_rows )
						current_label = st.session_state.get( 'stores_selected_label', '' )
						current_selected = get_selected_vector_store_id(
							selected_label=current_label,
							options=current_options )
						
						current_manual = st.session_state.get( 'stores_manual_id', '' )
						current_store_id = current_selected or (
								current_manual.strip( ) if isinstance( current_manual, str )
								                           and current_manual.strip( ) else st.session_state.get(
									'stores_id', '' ))
						
						answer = run_vector_store_answer(
							vector=vector,
							store_id=current_store_id,
							prompt=prompt )
						
						if isinstance( answer, str ) and answer.strip( ):
							st.markdown( answer )
							st.session_state.stores_messages.append(
								{
										'role': 'assistant',
										'content': answer.strip( ),
								} )
							st.session_state[ 'stores_last_answer' ] = answer.strip( )
							
							try:
								update_token_counters( getattr( vector, 'response', None ) )
							except Exception:
								pass
						else:
							message = 'No vector store answer was returned.'
							st.warning( message )
							st.session_state.stores_messages.append(
								{
										'role': 'assistant',
										'content': message,
								} )
					except Exception as exc:
						st.error( f'Vector store answer failed: {exc}' )
		
		# ------------------------------------------------------------------
		# Last Answer
		# ------------------------------------------------------------------
		last_answer = st.session_state.get( 'stores_last_answer', '' )
		if isinstance( last_answer, str ) and last_answer.strip( ):
			with st.expander( label='Last Vector Store Answer', icon='🧠',
					expanded=False, width='stretch' ):
				st.markdown( last_answer )
		
		# ------------------------------------------------------------------
		# Reset Buttons
		# ------------------------------------------------------------------
		reset_c1, reset_c2, reset_c3 = st.columns( [ 0.34, 0.33, 0.33 ] )
		
		with reset_c1:
			if st.button( 'Clear Messages', key='clear_vector_store_messages',
					width='stretch', on_click=clear_vector_store_messages ):
				st.rerun( )
		
		with reset_c2:
			if st.button( 'Clear Outputs', key='clear_vector_store_mode_outputs',
					width='stretch', on_click=clear_vector_store_outputs ):
				st.rerun( )
		
		with reset_c3:
			if st.button( 'Reset All', key='reset_vector_store_all',
					width='stretch', on_click=reset_vector_store_all ):
				st.rerun( )

# ======================================================================================
# PROMPT ENGINEERING MODE
# ======================================================================================
elif mode == 'Prompt Engineering':
	import sqlite3
	import math
	
	TABLE = 'Prompts'
	PAGE_SIZE = 10
	st.session_state.setdefault( 'pe_cascade_enabled', False )
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	with center:
		st.subheader( '📝 Prompt Engineering', help=cfg.PROMPT_ENGINEERING )
		st.divider( )
		st.checkbox( 'Cascade selection into System Instructions', key='pe_cascade_enabled' )
		
		# ------------------------------------------------------------------
		# Session state
		# ------------------------------------------------------------------
		st.session_state.setdefault( 'pe_page', 1 )
		st.session_state.setdefault( 'pe_search', '' )
		st.session_state.setdefault( 'pe_sort_col', 'PromptsId' )
		st.session_state.setdefault( 'pe_sort_dir', 'ASC' )
		st.session_state.setdefault( 'pe_selected_id', None )
		st.session_state.setdefault( 'pe_caption', '' )
		st.session_state.setdefault( 'pe_name', '' )
		st.session_state.setdefault( 'pe_text', '' )
		st.session_state.setdefault( 'pe_version', '' )
		st.session_state.setdefault( 'pe_id', 0 )
		
		# ------------------------------------------------------------------
		# DB helpers
		# ------------------------------------------------------------------
		def get_conn( ):
			return sqlite3.connect( cfg.DB_PATH )
		
		def reset_selection( ):
			st.session_state.pe_selected_id = None
			st.session_state.pe_caption = ''
			st.session_state.pe_name = ''
			st.session_state.pe_text = ''
			st.session_state.pe_version = ''
			st.session_state.pe_id = 0
		
		def load_prompt( pid: int ) -> None:
			with get_conn( ) as conn:
				_select = f"SELECT Caption, Name, Text, Version, ID FROM {TABLE} WHERE PromptsId=?"
				cur = conn.execute( _select, (pid,), )
				row = cur.fetchone( )
				if not row:
					return
				st.session_state.pe_caption = row[ 0 ]
				st.session_state.pe_name = row[ 1 ]
				st.session_state.pe_text = row[ 2 ]
				st.session_state.pe_version = row[ 3 ]
				st.session_state.pe_id = row[ 4 ]
		
		# ------------------------------------------------------------------
		# Filters
		# ------------------------------------------------------------------
		c1, c2, c3, c4 = st.columns( [ 4, 2, 2, 3 ] )
		
		with c1:
			st.text_input( 'Search (Name/Text contains)', key='pe_search' )
		
		with c2:
			st.selectbox( 'Sort by', [ 'PromptsId', 'Caption', 'Name', 'Text', 'Version', 'ID' ],
				key='pe_sort_col', )
		
		with c3:
			st.selectbox( 'Direction', [ 'ASC', 'DESC' ], key='pe_sort_dir' )
		
		with c4:
			st.markdown(
				"<div style='font-size:0.95rem;font-weight:600;margin-bottom:0.25rem;'>Go to ID</div>",
				unsafe_allow_html=True, )
			
			a1, a2, a3 = st.columns( [ 2, 1, 1 ] )
			
			with a1:
				jump_id = st.number_input( 'Go to ID', min_value=1,
					step=1, label_visibility='collapsed', )
			
			with a2:
				if st.button( 'Go' ):
					st.session_state.pe_selected_id = int( jump_id )
					load_prompt( int( jump_id ) )
			
			with a3:
				st.button( 'Clear', on_click=reset_selection )
		
		# ------------------------------------------------------------------
		# Load prompt table
		# ------------------------------------------------------------------
		where = ""
		params = [ ]
		if st.session_state.pe_search:
			where = 'WHERE Name LIKE ? OR Text LIKE ?'
			s = f"%{st.session_state.pe_search}%"
			params.extend( [ s, s ] )
		
		offset = (st.session_state.pe_page - 1) * PAGE_SIZE
		query = f"""
	        SELECT PromptsId, Caption, Name, Text, Version, ID
	        FROM {TABLE}
	        {where}
	        ORDER BY {st.session_state.pe_sort_col} {st.session_state.pe_sort_dir}
	        LIMIT {PAGE_SIZE} OFFSET {offset}
	    """
		
		count_query = f"SELECT COUNT(*) FROM {TABLE} {where}"
		
		with get_conn( ) as conn:
			rows = conn.execute( query, params ).fetchall( )
			total_rows = conn.execute( count_query, params ).fetchone( )[ 0 ]
		
		total_pages = max( 1, math.ceil( total_rows / PAGE_SIZE ) )
		
		# ------------------------------------------------------------------
		# Prompt table
		# ------------------------------------------------------------------
		table_rows = [ ]
		for r in rows:
			table_rows.append(
				{
						'Selected': r[ 0 ] == st.session_state.pe_selected_id,
						'PromptsId': r[ 0 ],
						'Caption': r[ 1 ],
						'Name': r[ 2 ],
						'Text': r[ 3 ],
						'Version': r[ 4 ],
						'ID': r[ 5 ],
				} )
		
		edited = st.data_editor( table_rows, hide_index=True, use_container_width=True,
			key="prompt_table", )
		
		# ------------------------------------------------------------------
		# SELECTION PROCESSING (must run BEFORE widgets below)
		# ------------------------------------------------------------------
		selected = [ r for r in edited if isinstance( r, dict ) and r.get( 'Selected' ) ]
		if len( selected ) == 1:
			pid = int( selected[ 0 ][ 'PromptsId' ] )
			if pid != st.session_state.pe_selected_id:
				st.session_state.pe_selected_id = pid
				load_prompt( pid )
		
		elif len( selected ) == 0:
			reset_selection( )
		
		elif len( selected ) > 1:
			st.warning( 'Select exactly one prompt row.' )
		
		# ------------------------------------------------------------------
		# Paging
		# ------------------------------------------------------------------
		p1, p2, p3 = st.columns( [ 0.25, 3.5, 0.25 ] )
		with p1:
			if st.button( "◀ Prev" ) and st.session_state.pe_page > 1:
				st.session_state.pe_page -= 1
		
		with p2:
			st.markdown( f"Page **{st.session_state.pe_page}** of **{total_pages}**" )
		
		with p3:
			if st.button( "Next ▶" ) and st.session_state.pe_page < total_pages:
				st.session_state.pe_page += 1
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ------------------------------------------------------------------
		# Edit Prompt
		# ------------------------------------------------------------------
		with st.expander( "🖊️ Edit Prompt", expanded=False ):
			st.text_input( "PromptsId", value=st.session_state.pe_selected_id or "",
				disabled=True, )
			st.text_input( 'Name', key='pe_name' )
			
			st.text_area( 'Text', key='pe_text', height=260 )
			st.text_input( 'Version', key='pe_version' )
			c1, c2, c3 = st.columns( 3 )
			
			with c1:
				if st.button( '💾 Save Changes'
				if st.session_state.pe_selected_id
				else '➕ Create Prompt' ):
					with get_conn( ) as conn:
						if st.session_state.pe_selected_id:
							conn.execute(
								f"""
	                            UPDATE {TABLE}
	                            SET Caption=?, Name=?, Text=?, Version=?, ID=?
	                            WHERE PromptsId=?
	                            """,
								(
										st.session_state.pe_caption,
										st.session_state.pe_name,
										st.session_state.pe_text,
										st.session_state.pe_version,
										st.session_state.pe_id,
										st.session_state.pe_selected_id
								), )
						else:
							conn.execute(
								f"""
	                            INSERT INTO {TABLE} (Caption, Name, Text, Version, ID)
	                            VALUES (?, ?, ?, ? , ?)
	                            """,
								(
										st.session_state.pe_caption,
										st.session_state.pe_name,
										st.session_state.pe_text,
										st.session_state.pe_version,
										st.session_state.pe_id
								),
							)
						conn.commit( )
					
					st.success( 'Saved.' )
					reset_selection( )
			
			with c2:
				if st.session_state.pe_selected_id and st.button( 'Delete' ):
					with get_conn( ) as conn:
						conn.execute(
							f'DELETE FROM {TABLE} WHERE PromptsId=?',
							(st.session_state.pe_selected_id,), )
						conn.commit( )
					reset_selection( )
					st.success( 'Deleted.' )
			
			with c3:
				st.button( '🧹 Clear Selection', on_click=reset_selection )

# ==============================================================================
# EXPORT MODE
# ==============================================================================
elif mode == 'Data Export':
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	with center:
		st.subheader( '📭  Export' )
		st.divider( )
		
		# -----------------------------------
		# Prompt export (System Instructions)
		st.caption( 'System Prompt' )
		export_format = st.radio( 'Export Format', options=[ 'XML-Delimited', 'Markdown' ],
			horizontal=True, help='Choose how system instructions should be exported.' )
		prompt_text: str = st.session_state.get( 'system_prompt', '' )
		if export_format == 'Markdown':
			try:
				export_text: str = convert_xml( prompt_text )
				export_filename: str = 'Buddy_Instructions.md'
			except Exception as exc:
				st.error( f'Markdown conversion failed: {exc}' )
				export_text = ''
				export_filename = ''
		else:
			export_text = prompt_text
			export_filename = 'Buddy_System_Instructions.xml'
		
		st.download_button( label='Download System Instructions', data=export_text,
			file_name=export_filename, mime='text/plain', disabled=not bool( export_text.strip( ) ) )
		
		# -----------------------------
		# Existing chat history export
		st.divider( )
		st.markdown( '###### Chat History' )
		
		hist = load_history( )
		md_history = '\n\n'.join( [ f'**{role.upper( )}**\n{content}' for role, content in hist ] )
		
		st.download_button( 'Download Chat History (Markdown)', md_history,
			'buddy_chat.md', mime='text/markdown' )
		
		buf = io.BytesIO( )
		pdf = canvas.Canvas( buf, pagesize=LETTER )
		y = 750
		
		for role, content in hist:
			pdf.drawString( 40, y, f'{role.upper( )}: {content[ :90 ]}' )
			y -= 14
			if y < 50:
				pdf.showPage( )
				y = 750
		
		pdf.save( )
		
		st.download_button( 'Download Chat History (PDF)', buf.getvalue( ),
			'buddy_chat.pdf', mime='application/pdf' )

# ==============================================================================
# DATA MANAGEMENT MODE
# ==============================================================================
elif mode == 'Data Management':
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	with center:
		st.subheader( '🏛️ Data Management', help=cfg.DATA_MANAGEMENT )
		tabs = st.tabs( [ 'Import', 'Browse', 'CRUD', 'Explore', 'Filter',
		                  'Aggregate', 'Visualize', 'Admin', 'SQL' ] )
		
		tables = list_tables( )
		if not tables:
			st.info( "No tables available." )
		
		# ------------------------------------------------------------------------------
		# UPLOAD TAB
		# ------------------------------------------------------------------------------
		with tabs[ 0 ]:
			uploaded_file = st.file_uploader( 'Upload Excel File', type=[ 'xlsx' ] )
			overwrite = st.checkbox( 'Overwrite existing tables', value=True )
			if uploaded_file:
				try:
					sheets = pd.read_excel( uploaded_file, sheet_name=None )
					with create_connection( ) as conn:
						conn.execute( 'BEGIN' )
						for sheet_name, df in sheets.items( ):
							table_name = create_identifier( sheet_name )
							if overwrite:
								conn.execute( f'DROP TABLE IF EXISTS "{table_name}"' )
							
							# --- Create Table ---
							columns = [ ]
							df.columns = [ create_identifier( c ) for c in df.columns ]
							for col in df.columns:
								sql_type = get_sqlite_type( df[ col ].dtype )
								columns.append( f'"{col}" {sql_type}' )
							
							create_stmt = (
									f'CREATE TABLE "{table_name}" '
									f'({", ".join( columns )});'
							)
							
							conn.execute( create_stmt )
							
							# --- Insert Data ---
							placeholders = ", ".join( [ "?" ] * len( df.columns ) )
							insert_stmt = (
									f'INSERT INTO "{table_name}" '
									f'VALUES ({placeholders});'
							)
							
							conn.executemany( insert_stmt,
								df.where( pd.notnull( df ), None ).values.tolist( ) )
						
						conn.commit( )
					
					st.success( 'Import completed successfully (transaction committed).' )
					st.rerun( )
				
				except Exception as e:
					try:
						conn.rollback( )
					except:
						pass
					st.error( f'Import failed — transaction rolled back.\n\n{e}' )
		
		# ------------------------------------------------------------------------------
		# BROWSE TAB
		# ------------------------------------------------------------------------------
		with tabs[ 1 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='table_name' )
				df = read_table( table )
				render_table( df )
			else:
				st.info( 'No tables available.' )
		
		# ------------------------------------------------------------------------------
		# CRUD (Schema-Aware)
		# ------------------------------------------------------------------------------
		with tabs[ 2 ]:
			tables = list_tables( )
			if not tables:
				st.info( 'No tables available.' )
			else:
				crud_header_c1, crud_header_c2, crud_header_c3 = st.columns(
					[ 0.45, 0.25, 0.30 ], border=True )
				
				with crud_header_c1:
					table = st.selectbox( 'Select Table', tables, key='crud_table' )
				
				df = read_table( table )
				schema = create_schema( table )
				
				type_map = { col[ 1 ]: col[ 2 ].upper( ) for col in schema if col[ 1 ] != 'rowid' }
				
				with crud_header_c2:
					st.metric( 'Rows', len( df.index ) )
				
				with crud_header_c3:
					st.metric( 'Columns', len( type_map ) )
				
				st.divider( )
				
				insert_col, update_col = st.columns( [ 0.50, 0.50 ], border=True )
				
				# ------------------------------------------------------------------
				# INSERT
				# ------------------------------------------------------------------
				with insert_col:
					st.markdown( '#### Insert Row' )
					insert_data = { }
					
					for column, col_type in type_map.items( ):
						if 'INT' in col_type:
							insert_data[ column ] = st.number_input(
								column,
								step=1,
								key=f'ins_{table}_{column}' )
						
						elif 'REAL' in col_type:
							insert_data[ column ] = st.number_input(
								column,
								format='%.6f',
								key=f'ins_{table}_{column}' )
						
						elif 'BOOL' in col_type:
							insert_data[ column ] = 1 if st.checkbox(
								column,
								key=f'ins_{table}_{column}' ) else 0
						
						else:
							insert_data[ column ] = st.text_input(
								column,
								key=f'ins_{table}_{column}' )
					
					if st.button( 'Insert Row', key=f'insert_row_{table}',
							use_container_width=True ):
						cols = list( insert_data.keys( ) )
						quoted_cols = [ f'"{c}"' for c in cols ]
						placeholders = ', '.join( [ '?' ] * len( cols ) )
						stmt = (
								f'INSERT INTO "{table}" ({", ".join( quoted_cols )}) '
								f'VALUES ({placeholders});')
						
						with create_connection( ) as conn:
							conn.execute( stmt, list( insert_data.values( ) ) )
							conn.commit( )
						
						st.success( 'Row inserted.' )
						st.rerun( )
				
				# ------------------------------------------------------------------
				# UPDATE
				# ------------------------------------------------------------------
				with update_col:
					st.markdown( '#### Update Row' )
					rowid = st.number_input(
						'Row ID',
						min_value=1,
						step=1,
						key=f'crud_update_rowid_{table}' )
					
					update_data = { }
					
					for column, col_type in type_map.items( ):
						if 'INT' in col_type:
							val = st.number_input(
								column,
								step=1,
								key=f'upd_{table}_{column}' )
							update_data[ column ] = val
						
						elif 'REAL' in col_type:
							val = st.number_input(
								column,
								format='%.6f',
								key=f'upd_{table}_{column}' )
							update_data[ column ] = val
						
						elif 'BOOL' in col_type:
							val = 1 if st.checkbox(
								column,
								key=f'upd_{table}_{column}' ) else 0
							update_data[ column ] = val
						
						else:
							val = st.text_input(
								column,
								key=f'upd_{table}_{column}' )
							update_data[ column ] = val
					
					if st.button( 'Update Row', key=f'update_row_{table}',
							use_container_width=True ):
						set_clause = ', '.join( [ f'"{c}"=?' for c in update_data ] )
						stmt = f'UPDATE "{table}" SET {set_clause} WHERE rowid=?;'
						
						with create_connection( ) as conn:
							conn.execute( stmt, list( update_data.values( ) ) + [ rowid ] )
							conn.commit( )
						
						st.success( 'Row updated.' )
						st.rerun( )
				
				st.divider( )
				
				delete_col, preview_col = st.columns( [ 0.35, 0.65 ], border=True )
				
				# ------------------------------------------------------------------
				# DELETE
				# ------------------------------------------------------------------
				with delete_col:
					st.markdown( '#### Delete Row' )
					delete_id = st.number_input(
						'Row ID to Delete',
						min_value=1,
						step=1,
						key=f'crud_delete_rowid_{table}' )
					
					if st.button( 'Delete Row', key=f'delete_row_{table}',
							use_container_width=True ):
						with create_connection( ) as conn:
							conn.execute( f'DELETE FROM "{table}" WHERE rowid=?;', (delete_id,) )
							conn.commit( )
						
						st.success( 'Row deleted.' )
						st.rerun( )
				
				# ------------------------------------------------------------------
				# PREVIEW
				# ------------------------------------------------------------------
				with preview_col:
					st.markdown( '#### Current Data Preview' )
					st.data_editor( df.head( 25 ), key=f'dm_crud_preview_{table}',
						use_container_width=True, disabled=True )
		
		# ------------------------------------------------------------------------------
		# EXPLORE
		# ------------------------------------------------------------------------------
		with tabs[ 3 ]:
			tables = list_tables( )
			if tables:
				exp_c1, exp_c2, exp_c3 = st.columns( [ 0.4, 0.4, 0.2 ], border=True )
				with exp_c1:
					table = st.selectbox( 'Table', tables, key='explore_table' )
				with exp_c2:
					page_size = st.slider( 'Rows per page', 10, 500, 50 )
				with exp_c3:
					page = st.number_input( 'Page', min_value=1, step=1 )
					offset = (page - 1) * page_size
					df_page = read_table( table, page_size, offset )
				
				st.data_editor( df_page )
		
		# ------------------------------------------------------------------------------
		# FILTER
		# ------------------------------------------------------------------------------
		with tabs[ 4 ]:
			tables = list_tables( )
			if tables:
				tbl_c1, tbl_c2, tbl_c3 = st.columns( [ 0.25, 0.25, 0.5 ], border=True )
				with tbl_c1:
					table = st.selectbox( 'Select Table', tables, key='filter_table' )
					df = read_table( table )
				with tbl_c2:
					column = st.selectbox( 'Select Field', df.columns )
				with tbl_c3:
					value = st.text_input( 'Contains', placeholder='Enter Text for Lookup' )
					if value:
						df = df[ df[ column ].astype( str ).str.contains( value ) ]
				
				st.data_editor( df )
		
		# ------------------------------------------------------------------------------
		# AGGREGATE
		# ------------------------------------------------------------------------------
		with tabs[ 5 ]:
			tables = list_tables( )
			if tables:
				agg_c1, agg_c2, agg_c3, agg_c4 = st.columns( [ 0.2, 0.2, 0.2, 0.4 ], border=True)
				with agg_c1:
					table = st.selectbox( 'Table', tables, key='agg_table' )
					df = read_table( table )
					numeric_cols = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
					with agg_c2:
						if numeric_cols:
							col = st.selectbox( 'Column', numeric_cols )
					with agg_c3:
						agg = st.selectbox( 'Function', [ 'SUM', 'AVG', 'COUNT' ] )
					with agg_c4:
						if agg == 'SUM':
							st.metric( 'Result', df[ col ].sum( ), width='stretch',
								format='accounting' )
							
						elif agg == 'AVG':
							st.metric( 'Result', df[ col ].mean( ), width='stretch',
								format='accounting' )
							
						elif agg == 'COUNT':
							st.metric( 'Result', df[ col ].count( ), width='stretch',
								format='accounting' )
		
		# ------------------------------------------------------------------------------
		# VISUALIZE
		# ------------------------------------------------------------------------------
		with tabs[ 6 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='viz_table' )
				df = read_table( table )
				create_visualization( df )
		
		# ------------------------------------------------------------------------------
		# ADMIN
		# ------------------------------------------------------------------------------
		with tabs[ 7 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='admin_table' )
			
			st.divider( )
			
			st.markdown( '#### Data Profiling' )
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table', tables, key='profile_table' )
				if st.button( 'Generate Profile' ):
					profile_df = create_profile_table( table )
					render_table( profile_df )
			
			st.markdown( '#### Drop Table' )
			
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table to Drop', tables, key='admin_drop_table' )
				
				# Initialize confirmation state
				if 'dm_confirm_drop' not in st.session_state:
					st.session_state.dm_confirm_drop = False
				
				# Step 1: Initial Drop click
				if st.button( 'Drop Table', key='admin_drop_button' ):
					st.session_state.dm_confirm_drop = True
				
				# Step 2: Confirmation UI
				if st.session_state.dm_confirm_drop:
					st.warning( f'You are about to permanently delete table {table}. '
					            'This action cannot be undone.' )
					
					col1, col2 = st.columns( 2 )
					
					if col1.button( 'Confirm Drop', key='admin_confirm_drop' ):
						try:
							drop_table( table )
							st.success( f'Table {table} dropped successfully.' )
						except Exception as e:
							st.error( f'Drop failed: {e}' )
						
						st.session_state.dm_confirm_drop = False
						st.rerun( )
					
					if col2.button( 'Cancel', key='admin_cancel_drop' ):
						st.session_state.dm_confirm_drop = False
						st.rerun( )
				
				df = read_table( table )
				col = st.selectbox( 'Create Index On', df.columns )
				
				if st.button( 'Create Index' ):
					create_index( table, col )
					st.success( 'Index created.' )
			
			st.divider( )
			
			st.markdown( '#### Create Custom Table' )
			new_table_name = st.text_input( 'Table Name' )
			column_count = st.number_input( 'Number of Columns', min_value=1, max_value=20, value=1 )
			columns = [ ]
			for i in range( column_count ):
				st.markdown( f'### Column {i + 1}' )
				col_name = st.text_input( 'Column Name', key=f'col_name_{i}' )
				col_type = st.selectbox( 'Column Type', [ 'INTEGER', 'REAL', 'TEXT' ],
					key=f'col_type_{i}' )
				
				not_null = st.checkbox( 'NOT NULL', key=f'not_null_{i}' )
				primary_key = st.checkbox( 'PRIMARY KEY', key=f'pk_{i}' )
				auto_inc = st.checkbox( 'AUTOINCREMENT (INTEGER only)', key=f'ai_{i}' )
				
				columns.append( {
						'name': col_name,
						'type': col_type,
						'not_null': not_null,
						'primary_key': primary_key,
						'auto_increment': auto_inc } )
			
			if st.button( 'Create Table' ):
				try:
					create_custom_table( new_table_name, columns )
					st.success( 'Table created successfully.' )
					st.rerun( )
				
				except Exception as e:
					st.error( f'Error: {e}' )
			
			st.divider( )
			st.markdown( '#### Schema Viewer' )
			
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table', tables, key='schema_view_table' )
				
				# Column schema
				schema = create_schema( table )
				schema_df = pd.DataFrame(
					schema,
					columns=[ 'cid', 'name', 'type', 'notnull', 'default', 'pk' ] )
				
				st.markdown( "### Columns" )
				st.data_editor(
					make_display_safe( schema_df ),
					hide_index=True,
					use_container_width=True,
					disabled=True )
				
				# Row count
				with create_connection( ) as conn:
					count = conn.execute(
						f'SELECT COUNT(*) FROM "{table}"'
					).fetchone( )[ 0 ]
				
				st.metric( "Row Count", f"{count:,}" )
				
				# Indexes
				indexes = get_indexes( table )
				if indexes:
					idx_df = pd.DataFrame(
						indexes,
						columns=[ 'seq', 'name', 'unique', 'origin', 'partial' ]
					)
					st.markdown( "### Indexes" )
					st.data_editor(
						make_display_safe( idx_df ),
						hide_index=True,
						use_container_width=True,
						disabled=True )
				else:
					st.info( "No indexes defined." )
			
			st.divider( )
			st.markdown( '#### ALTER TABLE Operations' )
			
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table', tables, key='alter_table_select' )
				operation = st.selectbox( 'Operation',
					[ 'Add Column', 'Rename Column', 'Rename Table', 'Drop Column' ] )
				
				if operation == 'Add Column':
					new_col = st.text_input( 'Column Name' )
					col_type = st.selectbox( 'Column Type', [ 'INTEGER', 'REAL', 'TEXT' ] )
					
					if st.button( 'Add Column' ):
						add_column( table, new_col, col_type )
						st.success( 'Column added.' )
						st.rerun( )
				
				elif operation == 'Rename Column':
					schema = create_schema( table )
					col_names = [ col[ 1 ] for col in schema ]
					
					old_col = st.selectbox( 'Column to Rename', col_names )
					new_col = st.text_input( 'New Column Name' )
					
					if st.button( 'Rename Column' ):
						rename_column( table, old_col, new_col )
						st.success( 'Column renamed.' )
						st.rerun( )
				
				elif operation == 'Rename Table':
					new_name = st.text_input( 'New Table Name' )
					
					if st.button( 'Rename Table' ):
						rename_table( table, new_name )
						st.success( 'Table renamed.' )
						st.rerun( )
				
				elif operation == 'Drop Column':
					schema = create_schema( table )
					col_names = [ col[ 1 ] for col in schema ]
					
					drop_col = st.selectbox( 'Column to Drop', col_names )
					
					if st.button( 'Drop Column' ):
						drop_column( table, drop_col )
						st.success( 'Column dropped.' )
						st.rerun( )
		
		# ------------------------------------------------------------------------------
		# SQL
		# ------------------------------------------------------------------------------
		with tabs[ 8 ]:
			st.subheader( 'SQL Console' )
			query = st.text_area( 'Enter SQL Query' )
			if st.button( 'Run Query' ):
				if not is_safe_query( query ):
					st.error( 'Query blocked: Only read-only SELECT statements are allowed.' )
				else:
					try:
						start_time = time.perf_counter( )
						with create_connection( ) as conn:
							result = pd.read_sql_query( query, conn )
						
						end_time = time.perf_counter( )
						elapsed = end_time - start_time
						
						# ----------------------------------------------------------
						# Display Results
						# ----------------------------------------------------------
						st.dataframe( result, use_container_width=True )
						row_count = len( result )
						
						# ----------------------------------------------------------
						# Execution Metrics
						# ----------------------------------------------------------
						col1, col2 = st.columns( 2 )
						col1.metric( 'Rows Returned', f'{row_count:,}' )
						col2.metric( 'Execution Time (seconds)', f'{elapsed:.6f}' )
						
						# Optional slow query warning
						if elapsed > 2.0:
							st.warning( 'Slow query detected (> 2 seconds). Consider indexing.' )
						
						# ----------------------------------------------------------
						# Download
						# ----------------------------------------------------------
						if not result.empty:
							csv = result.to_csv( index=False ).encode( 'utf-8' )
							st.download_button( 'Download CSV', csv,
								'query_results.csv', 'text/csv' )
					
					except Exception as e:
						st.error( f'Execution failed: {e}' )

# ======================================================================================
# FOOTER — SECTION
# ======================================================================================
st.markdown( """
	<style>
	.block-container {
		padding-bottom: 3rem;
	}
	</style>
	""", unsafe_allow_html=True )

# ---- Fixed Container
st.markdown(
	"""
	<style>
	.boo-status-bar {
		position: fixed;
		bottom: 0;
		left: 0;
		width: 100%;
		background-color: rgba(20, 20, 20, 0.95);
		border-top: 1px solid #5A5A5A;
		padding: 10px 16px;
		font-size: 0.80rem;
		color: #5181B0;
		z-index: 1000;
	}
	.boo-status-inner {
		display: flex;
		justify-content: space-between;
		align-items: center;
		max-width: 100%;
	}
	</style>
	""",
	unsafe_allow_html=True, )

# ======================================================================================
# FOOTER RENDERING
# ======================================================================================
_mode_to_model_key = \
{
	'Text': 'text_model',
	'Images': 'image_model',
	'Audio': 'audio_model',
	'Embeddings': 'embedding_model',
	'Document Q&A': 'docqna_model',
	'Files': 'files_model',
	'Vector Stores': 'stores_model',
	'Prompt Engineering': 'text_model',
	'Data Management': 'text_model'
}

provider_val = st.session_state.get( 'provider', '—' )
mode_val = mode or '—'
active_model = st.session_state.get( _mode_to_model_key.get( mode, '' ), None )
right_parts = [ ]
if active_model is not None:
	right_parts.append( f'Model: {active_model}' )

# ---- Rendered Variables
if mode == 'Text':
	number = st.session_state.get( 'text_number' )
	temperature = st.session_state.get( 'text_temperature' )
	top_p = st.session_state.get( 'text_top_percent' )
	freq = st.session_state.get( 'text_frequency_penalty' )
	presence = st.session_state.get( 'text_presence_penalty' )
	stream = st.session_state.get( 'text_stream' )
	parallel_tools = st.session_state.get( 'text_parallel_calls' )
	max_calls = st.session_state.get( 'text_max_calls' )
	store = st.session_state.get( 'text_store' )
	tools = st.session_state.get( 'text_tools' )
	include = st.session_state.get( 'text_include' )
	domains = st.session_state.get( 'text_domains' )
	input_mode = st.session_state.get( 'text_input' )
	tool_choice = st.session_state.get( 'text_tool_choice' )
	background = st.session_state.get( 'text_background' )
	messages = st.session_state.get( 'text_messages' )
	max_tokens = st.session_state.get( 'text_max_tokens' )
	
	if temperature is not None:
		right_parts.append( f'Temp: {temperature:.1%}' )
	if top_p is not None:
		right_parts.append( f'Top-P: {top_p:.1%}' )
	if freq is not None:
		right_parts.append( f'Freq: {freq:.2f}' )
	if presence is not None:
		right_parts.append( f'Presence: {presence:.2f}' )
	if number is not None:
		right_parts.append( f'N: {number}' )
	if max_tokens is not None:
		right_parts.append( f'Max Tokens: {max_tokens}' )
	
	if stream:
		right_parts.append( 'Stream: On' )
	if parallel_tools:
		right_parts.append( 'Parallel Tools: On' )
	if max_calls is not None:
		right_parts.append( f'Max Calls: {max_calls}' )
	if store:
		right_parts.append( 'Store: On' )
	if tools:
		right_parts.append( f'Tools: {len( tools )}' )
	if include:
		right_parts.append( 'Include: On' )
	if domains:
		right_parts.append( 'Domains: Set' )
	if input_mode:
		right_parts.append( 'Input: Set' )
	if tool_choice:
		right_parts.append( f'Tool Choice: On' )
	if background:
		right_parts.append( 'Background: On' )
	if messages:
		right_parts.append( 'Messages: Set' )

elif mode == 'Images':
	image_mode = st.session_state.get( 'image_mode' )
	image_size = st.session_state.get( 'image_size' )
	image_aspect = st.session_state.get( 'image_aspect' )
	image_style = st.session_state.get( 'image_style' )
	image_backcolor = st.session_state.get( 'image_backcolor' )
	image_quality = st.session_state.get( 'image_quality' )
	image_fmt = st.session_state.get( 'image_format' )
	image_reasoning = st.session_state.get( 'image_reasoning' )
	image_detail = st.session_state.get( 'image_detail' )
	image_number = st.session_state.get( 'image_number' )
	image_stream = st.session_state.get( 'image_stream' )
	image_store = st.session_state.get( 'image_store' )
	image_background = st.session_state.get( 'image_background' )
	image_include = st.session_state.get( 'image_include' )
	image_parallel_tools = st.session_state.get( 'image_parallel_tools' )
	image_max_calls = st.session_state.get( 'text_max_tools' )
	image_tools = st.session_state.get( 'image_tools' )
	
	if image_aspect is not None:
		right_parts.append( f'Aspect: {image_aspect}' )
	elif image_size is not None:
		right_parts.append( f'Size: {image_size}' )
	
	if image_mode is not None:
		right_parts.append( f'Mode: {image_mode}' )
	if image_reasoning is not None:
		right_parts.append( f'Reasoning: {image_reasoning}' )
	if image_style is not None:
		right_parts.append( f'Style: {image_style}' )
	if image_quality is not None:
		right_parts.append( f'Quality: {image_quality}' )
	if image_backcolor is not None:
		right_parts.append( f'Backcolor: {image_backcolor}' )
	if image_fmt is not None:
		right_parts.append( f'Format: {image_fmt}' )
	if image_detail is not None:
		right_parts.append( f'Detail: {image_detail}' )
	
	if image_number is not None:
		right_parts.append( f'N: {image_number}' )
	if image_parallel_tools:
		right_parts.append( 'Parallel Tools: On' )
	if image_max_calls is not None:
		right_parts.append( f'Max Calls: {image_max_calls}' )
	if image_tools:
		right_parts.append( f'Tools: {len( image_tools )}' )
	if image_include:
		right_parts.append( 'Include: On' )
	if image_stream:
		right_parts.append( 'Stream: On' )
	if image_store:
		right_parts.append( 'Store: On' )
	if image_background:
		right_parts.append( 'Background: On' )

elif mode == 'Audio':
	audio_model = st.session_state.get( 'audio_model' )
	audio_task = st.session_state.get( 'audio_task' )
	audio_format = st.session_state.get( 'audio_response_format' )
	audio_top_p = st.session_state.get( 'audio_top_percent' )
	audio_freq = st.session_state.get( 'audio_frequency_penalty' )
	audio_presence = st.session_state.get( 'audio_presence_penalty' )
	audio_number = st.session_state.get( 'audio_number' )
	audio_temperature = st.session_state.get( 'audio_temperature' )
	audio_stream = st.session_state.get( 'audio_stream' )
	audio_store = st.session_state.get( 'audio_store' )
	audio_input_mode = st.session_state.get( 'audio_input' )
	audio_reasoning = st.session_state.get( 'audio_reasoning' )
	audio_tool_choice = st.session_state.get( 'audio_tool_choice' )
	audio_messages = st.session_state.get( 'audio_messages' )
	audio_background = st.session_state.get( 'audio_background' )
	audio_file = st.session_state.get( 'audio_file' )
	audio_rate = st.session_state.get( 'audio_rate' )
	audio_start = st.session_state.get( 'audio_start' )
	audio_end = st.session_state.get( 'audio_end' )
	audio_loop = st.session_state.get( 'audio_loop' )
	audio_play = st.session_state.get( 'auto_play' )
	audio_voice = st.session_state.get( 'audio_voice' )
	
	if audio_task is not None:
		right_parts.append( f'Task: {audio_task}' )
	if audio_format is not None:
		right_parts.append( f'Format: {audio_format}' )
	
	if audio_temperature is not None:
		right_parts.append( f'Temp: {audio_temperature:.1%}' )
	if audio_top_p is not None:
		right_parts.append( f'Top-P: {audio_top_p:.1%}' )
	if audio_freq is not None:
		right_parts.append( f'Freq: {audio_freq:.2f}' )
	if audio_presence is not None:
		right_parts.append( f'Presence: {audio_presence:.2f}' )
	if audio_number is not None:
		right_parts.append( f'N: {audio_number}' )
	
	if audio_stream:
		right_parts.append( 'Stream: On' )
	if audio_store:
		right_parts.append( 'Store: On' )
	if audio_reasoning:
		right_parts.append( 'Reasoning: On' )
	if audio_input_mode:
		right_parts.append( 'Input: Set' )
	if audio_tool_choice:
		right_parts.append( f'Tool Choice: {audio_tool_choice}' )
	if audio_messages:
		right_parts.append( 'Messages: Set' )
	if audio_background:
		right_parts.append( 'Background: On' )
	
	if audio_voice:
		right_parts.append( f'Voice: {audio_voice}' )
	if audio_rate is not None:
		right_parts.append( f'Rate: {audio_rate}' )
	if (audio_start or audio_end) and audio_end >= audio_start:
		right_parts.append( f'Trim: {audio_start}s–{audio_end}s' )
	if audio_loop:
		right_parts.append( 'Loop: On' )
	if audio_play:
		right_parts.append( 'Autoplay: On' )
	if audio_file is not None:
		right_parts.append( 'File: Set' )

elif mode == 'Embeddings':
	model = st.session_state.get( 'embedding_model' )
	dimensions = st.session_state.get( 'embeddings_dimensions' )
	encoding = st.session_state.get( 'embeddings_encoding_format' )
	input_data = st.session_state.get( 'embedding_text_input' )
	
	if model is not None:
		right_parts.append( f'Model: {model}' )
	
	if dimensions is not None:
		right_parts.append( f'Dim: {dimensions}' )
	
	if encoding is not None:
		right_parts.append( f'Format: {encoding}' )
	
	if input_data:
		right_parts.append( 'Input: Set' )

elif mode == 'Document Q&A':
	temperature = st.session_state.get( 'docqna_temperature' )
	top_p = st.session_state.get( 'docqna_top_percent' )
	freq = st.session_state.get( 'docqna_frequency_penalty' )
	presence = st.session_state.get( 'docqna_presence_penalty' )
	number = st.session_state.get( 'docqna_number' )
	stream = st.session_state.get( 'docqna_stream' )
	parallel_tools = st.session_state.get( 'docqna_parallel_tools' )
	max_calls = st.session_state.get( 'docqna_max_tools' )
	store = st.session_state.get( 'docqna_store' )
	tools = st.session_state.get( 'docqna_tools' )
	include = st.session_state.get( 'docqna_include' )
	domains = st.session_state.get( 'docqna_domains' )
	input_mode = st.session_state.get( 'docqna_input' )
	tool_choice = st.session_state.get( 'docqna_tool_choice' )
	background = st.session_state.get( 'docqna_background' )
	messages = st.session_state.get( 'docqna_messages' )
	max_tokens = st.session_state.get( 'docqna_max_tokens' )
	
	if temperature is not None:
		right_parts.append( f'Temp: {temperature:.1%}' )
	if top_p is not None:
		right_parts.append( f'Top-P: {top_p:.1%}' )
	if freq is not None:
		right_parts.append( f'Freq: {freq:.2f}' )
	if presence is not None:
		right_parts.append( f'Presence: {presence:.2f}' )
	if number is not None:
		right_parts.append( f'N: {number}' )
	if max_tokens is not None:
		right_parts.append( f'Max Tokens: {max_tokens}' )
	
	if stream:
		right_parts.append( 'Stream: On' )
	if parallel_tools:
		right_parts.append( 'Parallel Tools: On' )
	if max_calls is not None:
		right_parts.append( f'Max Calls: {max_calls}' )
	if store:
		right_parts.append( 'Store: On' )
	if tools:
		right_parts.append( f'Tools: {len( tools )}' )
	if include:
		right_parts.append( 'Include: On' )
	if domains:
		right_parts.append( 'Domains: Set' )
	if input_mode:
		right_parts.append( 'Input: Set' )
	if tool_choice:
		right_parts.append( f'Tool Choice: On' )
	if background:
		right_parts.append( 'Background: On' )
	if messages:
		right_parts.append( 'Messages: Set' )

elif mode == 'Files':
	files_purpose = st.session_state.get( 'files_purpose' )
	files_type = st.session_state.get( 'files_type' )
	files_id = st.session_state.get( 'files_id' )
	files_url = st.session_state.get( 'files_url' )
	
	if files_purpose is not None:
		right_parts.append( f'Purpose: {files_purpose}' )
	
	if files_type is not None:
		right_parts.append( f'Type: {files_type}' )
	
	if files_id is not None:
		right_parts.append( f'File ID: {files_id}' )
	
	if files_url is not None:
		right_parts.append( 'URL: Set' )

elif mode == 'Vector Stores':
	model = st.session_state.get( 'stores_model' )
	fmt = st.session_state.get( 'stores_response_format' )
	temperature = st.session_state.get( 'stores_temperature' )
	top_p = st.session_state.get( 'stores_top_percent' )
	freq = st.session_state.get( 'stores_frequency_penalty' )
	presence = st.session_state.get( 'stores_presence_penalty' )
	number = st.session_state.get( 'stores_number' )
	stream = st.session_state.get( 'stores_stream' )
	store = st.session_state.get( 'stores_store' )
	input_data = st.session_state.get( 'stores_input' )
	reasoning = st.session_state.get( 'stores_reasoning' )
	tool_choice = st.session_state.get( 'stores_tool_choice' )
	messages = st.session_state.get( 'stores_messages' )
	background = st.session_state.get( 'stores_background' )
	
	if model is not None:
		right_parts.append( f'Model: {model}' )
	
	if fmt is not None:
		right_parts.append( f'Format: {fmt}' )
	
	if temperature is not None:
		right_parts.append( f'Temp: {temperature}' )
	
	if top_p is not None:
		right_parts.append( f'Top-P: {top_p}' )
	
	if freq is not None:
		right_parts.append( f'Freq: {freq}' )
	
	if presence is not None:
		right_parts.append( f'Presence: {presence}' )
	
	if number is not None:
		right_parts.append( f'N: {number}' )
	
	if stream:
		right_parts.append( 'Stream: On' )
	
	if store:
		right_parts.append( 'Store: On' )
	
	if reasoning is not None:
		right_parts.append( f'Reasoning: {reasoning}' )
	
	if tool_choice is not None:
		right_parts.append( f'Tool Choice: {tool_choice}' )
	
	if input_data:
		right_parts.append( 'Input: Set' )
	
	if messages:
		right_parts.append( 'Messages: Set' )
	
	if background:
		right_parts.append( 'Background: On' )

right_text = ' ◽ '.join( right_parts ) if right_parts else '—'

# ---- Rendering Method
st.markdown( f"""
    <div class="boo-status-bar">
        <div class="boo-status-inner">
            <span>{provider_val} — {mode_val}</span>
            <span>{right_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True, )

