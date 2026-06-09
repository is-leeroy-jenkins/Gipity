# GPT

## Purpose

This page documents the OpenAI provider wrapper module used by Gipity.

The wrapper module provides classes and helper functions for OpenAI-backed text, image, audio,
embedding, file, and vector-store workflows.

## Responsibilities

`gpt.py` is responsible for:

- Providing a shared GPT wrapper base class.
- Building provider request payloads.
- Normalizing model options and optional parameters.
- Coordinating text-generation workflows.
- Supporting image generation, editing, and analysis workflows.
- Supporting audio transcription, translation, and text-to-speech workflows.
- Supporting embedding generation workflows.
- Supporting OpenAI Files API operations.
- Supporting OpenAI Vector Store operations.
- Logging provider-wrapper exceptions.

## API Reference

::: gpt