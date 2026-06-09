# Configuration 

## Purpose

This page documents the Gipity configuration module.

The configuration module centralizes environment-variable handling, filesystem paths, runtime
constants, model lists, prompt constants, API key lookup, logging configuration, and
application-level defaults.

## Responsibilities

`config.py` is responsible for:

- Reading environment variables.
- Normalizing Boolean, integer, floating-point, text, and path values.
- Defining application paths.
- Defining OpenAI and Google service key variables.
- Defining model options and workflow constants.
- Defining local database and logging paths.
- Providing constants used by the Streamlit application and provider wrappers.

## API Reference

::: config