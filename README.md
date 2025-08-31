# Document Classification & Extraction (Prototype)

Minimal prototype to upload files, categorise into predefined classes, and extract key entities using LLMs via OpenRouter. Built for speed, clarity, and robustness without unnecessary complexity.

## Run

- Install: `pip install -r requirements.txt`
- Env: create `.env` with `OPENROUTER_API_KEY=...`
- Start: `streamlit run app.py`

## What It Does

- Categories: `invoice`, `marketplace_screenshot`, `chat_screenshot`, `website_screenshot`, `other`
- Files: PDF, PNG/JPG, TXT, CSV, DOC/DOCX (best results: PDF/images; text is inlined)
- Output: Standard JSON (category + entities)
- Judge: Second pass scores confidence (0–1) and reasoning

## How It Works

- `app.py`: Streamlit UI (upload, previews, results)
- `data_pipeline.py`: ingest → preprocess → model call → parse → judge
  - PDFs: sent as files with OpenRouter file-parser (engine configurable)
  - Images: sent as data URLs (`image_url`)
  - Text (txt/csv): base64-decoded and inlined to the prompt
  - Other: attached as generic files
- `config.py`: settings (models, limits, supported types), `.env` loading
- `schemas.py`: pydantic models (category, entities, results)

## Configuration

Set in `.env` (defaults in `config.py`):
- `OPENROUTER_API_KEY`: required
- `DEFAULT_MODEL`: default `gemini-flash-2.5`
- `JUDGE_MODEL`: default `gpt-4o`
- `PDF_ENGINE`: `pdf-text` (free) or `mistral-ocr` (scanned docs)
- `MAX_FILE_SIZE_MB`: default `10`
- `MAX_FILES_PER_BATCH`: default `20`
- `LOG_LEVEL`: `INFO` or `DEBUG`

Note: `.streamlit/config.toml` sets `server.maxUploadSize=10` to match app limits. If you change `MAX_FILE_SIZE_MB`, also update this to keep the UI consistent.

## Latency & Concurrency

- Classification and Judge run in parallel across files using a thread pool sized by `MAX_CONCURRENT_REQUESTS`.
- Judge uses structured outputs (JSON schema) for consistent `{confidence, reasoning}` and falls back automatically if a provider rejects schemas.
- Images are compressed; large text inputs are truncated to control payload size.

## Output Format

Example:
```
{
  "filename": "invoice.pdf",
  "category": "invoice",
  "entities": { "invoice_number": "INV-12345", "amounts": ["$199.99"], "dates": ["2024-08-10"] },
  "processing_time": 1.72,
  "model_used": "gemini-flash-2.5",
  "judge_confidence": 0.86,
  "judge_reasoning": "Correct category; key fields captured."
}
```

## Assumptions & Design

- OpenRouter used as the single API; model chosen from UI
- Minimal preprocessing; rely on model capabilities (with PDF parser plugin)
- Guardrails: file size/type checks, pointer resets, retries, clear errors
- Separation: UI and pipeline modules; schemas for typed results

## Limitations

- DOC/DOCX are attached as files (LLMs may not parse perfectly)
- Large/complex PDFs may benefit from  a better pdf engine
- Defaults target clarity over advanced batching/caching

## Improvements (Future)

- Add `.docx` text extraction (python-docx)
- Pluggable model selection per file type
- Optional caching of parsed PDFs (reuse annotations)
