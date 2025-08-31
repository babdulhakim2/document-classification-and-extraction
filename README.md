# Document Classification & Extraction

Minimal prototype to upload files, categorise into predefined classes with a focus of on fraud prevention , and extract key entities using LLMs via OpenRouter. Built for speed, clarity, and robustness.

## Run

- Create venv: `python3 -m venv .venv`
- Activate venv:
  - macOS/Linux: `source .venv/bin/activate`
  - Windows (PowerShell): `.venv\\Scripts\\Activate.ps1`
- Install deps: `pip install -r requirements.txt`
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
 
Tuning tips:
- Increase `MAX_CONCURRENT_REQUESTS` for more parallelism (watch rate limits).
- Prefer `PDF_ENGINE=pdf-text` for speed; use `mistral-ocr` only for scanned PDFs.

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
- Large/complex PDFs may benefit from a better PDF engine
- Defaults target clarity over advanced batching/caching
- Judge is probabilistic; may underrate a correct extraction. Use verifiable checks and/or better metrics to quantify reliability.
- Fraud scenarios: LLMs can be spoofed by synthesized or manipulated images; no cryptographic/authenticity checks are performed.
- PII handling: outputs may contain sensitive data; ensure appropriate governance.
- Rate limiting: Providers may throttle requests. Basic retries/backoff are implemented, but sustained high concurrency may still hit limits; consider adaptive throttling or a queue for production.
- Fixed categories: The classifier uses a predefined set (`invoice`, `marketplace_screenshot`, `chat_screenshot`, `website_screenshot`, `other`). Anything outside these maps to `other` by design. Extending categories requires updating the prompt/schema and downstream logic.

## Improvements (Future)

- Add `.docx` text extraction (python-docx)
- Pluggable model selection per file type
 - Verifiable checks: rule-based consistency (sums, line-item totals, date ranges), schema constraints, and sanity bounds.
 - Metrics: create a gold set; compute category accuracy and entity-level precision/recall/F1; track false positives/negatives and calibration of judge scores.
 - Dataset + fine-tuning: collect labeled examples for domain categories/entities; consider lightweight fine-tunes for the classifier head.
 - Prompt optimization: explore DSPy-style prompt programs to optimize extraction given known input/output schemas.
 - `.docx` extraction: parse text via `python-docx` before sending to model.
 - Anomaly/OOD detection: simple heuristics or embeddings to flag out-of-distribution files.
 - Caching: reuse parsed PDF annotations to avoid re-parse costs.
