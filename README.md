# Document Classification & Extraction

LLM-powered document categorization and content extraction system using OpenRouter API.

## Quick Start

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="your-api-key"
streamlit run app.py
```

## Features

- **Multi-format support**: PDF, PNG, JPG, JPEG (direct upload to LLM)
- **5 categories**: invoice, marketplace_screenshot, chat_screenshot, website_screenshot, other
- **3 models**: Gemini Flash 2.5, Mistral OCR, GPT-4o
- **Structured extraction**: dates, amounts, merchants, participants, etc.
- **Cost tracking**: Real-time usage monitoring
- **Batch processing**: Multiple file upload

## Files

- `app.py` - Streamlit UI
- `model.py` - OpenRouter API interface
- `processor.py` - File to base64 conversion
- `schemas.py` - Pydantic data models
- `benchmark.ipynb` - Model comparison notebook

## Benchmarking

Run the Jupyter notebook to compare models:

```bash
jupyter notebook benchmark.ipynb
```

Tests classification accuracy, extraction F1, speed, and cost.

## Architecture

1. **File ingestion**: Direct file to base64 encoding
2. **LLM processing**: Structured prompts via OpenRouter
3. **Validation**: Pydantic schema enforcement
4. **Retry logic**: JSON parsing error handling

## Assumptions

- Files sent directly to LLM for processing
- Standard document formats
- $100 OpenRouter credit limit

## Limitations

- LLM token limits (1000 max)
- Rate limiting (1 req/sec)
- No preprocessing or OCR fallback

## Future Improvements

- Add more document categories
- Implement confidence thresholding
- Support batch API calls
- Add vector embeddings for similarity