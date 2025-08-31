import json
import time
import base64
import io
import requests
import logging
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image

from config import settings, validate_file_size, validate_file_type, get_request_headers, IMAGE_SIZE_THRESHOLDS
from schemas import DocumentResult, ExtractedEntities, DocumentCategory

logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

class DocumentPipeline:
    """Complete document processing pipeline: ingest → preprocess → model → postprocess"""
    
    def __init__(self):
        self.headers = get_request_headers()
    
    # === DATA INGESTION ===
    
    def ingest(self, uploaded_files: List) -> List[Tuple[Any, Optional[str]]]:
        """Validate and load files, return (file, error) tuples"""
        results = []
        for file in uploaded_files[:settings.max_files_per_batch]:
            error = self._validate_file(file)
            results.append((file, error))
        return results
    
    def _validate_file(self, file) -> Optional[str]:
        """Validate single file, return error message or None"""
        try:
            if hasattr(file, 'size') and not validate_file_size(file.size):
                return f"File too large ({file.size / 1024 / 1024:.1f}MB > {settings.max_file_size_mb}MB)"
            if not validate_file_type(file.type):
                return f"Unsupported file type: {file.type}"
            return None
        except Exception as e:
            return f"Validation error: {str(e)}"
    
    # === PREPROCESSING ===
    
    def preprocess(self, file) -> str:
        """Convert file to base64 data URI with optimization"""
        try:
            # Ensure stream is at start before reading
            try:
                if hasattr(file, "seek"):
                    file.seek(0)
            except Exception:
                pass

            file_bytes = file.read()
            # If we got 0 bytes, attempt one more time from start
            if not file_bytes:
                try:
                    if hasattr(file, "seek"):
                        file.seek(0)
                    file_bytes = file.read()
                except Exception:
                    pass

            logger.info(f"Processing file: {file.name} ({len(file_bytes)} bytes)")

            if not file_bytes:
                raise Exception("File appears empty after read; ensure the file stream is reset and not corrupted.")
            
            if file.type == 'application/pdf':
                return f"data:application/pdf;base64,{base64.b64encode(file_bytes).decode()}"
            elif file.type.startswith('image/'):
                return self._preprocess_image(file_bytes)
            else:
                return f"data:{file.type};base64,{base64.b64encode(file_bytes).decode()}"
        except Exception as e:
            logger.error(f"Preprocessing failed for {file.name}: {str(e)}")
            raise Exception(f"Failed to preprocess {file.name}: {str(e)}")
    
    def _preprocess_image(self, image_bytes: bytes) -> str:
        """Optimize and convert image to base64"""
        try:
            with io.BytesIO(image_bytes) as buffer:
                img = Image.open(buffer)
                
                # Optimize large images
                if img.size[0] * img.size[1] > IMAGE_SIZE_THRESHOLDS["large_image_pixels"]:
                    img.thumbnail(IMAGE_SIZE_THRESHOLDS["thumbnail_size"], Image.Resampling.LANCZOS)
                    logger.info(f"Resized image to {img.size}")
                
                # Ensure RGB format
                if img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')
                
                return self._image_to_base64(img)
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise Exception(f"Failed to preprocess image: {str(e)}")
    
    def _image_to_base64(self, img: Image.Image) -> str:
        """Convert PIL image to optimized base64"""
        with io.BytesIO() as buffer:
            # Use JPEG for large images, PNG for small ones
            if img.size[0] * img.size[1] > IMAGE_SIZE_THRESHOLDS["jpeg_threshold_pixels"]:
                img.save(buffer, format='JPEG', quality=85, optimize=True)
                img_str = base64.b64encode(buffer.getvalue()).decode()
                return f"data:image/jpeg;base64,{img_str}"
            else:
                img.save(buffer, format='PNG', optimize=True)
                img_str = base64.b64encode(buffer.getvalue()).decode()
                return f"data:image/png;base64,{img_str}"
    
    # === MODEL CALLS ===
    
    def classify(self, file_base64: str, model_name: str, filename: str = "document") -> DocumentResult:
        """Classify document and extract entities"""
        start_time = time.time()
        
        prompt = self._get_extraction_prompt()
        messages = self._create_messages(prompt, file_base64, filename)
        
        try:
            response = self._make_request(messages, model_name)
            return self._parse_classification(response, model_name, time.time() - start_time)
        except Exception as e:
            logger.error(f"Classification failed for {model_name}: {str(e)}")
            return DocumentResult(
                category=DocumentCategory.OTHER,
                entities=ExtractedEntities(),
                processing_time=time.time() - start_time,
                model_used=model_name,
                raw_text=f"Error: {str(e)}"
            )
    
    def judge(self, file_base64: str, prediction: Dict[str, Any], filename: str = "document") -> Tuple[float, str]:
        """Evaluate prediction confidence"""
        prompt = self._get_judge_prompt(prediction)
        messages = self._create_messages(prompt, file_base64, filename)
        
        try:
            # Use structured outputs for judge only for consistent JSON
            response = self._make_request(messages, settings.judge_model, schema=self._get_judge_schema())
            content = response['choices'][0]['message']['content']
            logger.info(f"Judge raw response: {content[:200]}...")
            
            # Try to extract JSON from response
            if isinstance(content, str):
                if '{' in content and '}' in content:
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    json_str = content[start:end]
                    data = json.loads(json_str)
                else:
                    raise ValueError(f"No JSON found in judge response: {content[:100]}...")
            else:
                data = content
            
            confidence = max(0.0, min(1.0, float(data['confidence'])))
            reasoning = data['reasoning']
            
            logger.info(f"Judge: confidence={confidence}, reasoning='{reasoning[:50]}...'")
            return confidence, reasoning
        except Exception as e:
            logger.error(f"Judge evaluation failed: {str(e)}")
            return 0.0, f"Judge error: {str(e)[:50]}"
    
    def _make_request(self, messages: List, model_name: str, schema: Dict = None) -> Dict:
        """Make HTTP request to OpenRouter API with retry"""
        if model_name not in settings.available_models:
            raise ValueError(f"Model '{model_name}' not found in available models: {list(settings.available_models.keys())}")
        
        base_payload = {
            "model": settings.available_models[model_name],
            "messages": messages,
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature
        }
        
        # Add PDF processing plugin if messages contain PDF files
        has_pdf = any(
            any(
                item.get("type") == "file" and 
                item.get("file", {}).get("filename", "").endswith(".pdf")
                for item in msg.get("content", []) if isinstance(item, dict)
            )
            for msg in messages if isinstance(msg.get("content"), list)
        )
        
        if has_pdf:
            base_payload["plugins"] = [{
                "id": "file-parser",
                "pdf": {
                    "engine": settings.pdf_engine
                }
            }]
        
        last_error = None
        schema_in_payload = bool(schema)
        for attempt in range(settings.max_retries):
            try:
                payload = dict(base_payload)
                if schema_in_payload and schema:
                    payload["response_format"] = {
                        "type": "json_schema",
                        "json_schema": schema
                    }
                response = requests.post(
                    settings.openrouter_base_url,
                    json=payload,
                    headers=self.headers,
                    timeout=settings.request_timeout
                )
                
                # Log response details for debugging
                logger.info(f"Response status: {response.status_code}")
                if response.status_code != 200:
                    logger.error(f"Response body: {response.text[:500]}")
                
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                last_error = e
                if response.status_code >= 500:
                    logger.warning(f"API server error (attempt {attempt + 1}/{settings.max_retries}): {response.status_code}")
                elif response.status_code == 429:
                    logger.warning(f"Rate limited (attempt {attempt + 1}/{settings.max_retries})")
                else:
                    # If provider rejects structured outputs, retry once without schema
                    if schema_in_payload and response.status_code in (400, 422):
                        logger.warning("Structured output not accepted; retrying without schema")
                        schema_in_payload = False
                        continue
                    logger.error(f"Client error {response.status_code}: {response.text}")
                    break  # Don't retry other client errors
            except Exception as e:
                last_error = e
                logger.warning(f"Request error (attempt {attempt + 1}/{settings.max_retries}): {str(e)}")
            
            if attempt < settings.max_retries - 1:
                time.sleep(settings.retry_delay * (2 ** attempt))
        
        raise last_error

    def _get_judge_schema(self) -> Dict:
        """Structured output schema for judge responses"""
        return {
            "name": "judge_evaluation",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Confidence score between 0.0 and 1.0"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Concise reason for the confidence score"
                    }
                },
                "required": ["confidence", "reasoning"],
                "additionalProperties": False
            }
        }
    
    def _create_messages(self, prompt: str, file_base64: str, filename: str = "document") -> List[Dict]:
        """Create message format for OpenRouter API with proper file handling"""
        content = [{"type": "text", "text": prompt}]

        # PDF: send as file (OpenRouter file-parser can handle)
        if file_base64.startswith("data:application/pdf"):
            content.append({
                "type": "file",
                "file": {
                    "filename": filename if filename.endswith('.pdf') else f"{filename}.pdf",
                    "file_data": file_base64
                }
            })
        # Images: send as image_url data URL
        elif file_base64.startswith("data:image/"):
            content.append({"type": "image_url", "image_url": {"url": file_base64}})
        # Text files (e.g., CSV/TXT): inline decoded text
        elif file_base64.startswith("data:text/"):
            try:
                b64_marker = ",base64,"
                idx = file_base64.find(b64_marker)
                if idx != -1:
                    txt_b64 = file_base64[idx + len(b64_marker):]
                    decoded = base64.b64decode(txt_b64 or "").decode("utf-8", errors="replace")
                else:
                    decoded = ""
            except Exception as e:
                decoded = f"[Error decoding text file: {e}]"

            # Truncate overly large text to keep request reasonable
            max_chars = 200_000
            if len(decoded) > max_chars:
                logger.info(f"Text content too large ({len(decoded)} chars); truncating to {max_chars} chars")
                decoded = decoded[:max_chars]

            content.append({"type": "text", "text": f"File: {filename}\n\n" + decoded})
        else:
            # Fallback: attach as a generic file; models may not parse non-PDF files
            content.append({
                "type": "file",
                "file": {
                    "filename": filename,
                    "file_data": file_base64
                }
            })

        return [{"role": "user", "content": content}]
    
    def _parse_classification(self, response: Dict, model_name: str, processing_time: float) -> DocumentResult:
        """Parse classification response"""
        try:
            content = response['choices'][0]['message']['content']
            logger.info(f"Raw response: {content[:200]}...")
            
            # Try to extract JSON from response
            if isinstance(content, str):
                # Look for JSON block
                if '{' in content and '}' in content:
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    json_str = content[start:end]
                    data = json.loads(json_str)
                else:
                    raise ValueError(f"No JSON found in response: {content[:100]}...")
            else:
                data = content
            
            entities_dict = data.get('entities', {})
            entities = ExtractedEntities(**entities_dict)
            
            return DocumentResult(
                category=DocumentCategory(data['category']),
                confidence=None,
                entities=entities,
                processing_time=processing_time,
                model_used=model_name
            )
        except Exception as e:
            logger.error(f"Parse error for {model_name}: {str(e)}")
            return DocumentResult(
                category=DocumentCategory.OTHER,
                entities=ExtractedEntities(),
                processing_time=processing_time,
                model_used=model_name,
                raw_text=f"Parse error: {str(e)}"
            )
    
    # === POSTPROCESSING ===
    
    def postprocess(self, results: List[DocumentResult], confidences: List[Tuple[float, str]]) -> List[Dict[str, Any]]:
        """Format final results for output"""
        processed_results = []
        
        for i, (result, (confidence, reasoning)) in enumerate(zip(results, confidences)):
            processed_result = {
                "category": result.category.value,
                "entities": result.entities.model_dump() if result.entities else {},
                "processing_time": result.processing_time,
                "model_used": result.model_used,
                "judge_confidence": confidence,
                "judge_reasoning": reasoning
            }
            processed_results.append(processed_result)
        
        return processed_results
        
    # === HELPER METHODS ===
    
    def _get_extraction_prompt(self) -> str:
        """Get extraction prompt"""
        return """Analyze document thoroughly and extract ALL visible information.

Categories: invoice, marketplace_screenshot, chat_screenshot, website_screenshot, other
Entities: names, organizations, dates, addresses, phone numbers, license plates, usernames, prices, quantities, brands, etc.

Extract EVERYTHING you can see - text, numbers, dates, names, addresses, phone numbers, license plates, usernames, prices, quantities, brands, etc.
Examples:
- Chat: Extract all participants, messages, timestamps, platform indicators, phone numbers, profile names
- Marketplace: Get product details, prices, seller info, location, contact details, item condition, specifications
- Invoice: Capture all amounts, dates, line items, tax info, addresses, reference numbers
- Website: Note platform name, user details, account info, displayed content, navigation elements
- Other: Extract any visible text, numbers, dates, names, addresses, phone numbers, license plates, usernames, prices, quantities, brands, etc.

Be comprehensive and detailed.

IMPORTANT: Return your response as valid JSON in this exact format:
{
  "category": "one_of_the_categories_above",
  "entities": {
    "key": "value or array of values for any entities you find"
  }
}"""
    
    def _get_judge_prompt(self, prediction: Dict) -> str:
        """Get judge evaluation prompt"""
        return f"""Evaluate this prediction using the rubric below. Rate 0.0-1.0 confidence.

PREDICTION:
Category: {prediction['category']}
Entities: {json.dumps(prediction['entities'], indent=2)}

RUBRIC:
0.9-1.0: Perfect category + comprehensive extraction (all key details captured)
0.7-0.8: Correct category + good extraction (most important details captured) 
0.5-0.6: Correct category + partial extraction (some key details missing)
0.3-0.4: Wrong category OR major extraction errors
0.0-0.2: Completely wrong

Score and reason:

IMPORTANT: Return your response as valid JSON in this exact format:
{{
  "confidence": 0.75,
  "reasoning": "Brief explanation of why this confidence score"
}}"""
    
