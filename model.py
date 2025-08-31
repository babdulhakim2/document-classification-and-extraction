import json
import time
import requests
import asyncio
import concurrent.futures
from typing import Dict, Any, Optional
from schemas import DocumentResult, ExtractedEntities, DocumentCategory

class OpenRouterClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "Document Classifier"
        }
        
        self.models = {
            "gemini-flash-2.5": "google/gemini-2.5-flash",
            "gpt-4o-mini": "openai/gpt-4o-mini", 
            "gpt-4o": "openai/gpt-4o"
        }
    
    def classify_and_extract(self, file_base64: str, model_name: str) -> DocumentResult:
        start_time = time.time()
        
        prompt = self._create_prompt()
        messages = self._create_messages(prompt, file_base64)
        
        try:
            response = self._make_request(messages, model_name)
            result = self._parse_response(response, model_name, time.time() - start_time)
            return result
        except Exception as e:
            return DocumentResult(
                category=DocumentCategory.OTHER,
                entities=ExtractedEntities(),
                processing_time=time.time() - start_time,
                model_used=model_name,
                raw_text=f"Error: {str(e)}"
            )
    
    def _create_prompt(self) -> str:
        return """Analyze this document thoroughly and extract ALL visible information. Be comprehensive and detailed.

Categories: invoice, marketplace_screenshot, chat_screenshot, website_screenshot, other

Extract EVERYTHING you can see - text, numbers, dates, names, addresses, phone numbers, license plates, usernames, prices, quantities, brands, etc. Don't miss any detail.

Examples:
- Chat: Extract all participants, messages, timestamps, platform indicators, phone numbers, profile names
- Marketplace: Get product details, prices, seller info, location, contact details, item condition, specifications
- Invoice: Capture all amounts, dates, line items, tax info, addresses, reference numbers
- Website: Note platform name, user details, account info, displayed content, navigation elements

Be generous with extraction - capture everything visible even if unsure of relevance.

Response format:
{
  "category": "category_name", 
  "confidence": 0.85,
  "entities": {
    "key_detail": "value"
  }
}"""
    
    def _create_messages(self, prompt: str, file_base64: str) -> list:
        content = [{"type": "text", "text": prompt}]
        
        if file_base64.startswith("data:application/pdf"):
            # For PDFs, include as text attachment 
            content.append({"type": "text", "text": f"PDF Document (base64): {file_base64}"})
        else:
            # For images, use image_url
            content.append({"type": "image_url", "image_url": {"url": file_base64}})
        
        return [{"role": "user", "content": content}]
    
    def _make_request(self, messages: list, model_name: str) -> dict:
        payload = {
            "model": self.models[model_name],
            "messages": messages,
            "max_tokens": 3000,
            "temperature": 0
        }
        
        response = requests.post(self.base_url, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def _parse_response(self, response: dict, model_name: str, processing_time: float) -> DocumentResult:
        try:
            content = response['choices'][0]['message']['content']
            
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
            
            data = json.loads(content.strip())
            
            entities_dict = data.get('entities', {})
            entities = ExtractedEntities(**entities_dict)
            
            return DocumentResult(
                category=DocumentCategory(data['category']),
                confidence=data.get('confidence'),
                entities=entities,
                processing_time=processing_time,
                model_used=model_name
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return DocumentResult(
                category=DocumentCategory.OTHER,
                entities=ExtractedEntities(),
                processing_time=processing_time,
                model_used=model_name,
                raw_text=f"Parse error: {str(e)}"
            )
    
    def llm_judge(self, file_base64: str, prediction: dict, judge_model: str = "gpt-4o-mini") -> tuple:
        """Fast LLM judge to evaluate prediction confidence with reasoning"""
        judge_prompt = f"""Evaluate this prediction using the rubric below. Rate 0.0-1.0 confidence.

PREDICTION:
Category: {prediction['category']}
Entities: {json.dumps(prediction['entities'], indent=2)}

RUBRIC:
0.9-1.0: Perfect category + comprehensive extraction (all key details captured)
0.7-0.8: Correct category + good extraction (most important details captured) 
0.5-0.6: Correct category + partial extraction (some key details missing)
0.3-0.4: Wrong category OR major extraction errors
0.0-0.2: Completely wrong

Respond format: "0.85 - Correct category but missing license plate number"

Score and one-line reason:"""
        
        try:
            messages = self._create_messages(judge_prompt, file_base64)
            response = self._make_request(messages, judge_model)
            content = response['choices'][0]['message']['content'].strip()
            
            # Extract confidence score and reasoning
            try:
                parts = content.split(' - ', 1)
                confidence = float(parts[0])
                reasoning = parts[1] if len(parts) > 1 else "No reason provided"
                return max(0.0, min(1.0, confidence)), reasoning
            except:
                return 0.5, "Failed to parse response"
                
        except Exception as e:
            return 0.5, f"Judge error: {str(e)}"
    
    def evaluate_batch(self, files_and_predictions: list) -> list:
        """Parallel evaluation of multiple predictions"""
        def evaluate_single(item):
            file_base64, prediction = item
            confidence, reasoning = self.llm_judge(file_base64, prediction)
            return {"confidence": confidence, "reasoning": reasoning}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            evaluations = list(executor.map(evaluate_single, files_and_predictions))
        
        return evaluations
    
