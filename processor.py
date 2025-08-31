import io
import base64
from PIL import Image
from typing import Tuple

class DocumentProcessor:
    def __init__(self):
        self.supported_formats = {'.pdf', '.png', '.jpg', '.jpeg'}
    
    def process_uploaded_file(self, uploaded_file) -> Tuple[str, str]:
        if uploaded_file.type == 'application/pdf':
            return self._process_pdf_bytes(uploaded_file.read())
        else:
            return self._process_image_bytes(uploaded_file.read())
    
    def _process_pdf_bytes(self, pdf_bytes: bytes) -> Tuple[str, str]:
        # Send PDF directly as base64 to LLM
        pdf_base64 = base64.b64encode(pdf_bytes).decode()
        return "", f"data:application/pdf;base64,{pdf_base64}"
    
    def _process_image_bytes(self, image_bytes: bytes) -> Tuple[str, str]:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
            if img.mode not in ['RGB', 'L']:
                img = img.convert('RGB')
            base64_image = self._pil_to_base64(img)
            return "", base64_image
        except Exception as e:
            raise Exception(f"Failed to process image: {str(e)}")
    
    def _pil_to_base64(self, img: Image.Image) -> str:
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"