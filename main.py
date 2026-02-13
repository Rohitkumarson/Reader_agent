from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import io
import os
from typing import List, Dict, Optional
import tempfile
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
import logging
import edge_tts
import asyncio
from datetime import datetime
import uuid
from pydantic import BaseModel
from enum import Enum
import json
import aiofiles
import re
from pdf2image import convert_from_bytes
from PIL import Image
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MAX_PAGES = 50  # Maximum pages allowed in document
FREE_TIER_PAGES = 10  # Pages user can process without limit (can be changed)
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEMP_AUDIO_DIR = Path("temp_audio")
TEMP_AUDIO_DIR.mkdir(exist_ok=True)

# Voice configurations with natural names
class VoiceOption(str, Enum):
    INDIAN_FEMALE_WARM = "hi-IN-SwaraNeural"
    INDIAN_MALE_CALM = "hi-IN-MadhurNeural"
    US_FEMALE_NATURAL = "en-US-JennyNeural"
    US_MALE_PROFESSIONAL = "en-US-GuyNeural"
    UK_FEMALE_ELEGANT = "en-GB-SoniaNeural"
    UK_MALE_SOPHISTICATED = "en-GB-RyanNeural"
    AUSTRALIAN_FEMALE = "en-AU-NatashaNeural"
    CANADIAN_FEMALE = "en-CA-ClaraNeural"

# Voice display names for frontend
VOICE_DISPLAY_NAMES = {
    VoiceOption.INDIAN_FEMALE_WARM: "Indian Female (Warm & Clear)",
    VoiceOption.INDIAN_MALE_CALM: "Indian Male (Calm & Steady)",
    VoiceOption.US_FEMALE_NATURAL: "US Female (Natural & Friendly)",
    VoiceOption.US_MALE_PROFESSIONAL: "US Male (Professional & Clear)",
    VoiceOption.UK_FEMALE_ELEGANT: "UK Female (Elegant & Refined)",
    VoiceOption.UK_MALE_SOPHISTICATED: "UK Male (Sophisticated & Deep)",
    VoiceOption.AUSTRALIAN_FEMALE: "Australian Female (Bright & Energetic)",
    VoiceOption.CANADIAN_FEMALE: "Canadian Female (Smooth & Articulate)",
}

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("Groq client initialized for vision and text processing")
else:
    groq_client = None
    logger.warning("Running without Groq - text will not be AI-enhanced")

# Processing queue
processing_queue = asyncio.Queue()
active_jobs = {}


class PageProcessRequest(BaseModel):
    text: str
    page_num: int


def detect_language(text: str) -> str:
    """
    Detect if text is primarily Hindi or English
    Returns 'hindi' or 'english'
    """
    if not text:
        return 'english'
    
    # Count Devanagari characters (Hindi script)
    hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    # Count English alphabet characters
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    
    # If more than 30% Hindi characters, consider it Hindi
    total_chars = len(text.replace(' ', '').replace('\n', ''))
    if total_chars > 0:
        hindi_ratio = hindi_chars / total_chars
        if hindi_ratio > 0.3:
            return 'hindi'
    
    # If significantly more Hindi than English
    if hindi_chars > english_chars:
        return 'hindi'
    
    return 'english'


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


async def extract_text_from_image_with_groq(image: Image.Image, page_num: int) -> tuple:
    """
    Extract text from image using Groq Vision API
    Returns (text, language, confidence)
    """
    if not groq_client:
        return "", "english", 0
    
    try:
        # Convert image to base64
        image_data_url = image_to_base64(image)
        
        # Use Groq Vision API to extract text
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Extract ALL text from this image.

CRITICAL - RETURN ONLY THE TEXT:
- NO introductions like "Here is the text" or "The text reads"
- NO explanations or commentary
- NO markdown formatting or code blocks
- NO line breaks you add yourself
- NO "Here's the extracted text:" or similar phrases
- Just the actual text content from the image, nothing else

LANGUAGE:
- Keep the ORIGINAL language - do NOT translate
- Hindi text must stay in Hindi (Devanagari script)
- English text must stay in English

FORMATTING:
- Preserve paragraph structure from the image
- Remove headers, footers, page numbers
- Convert bullet points to flowing sentences
- Fix obvious typos or broken words

If no readable text: return empty (nothing, not even a message)

Extract now:"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,  # Very low for consistent extraction
            max_tokens=4096,
        )
        
        extracted_text = completion.choices[0].message.content.strip()
        
        # Clean up any LLM artifacts that might have been added
        # Remove common prefixes the LLM might add
        prefixes_to_remove = [
            "here is the text:",
            "here's the text:",
            "the text reads:",
            "extracted text:",
            "here is the extracted text:",
            "here's the extracted text:",
            "the text from the image:",
            "text content:",
            "the content is:",
            "here you go:",
            "sure, here's the text:",
        ]
        
        extracted_lower = extracted_text.lower()
        for prefix in prefixes_to_remove:
            if extracted_lower.startswith(prefix):
                # Remove the prefix (case-insensitive)
                extracted_text = extracted_text[len(prefix):].strip()
                break
        
        # Remove markdown code blocks if present
        if extracted_text.startswith("```") and extracted_text.endswith("```"):
            # Remove first and last lines
            lines = extracted_text.split('\n')
            if len(lines) > 2:
                extracted_text = '\n'.join(lines[1:-1]).strip()
        
        # Detect language
        language = detect_language(extracted_text)
        
        logger.info(f"Page {page_num}: Vision API extracted text ({language})")
        
        # Confidence is high since it's AI extraction, not OCR
        return extracted_text, language, 95
        
    except Exception as e:
        logger.error(f"Groq Vision API error on page {page_num}: {e}")
        return "", "english", 0


async def process_pdf_page_smart(pdf_bytes: bytes, page_num: int) -> tuple:
    """
    Smart PDF page processing:
    1. Try to extract text directly
    2. If no text, convert to image and use Groq Vision
    Returns (text, is_vision, language, confidence)
    """
    try:
        # First, try to extract text normally
        pdf_file = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        if page_num >= len(pdf_reader.pages):
            return "", False, "english", 0
        
        page = pdf_reader.pages[page_num]
        
        try:
            extracted_text = page.extract_text()
        except:
            extracted_text = ""
        
        # If we got meaningful text (more than 50 characters), use it
        if extracted_text and len(extracted_text.strip()) > 50:
            language = detect_language(extracted_text)
            logger.info(f"Page {page_num + 1}: Extracted text directly ({language})")
            return extracted_text.strip(), False, language, 100
        
        # If no text or very little text, use Groq Vision
        logger.info(f"Page {page_num + 1}: No extractable text, using Vision API...")
        
        # Convert PDF page to image
        images = convert_from_bytes(
            pdf_bytes,
            first_page=page_num + 1,
            last_page=page_num + 1,
            dpi=200  # Good balance between quality and speed
        )
        
        if not images:
            return "", False, "english", 0
        
        # Use Groq Vision API to extract text from image
        image = images[0]
        vision_text, language, confidence = await extract_text_from_image_with_groq(image, page_num + 1)
        
        return vision_text, True, language, confidence
        
    except Exception as e:
        logger.error(f"Error processing page {page_num + 1}: {e}")
        return "", False, "english", 0


async def enhance_text_for_audio(text: str, page_num: int, language: str = 'english') -> str:
    """
    Enhance extracted text to make it better for audio narration
    This is DIFFERENT from extraction - this makes it sound better
    """
    if not groq_client:
        return text
    
    if not text or not text.strip():
        return text
    
    try:
        # Enhancement prompt (NOT extraction)
        if language == 'hindi':
            system_prompt = """You are preparing text for audio narration. Make it flow naturally for listening.

CRITICAL - RETURN ONLY THE TEXT:
- NO introductions like "Here is the polished text" or "Ready for narration"
- NO explanations or meta-commentary
- NO phrases like "Here's the enhanced version" or similar
- Just return the improved text directly, nothing else

Your improvements:
1. Keep the SAME LANGUAGE (Hindi stays Hindi)
2. Make sentences flow smoothly
3. Add natural pauses with commas where appropriate
4. Expand abbreviations
5. Make it conversational and engaging
6. Fix grammar issues
7. Smooth transitions

Return the text only."""
        else:
            system_prompt = """You are preparing text for audio narration. Make it flow naturally for listening.

CRITICAL - RETURN ONLY THE TEXT:
- NO introductions like "Here is the polished text" or "Ready for narration"
- NO explanations or meta-commentary
- NO phrases like "Here's the enhanced version" or similar
- Just return the improved text directly, nothing else

Your improvements:
1. Make sentences flow smoothly
2. Add natural pauses with commas
3. Expand abbreviations (Dr. → Doctor)
4. Make it conversational and engaging
5. Fix grammar issues
6. Smooth transitions

Return the text only."""
        
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Prepare this text for audio narration (page {page_num}):\n\n{text}"
                }
            ],
            temperature=0.3,
            max_tokens=4096,
        )
        
        enhanced_text = completion.choices[0].message.content.strip()
        
        # Clean up any LLM artifacts that might have been added
        prefixes_to_remove = [
            "here is the polished text:",
            "here's the polished text:",
            "ready for narration:",
            "polished text:",
            "enhanced text:",
            "here is the text:",
            "here's the text:",
            "here you go:",
            "sure, here's the enhanced version:",
            "here's the enhanced version:",
        ]
        
        enhanced_lower = enhanced_text.lower()
        for prefix in prefixes_to_remove:
            if enhanced_lower.startswith(prefix):
                enhanced_text = enhanced_text[len(prefix):].strip()
                break
        
        # Remove markdown code blocks if present
        if enhanced_text.startswith("```") and enhanced_text.endswith("```"):
            lines = enhanced_text.split('\n')
            if len(lines) > 2:
                enhanced_text = '\n'.join(lines[1:-1]).strip()
        
        # Verify language preservation for Hindi
        if language == 'hindi':
            enhanced_lang = detect_language(enhanced_text)
            if enhanced_lang != 'hindi':
                logger.warning(f"Page {page_num}: Enhancement changed language, using original")
                return text
        
        logger.info(f"Enhanced page {page_num} for audio ({language})")
        return enhanced_text
    
    except Exception as e:
        logger.error(f"Enhancement error on page {page_num}: {e}")
        return text


@app.post("/api/parse-pdf")
async def parse_pdf(file: UploadFile = File(...)):
    """
    Parse PDF and return page-by-page structured data
    Uses Groq Vision API for image-based pages
    Supports both text-based and scanned PDFs
    Works with Hindi and English
    """
    try:
        # Read PDF file
        pdf_content = await file.read()
        pdf_file = io.BytesIO(pdf_content)
        
        # Create PDF reader with error handling
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
        except Exception as e:
            logger.warning(f"PDF reader error: {e}, attempting recovery...")
            try:
                pdf_file.seek(0)
                pdf_reader = PyPDF2.PdfReader(pdf_file, strict=False)
            except:
                return {
                    "success": True,
                    "page_count": 0,
                    "total_pages": 0,
                    "pages": [],
                    "message": "PDF could not be parsed, but file was received",
                    "has_limit": False
                }
        
        page_count = len(pdf_reader.pages)
        
        # Cap total pages to maximum
        total_pages = min(page_count, MAX_PAGES)
        
        # Check if exceeds free tier limit
        has_limit = page_count > FREE_TIER_PAGES
        accessible_pages = FREE_TIER_PAGES if has_limit else total_pages
        locked_pages = total_pages - accessible_pages if has_limit else 0
        
        # Process pages
        pages = []
        
        logger.info(f"Processing PDF: {total_pages} pages")
        
        for page_num in range(total_pages):
            is_accessible = page_num < accessible_pages
            
            try:
                # Smart processing: text extraction or vision API
                extracted_text, is_vision, detected_language, confidence = await process_pdf_page_smart(
                    pdf_content, page_num
                )
                
                if not extracted_text or not extracted_text.strip():
                    pages.append({
                        "page": page_num + 1,
                        "text": f"Page {page_num + 1} contains no readable text.",
                        "has_content": False,
                        "is_accessible": is_accessible,
                        "is_locked": not is_accessible,
                        "language": "english",
                        "extraction_method": "none",
                        "confidence": 0
                    })
                    continue
                
                # Clean text
                cleaned_text = extracted_text.strip()
                
                # Enhance for audio (only if Groq available)
                if groq_client:
                    enhanced_text = await enhance_text_for_audio(cleaned_text, page_num + 1, detected_language)
                else:
                    enhanced_text = cleaned_text
                
                # For locked pages, show preview
                display_text = enhanced_text if is_accessible else enhanced_text[:200] + "... [Locked - Upgrade to access]"
                
                pages.append({
                    "page": page_num + 1,
                    "text": display_text,
                    "full_text": enhanced_text,
                    "has_content": True,
                    "word_count": len(enhanced_text.split()),
                    "is_accessible": is_accessible,
                    "is_locked": not is_accessible,
                    "language": detected_language,
                    "extraction_method": "vision" if is_vision else "text",
                    "confidence": round(confidence, 1)
                })
                
                logger.info(f"Page {page_num + 1}: {'Vision' if is_vision else 'Text'} extraction ({detected_language})")
                
            except Exception as e:
                logger.error(f"Error processing page {page_num + 1}: {e}")
                pages.append({
                    "page": page_num + 1,
                    "text": f"Page {page_num + 1} could not be processed.",
                    "has_content": False,
                    "is_accessible": is_accessible,
                    "is_locked": not is_accessible,
                    "language": "english",
                    "extraction_method": "error",
                    "confidence": 0
                })
        
        if not pages:
            pages.append({
                "page": 1,
                "text": "No content could be extracted from this document.",
                "has_content": False,
                "is_accessible": True,
                "is_locked": False,
                "language": "english",
                "extraction_method": "none",
                "confidence": 0
            })
        
        # Calculate statistics
        vision_pages = sum(1 for p in pages if p.get("extraction_method") == "vision")
        text_pages = sum(1 for p in pages if p.get("extraction_method") == "text")
        
        logger.info(f"Successfully parsed PDF: {total_pages} pages ({vision_pages} vision, {text_pages} text)")
        
        return {
            "success": True,
            "page_count": total_pages,
            "total_pages": page_count,
            "accessible_pages": accessible_pages,
            "locked_pages": locked_pages,
            "pages": pages,
            "has_limit": has_limit,
            "limit_message": f"Free tier: {FREE_TIER_PAGES} pages. Upgrade to unlock all {page_count} pages!" if has_limit else "",
            "capped": page_count > MAX_PAGES,
            "extraction_stats": {
                "vision_pages": vision_pages,
                "text_pages": text_pages,
                "total_processed": len([p for p in pages if p.get("has_content")])
            }
        }
        
    except Exception as e:
        logger.error(f"Critical parse error: {e}")
        return {
            "success": True,
            "page_count": 0,
            "pages": [{
                "page": 1,
                "text": "An unexpected error occurred during processing, but your file was received.",
                "has_content": False
            }],
            "message": "File received but could not be fully processed"
        }


@app.get("/api/voices")
async def get_voices():
    """Get available voice options"""
    voices = []
    for voice_key, display_name in VOICE_DISPLAY_NAMES.items():
        voices.append({
            "id": voice_key.value,
            "name": display_name
        })
    return {"voices": voices}


@app.post("/api/generate-page-audio")
async def generate_page_audio(
    page: int,
    text: str,
    voice: str = VoiceOption.US_FEMALE_NATURAL.value
):
    """Generate audio for a single page and return as streaming response"""
    try:
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="No text provided")
        
        # Validate voice
        valid_voices = [v.value for v in VoiceOption]
        if voice not in valid_voices:
            voice = VoiceOption.US_FEMALE_NATURAL.value
        
        # Generate audio
        audio_data = await generate_audio_stream(text, voice)
        
        if not audio_data:
            raise HTTPException(status_code=500, detail="Audio generation failed")
        
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"inline; filename=page_{page}.mp3"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating audio for page {page}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/download-page-audio")
async def download_page_audio(
    page: int,
    text: str,
    voice: str = VoiceOption.US_FEMALE_NATURAL.value,
    filename: Optional[str] = None
):
    """Generate and download audio for a single page"""
    try:
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="No text provided")
        
        # Validate voice
        valid_voices = [v.value for v in VoiceOption]
        if voice not in valid_voices:
            voice = VoiceOption.US_FEMALE_NATURAL.value
        
        # Generate audio
        audio_data = await generate_audio_stream(text, voice)
        
        if not audio_data:
            raise HTTPException(status_code=500, detail="Audio generation failed")
        
        # Generate filename
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"page_{page}_{timestamp}"
        
        filename = filename.replace('.mp3', '').replace('.wav', '').replace('.', '_')
        
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}.mp3"'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading audio for page {page}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-full-document")
async def generate_full_document(
    pages: List[Dict],
    voice: str = VoiceOption.US_FEMALE_NATURAL.value,
    filename: Optional[str] = None
):
    """Generate audio for entire document with all pages combined"""
    try:
        if not pages:
            raise HTTPException(status_code=400, detail="No pages provided")
        
        # Validate voice
        valid_voices = [v.value for v in VoiceOption]
        if voice not in valid_voices:
            voice = VoiceOption.US_FEMALE_NATURAL.value
        
        # Combine all page texts
        full_text = ""
        for page_data in pages:
            page_num = page_data.get("page", 0)
            text = page_data.get("text", "")
            if text.strip():
                full_text += f"Page {page_num}. {text} ... "
        
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="No content to convert")
        
        # Generate audio
        logger.info(f"Generating full document audio with {len(pages)} pages")
        audio_data = await generate_audio_stream(full_text, voice)
        
        if not audio_data:
            raise HTTPException(status_code=500, detail="Audio generation failed")
        
        # Generate filename
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"full_document_{timestamp}"
        
        filename = filename.replace('.mp3', '').replace('.wav', '').replace('.', '_')
        
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}.mp3"'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating full document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def generate_audio_stream(text: str, voice: str) -> bytes:
    """Generate audio using edge-TTS"""
    try:
        communicate = edge_tts.Communicate(text=text, voice=voice)
        audio_data = b""
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        
        return audio_data
    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        raise
    
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF to Speech Pro - AI Enhanced Voice Generation</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --secondary: #8b5cf6;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --dark: #1f2937;
            --light: #f9fafb;
            --border: #e5e7eb;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            min-height: 100vh;
            padding: 20px;
            background-attachment: fixed;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        .glass {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        }

        header {
            text-align: center;
            padding: 40px;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
            border-radius: 24px 24px 0 0;
            border-bottom: 2px solid rgba(99, 102, 241, 0.1);
        }

        header h1 {
            font-size: 3em;
            font-weight: 800;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 12px;
            letter-spacing: -1px;
        }

        header p {
            color: #6b7280;
            font-size: 1.2em;
            font-weight: 500;
            margin-bottom: 20px;
        }

        .badge-container {
            display: flex;
            gap: 12px;
            justify-content: center;
            flex-wrap: wrap;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 20px;
            border-radius: 50px;
            font-size: 0.9em;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }

        .badge-primary {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
        }

        .badge-success {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
        }

        .main-content {
            padding: 40px;
        }

        .upload-section {
            margin-bottom: 40px;
        }

        .upload-box {
            border: 3px dashed var(--primary);
            border-radius: 20px;
            padding: 80px 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.03) 0%, rgba(139, 92, 246, 0.03) 100%);
            position: relative;
            overflow: hidden;
        }

        .upload-box::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.1), transparent);
            transition: left 0.5s;
        }

        .upload-box:hover::before {
            left: 100%;
        }

        .upload-box:hover {
            border-color: var(--secondary);
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(139, 92, 246, 0.08) 100%);
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(99, 102, 241, 0.2);
        }

        .upload-box.dragover {
            border-color: var(--success);
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
            transform: scale(1.02);
        }

        .upload-icon {
            font-size: 5em;
            margin-bottom: 20px;
            animation: float 3s ease-in-out infinite;
        }

        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }

        .upload-box h3 {
            font-size: 1.5em;
            color: var(--dark);
            margin-bottom: 10px;
            font-weight: 700;
        }

        .upload-box p {
            color: #6b7280;
            font-size: 1.1em;
        }

        .processing-status {
            margin-top: 30px;
            padding: 30px;
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border-radius: 16px;
            border-left: 5px solid var(--primary);
            display: none;
        }

        .processing-status.active {
            display: block;
            animation: slideIn 0.5s ease;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .status-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        .status-header h3 {
            font-size: 1.3em;
            color: var(--primary-dark);
            font-weight: 700;
        }

        .overall-progress {
            margin: 20px 0;
        }

        .progress-bar-container {
            height: 12px;
            background: rgba(99, 102, 241, 0.1);
            border-radius: 10px;
            overflow: hidden;
            position: relative;
        }

        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
            border-radius: 10px;
            transition: width 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .progress-bar-fill::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            animation: shimmer 2s infinite;
        }

        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        .progress-text {
            margin-top: 10px;
            display: flex;
            justify-content: space-between;
            font-size: 0.95em;
            color: #6b7280;
            font-weight: 600;
        }

        .pages-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }

        .page-item {
            background: white;
            border: 2px solid var(--border);
            border-radius: 16px;
            padding: 20px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .page-item::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: linear-gradient(180deg, #6366f1 0%, #8b5cf6 100%);
            transform: scaleY(0);
            transition: transform 0.3s ease;
        }

        .page-item:hover {
            border-color: var(--primary);
            box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15);
            transform: translateY(-4px);
        }

        .page-item:hover::before {
            transform: scaleY(1);
        }

        .page-item.processing {
            border-color: var(--warning);
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.05) 0%, rgba(251, 191, 36, 0.05) 100%);
        }

        .page-item.completed {
            border-color: var(--success);
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.05) 0%, rgba(5, 150, 105, 0.05) 100%);
        }

        .page-item.error {
            border-color: var(--danger);
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.05) 0%, rgba(220, 38, 38, 0.05) 100%);
        }

        .page-item.locked {
            border-color: #9ca3af;
            background: linear-gradient(135deg, rgba(156, 163, 175, 0.05) 0%, rgba(107, 114, 128, 0.05) 100%);
            opacity: 0.7;
            position: relative;
            overflow: hidden;
        }

        .page-item.locked::after {
            content: '🔒';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 4em;
            opacity: 0.1;
        }

        .page-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .page-number {
            font-size: 1.2em;
            font-weight: 700;
            color: var(--primary);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .page-status-badge {
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .status-pending {
            background: #f3f4f6;
            color: #6b7280;
        }

        .status-processing {
            background: #fef3c7;
            color: #d97706;
        }

        .status-completed {
            background: #d1fae5;
            color: #059669;
        }

        .status-error {
            background: #fee2e2;
            color: #dc2626;
        }

        .status-locked {
            background: #fff7ed;
            color: #ea580c;
            border: 2px solid #fed7aa;
        }

        .status-incomplete {
            background: #fef3c7;
            color: #d97706;
        }

        .page-item.locked {
            border-color: #fed7aa;
            background: linear-gradient(135deg, rgba(254, 215, 170, 0.05) 0%, rgba(253, 186, 116, 0.05) 100%);
            opacity: 0.85;
        }

        .page-item.locked:hover {
            opacity: 1;
            border-color: #fb923c;
            cursor: pointer;
        }

        .page-preview {
            color: #4b5563;
            font-size: 0.9em;
            line-height: 1.6;
            max-height: 80px;
            overflow: hidden;
            margin-bottom: 15px;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
        }

        .page-meta {
            display: flex;
            gap: 20px;
            margin-bottom: 15px;
            font-size: 0.85em;
            color: #6b7280;
        }

        .meta-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .page-actions {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 10px;
            font-size: 0.9em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            display: inline-flex;
            align-items: center;
            gap: 8px;
            position: relative;
            overflow: hidden;
        }

        .btn::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.3);
            transform: translate(-50%, -50%);
            transition: width 0.5s, height 0.5s;
        }

        .btn:hover::before {
            width: 300px;
            height: 300px;
        }

        .btn:active {
            transform: scale(0.95);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .btn-primary {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        }

        .btn-primary:hover:not(:disabled) {
            box-shadow: 0 6px 25px rgba(99, 102, 241, 0.4);
            transform: translateY(-2px);
        }

        .btn-success {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        }

        .btn-success:hover:not(:disabled) {
            box-shadow: 0 6px 25px rgba(16, 185, 129, 0.4);
            transform: translateY(-2px);
        }

        .btn-download {
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
        }

        .btn-download:hover:not(:disabled) {
            box-shadow: 0 6px 25px rgba(245, 158, 11, 0.4);
            transform: translateY(-2px);
        }

        .voice-control {
            margin: 30px 0;
            padding: 30px;
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            border-radius: 16px;
            border-left: 5px solid var(--warning);
        }

        .voice-control h3 {
            color: #92400e;
            margin-bottom: 20px;
            font-size: 1.3em;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .voice-control select {
            width: 100%;
            padding: 14px 18px;
            border: 2px solid #f59e0b;
            border-radius: 12px;
            font-size: 1em;
            font-weight: 500;
            background: white;
            cursor: pointer;
            transition: all 0.3s ease;
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23f59e0b' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 12px center;
            background-size: 20px;
            padding-right: 45px;
        }

        .voice-control select:hover {
            border-color: #d97706;
            box-shadow: 0 4px 15px rgba(245, 158, 11, 0.2);
        }

        .voice-control select:focus {
            outline: none;
            border-color: #d97706;
            box-shadow: 0 0 0 4px rgba(245, 158, 11, 0.1);
        }

        .download-all-section {
            margin-top: 30px;
            padding: 30px;
            background: linear-gradient(135deg, #ede9fe 0%, #ddd6fe 100%);
            border-radius: 16px;
            text-align: center;
        }

        .audio-player {
            margin-top: 15px;
            padding: 15px;
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border-radius: 12px;
            display: none;
        }

        .audio-player.active {
            display: block;
            animation: slideIn 0.3s ease;
        }

        .audio-player audio {
            width: 100%;
            height: 45px;
        }

        .inline-transcript {
            margin-top: 15px;
            padding: 15px;
            background: white;
            border-radius: 10px;
            max-height: 200px;
            overflow-y: auto;
            line-height: 1.8;
            font-size: 0.95em;
            color: #374151;
            box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.05);
            display: none;
        }

        .inline-transcript.active {
            display: block;
        }

        .inline-transcript .word {
            display: inline;
            padding: 2px 1px;
            transition: all 0.2s ease;
        }

        .inline-transcript .word.current {
            background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
            color: #78350f;
            padding: 3px 6px;
            border-radius: 4px;
            font-weight: 600;
            box-shadow: 0 2px 8px rgba(251, 191, 36, 0.3);
        }

        .inline-transcript .word.passed {
            color: #9ca3af;
        }

        .transcript-progress {
            height: 3px;
            background: #e5e7eb;
            border-radius: 3px;
            overflow: hidden;
            margin-top: 10px;
        }

        .transcript-progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #10b981 0%, #059669 100%);
            width: 0%;
            transition: width 0.3s ease;
        }

        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            backdrop-filter: blur(5px);
            z-index: 1000;
            align-items: center;
            justify-content: center;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }

        .modal.active {
            display: flex;
        }

        .modal-content {
            background: white;
            padding: 40px;
            border-radius: 20px;
            max-width: 550px;
            width: 90%;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            animation: modalSlide 0.3s ease;
        }

        @keyframes modalSlide {
            from {
                opacity: 0;
                transform: translateY(-50px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .modal-header h2 {
            color: var(--dark);
            font-size: 1.8em;
            margin-bottom: 10px;
            font-weight: 700;
        }

        .modal-header p {
            color: #6b7280;
            margin-bottom: 25px;
        }

        .modal-body input {
            width: 100%;
            padding: 14px 18px;
            border: 2px solid var(--border);
            border-radius: 12px;
            font-size: 1em;
            transition: all 0.3s ease;
        }

        .modal-body input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1);
        }

        .modal-footer {
            display: flex;
            gap: 12px;
            margin-top: 25px;
        }

        .modal-footer .btn {
            flex: 1;
        }

        .loading-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            z-index: 999;
            align-items: center;
            justify-content: center;
        }

        .loading-overlay.active {
            display: flex;
        }

        .loading-content {
            text-align: center;
        }

        .spinner {
            width: 70px;
            height: 70px;
            margin: 0 auto 25px;
            border: 6px solid rgba(99, 102, 241, 0.1);
            border-top: 6px solid var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .loading-text {
            font-size: 1.3em;
            color: var(--dark);
            font-weight: 600;
            margin-bottom: 10px;
        }

        .loading-subtext {
            color: #6b7280;
            font-size: 1em;
        }

        .toast-container {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1001;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .toast {
            padding: 16px 24px;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
            display: flex;
            align-items: center;
            gap: 12px;
            min-width: 300px;
            max-width: 500px;
            animation: toastSlide 0.3s ease;
            font-weight: 500;
        }

        @keyframes toastSlide {
            from {
                opacity: 0;
                transform: translateX(100%);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        .toast-success {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
        }

        .toast-error {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
        }

        .toast-info {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
        }

        .toast-warning {
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            color: white;
        }

        .hidden {
            display: none !important;
        }

        @media (max-width: 768px) {
            header h1 {
                font-size: 2em;
            }
            
            .pages-grid {
                grid-template-columns: 1fr;
            }

            .main-content {
                padding: 20px;
            }

            .modal-content {
                padding: 25px;
            }
        }

        .processing-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--warning);
            animation: pulse 1.5s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% {
                opacity: 1;
                transform: scale(1);
            }
            50% {
                opacity: 0.5;
                transform: scale(1.2);
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="glass">
            <header>
                <h1>🎙️ PDF to Speech Pro</h1>
                <p>Transform Your Documents into Natural AI-Powered Speech</p>
                <div class="badge-container">
                    <span class="badge badge-primary">🤖 AI Enhanced</span>
                    <span class="badge badge-success">🎵 Premium Voices</span>
                    <span class="badge badge-primary">⚡ Real-time Processing</span>
                </div>
            </header>

            <div class="main-content">
                <!-- Upload Section -->
                <div class="upload-section">
                    <div class="upload-box" id="uploadBox">
                        <div class="upload-icon">📄</div>
                        <h3>Drop your PDF here or click to browse</h3>
                        <p>Maximum 50 pages • AI-enhanced narration • Multiple voice options</p>
                        <input type="file" id="fileInput" accept=".pdf" hidden>
                    </div>
                </div>

                <!-- Voice Selection -->
                <div class="voice-control hidden" id="voiceControl">
                    <h3>🎤 Select Your Voice</h3>
                    <select id="voiceSelect">
                        <option value="">Loading voices...</option>
                    </select>
                </div>

                <!-- Processing Status -->
                <div class="processing-status" id="processingStatus">
                    <div class="status-header">
                        <h3>📊 Processing Status</h3>
                        <div id="overallStatus">
                            <span class="page-status-badge status-pending">Preparing...</span>
                        </div>
                    </div>
                    
                    <div class="overall-progress">
                        <div class="progress-bar-container">
                            <div class="progress-bar-fill" id="overallProgressFill" style="width: 0%"></div>
                        </div>
                        <div class="progress-text">
                            <span id="progressText">0 of 0 pages processed</span>
                            <span id="progressPercent">0%</span>
                        </div>
                    </div>

                    <div class="pages-grid" id="pagesGrid"></div>
                </div>

                <!-- Download All Section -->
                <div class="download-all-section hidden" id="downloadAllSection">
                    <h3 style="margin-bottom: 15px; color: var(--primary-dark); font-size: 1.3em;">🎧 Ready to Download</h3>
                    <p style="color: #6b7280; margin-bottom: 20px;">All pages have been processed and are ready for download</p>
                    <button class="btn btn-primary" id="downloadAllBtn" style="font-size: 1.1em; padding: 14px 30px;">
                        📥 Download Complete Document Audio
                    </button>
                </div>
            </div>
        </div>
    </div>

    <!-- Loading Overlay -->
    <div class="loading-overlay" id="loadingOverlay">
        <div class="loading-content">
            <div class="spinner"></div>
            <div class="loading-text" id="loadingText">Processing...</div>
            <div class="loading-subtext" id="loadingSubtext">Please wait while we process your document</div>
        </div>
    </div>

    <!-- Download Modal -->
    <div class="modal" id="downloadModal">
        <div class="modal-content">
            <div class="modal-header">
                <h2>💾 Save Audio File</h2>
                <p>Choose a name for your audio file</p>
            </div>
            <div class="modal-body">
                <input type="text" id="downloadFilename" placeholder="Enter filename (e.g., my-document)">
                <p style="margin-top: 12px; color: #6b7280; font-size: 0.9em;">
                    📝 The file will be saved as <strong>.mp3</strong> format automatically
                </p>
            </div>
            <div class="modal-footer">
                <button class="btn" style="background: #6b7280; color: white;" id="cancelDownload">
                    Cancel
                </button>
                <button class="btn btn-download" id="confirmDownload">
                    💾 Download
                </button>
            </div>
        </div>
    </div>

    <!-- Toast Container -->
    <div class="toast-container" id="toastContainer"></div>

    <script>
        const API_URL = window.location.origin;
        let pdfData = null;
        let selectedVoice = null;
        let currentDownloadData = null;
        let processedPages = 0;
        let totalPages = 0;
        let accessiblePages = 0;
        let lockedPages = 0;

        // DOM Elements
        const uploadBox = document.getElementById('uploadBox');
        const fileInput = document.getElementById('fileInput');
        const voiceControl = document.getElementById('voiceControl');
        const voiceSelect = document.getElementById('voiceSelect');
        const processingStatus = document.getElementById('processingStatus');
        const pagesGrid = document.getElementById('pagesGrid');
        const overallStatus = document.getElementById('overallStatus');
        const overallProgressFill = document.getElementById('overallProgressFill');
        const progressText = document.getElementById('progressText');
        const progressPercent = document.getElementById('progressPercent');
        const downloadAllSection = document.getElementById('downloadAllSection');
        const downloadAllBtn = document.getElementById('downloadAllBtn');
        const loadingOverlay = document.getElementById('loadingOverlay');
        const loadingText = document.getElementById('loadingText');
        const loadingSubtext = document.getElementById('loadingSubtext');
        const downloadModal = document.getElementById('downloadModal');
        const downloadFilename = document.getElementById('downloadFilename');
        const cancelDownload = document.getElementById('cancelDownload');
        const confirmDownload = document.getElementById('confirmDownload');
        const toastContainer = document.getElementById('toastContainer');

        // Load voices
        async function loadVoices() {
            try {
                const response = await fetch(`${API_URL}/api/voices`);
                const data = await response.json();
                
                voiceSelect.innerHTML = '';
                data.voices.forEach((voice, index) => {
                    const option = document.createElement('option');
                    option.value = voice.id;
                    option.textContent = voice.name;
                    if (index === 0) {
                        option.selected = true;
                        selectedVoice = voice.id;
                    }
                    voiceSelect.appendChild(option);
                });
            } catch (error) {
                console.error('Error loading voices:', error);
                showToast('Failed to load voices', 'error');
            }
        }

        // Voice selection
        voiceSelect.addEventListener('change', (e) => {
            selectedVoice = e.target.value;
            showToast('Voice changed successfully!', 'success');
        });

        // Upload handling
        uploadBox.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', handleFileSelect);

        uploadBox.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadBox.classList.add('dragover');
        });

        uploadBox.addEventListener('dragleave', () => {
            uploadBox.classList.remove('dragover');
        });

        uploadBox.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadBox.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                handleFileSelect({ target: fileInput });
            }
        });

        async function handleFileSelect(e) {
            const file = e.target.files[0];
            if (!file) return;
            
            if (file.type !== 'application/pdf') {
                showToast('Please select a valid PDF file', 'error');
                return;
            }

            showLoading('Parsing PDF with AI enhancement...', 'This may take a moment for large documents');
            
            try {
                const formData = new FormData();
                formData.append('file', file);
                
                const response = await fetch(`${API_URL}/api/parse-pdf`, {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success && data.pages.length > 0) {
                    pdfData = data;
                    totalPages = data.page_count;
                    accessiblePages = data.accessible_pages;
                    lockedPages = data.locked_pages || 0;
                    processedPages = 0;
                    
                    // Show UI sections
                    voiceControl.classList.remove('hidden');
                    processingStatus.classList.add('active');
                    
                    // Initialize pages grid
                    initializePagesGrid(data.pages);
                    
                    hideLoading();
                    
                    // Show info about page limits
                    if (data.has_limit) {
                        showToast(`📄 ${accessiblePages} pages ready. ${lockedPages} pages marked as incomplete - click to process!`, 'info');
                    } else {
                        showToast(`PDF loaded! ${accessiblePages} pages ready to process.`, 'success');
                    }
                    
                    updateOverallProgress();
                } else {
                    hideLoading();
                    showToast('No content could be extracted from PDF', 'error');
                }
                
            } catch (error) {
                hideLoading();
                console.error('Parse error:', error);
                showToast('Failed to parse PDF. Please try again.', 'error');
            }
        }

        function initializePagesGrid(pages) {
            pagesGrid.innerHTML = '';
            
            pages.forEach((page, index) => {
                const pageCard = createPageCard(page, index);
                pagesGrid.appendChild(pageCard);
                
                // Animate cards in sequence
                setTimeout(() => {
                    pageCard.style.animation = 'slideIn 0.5s ease forwards';
                }, index * 50);
            });
        }

        function createPageCard(page, index) {
            const card = document.createElement('div');
            card.className = 'page-item';
            if (page.is_locked) {
                card.classList.add('locked');
            }
            card.id = `page-${page.page}`;
            card.style.opacity = '0';
            
            const previewText = page.is_locked 
                ? (page.text.substring(0, 150) + (page.text.length > 150 ? '...' : ''))
                : (page.text.substring(0, 150) + (page.text.length > 150 ? '...' : ''));
            const wordCount = page.word_count || 0;
            
            const statusBadge = page.is_locked 
                ? '<span class="page-status-badge status-incomplete" id="status-' + page.page + '">⏸️ Incomplete</span>'
                : '<span class="page-status-badge status-pending" id="status-' + page.page + '">⏳ Pending</span>';
            
            const actionButtons = page.is_locked
                ? `<button class="btn btn-warning" id="process-${page.page}" onclick="processLockedPage(${page.page})" style="background: linear-gradient(135deg, #fb923c 0%, #f97316 100%); color: white;">
                        🔄 Process Page
                   </button>`
                : `<button class="btn btn-success" id="play-${page.page}" onclick="playPage(${page.page})">
                        ▶️ Play
                   </button>
                   <button class="btn btn-download" id="download-${page.page}" onclick="downloadPage(${page.page})">
                        💾 Download
                   </button>`;
            
            card.innerHTML = `
                <div class="page-header">
                    <div class="page-number">
                        📄 Page ${page.page}
                    </div>
                    ${statusBadge}
                </div>
                <div class="page-preview">${previewText}</div>
                <div class="page-meta">
                    <div class="meta-item">
                        <span>📝</span>
                        <span>${wordCount} words${page.is_locked ? ' (locked)' : ''}</span>
                    </div>
                    <div class="meta-item">
                        <span>⏱️</span>
                        <span id="time-${page.page}">--</span>
                    </div>
                </div>
                <div class="page-actions">
                    ${actionButtons}
                </div>
                <div class="audio-player" id="player-${page.page}">
                    <audio controls id="audio-${page.page}"></audio>
                    <div class="inline-transcript" id="transcript-${page.page}"></div>
                    <div class="transcript-progress">
                        <div class="transcript-progress-bar" id="transcript-progress-${page.page}"></div>
                    </div>
                </div>
            `;
            
            return card;
        }

        async function processLockedPage(pageNum) {
            const page = pdfData.pages.find(p => p.page === pageNum);
            if (!page) return;

            const card = document.getElementById(`page-${pageNum}`);
            const statusBadge = document.getElementById(`status-${pageNum}`);
            const processBtn = document.getElementById(`process-${pageNum}`);
            
            // Update to processing
            card.classList.remove('locked');
            card.classList.add('processing');
            statusBadge.className = 'page-status-badge status-processing';
            statusBadge.innerHTML = '<span class="processing-indicator"></span> Processing...';
            processBtn.disabled = true;
            processBtn.textContent = '⏳ Processing...';
            
            showToast(`Processing page ${pageNum} in background...`, 'info');
            
            // Simulate processing (in real app, you'd call backend to process)
            setTimeout(() => {
                // Use full_text for unlocked page
                page.text = page.full_text;
                page.is_locked = false;
                page.is_accessible = true;
                
                // Update card to ready state
                card.classList.remove('processing');
                card.classList.add('completed');
                statusBadge.className = 'page-status-badge status-pending';
                statusBadge.innerHTML = '⏳ Ready';
                
                // Update word count display
                const wordCountMeta = card.querySelector('.page-meta .meta-item span:nth-child(2)');
                if (wordCountMeta) {
                    wordCountMeta.textContent = page.word_count + ' words';
                }
                
                // Update preview text
                const previewDiv = card.querySelector('.page-preview');
                if (previewDiv) {
                    const previewText = page.full_text.substring(0, 150) + (page.full_text.length > 150 ? '...' : '');
                    previewDiv.textContent = previewText;
                }
                
                // Replace button with play/download
                const actionsDiv = card.querySelector('.page-actions');
                actionsDiv.innerHTML = `
                    <button class="btn btn-success" id="play-${pageNum}" onclick="playPage(${pageNum})">
                        ▶️ Play
                    </button>
                    <button class="btn btn-download" id="download-${pageNum}" onclick="downloadPage(${pageNum})">
                        💾 Download
                    </button>
                `;
                
                showToast(`Page ${pageNum} is now ready! ${page.word_count} words available.`, 'success');
                
                // Update accessible count
                accessiblePages++;
                updateOverallProgress();
            }, 2000); // 2 second simulated processing
        }

        async function playPage(pageNum) {
            if (!selectedVoice) {
                showToast('Please select a voice first', 'error');
                return;
            }

            const page = pdfData.pages.find(p => p.page === pageNum);
            if (!page || page.is_locked) {
                showToast('This page is incomplete. Click "Process Page" first!', 'warning');
                return;
            }

            const card = document.getElementById(`page-${pageNum}`);
            const statusBadge = document.getElementById(`status-${pageNum}`);
            const audioPlayer = document.getElementById(`player-${pageNum}`);
            const audioElement = document.getElementById(`audio-${pageNum}`);
            const transcriptDiv = document.getElementById(`transcript-${pageNum}`);
            const transcriptProgress = document.getElementById(`transcript-progress-${pageNum}`);
            const playBtn = document.getElementById(`play-${pageNum}`);
            const timeDisplay = document.getElementById(`time-${pageNum}`);
            
            // Scroll page card into view smoothly
            card.scrollIntoView({ behavior: 'smooth', block: 'center' });
            
            // Update status to processing
            card.classList.add('processing');
            statusBadge.className = 'page-status-badge status-processing';
            statusBadge.innerHTML = '<span class="processing-indicator"></span> Generating...';
            playBtn.disabled = true;
            
            const startTime = Date.now();
            
            try {
                const response = await fetch(`${API_URL}/api/generate-page-audio?page=${pageNum}&voice=${selectedVoice}&text=${encodeURIComponent(page.text)}`, {
                    method: 'POST'
                });
                
                if (!response.ok) {
                    throw new Error('Audio generation failed');
                }
                
                const blob = await response.blob();
                const audioUrl = URL.createObjectURL(blob);
                
                audioElement.src = audioUrl;
                audioPlayer.classList.add('active');
                
                // Setup inline transcript
                setupInlineTranscript(page.text, audioElement, transcriptDiv, transcriptProgress, pageNum);
                
                // Auto-play the audio
                await audioElement.play();
                
                // Update status to completed
                card.classList.remove('processing');
                card.classList.add('completed');
                statusBadge.className = 'page-status-badge status-completed';
                statusBadge.innerHTML = '✅ Playing';
                
                const processingTime = ((Date.now() - startTime) / 1000).toFixed(1);
                timeDisplay.textContent = `${processingTime}s`;
                
                playBtn.disabled = false;
                
                // Update progress
                if (!card.dataset.processed) {
                    card.dataset.processed = 'true';
                    processedPages++;
                    updateOverallProgress();
                }
                
                // Update status when audio ends
                audioElement.onended = () => {
                    statusBadge.innerHTML = '✅ Ready';
                    showToast(`Page ${pageNum} playback completed`, 'success');
                };
                
                showToast(`Playing page ${pageNum}`, 'success');
                
            } catch (error) {
                console.error('Play error:', error);
                card.classList.remove('processing');
                card.classList.add('error');
                statusBadge.className = 'page-status-badge status-error';
                statusBadge.innerHTML = '❌ Failed';
                playBtn.disabled = false;
                showToast(`Failed to generate audio for page ${pageNum}`, 'error');
            }
        }

        function setupInlineTranscript(text, audioElement, transcriptDiv, progressBar, pageNum) {
            // Split text into words
            const words = text.split(/\s+/);
            let currentWordIndex = 0;
            let transcriptInterval = null;
            
            // Create word spans
            transcriptDiv.innerHTML = words.map((word, idx) => 
                `<span class="word" id="word-${pageNum}-${idx}">${word}</span>`
            ).join(' ');
            
            transcriptDiv.classList.add('active');
            
            // Calculate words per second (average speaking rate)
            const wordsPerSecond = 2.5;
            const msPerWord = 1000 / wordsPerSecond;
            
            // Handle play event
            const handlePlay = () => {
                currentWordIndex = 0;
                
                // Clear existing interval
                if (transcriptInterval) {
                    clearInterval(transcriptInterval);
                }
                
                // Start highlighting words
                transcriptInterval = setInterval(() => {
                    if (currentWordIndex < words.length && !audioElement.paused) {
                        // Mark previous word as passed
                        if (currentWordIndex > 0) {
                            const prevWord = document.getElementById(`word-${pageNum}-${currentWordIndex - 1}`);
                            if (prevWord) {
                                prevWord.classList.remove('current');
                                prevWord.classList.add('passed');
                            }
                        }
                        
                        // Highlight current word
                        const currentWord = document.getElementById(`word-${pageNum}-${currentWordIndex}`);
                        if (currentWord) {
                            currentWord.classList.add('current');
                            
                            // Auto-scroll transcript to current word
                            const transcriptRect = transcriptDiv.getBoundingClientRect();
                            const wordRect = currentWord.getBoundingClientRect();
                            
                            if (wordRect.top < transcriptRect.top || wordRect.bottom > transcriptRect.bottom) {
                                currentWord.scrollIntoView({ behavior: 'smooth', block: 'center' });
                            }
                        }
                        
                        // Update progress bar
                        const progress = ((currentWordIndex + 1) / words.length) * 100;
                        progressBar.style.width = progress + '%';
                        
                        currentWordIndex++;
                    }
                }, msPerWord);
            };
            
            // Handle pause event
            const handlePause = () => {
                if (transcriptInterval) {
                    clearInterval(transcriptInterval);
                }
            };
            
            // Handle ended event
            const handleEnded = () => {
                if (transcriptInterval) {
                    clearInterval(transcriptInterval);
                }
                
                // Mark all words as passed
                document.querySelectorAll(`#transcript-${pageNum} .word`).forEach(word => {
                    word.classList.remove('current');
                    word.classList.add('passed');
                });
                
                progressBar.style.width = '100%';
            };
            
            // Handle seeking
            const handleSeeked = () => {
                // Reset all words
                document.querySelectorAll(`#transcript-${pageNum} .word`).forEach(word => {
                    word.classList.remove('current', 'passed');
                });
                
                // Estimate current word based on time
                const currentTime = audioElement.currentTime;
                const duration = audioElement.duration;
                if (duration > 0) {
                    const estimatedWordIndex = Math.floor((currentTime / duration) * words.length);
                    currentWordIndex = estimatedWordIndex;
                    
                    // Mark words before current as passed
                    for (let i = 0; i < currentWordIndex; i++) {
                        const word = document.getElementById(`word-${pageNum}-${i}`);
                        if (word) {
                            word.classList.add('passed');
                        }
                    }
                    
                    progressBar.style.width = ((currentWordIndex / words.length) * 100) + '%';
                }
            };
            
            // Add event listeners
            audioElement.addEventListener('play', handlePlay);
            audioElement.addEventListener('pause', handlePause);
            audioElement.addEventListener('ended', handleEnded);
            audioElement.addEventListener('seeked', handleSeeked);
        }

        function downloadPage(pageNum) {
            const page = pdfData.pages.find(p => p.page === pageNum);
            if (!page) return;
            
            currentDownloadData = { 
                pageNum, 
                text: page.text,
                type: 'single'
            };
            
            downloadFilename.value = `page_${pageNum}_${Date.now()}`;
            downloadModal.classList.add('active');
            downloadFilename.focus();
            downloadFilename.select();
        }

        cancelDownload.addEventListener('click', () => {
            downloadModal.classList.remove('active');
            currentDownloadData = null;
        });

        confirmDownload.addEventListener('click', async () => {
            if (!currentDownloadData) return;
            
            const filename = downloadFilename.value.trim() || `page_${currentDownloadData.pageNum}`;
            downloadModal.classList.remove('active');
            
            if (currentDownloadData.type === 'single') {
                await downloadSinglePage(currentDownloadData.pageNum, currentDownloadData.text, filename);
            } else {
                await downloadFullDocument(filename);
            }
            
            currentDownloadData = null;
        });

        async function downloadSinglePage(pageNum, text, filename) {
            if (!selectedVoice) {
                showToast('Please select a voice first', 'error');
                return;
            }

            showLoading('Generating audio file...', 'Preparing your download');
            
            try {
                const response = await fetch(`${API_URL}/api/download-page-audio?page=${pageNum}&voice=${selectedVoice}&filename=${encodeURIComponent(filename)}&text=${encodeURIComponent(text)}`, {
                    method: 'POST'
                });
                
                if (!response.ok) {
                    throw new Error('Download failed');
                }
                
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${filename}.mp3`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
                hideLoading();
                showToast(`Page ${pageNum} downloaded successfully!`, 'success');
                
            } catch (error) {
                hideLoading();
                console.error('Download error:', error);
                showToast(`Failed to download page ${pageNum}`, 'error');
            }
        }

        downloadAllBtn.addEventListener('click', () => {
            if (!pdfData || !selectedVoice) {
                showToast('Please select a voice first', 'error');
                return;
            }

            currentDownloadData = {
                type: 'full'
            };
            
            downloadFilename.value = `full_document_${Date.now()}`;
            downloadModal.classList.add('active');
            downloadFilename.focus();
            downloadFilename.select();
        });

        async function downloadFullDocument(filename) {
            if (!pdfData || !selectedVoice) {
                showToast('Please select a voice first', 'error');
                return;
            }

            showLoading('Generating complete document audio...', 'This may take several minutes for large documents');
            
            try {
                const response = await fetch(`${API_URL}/api/generate-full-document?voice=${selectedVoice}&filename=${encodeURIComponent(filename)}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(pdfData.pages)
                });
                
                if (!response.ok) {
                    throw new Error('Generation failed');
                }
                
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${filename}.mp3`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
                hideLoading();
                showToast('Full document downloaded successfully!', 'success');
                
            } catch (error) {
                hideLoading();
                console.error('Full document error:', error);
                showToast('Failed to generate full document audio', 'error');
            }
        }

        function updateOverallProgress() {
            const percent = accessiblePages > 0 ? Math.round((processedPages / accessiblePages) * 100) : 0;
            
            overallProgressFill.style.width = percent + '%';
            progressText.textContent = `${processedPages} of ${accessiblePages} pages processed`;
            progressPercent.textContent = percent + '%';
            
            if (processedPages === 0) {
                overallStatus.innerHTML = '<span class="page-status-badge status-pending">Ready to start</span>';
            } else if (processedPages < accessiblePages) {
                overallStatus.innerHTML = '<span class="page-status-badge status-processing"><span class="processing-indicator"></span> Processing...</span>';
            } else {
                overallStatus.innerHTML = '<span class="page-status-badge status-completed">✅ All pages completed</span>';
                downloadAllSection.classList.remove('hidden');
                downloadAllSection.style.animation = 'slideIn 0.5s ease';
                
                if (lockedPages > 0) {
                    showToast(`🎉 All ${accessiblePages} pages complete! ${lockedPages} incomplete pages can be processed.`, 'success');
                }
            }
        }

        function showLoading(text, subtext = '') {
            loadingText.textContent = text;
            loadingSubtext.textContent = subtext;
            loadingOverlay.classList.add('active');
        }

        function hideLoading() {
            loadingOverlay.classList.remove('active');
        }

        function showToast(message, type = 'info') {
            const toast = document.createElement('div');
            toast.className = `toast toast-${type}`;
            
            const icons = {
                success: '✅',
                error: '❌',
                warning: '⚠️',
                info: 'ℹ️'
            };
            
            toast.innerHTML = `
                <span style="font-size: 1.3em;">${icons[type]}</span>
                <span>${message}</span>
            `;
            
            toastContainer.appendChild(toast);
            
            setTimeout(() => {
                toast.style.animation = 'toastSlide 0.3s ease reverse';
                setTimeout(() => toast.remove(), 300);
            }, 5000);
        }

        // Initialize
        loadVoices();
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting PDF to Speech Pro with Edge TTS + Groq AI...")
    logger.info("🚀 Features: AI-enhanced narration, multiple voices, page-wise streaming, downloads")
    uvicorn.run(app, host="0.0.0.0", port=8000)
