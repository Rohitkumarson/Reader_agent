from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import PyPDF2
import io
import os
from typing import List, Dict
import tempfile
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
import logging

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

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration
MAX_PAGES = 50
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Initialize Groq client (optional - for text processing)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("Groq client initialized for text processing")
else:
    groq_client = None
    logger.info("Running without Groq - text will not be AI-processed")

def process_text_with_groq(text: str, page_num: int) -> str:
    """Process extracted PDF text with Groq LLM to clean and structure it"""
    if not groq_client:
        return text  # Return original if Groq not available
    
    if not text or not text.strip():
        return text
    
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a text processing assistant. Clean up and structure the extracted PDF text to make it more readable and suitable for text-to-speech conversion. Remove formatting artifacts, fix broken sentences, and ensure smooth flow. Keep all important content but make it natural for listening. Return ONLY the cleaned text without any commentary."
                },
                {
                    "role": "user",
                    "content": f"Clean this text from page {page_num}:\n\n{text}"
                }
            ],
            temperature=0.3,
            max_completion_tokens=2048,
            top_p=1,
            stream=False
        )
        
        processed_text = completion.choices[0].message.content
        logger.info(f"Successfully processed page {page_num} with Groq")
        return processed_text.strip()
    
    except Exception as e:
        logger.error(f"Groq processing error on page {page_num}: {e}")
        return text  # Fallback to original text

@app.post("/api/parse-pdf")
async def parse_pdf(file: UploadFile = File(...)):
    """
    Parse PDF and return all text line by line
    Frontend will use browser's Speech Synthesis API to speak
    """
    try:
        # Read PDF file
        pdf_content = await file.read()
        pdf_file = io.BytesIO(pdf_content)
        
        # Create PDF reader
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        page_count = len(pdf_reader.pages)
        
        # Check page limit
        if page_count > MAX_PAGES:
            raise HTTPException(
                status_code=400, 
                detail=f"PDF has {page_count} pages. Maximum allowed is {MAX_PAGES} pages."
            )
        
        # Extract all text line by line
        all_lines = []
        
        for page_num in range(page_count):
            page = pdf_reader.pages[page_num]
            raw_text = page.extract_text().strip()
            
            if raw_text:
                # Process with Groq if available
                processed_text = process_text_with_groq(raw_text, page_num + 1)
                
                # Split into sentences for better speech
                # Split by periods, exclamation marks, and question marks
                sentences = []
                current_sentence = ""
                
                for char in processed_text:
                    current_sentence += char
                    if char in '.!?' and len(current_sentence.strip()) > 5:
                        sentences.append(current_sentence.strip())
                        current_sentence = ""
                
                # Add remaining text
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                
                # Add to all lines
                for sentence in sentences:
                    if sentence:  # Only add non-empty sentences
                        all_lines.append({
                            "page": page_num + 1,
                            "text": sentence
                        })
        
        if not all_lines:
            raise HTTPException(status_code=400, detail="No text found in PDF")
        
        logger.info(f"Successfully parsed PDF: {page_count} pages, {len(all_lines)} lines")
        
        return {
            "success": True,
            "page_count": page_count,
            "line_count": len(all_lines),
            "lines": all_lines
        }
        
    except PyPDF2.errors.PdfReadError:
        raise HTTPException(status_code=400, detail="Invalid PDF file")
    except Exception as e:
        logger.error(f"Parse PDF error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    try:
        with open("static/index.html") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Static files not found. Please ensure index.html is in the static folder.</h1>"

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "tts_engine": "Browser Speech Synthesis API",
        "groq_enabled": groq_client is not None,
        "max_pages": MAX_PAGES
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting PDF to Speech Converter with Browser TTS...")
    logger.info("🎉 Using browser's native speech - no backend TTS needed!")
    uvicorn.run(app, host="0.0.0.0", port=8000)