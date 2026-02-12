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
# app.mount("/static", StaticFiles(directory="static"), name="static")

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
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF to Speech - Browser Edition</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            max-width: 700px;
            width: 100%;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }

        header {
            text-align: center;
            margin-bottom: 30px;
        }

        header h1 {
            color: #333;
            font-size: 2em;
            margin-bottom: 10px;
        }

        header p {
            color: #666;
            font-size: 0.95em;
        }

        .browser-badge {
            display: inline-block;
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin-top: 10px;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }

        .upload-section {
            margin-bottom: 30px;
        }

        .upload-box {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 50px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #f8f9ff;
        }

        .upload-box:hover {
            border-color: #764ba2;
            background: #f0f1ff;
            transform: translateY(-2px);
        }

        .upload-box.dragover {
            border-color: #764ba2;
            background: #e8e9ff;
            transform: scale(1.02);
        }

        .upload-icon {
            width: 60px;
            height: 60px;
            color: #667eea;
            margin-bottom: 15px;
        }

        .file-info {
            margin-top: 20px;
            padding: 15px;
            background: #f0f9ff;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }

        .error-message, .success-message {
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 4px solid;
        }

        .error-message {
            background: #fee;
            color: #c33;
            border-color: #c33;
        }

        .success-message {
            background: #efe;
            color: #3c3;
            border-color: #3c3;
        }

        .controls {
            margin: 30px 0;
        }

        .btn {
            width: 100%;
            padding: 15px 25px;
            border: none;
            border-radius: 10px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .player {
            background: #f8f9ff;
            padding: 25px;
            border-radius: 15px;
            margin-top: 30px;
        }

        .player-controls {
            display: flex;
            gap: 10px;
            margin-top: 15px;
            flex-wrap: wrap;
        }

        .player-btn {
            flex: 1;
            min-width: 100px;
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .player-btn.play {
            background: #10b981;
            color: white;
        }

        .player-btn.pause {
            background: #f59e0b;
            color: white;
        }

        .player-btn.stop {
            background: #ef4444;
            color: white;
        }

        .player-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }

        .progress-info {
            margin-top: 15px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
        }

        .progress-bar {
            height: 8px;
            background: #e5e7eb;
            border-radius: 4px;
            overflow: hidden;
            margin: 10px 0;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
        }

        .current-text {
            margin-top: 15px;
            padding: 15px;
            background: #fef3c7;
            border-radius: 8px;
            border-left: 4px solid #f59e0b;
            font-style: italic;
            color: #92400e;
            max-height: 150px;
            overflow-y: auto;
        }

        .loading {
            text-align: center;
            margin: 30px 0;
        }

        .spinner {
            width: 50px;
            height: 50px;
            margin: 0 auto 15px;
            border: 5px solid #f3f3f3;
            border-top: 5px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .voice-selector {
            margin: 15px 0;
            padding: 15px;
            background: white;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
        }

        .voice-selector select {
            width: 100%;
            padding: 10px;
            border: 2px solid #667eea;
            border-radius: 8px;
            font-size: 1em;
            background: white;
            cursor: pointer;
        }

        .speed-control {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 10px;
        }

        .speed-control input {
            flex: 1;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📄 PDF to Speech</h1>
            <p>Upload PDF - Browser Reads It Aloud</p>
            <span class="browser-badge">🎤 Browser Native Speech</span>
        </header>

        <div class="upload-section">
            <div class="upload-box" id="uploadBox">
                <svg class="upload-icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <p>Drag & drop PDF or click to browse</p>
                <input type="file" id="fileInput" accept=".pdf" hidden>
            </div>
            <div class="file-info" id="fileInfo" style="display: none;">
                <p><strong>File:</strong> <span id="fileName"></span></p>
                <p><strong>Pages:</strong> <span id="pageCount"></span></p>
                <p><strong>Lines:</strong> <span id="lineCount"></span></p>
            </div>
        </div>

        <div class="error-message" id="errorMessage" style="display: none;"></div>
        <div class="success-message" id="successMessage" style="display: none;"></div>

        <div class="controls" id="controls" style="display: none;">
            <button class="btn" id="startBtn">
                <span>▶️ Start Reading</span>
            </button>
        </div>

        <div class="player" id="player" style="display: none;">
            <h3>🎧 Now Playing</h3>
            
            <div class="voice-selector">
                <label><strong>Voice:</strong></label>
                <select id="voiceSelect"></select>
                <div class="speed-control">
                    <label><strong>Speed:</strong></label>
                    <input type="range" id="speedControl" min="0.5" max="2" step="0.1" value="1">
                    <span id="speedValue">1.0x</span>
                </div>
            </div>

            <div class="progress-info">
                <div><strong>Progress:</strong> <span id="progressText">0 / 0</span></div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <div><strong>Page:</strong> <span id="currentPage">-</span></div>
            </div>

            <div class="current-text" id="currentText">Ready to start...</div>

            <div class="player-controls">
                <button class="player-btn play" id="playBtn">▶️ Play</button>
                <button class="player-btn pause" id="pauseBtn">⏸️ Pause</button>
                <button class="player-btn stop" id="stopBtn">⏹️ Stop</button>
            </div>
        </div>

        <div class="loading" id="loading" style="display: none;">
            <div class="spinner"></div>
            <p id="loadingText">Processing...</p>
        </div>
    </div>

    <script>
        const API_URL = window.location.origin;

        let pdfLines = [];
        let currentIndex = 0;
        let isPaused = false;
        let isSpeaking = false;
        let speechSynthesis = window.speechSynthesis;
        let currentUtterance = null;
        let availableVoices = [];
        let selectedVoice = null;
        let speechSpeed = 1.0;

        // DOM Elements
        const uploadBox = document.getElementById('uploadBox');
        const fileInput = document.getElementById('fileInput');
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');
        const pageCount = document.getElementById('pageCount');
        const lineCount = document.getElementById('lineCount');
        const controls = document.getElementById('controls');
        const startBtn = document.getElementById('startBtn');
        const player = document.getElementById('player');
        const playBtn = document.getElementById('playBtn');
        const pauseBtn = document.getElementById('pauseBtn');
        const stopBtn = document.getElementById('stopBtn');
        const currentText = document.getElementById('currentText');
        const progressText = document.getElementById('progressText');
        const progressFill = document.getElementById('progressFill');
        const currentPage = document.getElementById('currentPage');
        const loading = document.getElementById('loading');
        const loadingText = document.getElementById('loadingText');
        const errorMessage = document.getElementById('errorMessage');
        const successMessage = document.getElementById('successMessage');
        const voiceSelect = document.getElementById('voiceSelect');
        const speedControl = document.getElementById('speedControl');
        const speedValue = document.getElementById('speedValue');

        // Load voices
        function loadVoices() {
            availableVoices = speechSynthesis.getVoices();
            voiceSelect.innerHTML = '';
            
            // Filter English voices
            const englishVoices = availableVoices.filter(voice => voice.lang.startsWith('en'));
            
            if (englishVoices.length === 0) {
                englishVoices.push(...availableVoices.slice(0, 5)); // Fallback to first 5 voices
            }
            
            englishVoices.forEach((voice, index) => {
                const option = document.createElement('option');
                option.value = index;
                option.textContent = `${voice.name} (${voice.lang})`;
                if (voice.default) {
                    option.selected = true;
                    selectedVoice = voice;
                }
                voiceSelect.appendChild(option);
            });
            
            if (!selectedVoice && englishVoices.length > 0) {
                selectedVoice = englishVoices[0];
            }
        }

        // Initialize voices
        loadVoices();
        if (speechSynthesis.onvoiceschanged !== undefined) {
            speechSynthesis.onvoiceschanged = loadVoices;
        }

        // Voice selection
        voiceSelect.addEventListener('change', (e) => {
            const voices = speechSynthesis.getVoices().filter(v => v.lang.startsWith('en'));
            selectedVoice = voices[e.target.value] || voices[0];
        });

        // Speed control
        speedControl.addEventListener('input', (e) => {
            speechSpeed = parseFloat(e.target.value);
            speedValue.textContent = speechSpeed.toFixed(1) + 'x';
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
            if (files.length > 0 && files[0].type === 'application/pdf') {
                fileInput.files = files;
                handleFileSelect({ target: fileInput });
            } else {
                showError('Please drop a valid PDF file');
            }
        });

        async function handleFileSelect(e) {
            const file = e.target.files[0];
            if (!file) return;
            
            if (file.type !== 'application/pdf') {
                showError('Please select a PDF file');
                return;
            }
            
            hideMessages();
            showLoading('Parsing PDF...');
            
            try {
                const formData = new FormData();
                formData.append('file', file);
                
                const response = await fetch(`${API_URL}/api/parse-pdf`, {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.detail || 'Failed to parse PDF');
                }
                
                pdfLines = data.lines;
                
                fileName.textContent = file.name;
                pageCount.textContent = data.page_count;
                lineCount.textContent = data.line_count;
                fileInfo.style.display = 'block';
                controls.style.display = 'block';
                
                hideLoading();
                showSuccess(`PDF loaded! ${data.line_count} lines ready to read.`);
                
            } catch (error) {
                hideLoading();
                showError(error.message);
            }
        }

        startBtn.addEventListener('click', () => {
            if (pdfLines.length === 0) return;
            
            controls.style.display = 'none';
            player.style.display = 'block';
            currentIndex = 0;
            speakNext();
        });

        function speakNext() {
            if (currentIndex >= pdfLines.length) {
                currentText.textContent = '✅ Finished reading all text!';
                isSpeaking = false;
                return;
            }
            
            const line = pdfLines[currentIndex];
            
            // Update UI
            currentText.textContent = line.text;
            currentPage.textContent = `Page ${line.page}`;
            progressText.textContent = `${currentIndex + 1} / ${pdfLines.length}`;
            const progress = ((currentIndex + 1) / pdfLines.length) * 100;
            progressFill.style.width = progress + '%';
            
            // Create speech
            currentUtterance = new SpeechSynthesisUtterance(line.text);
            currentUtterance.voice = selectedVoice;
            currentUtterance.rate = speechSpeed;
            
            currentUtterance.onend = () => {
                if (!isPaused) {
                    currentIndex++;
                    speakNext();
                }
            };
            
            currentUtterance.onerror = (e) => {
                console.error('Speech error:', e);
                currentIndex++;
                speakNext();
            };
            
            speechSynthesis.speak(currentUtterance);
            isSpeaking = true;
        }

        playBtn.addEventListener('click', () => {
            if (isPaused) {
                isPaused = false;
                speechSynthesis.resume();
            } else if (!isSpeaking) {
                speakNext();
            }
        });

        pauseBtn.addEventListener('click', () => {
            if (isSpeaking) {
                isPaused = true;
                speechSynthesis.pause();
            }
        });

        stopBtn.addEventListener('click', () => {
            speechSynthesis.cancel();
            currentIndex = 0;
            isPaused = false;
            isSpeaking = false;
            currentText.textContent = 'Stopped. Click Play to start from beginning.';
            progressFill.style.width = '0%';
            progressText.textContent = '0 / ' + pdfLines.length;
        });

        function showLoading(text) {
            loadingText.textContent = text;
            loading.style.display = 'block';
        }

        function hideLoading() {
            loading.style.display = 'none';
        }

        function showError(message) {
            errorMessage.textContent = '❌ ' + message;
            errorMessage.style.display = 'block';
            setTimeout(() => errorMessage.style.display = 'none', 5000);
        }

        function showSuccess(message) {
            successMessage.textContent = '✅ ' + message;
            successMessage.style.display = 'block';
            setTimeout(() => successMessage.style.display = 'none', 5000);
        }

        function hideMessages() {
            errorMessage.style.display = 'none';
            successMessage.style.display = 'none';
        }
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting PDF to Speech Converter with Browser TTS...")
    logger.info("🎉 Using browser's native speech - no backend TTS needed!")
    uvicorn.run(app, host="0.0.0.0", port=8000)
