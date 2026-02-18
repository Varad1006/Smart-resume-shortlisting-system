AI Resume Shortlisting System ✨ Updated: Open-Source APIs

Handwritten • Multilingual • Context-Aware • Social Score

Overview

An advanced AI-powered resume shortlisting system that intelligently ranks candidates based on:

Contextual Relevance: Using Bi-Encoder and Cross-Encoder models
Social Performance: GitHub, LeetCode, and CodeChef profiles
Multilingual Support: Automatic translation via free open-source APIs ✨
Document OCR: Tesseract OCR for PDFs and image-based resumes ✨
What's New ✨

Latest Update (Nov 2025): Migrated from Google Cloud APIs to open-source alternatives!

Feature	Before	After
OCR	Google Vision	Tesseract (Free) ✨
Translation	Google Translate API	translate library (Free) ✨
Credentials	Required in Frontend	Not needed ✨
Setup Time	15-30 min	5-10 min ✨
Cost	$0.60 per 100 images	$0 (Free) ✨
Features

1. Resume Processing

Accepts PDF and image files (JPG, PNG)
Extracts text from PDFs using PyPDF2
OCR for handwritten/image-based resumes using Tesseract ✨ (free, open-source)
Automatic language detection and translation to English (free API) ✨
2. AI-Powered Ranking

Bi-Encoder: all-MiniLM-L6-v2 for initial semantic similarity scoring
Cross-Encoder: cross-encoder/ms-marco-MiniLM-L-6-v2 for reranking top candidates
Combined scoring: 70% contextual relevance + 30% social score
3. Social Score Calculation

GitHub: Public repos, followers, and stars
LeetCode: Total problems solved
CodeChef: Problems solved and rating
The system automatically extracts social media usernames from resumes.

4. Results

Returns top 10 candidates ranked by combined score with detailed breakdowns.

Prerequisites

Required

Python 3.12+ (with venv)
Node.js 24+ & npm 11+
Tesseract OCR (required for document processing) ✨
Optional

GitHub Token (for social scoring, not required for basic usage)
Quick Start

1. Install Tesseract OCR

# Windows (using Chocolatey)
choco install tesseract

# Windows (alternative: download from)
# https://github.com/UB-Mannheim/tesseract/wiki

# Linux
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
2. Setup Backend

cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
3. Setup Frontend

cd frontend
npm install --legacy-peer-deps
4. Start Services

Terminal 1 - Backend:

cd backend
.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
Terminal 2 - Frontend:

cd frontend
npm start
5. Open Browser

http://localhost:3000
How to Use

Step 1: Enter Job Description

Simply type or paste the job description in the text area.

Step 2: Configure API Keys (Optional)

GitHub Token (optional): Add for better social scoring
Google Cloud Credentials: No longer needed! ✨
Step 3: Upload Resumes

Click the upload area and select multiple resume files (PDF, JPG, PNG)
Supports handwritten and printed documents ✨
Step 4: Process

Click "Process" to start the analysis. The system will:

Extract text from PDFs/images (Tesseract) ✨
Detect and translate to English (free API) ✨
Generate AI embeddings
Rank candidates by relevance
Fetch social stats (GitHub, LeetCode, CodeChef)
Step 5: View Results

See top 10 candidates with:

Combined score (contextual + social)
Individual score breakdowns
Resume preview (click to expand)
Social profile links
Resume Format Guidelines

Include social profiles in your resume for better scoring:

GitHub: github.com/username
LeetCode: leetcode.com/username
CodeChef: codechef.com/users/username
Technical Stack

Backend:

FastAPI (REST API)
MongoDB (Results Storage)
Tesseract OCR (Document Processing) ✨
Sentence-Transformers (AI Ranking)
Free Translation API ✨
Frontend:

React 19
Shadcn/UI Components
Tailwind CSS
Axios (HTTP Client)
Scoring Algorithm

Bi-Encoder Score: Semantic similarity (0-100)
Cross-Encoder Score: Deep relevance (0-100)
Social Score: GitHub/LeetCode/CodeChef (0-100)
Combined: Contextual (70%) + Social (30%)
API Documentation

Running

Backend: http://127.0.0.1:8000
Swagger UI: http://127.0.0.1:8000/docs
ReDoc: http://127.0.0.1:8000/redoc
Key Endpoints

POST /api/process - Main resume processing
POST /api/ocr - Direct document OCR ✨
POST /api/translate - Direct translation ✨
GET /api/results - Fetch saved results
Troubleshooting

Tesseract Not Found

# Verify installation
tesseract --version

# Install if missing
choco install tesseract

# Update path in backend/.env if installed elsewhere
TESSERACT_PATH="C:\\Custom\\Path\\tesseract.exe"
OCR Returns Empty Text

Check image/PDF quality
Verify Tesseract installed correctly
Try different resume format
Translation Fails

Check internet connection (needed for language detection)
Verify translate library installed: pip show translate
Backend Won't Start

Check port 8000 is free
Verify all dependencies installed
Check MongoDB connection string in .env
Documentation

For detailed information, see:

QUICK_START.md - Fast setup guide
SETUP_GUIDE.md - Detailed setup & troubleshooting
MIGRATION_NOTES.md - Technical migration details
COMPLETE_SUMMARY.md - Full project summary
DETAILED_CHANGELOG.md - All changes made
ARCHITECTURE_DIAGRAMS.md - System design diagrams
Benefits of Migration ✨

✅ No external API credentials needed ✅ No API costs (100% free) ✅ No quota limits ✅ Works mostly offline ✅ Better privacy & security ✅ Simpler setup ✅ Faster deployment

Future Enhancements

 Add OCR confidence scoring
 Implement result caching
 Support batch translation
 Add more language support
 Resume structure parsing
 API rate limiting
 User authentication
Support

Having Issues?

Check the troubleshooting section above
Review documentation files (QUICK_START.md, SETUP_GUIDE.md)
Check backend logs: http://127.0.0.1:8000/docs
Verify Tesseract installation: tesseract --version
License

This project is open-source and uses free, open-source libraries.
