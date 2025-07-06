# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
import os
import PyPDF2
from typing import List
import re # Import regex module

# For HuggingFace Transformers (T5, BART, Pegasus)
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration

# For Google Gemini API
import google.generativeai as genai

# For OpenAI API
import openai # Import openai library

# --- Configuration ---
# Set your Google API Key for Gemini.
# IMPORTANT: In a real application, do NOT hardcode API keys.
# Use environment variables or a secure configuration management system.
# For local testing, you can set it here or via an environment variable.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Set your OpenAI API Key.
# IMPORTANT: Use environment variables for API keys in production.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Initialize OpenAI client globally (or within a startup event if preferred)
# This avoids re-initializing the client on every request.
openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY) # Use AsyncOpenAI for async calls
        print("OpenAI API configured.")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        print("OpenAI summarization will not be available.")
else:
    print("WARNING: OPENAI_API_KEY not set. Online OpenAI summarization will not work.")


# Initialize Gemini API
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Gemini API configured.")
else:
    print("WARNING: GOOGLE_API_KEY not set. Online Gemini summarization will not work.")

# Initialize FastAPI app
app = FastAPI(
    title="PDF Summarizer Backend",
    description="Backend for Offline & Online AI-Powered PDF Summarizer",
    version="1.0.0"
)

# --- CORS Configuration ---
origins = [
    "http://localhost",
    "http://localhost:3000",
    "file://",
    "app://.",
    "capacitor://localhost",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading (Offline Models) ---
# Load HuggingFace models once when the app starts.
# These will download on first run if not cached.

# T5-small summarization pipeline
t5_summarizer = None
try:
    print("Loading t5-small summarization model...")
    t5_summarizer = pipeline("summarization", model="t5-small")
    print("t5-small model loaded successfully.")
except Exception as e:
    print(f"Error loading t5-small model: {e}")
    print("T5-small summarization will not be available.")

# BART (distilbart-cnn-12-6) model and tokenizer
bart_tokenizer = None
bart_model = None
try:
    print("Loading distilbart-cnn-12-6 summarization model...")
    bart_tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    bart_model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6")
    print("distilbart-cnn-12-6 model loaded successfully.")
except Exception as e:
    print(f"Error loading distilbart-cnn-12-6 model: {e}")
    print("BART summarization will not be available.")

# Pegasus summarization pipeline
pegasus_summarizer = None
try:
    print("Loading google/pegasus-cnn_dailymail summarization model...")
    pegasus_summarizer = pipeline("summarization", model="google/pegasus-cnn_dailymail")
    print("Pegasus model loaded successfully.")
except Exception as e:
    print(f"Error loading Pegasus model: {e}")
    print("Pegasus summarization will not be available.")


# --- Helper Functions ---
def extract_text_from_pdf(pdf_file_content: bytes) -> str:
    """Extracts text from a PDF file using PyPDF2."""
    try:
        pdf_file = io.BytesIO(pdf_file_content)
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
            else:
                print("WARNING: A page in the PDF may contain scanned images or no selectable text.")
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text from PDF with PyPDF2: {e}")

async def call_gemini_api(prompt: str, model_name: str = "gemini-1.5-flash") -> str:
    """Calls the Gemini API to generate content."""
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=503, detail="Gemini API key not configured.")

    try:
        model = genai.GenerativeModel(model_name)
        response = await model.generate_content_async(prompt)
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get response from Gemini API: {e}")

async def call_openai_api(text: str, summary_level: str) -> str:
    """Calls the OpenAI API to generate content."""
    if not openai_client: # Check if client was initialized
        raise HTTPException(status_code=503, detail="OpenAI API client not configured or initialized.")

    prompt_map = {
        "high": "Give a very short summary:",
        "medium": "Summarize the following text with moderate detail:",
        "low": "Give a detailed summary:"
    }
    
    prompt_prefix = prompt_map.get(summary_level, "Summarize the following text:")
    full_prompt = f"{prompt_prefix}\n\n{text}"

    try:
        response = await openai_client.chat.completions.create( # Updated API call
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        # Access content from the new response object structure
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get response from OpenAI API: {e}")


def get_hf_summary_lengths(original_doc_word_count: int, summary_level: str):
    """
    Determines min/max length for HuggingFace summarization *output* for both chunk-level
    and final summary, based on desired compression percentages for the final summary.
    """
    # Fixed output token ranges for a single chunk summary (first stage)
    # Making these extremely aggressive to force minimal info for the second stage
    min_len_chunk_summary = 5
    max_len_chunk_summary = 20

    # Target percentages for the final summary (second stage)
    if summary_level == "low": # ~10% of original document words
        target_percentage = 0.10
    elif summary_level == "medium": # ~5% of original document words
        target_percentage = 0.05
    elif summary_level == "high": # ~2% of original document words
        target_percentage = 0.02
    else: # Default to medium
        target_percentage = 0.05

    # Calculate min/max length for the FINAL summary based on original document word count
    min_len_final_summary = int(original_doc_word_count * target_percentage * 0.8) # 80% of target
    max_len_final_summary = int(original_doc_word_count * target_percentage * 1.2) # 120% of target

    # Ensure final summary lengths are within reasonable bounds for summarization models
    # and also within the typical output limits (e.g., 512 tokens for HF models)
    min_len_final_summary = max(20, min_len_final_summary) # Minimum of 20 words
    max_len_final_summary = max(min_len_final_summary + 20, max_len_final_summary) # Ensure max > min
    max_len_final_summary = min(512, max_len_final_summary) # Cap at 512 tokens (approx words)

    return (min_len_chunk_summary, max_len_chunk_summary,
            min_len_final_summary, max_len_final_summary)


def split_into_chunks_by_chars(text: str, max_chars: int = 3500) -> List[str]:
    """
    Splits text into chunks based on character count.
    Aims to keep chunks within a maximum character limit.
    """
    chunks = []
    current_chunk = []
    current_length = 0

    words = text.split()
    for word in words:
        if current_length + len(word) + (1 if current_chunk else 0) <= max_chars:
            current_chunk.append(word)
            current_length += len(word) + (1 if current_chunk else 0)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def chunk_text_semantically(text: str, model_max_words_per_chunk: int) -> List[str]:
    """
    Splits text into chunks, attempting to preserve paragraph boundaries.
    If a paragraph is too long, it will be split at the closest full stop
    before the `model_max_words_per_chunk` limit. If no full stop is found,
    it splits at the limit.
    """
    paragraphs = text.split('\n\n') # Split by double newline for paragraphs
    chunks = []
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        words = para.split()
        para_word_count = len(words)

        current_pos = 0
        while current_pos < para_word_count:
            # Determine the segment to consider for the current chunk
            segment_words = words[current_pos : current_pos + model_max_words_per_chunk]
            segment_len = len(segment_words)

            if segment_len == 0: # Should not happen if current_pos < para_word_count
                break

            # If the remaining part of the paragraph fits, take it
            if current_pos + segment_len <= para_word_count and segment_len < model_max_words_per_chunk:
                chunks.append(" ".join(segment_words).strip())
                current_pos = para_word_count
                break

            # Find the last full stop within the current segment
            split_idx_in_segment = -1
            # Search backwards from the end of the segment to find a full stop
            for i in range(segment_len - 1, -1, -1):
                if re.search(r'[.!?]$', segment_words[i]):
                    split_idx_in_segment = i
                    break
            
            # If a full stop was found, split there
            if split_idx_in_segment != -1 and split_idx_in_segment > 0: # Ensure it's not the very first word
                chunk_to_add = " ".join(segment_words[:split_idx_in_segment + 1]).strip()
                next_pos_increment = split_idx_in_segment + 1
            else:
                # No suitable full stop found in segment, split at max_words_per_chunk
                chunk_to_add = " ".join(segment_words).strip()
                next_pos_increment = segment_len
            
            chunks.append(chunk_to_add)
            current_pos += next_pos_increment
            
            # Safety break for very long words or unusual text that might cause infinite loops
            if next_pos_increment == 0:
                print(f"WARNING: Chunking made no progress on a paragraph. Forcing split.")
                chunks.append(" ".join(words[current_pos : current_pos + model_max_words_per_chunk]).strip())
                current_pos += model_max_words_per_chunk
                break
    return chunks

def clean_summary_text(summary: str) -> str:
    """
    Cleans the summary text by removing common unwanted tokens like '<n>'
    and normalizing whitespace.
    """
    # Replace '<n>' with a single space or newline, depending on desired formatting
    # A single space is often better for continuous text, newlines for paragraph breaks
    cleaned_summary = summary.replace('<n>', ' ')
    
    # Remove multiple spaces/newlines and strip leading/trailing whitespace
    cleaned_summary = re.sub(r'\s+', ' ', cleaned_summary).strip()
    
    return cleaned_summary

# --- API Endpoints ---

@app.get("/")
async def read_root():
    return {"message": "Welcome to the PDF Summarizer Backend!"}

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Uploads a PDF file and extracts its text content.
    """
    if not file.content_type == "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    try:
        pdf_content = await file.read()
        extracted_text = extract_text_from_pdf(pdf_content)
        return {"filename": file.filename, "text_content": extracted_text}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

@app.post("/summarize/")
async def summarize_text(
    text: str = Form(...),
    model_type: str = Form(...),
    summary_level: str = Form("medium")
):
    """
    Summarizes the provided text using the specified model and summary level.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text to summarize cannot be empty.")

    summary = ""
    original_text_word_count = len(text.split())
    
    try:
        if model_type == "offline-t5":
            if not t5_summarizer:
                raise HTTPException(status_code=503, detail="T5-small model not loaded or available.")
            
            (min_len_chunk_summary, max_len_chunk_summary,
             min_len_final_summary, max_len_final_summary) = get_hf_summary_lengths(original_text_word_count, summary_level)
            
            # Use semantic chunking for T5
            chunks = chunk_text_semantically(text, model_max_words_per_chunk=450) # T5-small max input is 512 tokens (approx 450 words)
            
            first_stage_summaries = []
            for chunk in chunks:
                summary_result = t5_summarizer(
                    chunk,
                    min_length=min_len_chunk_summary,
                    max_length=max_len_chunk_summary,
                    do_sample=False
                )
                if summary_result and summary_result[0] and 'summary_text' in summary_result[0]:
                    first_stage_summaries.append(summary_result[0]['summary_text'])
                else:
                    print(f"WARNING: T5-small produced an empty or invalid summary for a chunk.")

            if not first_stage_summaries:
                raise HTTPException(status_code=500, detail="T5-small failed to produce any chunk summaries.")
            
            combined_first_stage_summary = "\n\n".join(first_stage_summaries)
            
            # Second stage summarization: Summarize the concatenated chunk summaries
            # Use semantic chunking for the second stage input as well
            second_stage_chunks = chunk_text_semantically(combined_first_stage_summary, model_max_words_per_chunk=450)
            
            final_summaries_parts = []
            for s_chunk in second_stage_chunks:
                final_summary_result = t5_summarizer(
                    s_chunk,
                    min_length=min_len_final_summary, # Use final summary length params
                    max_length=max_len_final_summary,
                    do_sample=False
                )
                if final_summary_result and final_summary_result[0] and 'summary_text' in final_summary_result[0]:
                    final_summaries_parts.append(final_summary_result[0]['summary_text'])
                else:
                    print(f"WARNING: T5-small produced an empty or invalid summary for a second-stage chunk.")
            
            summary = "\n\n".join(final_summaries_parts)
            if not summary.strip(): # Fallback if second stage produces empty summary
                summary = combined_first_stage_summary # Use the first stage combined summary

        elif model_type == "offline-bart":
            if not bart_model or not bart_tokenizer:
                raise HTTPException(status_code=503, detail="BART model not loaded or available.")
            
            (min_len_chunk_summary, max_len_chunk_summary,
             min_len_final_summary, max_len_final_summary) = get_hf_summary_lengths(original_text_word_count, summary_level)
            
            # Use semantic chunking for BART
            chunks = chunk_text_semantically(text, model_max_words_per_chunk=800) # BART max input is 1024 tokens (approx 800 words)
            
            first_stage_summaries = []
            for chunk in chunks:
                inputs = bart_tokenizer.batch_encode_plus(
                    [chunk], max_length=1024, return_tensors="pt", truncation=True
                )
                summary_ids = bart_model.generate(
                    inputs["input_ids"],
                    num_beams=4,
                    length_penalty=2.0,
                    max_length=max_len_chunk_summary, # Use chunk summary length params
                    min_length=min_len_chunk_summary,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
                first_stage_summaries.append(bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True))
            
            if not first_stage_summaries:
                raise HTTPException(status_code=500, detail="BART failed to produce any chunk summaries.")

            combined_first_stage_summary = "\n\n".join(first_stage_summaries)
            
            # Second stage summarization for BART
            second_stage_chunks = chunk_text_semantically(combined_first_stage_summary, model_max_words_per_chunk=800)
            
            final_summaries_parts = []
            for s_chunk in second_stage_chunks:
                inputs = bart_tokenizer.batch_encode_plus(
                    [s_chunk], max_length=1024, return_tensors="pt", truncation=True
                )
                final_summary_ids = bart_model.generate(
                    inputs["input_ids"],
                    num_beams=4,
                    length_penalty=2.0,
                    max_length=max_len_final_summary, # Use final summary length params
                    min_length=min_len_final_summary,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
                final_summaries_parts.append(bart_tokenizer.decode(final_summary_ids[0], skip_special_tokens=True))
            
            summary = "\n\n".join(final_summaries_parts)
            if not summary.strip(): # Fallback if second stage produces empty summary
                summary = combined_first_stage_summary

        elif model_type == "offline-pegasus":
            if not pegasus_summarizer:
                raise HTTPException(status_code=503, detail="Pegasus model not loaded or available.")
            
            (min_len_chunk_summary, max_len_chunk_summary,
             min_len_final_summary, max_len_final_summary) = get_hf_summary_lengths(original_text_word_count, summary_level)
            
            # Use semantic chunking for Pegasus
            chunks = chunk_text_semantically(text, model_max_words_per_chunk=700) # Pegasus max input is typically 1024 tokens (approx 700 words)
            
            first_stage_summaries = []
            for chunk in chunks:
                summary_result = pegasus_summarizer(
                    chunk,
                    min_length=min_len_chunk_summary,
                    max_length=max_len_chunk_summary,
                    do_sample=False
                )
                if summary_result and summary_result[0] and 'summary_text' in summary_result[0]:
                    first_stage_summaries.append(summary_result[0]['summary_text'])
                else:
                    print(f"WARNING: Pegasus produced an empty or invalid summary for a chunk.")
            
            if not first_stage_summaries:
                raise HTTPException(status_code=500, detail="Pegasus failed to produce any chunk summaries.")

            combined_first_stage_summary = "\n\n".join(first_stage_summaries)
            
            # Second stage summarization for Pegasus
            second_stage_chunks = chunk_text_semantically(combined_first_stage_summary, model_max_words_per_chunk=700)
            
            final_summaries_parts = []
            for s_chunk in second_stage_chunks:
                final_summary_result = pegasus_summarizer(
                    s_chunk,
                    min_length=min_len_final_summary,
                    max_length=max_len_final_summary,
                    do_sample=False
                )
                if final_summary_result and final_summary_result[0] and 'summary_text' in final_summary_result[0]:
                    final_summaries_parts.append(final_summary_result[0]['summary_text'])
                else:
                    print(f"WARNING: Pegasus produced an empty or invalid summary for a second-stage chunk.")
            
            summary = "\n\n".join(final_summaries_parts)
            if not summary.strip(): # Fallback if second stage produces empty summary
                summary = combined_first_stage_summary

        elif model_type == "online-gemini":
            if not GOOGLE_API_KEY:
                raise HTTPException(status_code=503, detail="Gemini API key not configured.")
            
            # Determine target percentage for Gemini
            target_percentage_gemini = 0.05 # Default medium
            if summary_level == "low":
                target_percentage_gemini = 0.10
            elif summary_level == "medium":
                target_percentage_gemini = 0.05
            elif summary_level == "high":
                target_percentage_gemini = 0.02
            
            # Calculate target word count for Gemini's prompt
            target_word_count_gemini = int(original_text_word_count * target_percentage_gemini)
            
            # Adjust word count range for the prompt
            min_words_prompt = max(40, int(target_word_count_gemini * 0.8))
            max_words_prompt = max(min_words_prompt + 20, int(target_word_count_gemini * 1.2))


            chunks = split_into_chunks_by_chars(text, max_chars=3500)
            summaries = []
            gemini_model_name = "gemini-1.5-flash"

            for i, chunk in enumerate(chunks):
                # Use the prompt format from the user's Colab script, now with dynamic word count
                prompt = f"Summarize the following text concisely in {min_words_prompt}-{max_words_prompt} words, avoiding repetition:\n\n{chunk}"
                print(f"Summarizing chunk {i+1}/{len(chunks)} for Gemini...")
                chunk_summary = await call_gemini_api(prompt, model_name=gemini_model_name)
                summaries.append(chunk_summary)
            
            # For Gemini, a single pass is often sufficient. If the combined summaries are still too long,
            # you could add a second Gemini call here to summarize the 'summaries' list.
            summary = "\n\n".join(summaries)

        elif model_type == "online-openai": # NEW: OpenAI Summarization
            summary = await call_openai_api(text, summary_level)
        
        else:
            raise HTTPException(status_code=400, detail="Invalid model type selected.")

        # Clean the final summary before returning
        summary = clean_summary_text(summary)

        return {"model_type": model_type, "summary_level": summary_level, "summary": summary}

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")

# To run the app directly using `python main.py`
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
