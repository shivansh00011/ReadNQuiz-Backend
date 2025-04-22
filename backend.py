from fastapi import FastAPI, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import tempfile
import shutil
import json
from typing import Dict, Any
import pdfplumber
from pypdf import PdfReader

app = FastAPI(
    title="PDF Q&A API",
    description="API for processing PDFs and answering questions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Replace with your Gemini API key
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize the embedding model
embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Directory to store the vector database
VECTOR_STORE_DIR = "vector_db"

# Create vector store directory if it doesn't exist
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

def extract_text_from_pdf(file_path: str) -> list:
    """Extract text from PDF using multiple methods."""
    texts = []
    
    # Try pdfplumber first
    try:
        print("[DEBUG] Attempting to read PDF with pdfplumber...")
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text.strip():
                    texts.append(text)
        if texts:
            print("[DEBUG] Successfully extracted text with pdfplumber")
            return texts
    except Exception as e:
        print(f"[DEBUG] pdfplumber failed: {str(e)}")

    # Try pypdf
    try:
        print("[DEBUG] Attempting to read PDF with pypdf...")
        reader = PdfReader(file_path)
        for page in reader.pages:
            text = page.extract_text()
            if text.strip():
                texts.append(text)
        if texts:
            print("[DEBUG] Successfully extracted text with pypdf")
            return texts
    except Exception as e:
        print(f"[DEBUG] pypdf failed: {str(e)}")

    # Try PDFMinerLoader
    try:
        print("[DEBUG] Attempting to read PDF with PDFMinerLoader...")
        loader = PDFMinerLoader(file_path)
        documents = loader.load()
        if documents:
            texts = [doc.page_content for doc in documents]
            print("[DEBUG] Successfully extracted text with PDFMinerLoader")
            return texts
    except Exception as e:
        print(f"[DEBUG] PDFMinerLoader failed: {str(e)}")

    # Try PyPDFLoader as last resort
    try:
        print("[DEBUG] Attempting to read PDF with PyPDFLoader...")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        if documents:
            texts = [doc.page_content for doc in documents]
            print("[DEBUG] Successfully extracted text with PyPDFLoader")
            return texts
    except Exception as e:
        print(f"[DEBUG] PyPDFLoader failed: {str(e)}")

    return texts

def safe_load_vector_store(folder_path: str, embeddings: Any) -> FAISS:
    """Safely load the vector store with proper deserialization settings."""
    try:
        return FAISS.load_local(
            folder_path=folder_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"[DEBUG] Vector store load failed: {str(e)}")
        raise e

@app.get("/")
async def root():
    """Root endpoint to check if the API is running"""
    return {
        "status": "ok",
        "message": "PDF Q&A API is running",
        "endpoints": {
            "/": "Root endpoint (this one)",
            "/health": "Health check endpoint",
            "/upload_pdf/": "Upload and process a PDF file",
            "/ask_question/": "Ask a question about the uploaded PDF"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile):
    """
    Upload and process a PDF file
    """
    print(f"[DEBUG] Received PDF upload request for file: {file.filename}")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Create a temporary directory to store the uploaded file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        try:
            # Save uploaded file to temporary directory
            print(f"[DEBUG] Saving uploaded file to: {temp_file_path}")
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Extract text from PDF
            print("[DEBUG] Extracting text from PDF...")
            texts = extract_text_from_pdf(temp_file_path)
            
            if not texts:
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract any text from the PDF"
                )

            print(f"[DEBUG] Extracted {len(texts)} pages of text")
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.create_documents(texts)
            
            print(f"[DEBUG] Split text into {len(chunks)} chunks")

            # Create and save the vector store
            print("[DEBUG] Creating vector store...")
            vector_store = FAISS.from_documents(chunks, embedding)
            
            # Clear existing vector store directory
            if os.path.exists(VECTOR_STORE_DIR):
                shutil.rmtree(VECTOR_STORE_DIR)
            
            # Save the vector store
            print(f"[DEBUG] Saving vector store to: {VECTOR_STORE_DIR}")
            vector_store.save_local(VECTOR_STORE_DIR)
            
            return JSONResponse(
                content={
                    "message": "PDF processed successfully",
                    "num_pages": len(texts),
                    "num_chunks": len(chunks)
                },
                status_code=200
            )
            
        except Exception as e:
            print(f"[ERROR] Failed to process PDF: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process PDF: {str(e)}"
            )
        finally:
            # Clean up: close the uploaded file
            file.file.close()

@app.post("/ask_question/")
async def ask_question(request: Request):
    """
    Ask a question about the uploaded PDF
    """
    try:
        try:
            data = await request.json()
        except json.JSONDecodeError:
            print("[ERROR] Invalid JSON in request body")
            raise HTTPException(
                status_code=400,
                detail="Invalid JSON in request body. Request must be a JSON object with a 'question' field"
            )

        question = data.get("question")
        if not question:
            print("[ERROR] No question field found in request")
            raise HTTPException(
                status_code=400,
                detail="Request must include a 'question' field"
            )

        print(f"[DEBUG] Received question: {question}")
        
        if not os.path.exists(VECTOR_STORE_DIR):
            print("[ERROR] No vector store found")
            raise HTTPException(
                status_code=400,
                detail="No PDF has been uploaded yet"
            )
        
        # Load the vector store
        print("[DEBUG] Loading vector store...")
        vector_store = safe_load_vector_store(VECTOR_STORE_DIR, embedding)
        
        # Get relevant documents first
        print("[DEBUG] Retrieving relevant documents...")
        relevant_docs = vector_store.similarity_search(
            question,
            k=5,
            fetch_k=10
        )

        if not relevant_docs:
            return JSONResponse(
                content={
                    "answer": "I couldn't find any relevant information in the document to answer your question. Please try rephrasing or ask something else.",
                    "status": "no_answer"
                },
                status_code=200
            )

        # Initialize the Gemini model
        print("[DEBUG] Initializing Gemini model...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3
        )

        # Prepare the context and question
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create a more specific prompt
        prompt = f"""Based on the following context from a PDF document, please answer the question.
        If you cannot find relevant information to answer the question, say so clearly.

        Context:
        {context}

        Question: {question}

        Answer:"""

        print("[DEBUG] Generating answer...")
        # Use the new invoke method instead of __call__
        messages = [{"role": "user", "content": prompt}]
        response = llm.invoke(messages)
        
        answer = response.content.strip()
        
        if not answer or "i cannot" in answer.lower() or "i don't have" in answer.lower():
            return JSONResponse(
                content={
                    "answer": "I couldn't find relevant information to answer your question. Please try rephrasing or ask something else.",
                    "status": "no_answer"
                },
                status_code=200
            )
        
        return JSONResponse(
            content={
                "answer": answer,
                "status": "success"
            },
            status_code=200
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"[ERROR] Failed to process question: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
