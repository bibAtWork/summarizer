import streamlit as st
import os
import tempfile
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from llama_index.core.schema import Document
from llama_index.llms.ollama import Ollama
import chromadb
import ollama

# Setup directories
if not os.path.exists("uploads"):
    os.makedirs("uploads")
if not os.path.exists("summaries"):
    os.makedirs("summaries")

# Initialize Ollama client and LLM
llm = Ollama(model="llama3", temperature=0.1, request_timeout=300.0)

# Initialize embeddings model
embed_model = HuggingFaceEmbedding(model_name="intfloat/e5-base-v2")

# Configure global settings
Settings.embed_model = embed_model
Settings.llm = llm
Settings.chunk_size = 1024

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "pdf_summaries"
chroma_collection = chroma_client.get_or_create_collection(name=collection_name)

# Setup vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

def extract_text_and_highlights(pdf_path):
    """Extract text and highlighted portions from PDF."""
    doc = fitz.open(pdf_path)
    text_with_importance = []
    
    for page_num, page in enumerate(doc):
        # Extract text
        text = page.get_text()
        
        # Extract highlights (annotations)
        highlights = []
        for annot in page.annots():
            if annot.type[0] == 8:  # Highlight annotation
                highlight_rect = annot.rect
                words = page.get_text("words")
                highlighted_text = ""
                
                for word in words:
                    word_rect = fitz.Rect(word[:4])
                    if highlight_rect.intersects(word_rect):
                        highlighted_text += word[4] + " "
                
                if highlighted_text:
                    highlights.append(highlighted_text.strip())
        
        # Add page text and any highlights
        if highlights:
            for highlight in highlights:
                text_with_importance.append({"text": highlight, "importance": 2})
                
            # Add non-highlighted text with normal importance
            regular_text = text
            for highlight in highlights:
                regular_text = regular_text.replace(highlight, "")
            text_with_importance.append({"text": regular_text, "importance": 1})
        else:
            text_with_importance.append({"text": text, "importance": 1})
    
    return text_with_importance

def extract_images_and_ocr(pdf_path):
    """Extract images from PDF and perform OCR."""
    doc = fitz.open(pdf_path)
    image_texts = []
    
    for page_num, page in enumerate(doc):
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Convert image bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # For a real implementation, you would use OCR here
            # For example with pytesseract: text = pytesseract.image_to_string(img)
            # For simplicity, we'll just note that an image was found
            image_texts.append(f"[Image {page_num+1}-{img_index+1}: Image content description would be here with OCR]")
    
    return image_texts

def generate_summary(text_chunks, image_texts):
    """Generate summary using Ollama LLM."""
    # Combine text chunks based on importance
    sorted_chunks = sorted(text_chunks, key=lambda x: x["importance"], reverse=True)
    text_content = " ".join([chunk["text"] for chunk in sorted_chunks])
    
    # Add image descriptions
    if image_texts:
        image_content = "\n".join(image_texts)
        full_content = text_content + "\n\nImages found in document:\n" + image_content
    else:
        full_content = text_content
    
    # Create a prompt for summarization
    prompt = f"""
    Please provide a comprehensive summary of the following document. 
    Pay special attention to highlighted text (which appears first in the content).
    Include key points, main arguments, and important details.
    
    Document content:
    {full_content}
    
    Summary:
    """
    
    # Call Ollama for summarization
    response = llm_client.generate(model="llama3", prompt=prompt)
    summary = response['response']
    
    return summary

def save_summary_to_vector_store(file_name, summary, original_text):
    """Save summary to vector store for later querying."""
    # Create documents for indexing
    documents = [
        Document(text=summary, metadata={"filename": file_name, "type": "summary"}),
        Document(text=original_text, metadata={"filename": file_name, "type": "full_text"})
    ]
    
    # Create index from documents
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context
    )
    
    return index

def query_summary(query_text):
    """Query summaries in the vector store."""
    # Load index from vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store
    )
    
    # Create query engine
    query_engine = index.as_query_engine()
    response = query_engine.query(query_text)
    
    return response.response

def get_available_summaries():
    """Get list of available summaries."""
    if os.path.exists("summaries"):
        return sorted(os.listdir("summaries"))
    return []

# Streamlit UI
st.title("PDF Summarizer")

# Add a refresh button for summaries
if st.button("Refresh Available Summaries"):
    st.session_state.summaries = get_available_summaries()

# Initialize summaries in session state if not present
if 'summaries' not in st.session_state:
    st.session_state.summaries = get_available_summaries()

# File upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name
    
    st.success(f"PDF '{uploaded_file.name}' successfully uploaded")
    
    # Process Button
    if st.button("Generate Summary"):
        with st.spinner("Processing PDF..."):
            # Extract text and highlights
            text_chunks = extract_text_and_highlights(pdf_path)
            
            # Extract images and perform OCR
            image_texts = extract_images_and_ocr(pdf_path)
            
            # Generate summary
            summary = generate_summary(text_chunks, image_texts)
            
            # Save to vector store
            full_text = " ".join([chunk["text"] for chunk in text_chunks])
            save_summary_to_vector_store(uploaded_file.name, summary, full_text)
            
            # Display summary
            st.subheader("Summary")
            st.write(summary)
            
            # Save summary to file for persistence
            summary_path = os.path.join("summaries", f"{uploaded_file.name.split('.')[0]}_summary.txt")
            with open(summary_path, "w") as f:
                f.write(summary)
            
            st.success(f"Summary saved to {summary_path}")
    
    # Clean up the temporary file
    os.unlink(pdf_path)

# Query interface
st.subheader("Query Summaries")
query = st.text_input("Enter your query about previously summarized documents")

if query:
    with st.spinner("Searching..."):
        result = query_summary(query)
        st.write(result)

# List available summaries with auto-refresh
st.subheader("Available Summaries")
for summary_file in st.session_state.summaries:
    st.write(summary_file)
    with open(os.path.join("summaries", summary_file), "r") as f:
        if st.button(f"View {summary_file}"):
            st.text_area("Summary Content", f.read(), height=300)