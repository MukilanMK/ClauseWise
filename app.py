import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import base64
import os
from datetime import datetime
import re

# Set page configuration
st.set_page_config(
    page_title="StudyMate - AI-Powered PDF Q&A",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .answer-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .question-card {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #1976d2;
    }
    .reference-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .stButton > button {
        background-color: #2E86AB;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class StudyMate:
    def _init_(self):
        self.embedding_model = None
        self.llm_model = None
        self.tokenizer = None
        self.chunks = []
        self.embeddings = None
        self.faiss_index = None
        self.chunk_metadata = []

    @st.cache_resource
    def load_models(_self):
        """Load the embedding and LLM models"""
        try:
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            # Changed from granite-3.2-2b-instruct to granite-3.1-2b-instruct
            model_name = "ibm-granite/granite-3.1-2b-instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            return embedding_model, llm_model, tokenizer
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return None, None, None

    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF file"""
        try:
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = ""
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text += page.get_text()
            pdf_document.close()
            return text
        except Exception as e:
            st.error(f"Error extracting text from {pdf_file.name}: {str(e)}")
            return ""

    def chunk_text(self, text, filename, chunk_size=500, overlap=100):
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        metadata = []
        if len(words) <= chunk_size:
            chunks.append(text)
            metadata.append({"filename": filename, "chunk_id": 0})
            return chunks, metadata

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)
            metadata.append({"filename": filename, "chunk_id": len(chunks) - 1})
            if i + chunk_size >= len(words):
                break
        return chunks, metadata

    def process_pdfs(self, uploaded_files):
        """Process uploaded PDF files"""
        if not uploaded_files:
            return False

        self.chunks = []
        self.chunk_metadata = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, pdf_file in enumerate(uploaded_files):
            status_text.text(f"Processing {pdf_file.name}...")
            text = self.extract_text_from_pdf(pdf_file)
            if not text.strip():
                st.warning(f"No text found in {pdf_file.name}")
                continue
            file_chunks, file_metadata = self.chunk_text(text, pdf_file.name)
            self.chunks.extend(file_chunks)
            self.chunk_metadata.extend(file_metadata)
            progress_bar.progress((idx + 1) / len(uploaded_files))

        status_text.text("Creating embeddings...")
        if self.chunks:
            try:
                self.embeddings = self.embedding_model.encode(self.chunks, show_progress_bar=True)
                dimension = self.embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatL2(dimension)
                self.faiss_index.add(self.embeddings.astype('float32'))
                progress_bar.progress(1.0)
                status_text.text(f"✅ Successfully processed {len(uploaded_files)} PDF(s) with {len(self.chunks)} chunks")
                return True
            except Exception as e:
                st.error(f"Error creating embeddings: {str(e)}")
                return False
        return False

    def retrieve_relevant_chunks(self, query, k=3):
        """Retrieve most relevant chunks for the query"""
        if not self.faiss_index or not self.chunks:
            return []
        try:
            query_embedding = self.embedding_model.encode([query])
            scores, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
            relevant_chunks = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.chunks):
                    relevant_chunks.append({
                        'text': self.chunks[idx],
                        'metadata': self.chunk_metadata[idx],
                        'score': scores[0][i]
                    })
            return relevant_chunks
        except Exception as e:
            st.error(f"Error retrieving chunks: {str(e)}")
            return []

    def generate_answer(self, query, relevant_chunks):
        """Generate answer using the Granite 3.1 model with attention mask fix"""
        if not relevant_chunks:
            return "I couldn't find relevant information in the uploaded documents to answer your question."

        context = "\n\n".join([chunk['text'] for chunk in relevant_chunks[:3]])

        prompt = f"""Based on the following context from academic documents, please provide a clear and accurate answer to the question. Only use information from the provided context.

Context:
{context}

Question: {query}

Answer:"""

        try:
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1500,
                padding="max_length"
            )
            attention_mask = inputs["attention_mask"]

            with torch.no_grad():
                outputs = self.llm_model.generate(
                    inputs.input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response.split("Answer:")[-1].strip()
            return answer

        except Exception as e:
            return f"Error generating answer: {str(e)}"

def main():
    st.markdown('<h1 class="main-header">📚 StudyMate</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered PDF-Based Q&A System for Students</p>', unsafe_allow_html=True)

    if 'studymate' not in st.session_state:
        st.session_state.studymate = StudyMate()
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []

    studymate = st.session_state.studymate

    if studymate.embedding_model is None:
        with st.spinner("Loading AI models... This may take a few moments."):
            studymate.embedding_model, studymate.llm_model, studymate.tokenizer = studymate.load_models()
        if studymate.embedding_model is None:
            st.error("Failed to load models. Please check your internet connection and try again.")
            return

    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📁 Upload Your Study Materials")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files containing your study materials"
    )

    if uploaded_files:
        if st.button("🔄 Process PDFs"):
            with st.spinner("Processing your documents..."):
                success = studymate.process_pdfs(uploaded_files)
                if success:
                    st.success(f"✅ Successfully processed {len(uploaded_files)} PDF(s)!")
                else:
                    st.error("❌ Failed to process PDFs. Please try again.")

    st.markdown('</div>', unsafe_allow_html=True)

    if studymate.chunks:
        st.markdown("### 💭 Ask Your Question")
        col1, col2 = st.columns([4, 1])
        with col1:
            user_question = st.text_input(
                "Enter your question about the uploaded documents:",
                placeholder="e.g., What is machine learning? Explain classification vs regression.",
                key="question_input"
            )
        with col2:
            ask_button = st.button("🔍 Ask", type="primary")

        if ask_button and user_question.strip():
            with st.spinner("Searching for relevant information and generating answer..."):
                relevant_chunks = studymate.retrieve_relevant_chunks(user_question)
                if relevant_chunks:
                    answer = studymate.generate_answer(user_question, relevant_chunks)
                    st.session_state.qa_history.append({
                        'question': user_question,
                        'answer': answer,
                        'chunks': relevant_chunks,
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    })

                    st.markdown('<div class="answer-card">', unsafe_allow_html=True)
                    st.markdown("*Answer:*")
                    st.markdown(answer)
                    st.markdown('</div>', unsafe_allow_html=True)

                    with st.expander("📖 Referenced Paragraphs"):
                        for i, chunk in enumerate(relevant_chunks):
                            st.markdown(f"*Source: {chunk['metadata']['filename']} (Chunk {chunk['metadata']['chunk_id']})*")
                            st.markdown(f'<div class="reference-card">{chunk["text"]}</div>', unsafe_allow_html=True)
                else:
                    st.warning("No relevant information found in the uploaded documents.")

    if st.session_state.qa_history:
        st.markdown("### 📝 Q&A History")
        if st.button("📥 Download Q&A History"):
            history_text = f"StudyMate Q&A Session - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            history_text += "=" * 50 + "\n\n"
            for i, qa in enumerate(st.session_state.qa_history, 1):
                history_text += f"Q{i} ({qa['timestamp']}): {qa['question']}\n"
                history_text += f"A{i}: {qa['answer']}\n"
                history_text += "-" * 30 + "\n\n"

            st.download_button(
                label="📄 Download as Text File",
                data=history_text,
                file_name=f"studymate_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

        for i, qa in enumerate(reversed(st.session_state.qa_history)):
            with st.container():
                st.markdown(f'<div class="question-card"><strong>Q:</strong> {qa["question"]} <small>({qa["timestamp"]})</small></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-card"><strong>A:</strong> {qa["answer"]}</div>', unsafe_allow_html=True)

    if not studymate.chunks:
        st.markdown("### 🚀 How to Use StudyMate")
        st.markdown("""
        1. *Upload PDFs*: Use the file uploader above to upload your study materials (textbooks, notes, papers)
        2. *Process Documents*: Click 'Process PDFs' to prepare your documents for Q&A
        3. *Ask Questions*: Type your questions in natural language
        4. *Get Answers*: Receive AI-generated answers with source references
        5. *Download History*: Save your Q&A session for future reference

        *Example Questions:*
        - What is overfitting in machine learning?
        - Explain the difference between classification and regression
        - What are the key principles of data preprocessing?
        - How does cross-validation work?
        """)

    st.markdown("---")
    st.markdown("Made with ❤ using Streamlit, Sentence Transformers, FAISS, and Hugging Face Transformers")

if _name_ == "_main_":
    main()
