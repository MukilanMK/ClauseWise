import streamlit as st
import requests
import PyPDF2
from docx import Document

# --- CONFIGURATION ---
# Fetch the API token from Streamlit's secrets management
# For local development, this reads from .streamlit/secrets.toml
# For deployment, this reads from the secrets set in the Streamlit Cloud dashboard
HUGGINGFACE_API_TOKEN = st.secrets.get("HUGGINGFACE_API_TOKEN")

# Use the specified IBM Granite instruction-tuned model
MODEL_API_URL = "https://api-inference.huggingface.co/models/ibm/granite-13b-instruct-v2"

# --- HELPER FUNCTIONS ---

def query_huggingface_model(prompt):
    """
    Sends a prompt to the Hugging Face Inference API and returns the model's response.
    """
    if not HUGGINGFACE_API_TOKEN:
        st.error("Hugging Face API token is not configured. Please add it to your Streamlit secrets.")
        return None

    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 1024, # Adjust as needed for longer responses
            "temperature": 0.7,
            "return_full_text": False,
        }
    }
    try:
        response = requests.post(MODEL_API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
        return response.json()[0]['generated_text']
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None
    except (KeyError, IndexError) as e:
        st.error(f"Failed to parse API response: {e} - Response: {response.text}")
        return None


def extract_text(uploaded_file):
    """
    Extracts text from uploaded PDF, DOCX, or TXT files.
    """
    if uploaded_file.name.endswith('.pdf'):
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            st.error(f"Error reading PDF file: {e}")
            return None
    elif uploaded_file.name.endswith('.docx'):
        try:
            doc = Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            st.error(f"Error reading DOCX file: {e}")
            return None
    elif uploaded_file.name.endswith('.txt'):
        try:
            return uploaded_file.getvalue().decode("utf-8")
        except Exception as e:
            st.error(f"Error reading TXT file: {e}")
            return None
    return None

# --- STREAMLIT APP LAYOUT ---

st.set_page_config(page_title="ClauseWise Legal Analyzer", layout="wide", initial_sidebar_state="collapsed")

st.title("ClauseWise: AI Legal Document Analyzer ⚖️")
st.markdown("Simplify, decode, and classify complex legal texts using IBM's Granite AI model.")

# --- Main Tabs ---
tab1, tab2 = st.tabs(["📄 Document Analyzer", "💬 Legal Q&A Chatbot"])


# --- TAB 1: DOCUMENT ANALYZER ---
with tab1:
    st.header("Upload and Analyze Your Legal Document")
    st.markdown("Supports **PDF, DOCX, and TXT** formats. The model will classify the document, simplify its clauses, identify risks, and extract key entities.")

    uploaded_file = st.file_uploader("Choose a document", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        if st.button("Analyze Document", key="analyze_doc"):
            with st.spinner("Reading and analyzing the document... This may take a moment."):
                # 1. Extract text from the document
                document_text = extract_text(uploaded_file)

                if document_text:
                    st.success("Document text extracted successfully!")

                    # 2. Define prompts for different analysis tasks
                    prompt_classify = f"Based on the content of the following legal document, what is the most likely document type? Examples: Non-Disclosure Agreement, Employment Contract, Lease Agreement, Service Agreement. Provide only the document type.\n\nDocument:\n{document_text[:4000]}\n\nDocument Type:"
                    prompt_simplify = f"Rewrite the following legal clauses in simple, plain English that a non-lawyer can easily understand. Break it down clause by clause.\n\nDocument:\n{document_text}\n\nSimplified Clauses:"
                    prompt_risk = f"Analyze the following legal text and identify potential risks, liabilities, and unfavorable clauses for a party signing this document. Present the analysis in a clear, structured format with bullet points.\n\nDocument:\n{document_text}\n\nRisk Analysis:"
                    prompt_ner = f"From the following legal text, extract the key entities. List them under these specific categories: Parties (names of people or companies involved), Obligations (key duties or actions required), Key Dates (specific dates or deadlines mentioned), and Governing Law (the jurisdiction mentioned).\n\nDocument:\n{document_text}\n\nExtracted Entities:"

                    # 3. Query the model for each task
                    doc_type = query_huggingface_model(prompt_classify)
                    simplified_text = query_huggingface_model(prompt_simplify)
                    risk_analysis = query_huggingface_model(prompt_risk)
                    entities = query_huggingface_model(prompt_ner)

                    # 4. Display results
                    st.subheader(f"Analysis Results for: `{uploaded_file.name}`")

                    if doc_type:
                        st.info(f"**Predicted Document Type:** {doc_type.strip()}")

                    if simplified_text:
                        with st.expander("🔍 Simplified Clauses (Plain English Version)", expanded=True):
                            st.markdown(simplified_text)

                    if risk_analysis:
                        with st.expander("⚠️ Risk Analysis", expanded=True):
                            st.markdown(risk_analysis)

                    if entities:
                        with st.expander("📌 Key Entities Extracted", expanded=True):
                            st.markdown(entities)
                else:
                    st.error("Could not extract text from the document. The file might be corrupted, empty, or an image-based PDF.")


# --- TAB 2: LEGAL Q&A CHATBOT ---
with tab2:
    st.header("Ask a General Legal Question")
    st.markdown("Get general information about **criminal law, traffic law, and human rights**. This chatbot does not provide legal advice.")
    
    # Disclaimer
    st.warning("🚨 **Disclaimer:** I am an AI assistant, not a lawyer. The information provided here is for general informational purposes only and does not constitute legal advice. Always consult with a qualified legal professional for advice on your specific situation.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question about criminal, traffic, or human rights law..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chatbot_prompt = f"""You are a helpful legal AI assistant named ClauseWise. Your purpose is to provide general information about criminal, traffic, and human rights law in simple terms. You must not give legal advice. You must include a brief disclaimer at the end of every response stating that the user should consult with a qualified legal professional.

                User question: {prompt}

                Answer:"""
                
                response = query_huggingface_model(chatbot_prompt)
                st.markdown(response or "Sorry, I couldn't process that request.")
        
        # Add assistant response to chat history
        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})