import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import joblib
import os

# 1. Page Configuration
st.set_page_config(page_title="AI Resume Matcher", layout="wide")

# 2. Load my pretrained SBERT Model (Cached for performance)
@st.cache_resource
def load_model():
    # model_path = 'model_assets/sbert_model'
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load the MinMaxScaler
    scaler_path = 'semantic_scaler.pkl'
    scaler = joblib.load(scaler_path)

    return model, scaler

model,scaler = load_model()

# 3. Update scoring logic 
def calculate_calibrated_score(jd_text, resume_text):
    # Get embeddings
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    res_emb = model.encode(resume_text, convert_to_tensor=True)
    
    # Raw Cosine Similarity
    raw_score = util.cos_sim(jd_emb, res_emb).item()
    
    # Apply your Scaler (reshape is needed for a single value)
    calibrated_score = scaler.transform(np.array([[raw_score]]))[0][0]
    
    return round(calibrated_score * 100, 2)


# Extract text from PDF
def extract_text(file):
    pdf = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

# --- UI LAYOUT ---
st.title("🎯 Semantic Resume-JD Matcher")
st.markdown("Rank resumes based on **meaning**, not just keywords.")

with st.sidebar:
    st.header("Step 1: Job Description")
    jd_text = st.text_area("Paste the Job Description here:", height=300)

st.header("Step 2: Upload Resumes")
uploaded_files = st.file_uploader("Upload candidate PDFs", type="pdf", accept_multiple_files=True)

if st.button("🚀 Analyze & Rank"):
    if not jd_text or not uploaded_files:
        st.error("Please provide both a JD and at least one resume.")
    else:
        results = []
        with st.spinner("Calculating semantic scores..."):
            # Compute JD embedding once
            jd_embedding = model.encode(jd_text, convert_to_tensor=True)
            
            for file in uploaded_files:
                # Extract and Embed
                resume_text = extract_text(file)
                resume_embedding = model.encode(resume_text, convert_to_tensor=True)
                
                # Calculate Cosine Similarity
                score = util.cos_sim(jd_embedding, resume_embedding).item()
                
                results.append({
                    "Candidate Name": file.name,
                    "Match Score": round(score * 100, 2),
                    "Raw Text": resume_text[:500] + "..." # Preview
                })
        
        # 3. Display Results
        df = pd.DataFrame(results).sort_values(by="Match Score", ascending=False)
        
        st.subheader("📊 Ranked Candidates")
        # Highlighting the top match
        st.dataframe(df.style.highlight_max(axis=0, subset=['Match Score'], color='lightgreen'), 
                     use_container_width=True)
        
        # Success Metric
        st.success(f"Top candidate: {df.iloc[0]['Candidate Name']} with {df.iloc[0]['Match Score']}% match!")