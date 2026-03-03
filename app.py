import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import joblib
import re
import torch

# 1. Page Configuration
st.set_page_config(page_title="AI Resume Job Matcher", layout="wide")

# 2. Load Assets
@st.cache_resource
def load_assets():
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer('all-mpnet-base-v2')
    scaler = joblib.load('semantic_scaler_2.pkl')
    return model, scaler
    return model

model, scaler = load_assets()
# model = load_assets()

# --- HELPER FUNCTIONS ---
def extract_text(file):
    pdf = PyPDF2.PdfReader(file)
    return " ".join([page.extract_text() or "" for page in pdf.pages])

def chunk_resume(resume_text):
    """Simple cleaner to extract meaningful chunks for comparison"""
    # Removes special chars and splits by commas/bullets/newlines
    chunks = re.split(r'[,.\n•●/-]', resume_text)
    return [c.strip() for c in chunks if len(c.strip()) > 3]

# def extract_jd(jd_text):
#     """
#     Robust JD extraction without fragile regex blocks.
#     """

#     jd_text = jd_text.lower()

#     # Keep only the meaningful section onward
#     trigger_words = [
#         "key responsibilities",
#         "required skills",
#         "required qualifications",
#         "preferred qualifications"
#     ]

#     start_idx = None
#     for word in trigger_words:
#         idx = jd_text.find(word)
#         if idx != -1:
#             start_idx = idx
#             break

#     if start_idx is None:
#         return []

#     relevant_text = jd_text[start_idx:]

#     # Split into lines
#     lines = relevant_text.split("\n")

#     requirements = []
#     for line in lines:
#         line = line.strip()

#         if (
#             len(line) > 25
#             and not any(x in line for x in ["about the job", "rate:", "engagement", "title:"])
#         ):
#             requirements.append(line)

#     return requirements[:15]

def extract_jd(jd_text):
    """
    Enhanced JD extraction that handles varied formatting and missing headers.
    """
    jd_clean = jd_text.lower()
    
    # 1. Expanded Triggers (covering more industries)
    trigger_words = [
        "key responsibilities", "required skills", "qualifications", 
        "what you will do", "experience you'll need", "requirements",
        "job description", "core competencies", "technical skills"
    ]

    start_idx = None
    for word in trigger_words:
        idx = jd_clean.find(word)
        if idx != -1:
            start_idx = idx
            break

    # 2. Fallback: If no headers, take the whole text but skip "About" intro
    if start_idx == None:
        return []
    
    relevant_text = jd_clean[start_idx:]

    # 3. Smart Filtering of lines
    lines = relevant_text.split("\n")
    requirements = []
    
    # List of "Noise" phrases to discard
    stop_phrases = ["about the company", "equal opportunity", "how to apply", "salary range", "rate:", "engagement", "title:"]

    for line in lines:
        line = line.strip()
        # Filter for quality: not too short, not a header, not noise
        if (
            # 15 < len(line) < 300  # Requirements are usually sentences
            len(line) >  25  # Requirements are usually sentences
            and not any(stop in line for stop in stop_phrases)
            and (line.startswith(('-', '•', '*', '○', '●')) or any(char.isdigit() for char in line[:2]))
        ):
            # Clean bullet points for better SBERT encoding
            cleaned_line = re.sub(r'^[\-\•\*\○\●\d\.\s]+', '', line)
            requirements.append(cleaned_line)

    # 4. Final Fallback: If no bullet points found, take the top 10 meaningful sentences
    # if not requirements:
    #     sentences = re.split(r'(?<=[.!?]) +', relevant_text)
    #     requirements = [s.strip() for s in sentences if 30 < len(s.strip()) < 200][:10]

    return requirements[:20]


def analyze_skills(jd_text, resume_text):
    """Matches JD requirements to Resume text using SBERT"""
    jd_requirements = extract_jd(jd_text)[:20] # Focus on top 20 chunks
    resume_chunks = chunk_resume(resume_text)
    
    if not jd_requirements or not resume_chunks:
        return [], jd_requirements
    
    # batch-encode for speed
    jd_embs = model.encode(jd_requirements, convert_to_tensor=True)
    res_embs = model.encode(resume_chunks, convert_to_tensor=True)    
    
    # Use matrix multiplication for all scores at once
    cosine_scores = util.cos_sim(jd_embs, res_embs) 
    # Now find the max score for each JD requirement
    max_scores, _ = torch.max(cosine_scores, dim=1)

    matched = []
    gaps = []

    for i, score in enumerate(max_scores):
        skill_name = jd_requirements[i]
        if score.item() > 0.65: # .item() converts tensor to float
        # if score.item() > 0.72: # .item() converts tensor to float
        # if score.item() > strictness: # .item() converts tensor to float
            # matched.append(skill_name)
            matched.append((skill_name, round(score.item()*100,2)))
        else:
            gaps.append(skill_name)
            
    return list(set(matched)), list(set(gaps))


# 4 . Generate downloadable output format
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import units
from reportlab.lib.pagesizes import A4
from io import BytesIO
def generate_pdf_report(df):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>Ranked Candidate Audit Report</b>", styles['Heading1']))
    elements.append(Spacer(1, 12))

    for idx, row in df.reset_index(drop=True).iterrows():

        elements.append(Paragraph(
            f"<b>Rank {idx+1}: {row['Candidate']}</b>",
            styles['Heading2']
        ))
        elements.append(Spacer(1, 6))

        elements.append(Paragraph(
            f"<b>Overall Match Score:</b> {row['Match Score']}%",
            styles['Normal']
        ))
        elements.append(Spacer(1, 10))

        # Matches
        elements.append(Paragraph("<b>Semantic Matches:</b>", styles['Normal']))
        elements.append(Spacer(1, 4))

        if row["Full_Matches"]:
            for m in row["Full_Matches"]:
                if isinstance(m, tuple):
                    elements.append(Paragraph(
                        f"• {m[0]}  (Confidence: {m[1]}%)",
                        styles['Normal']
                    ))
                else:
                    elements.append(Paragraph(f"• {m}", styles['Normal']))
        else:
            elements.append(Paragraph("• No strong semantic matches detected.", styles['Normal']))

        elements.append(Spacer(1, 8))

        # Gaps
        elements.append(Paragraph("<b>Identified Gaps:</b>", styles['Normal']))
        elements.append(Spacer(1, 4))

        if row["Full_Gaps"]:
            for g in row["Full_Gaps"]:
                elements.append(Paragraph(f"• {g}", styles['Normal']))
        else:
            elements.append(Paragraph("• No significant gaps identified.", styles['Normal']))

        elements.append(Spacer(1, 20))

    doc.build(elements)
    buffer.seek(0)
    return buffer


import math

def sigmoid_calibration(score):
    # This keeps the score mapping 1:1 in the middle but 
    # prevents it from exploding at the top/bottom
    return 1 / (1 + math.exp(-score))


# --- UI LAYOUT ---
st.title("🎯 Semantic Resume-JD Matcher")
st.markdown("Rank resumes based on **meaning**, and see exactly what's missing.")

with st.sidebar:
    st.header("Step 1: Job Description")
    jd_input = st.text_area("Paste the Job Description here:", height=300)

st.header("Step 2: Upload Resumes")
uploaded_files = st.file_uploader("Upload candidate PDFs", type="pdf", accept_multiple_files=True)

if st.button("🚀 Analyze & Rank"):
    if not jd_input or not uploaded_files:
        st.error("Please provide both a JD and at least one resume.")
    else:
        results = []
        with st.spinner("Performing Deep Semantic Audit..."):
            for file in uploaded_files:
                text = extract_text(file)
                
                # 1. Global Similarity
                jd_emb = model.encode(jd_input, convert_to_tensor=True)
                res_emb = model.encode(text, convert_to_tensor=True)
                raw_sim = util.cos_sim(jd_emb, res_emb).item()
                
                # Calibrated Score
                calibrated = scaler.transform(np.array([[raw_sim]]))[0][0]
                final_score = sigmoid_calibration(calibrated) * 100

                # 2. Skill-Level Audit (Using our optimized batch function)
                matches, gaps = analyze_skills(jd_input, text)

                results.append({
                    "Candidate": file.name,
                    "Match Score": final_score,
                    # "Matches": ", ".join(matches[:5]),
                    # "Matches": ", ".join(matches),
                    "Matches": ", ".join([f"{m[0]} ({m[1]}%)" for m in matches]),
                    # "Gaps": ", ".join(gaps[:5]),
                    "Gaps": ", ".join(gaps),
                    "Full_Matches": matches,
                    "Full_Gaps": gaps
                })
        
        # Create Result DataFrame
        df = pd.DataFrame(results).sort_values(by="Match Score", ascending=False)

        # --- UI DISPLAY ---
        st.success(f"✅ Analysis Complete! Top candidate: {df.iloc[0]['Candidate']}")

        # Download Button
        pdf_buffer = generate_pdf_report(df)
        st.download_button(
            label="📄 Download Full Ranking Report (PDF)",
            data=pdf_buffer,
            file_name="Ranked_Candidates_Audit.pdf",
            mime="application/pdf"
        )

        st.divider()

        # 1. High-Level Summary Table
        st.subheader("📊 Ranking Overview")
        st.dataframe(
            df[['Candidate', 'Match Score', 'Matches', 'Gaps']].style.highlight_max(axis=0, subset=['Match Score'], color='lightgreen'),
            use_container_width=True
        )

        # 2. Deep Dive Expanders
        st.subheader("🔍 Individual Candidate Audit")
        for _, row in df.iterrows():
            with st.expander(f"Audit: {row['Candidate']} ({row['Match Score']}%)"):
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**✅ Semantic Matches**")
                    for m in row['Full_Matches']: st.write(f"- {m}")
                with c2:
                    st.write("**⚠️ Found Gaps**")
                    for g in row['Full_Gaps']: st.write(f"- {g}")
                
                # Strength Chart
                if row['Full_Matches']:
                    st.write("---")
                    st.write("**Match Confidence Profile**")

                    requirements = [m[0] for m in row['Full_Matches']]
                    confidences = [m[1] for m in row['Full_Matches']]
                    # Simulating confidence levels for the UI
                    conf_data = pd.DataFrame({
                        # "Requirement": row['Full_Matches'],
                        "Requirement": requirements,
                        # "Confidence": np.random.uniform(75, 99, len(row['Full_Matches']))
                        "Confidence": confidences
                    }).set_index("Requirement")
                    st.bar_chart(conf_data)