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

model, scaler = load_assets()

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
#     Extract meaningful skill/responsibility requirements 
#     from JD using structured section parsing.
#     """
#     # Normalize whitespace
#     jd_text = re.sub(r'\r', '', jd_text)

#     # Capture content under relevant sections using regex blocks
#     pattern = re.compile(
#         r'(Key Responsibilities|Responsibilities|Required Skills.*?|Qualifications|Preferred Qualifications)(.*?)(?=\n[A-Z][^\n]+\n|$)',
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = pattern.findall(jd_text)

#     extracted = []

#     for _, section_text in matches:
#         # Split by bullet symbols or line breaks (NOT hyphen)
#         chunks = re.split(r'\n|•|●', section_text)

#         for chunk in chunks:
#             chunk = chunk.strip()

#             # Remove leading dashes only (not internal ones)
#             chunk = re.sub(r'^-\s*', '', chunk)

#             # Filter noise
#             if len(chunk) > 20 and not chunk.lower().startswith(
#                 ("about", "title", "rate", "engagement", "job summary")
#             ):
#                 extracted.append(chunk)

#     # Remove near-duplicates while preserving order
#     seen = set()
#     cleaned = []
#     for item in extracted:
#         norm = item.lower()
#         if norm not in seen:
#             seen.add(norm)
#             cleaned.append(item)

#     return cleaned[:15]

def extract_jd(jd_text):
    """
    Robust JD extraction without fragile regex blocks.
    """

    jd_text = jd_text.lower()

    # Keep only the meaningful section onward
    trigger_words = [
        "key responsibilities",
        "required skills",
        "required qualifications",
        "preferred qualifications"
    ]

    start_idx = None
    for word in trigger_words:
        idx = jd_text.find(word)
        if idx != -1:
            start_idx = idx
            break

    if start_idx is None:
        return []

    relevant_text = jd_text[start_idx:]

    # Split into lines
    lines = relevant_text.split("\n")

    requirements = []
    for line in lines:
        line = line.strip()

        if (
            len(line) > 25
            and not any(x in line for x in ["about the job", "rate:", "engagement", "title:"])
        ):
            requirements.append(line)

    return requirements[:15]


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
        # if score.item() > 0.65: # .item() converts tensor to float
        if score.item() > 0.72: # .item() converts tensor to float
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

# def generate_pdf_report(df):
#     buffer = BytesIO()
#     doc = SimpleDocTemplate(buffer, pagesize=A4)
#     elements = []
#     table_data = [["Rank", "Candidate", "Match Score (%)"]]
#     styles = getSampleStyleSheet()

#     # Title
#     elements.append(Paragraph("<b>Ranked Candidate Report</b>", styles['Heading1']))
#     elements.append(Spacer(1, 12))

#     elements.append(Paragraph("Generated by AI Semantic Resume Matcher", styles['Normal']))
#     elements.append(Spacer(1, 20))

#     for idx, row in df.reset_index(drop=True).iterrows():
#         table_data.append([
#             idx + 1,
#             row["Candidate"],
#             f"{row['Match Score']}%"
#         ])

#     table = Table(table_data, colWidths=[50, 250, 120])

#     table.setStyle(TableStyle([
#         ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
#         ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
#         ('ALIGN', (2,1), (2,-1), 'CENTER'),
#         ('FONTNAME', (0,0), (-1,-1), 'Helvetica')
#     ]))

#     elements.append(table)
#     doc.build(elements)

#     buffer.seek(0)
#     return 

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
                final_score = float(np.clip(calibrated * 100, 0, 100))
                final_score = round(final_score, 2)
                
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