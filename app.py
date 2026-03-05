import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import joblib
import re
import torch
# import pytesseract
# import fitz
# from PIL import Image
# import io
# from pdf2image import convert_from_bytes

# 1. Page Configuration
st.set_page_config(page_title="AI Resume Job Matcher", layout="wide")

# 2. Load Assets
@st.cache_resource
def load_assets():
    # # Model now loads from Hugging Face instead of a local path
    model = SentenceTransformer('iwamu/bert-data-analyst-matcher')
    scaler = joblib.load('semantic_scaler_2.pkl')
    return model, scaler

model, scaler = load_assets()

# --- HELPER FUNCTIONS ---
def extract_text(file):
    pdf = PyPDF2.PdfReader(file)
    return " ".join([page.extract_text() or "" for page in pdf.pages])

# extract image pdf, scanned and exported resumes
# def extract_text(file):
#     try:
#         pdf = PyPDF2.PdfReader(file)
#         text = " ".join([page.extract_text() or "" for page in pdf.pages])
        
#         if len(text.strip()) > 50:
#             return text
#     except:
#         pass

#     # OCR fallback
#     images = convert_from_bytes(file.read())
#     ocr_text = ""

#     for img in images:
#         ocr_text += pytesseract.image_to_string(img)

#     return ocr_text

# def extract_text(file):
#     try:
#         pdf = PyPDF2.PdfReader(file)
#         text = " ".join([page.extract_text() or "" for page in pdf.pages])
        
#         if len(text.strip()) > 50:
#             return text
#     except:
#         pass

#     # OCR fallback
#     file.seek(0)
#     pdf = fitz.open(stream=file.read(), filetype="pdf")

#     ocr_text = ""

#     for page in pdf:
#         pix = page.get_pixmap()
#         img_bytes = pix.tobytes("png")
#         img = Image.open(io.BytesIO(img_bytes))

#         ocr_text += pytesseract.image_to_string(img)

#     return ocr_text


def chunk_resume(resume_text):
    """Simple cleaner to extract meaningful chunks for comparison"""
    # Removes special chars and splits by commas/bullets/newlines
    chunks = re.split(r'[,.\n•●/-]', resume_text)
    return [c.strip() for c in chunks if len(c.strip()) > 3]



def extract_jd(jd_text):
    jd_clean = jd_text.lower()
    # trigger_words = ["education and experience", "technical skills", "requirements", "responsibilities"]
    trigger_words = [
        "responsibilities",
        "requirements",
        "qualifications",
        "skills",
        "experience",
        "job responsibilities",
        "job requirements"
        ]

    start_idx = 0
    for word in trigger_words:
        idx = jd_clean.find(word)
        if idx != -1:
            start_idx = idx
            break

    relevant_text = jd_clean[start_idx:]
    lines = relevant_text.split("\n")
    
    requirements = []
    # Add triggers to a set for fast lookup
    blacklisted_headers = set(trigger_words)
    
    for l in lines:
        clean_l = l.strip()
        if len(clean_l) < 5 or clean_l in blacklisted_headers or clean_l.endswith(':'):
            continue

        final_line = re.sub(r'^[\-\•\*\○\●\d\.\s]+', '', clean_l)
        if len(final_line) > 10: # Ensure we aren't matching on single words
            requirements.append(final_line)
            
    return requirements[:20]


def analyze_skills(jd_text, resume_text):
    jd_requirements = extract_jd(jd_text)
    resume_chunks = chunk_resume(resume_text)
    
    if not jd_requirements or not resume_chunks:
        return [], {"Technical": [], "Stats": [], "Soft Skills": []}, 0.0
    
    jd_embs = model.encode(jd_requirements, convert_to_tensor=True)
    res_embs = model.encode(resume_chunks, convert_to_tensor=True)    
    cosine_scores = util.cos_sim(jd_embs, res_embs) 
    max_scores, _ = torch.max(cosine_scores, dim=1)

    matched = []
    gaps = {"Technical": [], "Stats": [], "Soft Skills": []}
    total_quality_score = 0

    for i, score in enumerate(max_scores):
        skill_name = jd_requirements[i]
        score_val = score.item() * 100
        
        if score_val > 60: # Strong match
            matched.append((skill_name, round(score_val, 2)))
            total_quality_score += 1.0 
        elif score_val > 45: # Partial Match
            matched.append((skill_name + " (Partial)", round(score_val, 2)))
            total_quality_score += 0.5
        else:
            # Smart Categorization for Gaps
            low_s = skill_name.lower()
            if any(k in low_s for k in ['python', 'sql', 'machine learning', 'code', 'production']):
                gaps["Technical"].append(skill_name)
            elif any(k in low_s for k in ['statistics', 'causal', 'inference', 'probability', 'math']):
                gaps["Stats"].append(skill_name)
            else:
                gaps["Soft Skills"].append(skill_name) 

    quality_ratio = total_quality_score / len(jd_requirements)
    return matched, gaps, quality_ratio

# regex for years of experience
def calculate_seniority_bonus(text):
    # Search for "X+ years", "X years of experience", etc.
    experience_matches = re.findall(r'(\d+)\+?\s*(?:years|yrs)', text.lower())
    if experience_matches:
        years = max([int(x) for x in experience_matches])
        if years >= 10: return 15  # Senior Leader Bonus
        if years >= 3: return 10   # Met JD Requirement Bonus
    return 0


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
        elements.append(Paragraph("<b>🔍 Gap Analysis by Category:</b>", styles['Normal']))
        elements.append(Spacer(1, 4))

        has_gaps = False
        for category, gap_list in row["Full_Gaps"].items():
            if gap_list:
                has_gaps = True
                elements.append(Paragraph(f"<i>{category}:</i>", styles['Normal']))
                for g in gap_list:
                    elements.append(Paragraph(f"• {g}", styles['Normal']))
                elements.append(Spacer(1, 4))

        if not has_gaps:
            elements.append(Paragraph("• No significant gaps identified.", styles['Normal']))

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
            #  place outside for performance since within loop recomputes it for every resune
            jd_emb = model.encode(jd_input, convert_to_tensor=True)
            for file in uploaded_files:
                text = extract_text(file)
                
                # 1. Global Similarity
                res_emb = model.encode(text, convert_to_tensor=True)
                raw_sim = util.cos_sim(jd_emb, res_emb).item()
                

                # 2. Skill-Level Audit (Using our optimized batch function)
                # matches, gaps = analyze_skills(jd_input, text)
                # skill audit with fuzzy logic
                matches, gaps, quality_ratio = analyze_skills(jd_input, text)
                

                # 3. Seniority Detection
                bonus = calculate_seniority_bonus(text)

                # 4. Final Differentiated Formula
                # 30% Vibe + 50% Quality Coverage + 20% Seniority/Bonus
                final_score = (raw_sim * 30) + (quality_ratio * 50) + bonus

                # Ensure it doesn't exceed 100
                final_score = float(np.clip(final_score, 0, 100))

                results.append({
                    "Candidate": file.name,
                    "Match Score": final_score,
                    # "Matches": ", ".join(matches[:5]),
                    # "Matches": ", ".join(matches),
                    "Matches": ", ".join([f"{m[0]} ({m[1]}%)" for m in matches]),
                    # "Gaps": ", ".join(gaps[:5]),
                    "Gaps": ", ".join([item for sublist in gaps.values() for item in sublist][:3]),
                    # "Gaps": ", ".join(gaps),
                    "Full_Matches": matches,
                    # "Full_Gaps": gaps
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

