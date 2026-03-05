# import streamlit as st
# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer, util
# import PyPDF2
# import joblib
# import re
# import torch
# # import pytesseract
# # import fitz
# # from PIL import Image
# # import io
# # from pdf2image import convert_from_bytes

# # 1. Page Configuration
# st.set_page_config(page_title="AI Resume Job Matcher", layout="wide")

# # 2. Load Assets
# @st.cache_resource
# def load_assets():
#     # # Model now loads from Hugging Face instead of a local path
#     model = SentenceTransformer('iwamu/bert-data-analyst-matcher')
#     scaler = joblib.load('semantic_scaler_2.pkl')
#     return model, scaler

# model, scaler = load_assets()

# # --- HELPER FUNCTIONS ---
# def extract_text(file):
#     pdf = PyPDF2.PdfReader(file)
#     return " ".join([page.extract_text() or "" for page in pdf.pages])


# # def chunk_resume(resume_text):
# #     """Simple cleaner to extract meaningful chunks for comparison"""
# #     # Removes special chars and splits by commas/bullets/newlines
# #     chunks = re.split(r'[,.\n•●/-]', resume_text)
# #     return [c.strip() for c in chunks if len(c.strip()) > 3]


# def chunk_resume(resume_text):
#     """Extract meaningful resume chunks"""
    
#     # Split by section headers first
#     sections = re.split(r'\n(?=\b(?:PROFESSIONAL|TECHNICAL|EDUCATION|SKILLS|EXPERIENCE|PROJECTS)\b)', 
#                        resume_text, flags=re.IGNORECASE)
    
#     chunks = []
#     for section in sections:
#         # Within each section, split by bullets/lines
#         lines = section.split('\n')
#         for line in lines:
#             clean = re.sub(r'^[\s\-\•\*\d\.\)]+', '', line).strip()
#             # Keep full bullet points, don't over-split
#             if len(clean) > 5:
#                 chunks.append(clean)
    
#     return [c for c in chunks if len(c) > 8]

# # def extract_jd(jd_text):
# #     jd_clean = jd_text.lower()
# #     # trigger_words = ["education and experience", "technical skills", "requirements", "responsibilities"]
# #     trigger_words = [
# #         "responsibilities",
# #         "requirements",
# #         "qualifications",
# #         "skills",
# #         "experience",
# #         "job responsibilities",
# #         "job requirements"
# #         ]

# #     start_idx = 0
# #     for word in trigger_words:
# #         idx = jd_clean.find(word)
# #         if idx != -1:
# #             start_idx = idx
# #             break

# #     relevant_text = jd_clean[start_idx:]
# #     lines = relevant_text.split("\n")
    
# #     requirements = []
# #     # Add triggers to a set for fast lookup
# #     blacklisted_headers = set(trigger_words)
    
# #     for l in lines:
# #         clean_l = l.strip()
# #         if len(clean_l) < 5 or clean_l in blacklisted_headers or clean_l.endswith(':'):
# #             continue

# #         final_line = re.sub(r'^[\-\•\*\○\●\d\.\s]+', '', clean_l)
# #         if len(final_line) > 10: # Ensure we aren't matching on single words
# #             requirements.append(final_line)
            
# #     return requirements[:20]

# def extract_jd(jd_text):
#     jd_clean = jd_text.lower()
    
#     # Find section starts more robustly
#     section_keywords = [
#         "responsibilities", "requirements", "qualifications", 
#         "skills", "experience", "what we look for", "required qualifications",
#         "preferred qualifications", "key responsibilities", "what we offer"
#     ]
    
#     # Extract from first section keyword onward
#     earliest_idx = len(jd_text)
#     for keyword in section_keywords:
#         idx = jd_clean.find(keyword)
#         if idx != -1 and idx < earliest_idx:
#             earliest_idx = idx
    
#     relevant_text = jd_text[earliest_idx:] if earliest_idx < len(jd_text) else jd_text
    
#     # Better bullet detection
#     lines = relevant_text.split("\n")
#     requirements = []
    
#     for line in lines:
#         clean_line = line.strip()
        
#         # Skip empty, header-like, or too-short lines
#         if not clean_line or len(clean_line) < 8 or clean_line.endswith(':'):
#             continue
        
#         # Remove bullets more aggressively
#         cleaned = re.sub(r'^[\s\-\•\*\○\●\d\.\)]+\s*', '', clean_line)
#         cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
#         if len(cleaned) > 12:  
#             requirements.append(cleaned)
    
#     return requirements[:25] 


# def analyze_skills(jd_text, resume_text):
#     jd_requirements = extract_jd(jd_text)
#     resume_chunks = chunk_resume(resume_text)
    
#     if not jd_requirements or not resume_chunks:
#         return [], {"Technical": [], "Stats": [], "Soft Skills": []}, 0.0
    
#     jd_embs = model.encode(jd_requirements, convert_to_tensor=True)
#     res_embs = model.encode(resume_chunks, convert_to_tensor=True)    
#     cosine_scores = util.cos_sim(jd_embs, res_embs) 
#     max_scores, _ = torch.max(cosine_scores, dim=1)

#     matched = []
#     # gaps = {"Technical": [], "Stats": [], "Soft Skills": []}
#     gaps = {
#     "Technical": [],
#     "Stats": [],
#     "Soft Skills": [],
#     "Critical": []  # Requirements > 70% importance
# }
#     total_quality_score = 0

#     for i, score in enumerate(max_scores):
#         skill_name = jd_requirements[i]
#         score_val = score.item() * 100
        
#         if score_val > 60: # Strong match
#             matched.append((skill_name, round(score_val, 2)))
#             total_quality_score += 1.0 
#         elif score_val > 45: # Partial Match
#             matched.append((skill_name + " (Partial)", round(score_val, 2)))
#             total_quality_score += 0.5
#         else:
#             # Smart Categorization for Gaps
#             low_s = skill_name.lower()
#             if any(k in low_s for k in ['python', 'sql', 'machine learning', 'code', 'production']):
#                 gaps["Technical"].append(skill_name)
#             elif any(k in low_s for k in ['statistics', 'causal', 'inference', 'probability', 'math']):
#                 gaps["Stats"].append(skill_name)
#             else:
#                 gaps["Soft Skills"].append(skill_name) 

#     quality_ratio = total_quality_score / len(jd_requirements)
#     return matched, gaps, quality_ratio

# # regex for years of experience
# def calculate_seniority_bonus(text):
#     # Search for "X+ years", "X years of experience", etc.
#     experience_matches = re.findall(r'(\d+)\+?\s*(?:years|yrs)', text.lower())
#     if experience_matches:
#         years = max([int(x) for x in experience_matches])
#         if years >= 10: return 15  # Senior Leader Bonus
#         if years >= 3: return 10   # Met JD Requirement Bonus
#     return 0


# # 4 . Generate downloadable output format
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
# from reportlab.lib import colors
# from reportlab.lib.styles import getSampleStyleSheet
# from reportlab.lib import units
# from reportlab.lib.pagesizes import A4
# from io import BytesIO
# def generate_pdf_report(df):
#     buffer = BytesIO()
#     doc = SimpleDocTemplate(buffer, pagesize=A4)
#     elements = []
#     styles = getSampleStyleSheet()

#     elements.append(Paragraph("<b>Ranked Candidate Audit Report</b>", styles['Heading1']))
#     elements.append(Spacer(1, 12))

#     for idx, row in df.reset_index(drop=True).iterrows():

#         elements.append(Paragraph(
#             f"<b>Rank {idx+1}: {row['Candidate']}</b>",
#             styles['Heading2']
#         ))
#         elements.append(Spacer(1, 6))

#         elements.append(Paragraph(
#             f"<b>Overall Match Score:</b> {row['Match Score']}%",
#             styles['Normal']
#         ))
#         elements.append(Spacer(1, 10))

#         # Matches
#         elements.append(Paragraph("<b>Semantic Matches:</b>", styles['Normal']))
#         elements.append(Spacer(1, 4))

#         if row["Full_Matches"]:
#             for m in row["Full_Matches"]:
#                 if isinstance(m, tuple):
#                     elements.append(Paragraph(
#                         f"• {m[0]}  (Confidence: {m[1]}%)",
#                         styles['Normal']
#                     ))
#                 else:
#                     elements.append(Paragraph(f"• {m}", styles['Normal']))
#         else:
#             elements.append(Paragraph("• No strong semantic matches detected.", styles['Normal']))

#         elements.append(Spacer(1, 8))

#         # Gaps
#         elements.append(Paragraph("<b>🔍 Gap Analysis by Category:</b>", styles['Normal']))
#         elements.append(Spacer(1, 4))

#         has_gaps = False
#         for category, gap_list in row["Full_Gaps"].items():
#             if gap_list:
#                 has_gaps = True
#                 elements.append(Paragraph(f"<i>{category}:</i>", styles['Normal']))
#                 for g in gap_list:
#                     elements.append(Paragraph(f"• {g}", styles['Normal']))
#                 elements.append(Spacer(1, 4))

#         if not has_gaps:
#             elements.append(Paragraph("• No significant gaps identified.", styles['Normal']))

#     doc.build(elements)
#     buffer.seek(0)
#     return buffer



# # --- UI LAYOUT ---
# st.title("🎯 Semantic Resume-JD Matcher")
# st.markdown("Rank resumes based on **meaning**, and see exactly what's missing.")

# with st.sidebar:
#     st.header("Step 1: Job Description")
#     jd_input = st.text_area("Paste the Job Description here:", height=300)

# st.header("Step 2: Upload Resumes")
# uploaded_files = st.file_uploader("Upload candidate PDFs", type="pdf", accept_multiple_files=True)

# if st.button("🚀 Analyze & Rank"):
#     if not jd_input or not uploaded_files:
#         st.error("Please provide both a JD and at least one resume.")
#     else:
#         results = []
#         with st.spinner("Performing Deep Semantic Audit..."):
#             #  place outside for performance since within loop recomputes it for every resune
#             jd_emb = model.encode(jd_input, convert_to_tensor=True)
#             for file in uploaded_files:
#                 text = extract_text(file)
                
#                 # 1. Global Similarity
#                 res_emb = model.encode(text, convert_to_tensor=True)
#                 raw_sim = util.cos_sim(jd_emb, res_emb).item()
                

#                 # 2. Skill-Level Audit (Using our optimized batch function)
#                 # matches, gaps = analyze_skills(jd_input, text)
#                 # skill audit with fuzzy logic
#                 matches, gaps, quality_ratio = analyze_skills(jd_input, text)
                

#                 # 3. Seniority Detection
#                 bonus = calculate_seniority_bonus(text)
#                 quality_bonus = quality_ratio * 70
#                 sim_bonus = raw_sim * 20
#                 # 4. Final Differentiated Formula
#                 # 30% Vibe + 50% Quality Coverage + 20% Seniority/Bonus
#                 # final_score = (raw_sim * 30) + (quality_ratio * 50) + bonus
#                 # final_score = (raw_sim * 20) + (quality_ratio * 60) + bonus
#                 final_score = quality_bonus + sim_bonus + bonus

#                 # Ensure it doesn't exceed 100
#                 # final_score = float(np.clip(final_score, 0, 100))

#                 results.append({
#                     "Candidate": file.name,
#                     "Match Score": final_score,
#                     # "Matches": ", ".join(matches[:5]),
#                     # "Matches": ", ".join(matches),
#                     "Matches": ", ".join([f"{m[0]} ({m[1]}%)" for m in matches]),
#                     # "Gaps": ", ".join(gaps[:5]),
#                     "Gaps": ", ".join([item for sublist in gaps.values() for item in sublist][:3]),
#                     # "Gaps": ", ".join(gaps),
#                     "Full_Matches": matches,
#                     # "Full_Gaps": gaps
#                     "Full_Gaps": gaps      
#                 })
        
#         # Create Result DataFrame
#         df = pd.DataFrame(results).sort_values(by="Match Score", ascending=False)

#         # --- UI DISPLAY ---
#         st.success(f"✅ Analysis Complete! Top candidate: {df.iloc[0]['Candidate']}")

#         # Download Button
#         pdf_buffer = generate_pdf_report(df)
#         st.download_button(
#             label="📄 Download Full Ranking Report (PDF)",
#             data=pdf_buffer,
#             file_name="Ranked_Candidates_Audit.pdf",
#             mime="application/pdf"
#         )

#         st.divider()

#         # 1. High-Level Summary Table
#         st.subheader("📊 Ranking Overview")
#         st.dataframe(
#             df[['Candidate', 'Match Score', 'Matches', 'Gaps']].style.highlight_max(axis=0, subset=['Match Score'], color='lightgreen'),
#             use_container_width=True
#         )

#         # 2. Deep Dive Expanders
#         st.subheader("🔍 Individual Candidate Audit")
#         for _, row in df.iterrows():
#             with st.expander(f"Audit: {row['Candidate']} ({row['Match Score']}%)"):
#                 c1, c2 = st.columns(2)
#                 with c1:
#                     st.write("**✅ Semantic Matches**")
#                     for m in row['Full_Matches']: st.write(f"- {m}")
#                 with c2:
#                     st.write("**⚠️ Found Gaps**")
#                     for g in row['Full_Gaps']: st.write(f"- {g}")
                
#                 # Strength Chart
#                 if row['Full_Matches']:
#                     st.write("---")
#                     st.write("**Match Confidence Profile**")

#                     requirements = [m[0] for m in row['Full_Matches']]
#                     confidences = [m[1] for m in row['Full_Matches']]
#                     # Simulating confidence levels for the UI
#                     conf_data = pd.DataFrame({
#                         # "Requirement": row['Full_Matches'],
#                         "Requirement": requirements,
#                         # "Confidence": np.random.uniform(75, 99, len(row['Full_Matches']))
#                         "Confidence": confidences
#                     }).set_index("Requirement")
#                     st.bar_chart(conf_data)



import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import joblib
import re
import torch
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ============================================
# GAP SEVERITY CLASSES (Integrated)
# ============================================

class GapSeverity(Enum):
    """Gap severity levels"""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass
class GapItem:
    """Structured gap representation"""
    skill: str
    category: str
    severity: GapSeverity
    reason: str
    frequency: int
    position: int


class GapSeverityAnalyzer:
    """Analyzes and categorizes gaps by severity"""
    
    def __init__(self):
        self.critical_keywords = {
            'must have', 'required', 'essential', 'mandatory',
            '7+', '8+', '10+',
            'lead', 'architect', 'design', 'responsible for'
        }
        
        self.high_keywords = {
            'strong', 'proven', 'demonstrated', 'extensive',
            '3+', '5+', 'preferred', 'highly desired'
        }
        
        self.medium_keywords = {
            'experience with', 'familiarity', 'knowledge of',
            'understanding of', 'ability to', 'nice to have'
        }
        
        self.technical_keywords = {
            'python', 'sql', 'machine learning', 'code', 'production',
            'api', 'database', 'framework', 'architecture', 'system',
            'backend', 'frontend', 'full-stack', 'deploy', 'git',
            'docker', 'kubernetes', 'aws', 'gcp', 'azure',
            'rest', 'graphql', 'grpc', 'microservice',
            'testing', 'optimization', 'scalability'
        }
        
        self.stats_keywords = {
            'statistics', 'causal', 'inference', 'probability',
            'math', 'mathematics', 'statistical', 'experiment',
            'a/b test', 'hypothesis', 'regression', 'analytical'
        }
        
        self.soft_skills_keywords = {
            'communication', 'leadership', 'mentoring', 'collaboration',
            'team', 'cross-functional', 'management', 'delegation',
            'presentation', 'writing', 'interpersonal'
        }
    
    def get_severity_level(self, requirement: str, position: int, 
                          frequency: int, jd_text: str) -> Tuple[GapSeverity, str]:
        """Determine severity level of a gap"""
        reasons = []
        severity_score = 0
        
        req_lower = requirement.lower()
        
        # 1. POSITION-BASED SEVERITY
        if position < 3:
            severity_score += 30
            reasons.append("Listed in top 3 requirements")
        elif position < 7:
            severity_score += 20
            reasons.append("Listed in top 7 requirements")
        elif position < 15:
            severity_score += 10
            reasons.append("Listed in requirements")
        
        # 2. CRITICAL KEYWORDS
        if any(kw in req_lower for kw in self.critical_keywords):
            severity_score += 35
            matching_kw = [kw for kw in self.critical_keywords if kw in req_lower][0]
            reasons.append(f"Contains critical keyword: '{matching_kw}'")
        
        # 3. HIGH KEYWORDS
        elif any(kw in req_lower for kw in self.high_keywords):
            severity_score += 20
            matching_kw = [kw for kw in self.high_keywords if kw in req_lower][0]
            reasons.append(f"Contains high-priority keyword: '{matching_kw}'")
        
        # 4. FREQUENCY-BASED
        if frequency >= 3:
            severity_score += 25
            reasons.append(f"Mentioned {frequency} times in JD")
        elif frequency == 2:
            severity_score += 15
            reasons.append("Mentioned multiple times")
        
        # 5. YEARS OF EXPERIENCE
        exp_match = re.search(r'(\d+)\+?\s*(?:years|yrs)', req_lower)
        if exp_match:
            years = int(exp_match.group(1))
            severity_score += 25
            if years >= 10:
                severity_score += 10
                reasons.append(f"Requires {years}+ years (senior-level)")
            elif years >= 5:
                reasons.append(f"Requires {years}+ years")
        
        # 6. CONTEXT IN JD
        jd_lower = jd_text.lower()
        resp_section = jd_lower.find('responsibility')
        pref_section = jd_lower.find('preferred')
        req_section = jd_lower.find('required')
        
        if req_section != -1:
            severity_score += 15
            reasons.append("In 'Requirements' section")
        elif resp_section != -1:
            severity_score += 10
            reasons.append("In 'Responsibilities' section")
        elif pref_section != -1:
            severity_score -= 10
            reasons.append("In 'Preferred' section (not required)")
        
        # 7. COMPLEXITY/IMPACT WORDS
        impact_keywords = ['lead', 'architect', 'design', 'manage', 'mentor',
                          'implement', 'optimize', 'scale', 'improve', 'reduce']
        if any(kw in req_lower for kw in impact_keywords):
            severity_score += 15
            reasons.append("High-impact responsibility")
        
        # Determine final severity
        if severity_score >= 85:
            return GapSeverity.CRITICAL, "; ".join(reasons)
        elif severity_score >= 60:
            return GapSeverity.HIGH, "; ".join(reasons)
        elif severity_score >= 35:
            return GapSeverity.MEDIUM, "; ".join(reasons)
        else:
            return GapSeverity.LOW, "; ".join(reasons)
    
    def categorize_gap(self, gap: str) -> str:
        """Categorize gap into Technical, Stats, or Soft Skills"""
        gap_lower = gap.lower()
        
        if any(kw in gap_lower for kw in self.technical_keywords):
            return "Technical"
        elif any(kw in gap_lower for kw in self.stats_keywords):
            return "Statistics"
        elif any(kw in gap_lower for kw in self.soft_skills_keywords):
            return "Soft Skills"
        else:
            return "Domain Knowledge"
    
    def count_requirement_frequency(self, requirement: str, jd_text: str) -> int:
        """Count how many times a requirement appears in JD"""
        keywords = re.findall(r'\b\w{4,}\b', requirement.lower())
        
        if not keywords:
            return 1
        
        jd_lower = jd_text.lower()
        total_count = 0
        
        for kw in keywords:
            count = len(re.findall(r'\b' + re.escape(kw) + r'\b', jd_lower))
            total_count += count
        
        return max(1, total_count // len(keywords))
    
    def analyze_all_gaps(self, gaps_dict: Dict[str, List[str]], 
                         jd_requirements: List[str], jd_text: str) -> Dict[str, List[GapItem]]:
        """Analyze all gaps and return structured data with severity"""
        severity_organized = {
            "Critical": [],
            "High": [],
            "Medium": [],
            "Low": []
        }
        
        all_gaps = []
        for category, gap_list in gaps_dict.items():
            for gap in gap_list:
                all_gaps.append((gap, category))
        
        for gap, original_category in all_gaps:
            position = self._find_requirement_position(gap, jd_requirements)
            frequency = self.count_requirement_frequency(gap, jd_text)
            severity, reason = self.get_severity_level(gap, position, frequency, jd_text)
            category = self.categorize_gap(gap)
            
            gap_item = GapItem(
                skill=gap,
                category=category,
                severity=severity,
                reason=reason,
                frequency=frequency,
                position=position
            )
            
            severity_organized[severity.value].append(gap_item)
        
        return severity_organized
    
    def _find_requirement_position(self, gap: str, requirements: List[str]) -> int:
        """Find approximate position of gap in requirements list"""
        gap_lower = gap.lower()
        
        for i, req in enumerate(requirements):
            if gap_lower in req.lower() or req.lower() in gap_lower:
                return i
        
        return len(requirements)
    
    def format_gap_report(self, severity_organized: Dict[str, List[GapItem]]) -> str:
        """Generate human-readable gap report"""
        report = []
        
        severity_order = ["Critical", "High", "Medium", "Low"]
        severity_colors = {
            "Critical": "🔴",
            "High": "🟠",
            "Medium": "🟡",
            "Low": "🟢"
        }
        
        for severity_level in severity_order:
            gaps = severity_organized.get(severity_level, [])
            
            if not gaps:
                continue
            
            report.append(f"\n{severity_colors[severity_level]} {severity_level} Gaps ({len(gaps)})")
            report.append("-" * 60)
            
            for gap in gaps:
                report.append(f"\n  • {gap.skill}")
                report.append(f"    Category: {gap.category}")
                report.append(f"    Frequency: Mentioned {gap.frequency}x in JD")
                report.append(f"    Rationale: {gap.reason}")
        
        return "\n".join(report)


# ============================================
# 1. PAGE CONFIGURATION
# ============================================
st.set_page_config(page_title="AI Resume Job Matcher", layout="wide")


# ============================================
# 2. LOAD ASSETS
# ============================================
@st.cache_resource
def load_assets():
    model = SentenceTransformer('iwamu/bert-data-analyst-matcher')
    scaler = joblib.load('semantic_scaler_2.pkl')
    severity_analyzer = GapSeverityAnalyzer()
    return model, scaler, severity_analyzer

model, scaler, severity_analyzer = load_assets()


# ============================================
# 3. HELPER FUNCTIONS
# ============================================

def extract_text(file):
    """Extract text from PDF"""
    pdf = PyPDF2.PdfReader(file)
    return " ".join([page.extract_text() or "" for page in pdf.pages])


def chunk_resume(resume_text):
    """Split resume into meaningful chunks"""
    chunks = re.split(r'[,.\n•●/-]', resume_text)
    return [c.strip() for c in chunks if len(c.strip()) > 3]


def extract_jd(jd_text):
    """Extract requirements from job description"""
    jd_clean = jd_text.lower()
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
    blacklisted_headers = set(trigger_words)
    
    for l in lines:
        clean_l = l.strip()
        if len(clean_l) < 5 or clean_l in blacklisted_headers or clean_l.endswith(':'):
            continue

        final_line = re.sub(r'^[\-\•\*\○\●\d\.\s]+', '', clean_l)
        if len(final_line) > 10:
            requirements.append(final_line)
            
    return requirements[:20]


def analyze_skills_with_severity(jd_text, resume_text):
    """
    Enhanced analyze_skills function with gap severity
    """
    jd_requirements = extract_jd(jd_text)
    resume_chunks = chunk_resume(resume_text)
    
    if not jd_requirements or not resume_chunks:
        return {
            "matches": [],
            "gaps_by_severity": {"Critical": [], "High": [], "Medium": [], "Low": []},
            "quality_ratio": 0.0,
            "gap_summary": "Unable to extract requirements or resume content"
        }
    
    # Semantic matching
    jd_embs = model.encode(jd_requirements, convert_to_tensor=True)
    res_embs = model.encode(resume_chunks, convert_to_tensor=True)
    cosine_scores = util.cos_sim(jd_embs, res_embs)
    max_scores, _ = torch.max(cosine_scores, dim=1)
    
    # Collect matches and gaps
    matched = []
    gaps_basic = {"Technical": [], "Stats": [], "Soft Skills": []}
    total_quality_score = 0
    
    for i, score in enumerate(max_scores):
        skill_name = jd_requirements[i]
        score_val = score.item() * 100
        
        if score_val > 60:  # Strong match
            matched.append((skill_name, round(score_val, 2)))
            total_quality_score += 1.0
        elif score_val > 45:  # Partial match
            matched.append((skill_name + " (Partial)", round(score_val, 2)))
            total_quality_score += 0.5
        else:
            # Categorize for gaps
            low_s = skill_name.lower()
            if any(k in low_s for k in ['python', 'sql', 'machine learning', 'code', 'production']):
                gaps_basic["Technical"].append(skill_name)
            elif any(k in low_s for k in ['statistics', 'causal', 'inference', 'probability', 'math']):
                gaps_basic["Stats"].append(skill_name)
            else:
                gaps_basic["Soft Skills"].append(skill_name)
    
    quality_ratio = total_quality_score / len(jd_requirements) if jd_requirements else 0
    
    # Analyze severity
    severity_organized = severity_analyzer.analyze_all_gaps(gaps_basic, jd_requirements, jd_text)
    
    return {
        "matches": matched,
        "gaps_by_severity": severity_organized,
        "gaps_basic": gaps_basic,
        "quality_ratio": quality_ratio,
        "gap_summary": severity_analyzer.format_gap_report(severity_organized)
    }


def calculate_seniority_bonus(text):
    """Calculate experience bonus"""
    experience_matches = re.findall(r'(\d+)\+?\s*(?:years|yrs)', text.lower())
    if experience_matches:
        years = max([int(x) for x in experience_matches])
        if years >= 10:
            return 15
        if years >= 3:
            return 10
    return 0


def generate_pdf_report(df):
    """Generate downloadable PDF report"""
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from io import BytesIO
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>Ranked Candidate Audit Report (With Gap Severity)</b>", styles['Heading1']))
    elements.append(Spacer(1, 12))

    for idx, row in df.reset_index(drop=True).iterrows():
        
        elements.append(Paragraph(
            f"<b>Rank {idx+1}: {row['Candidate']}</b>",
            styles['Heading2']
        ))
        
        # Score + Severity Summary
        elements.append(Paragraph(
            f"<b>Overall Match Score:</b> {row['Match Score']:.2f}%",
            styles['Normal']
        ))
        
        elements.append(Paragraph(
            f"<b>Gap Severity Summary:</b> 🔴 Critical: {row['Critical_Gaps']} | "
            f"🟠 High: {row['High_Gaps']} | 🟡 Medium: {row['Medium_Gaps']}",
            styles['Normal']
        ))
        elements.append(Spacer(1, 10))
        
        # Matches
        elements.append(Paragraph("<b>✅ Semantic Matches:</b>", styles['Normal']))
        if row["Full_Matches"]:
            for m in row["Full_Matches"]:
                elements.append(Paragraph(
                    f"• {m[0]} (Confidence: {m[1]}%)",
                    styles['Normal']
                ))
        else:
            elements.append(Paragraph("• No strong matches detected.", styles['Normal']))
        
        elements.append(Spacer(1, 12))
        
        # Severity-organized gaps
        elements.append(Paragraph("<b>⚠️ Gaps by Severity:</b>", styles['Normal']))
        elements.append(Spacer(1, 4))
        
        gaps_by_severity = row["Full_Gaps"]
        
        if gaps_by_severity.get("Critical"):
            elements.append(Paragraph("<b><font color=red>🔴 CRITICAL (Deal-breakers):</font></b>", styles['Normal']))
            for gap in gaps_by_severity["Critical"]:
                elements.append(Paragraph(
                    f"• {gap.skill}<br/><i>Why: {gap.reason}</i>",
                    styles['Normal']
                ))
            elements.append(Spacer(1, 6))
        
        if gaps_by_severity.get("High"):
            elements.append(Paragraph("<b><font color=orange>🟠 HIGH (Strongly Preferred):</font></b>", styles['Normal']))
            for gap in gaps_by_severity["High"]:
                elements.append(Paragraph(
                    f"• {gap.skill} (Category: {gap.category})",
                    styles['Normal']
                ))
            elements.append(Spacer(1, 6))
        
        if gaps_by_severity.get("Medium"):
            elements.append(Paragraph("<b>🟡 MEDIUM (Nice-to-Have):</b>", styles['Normal']))
            for gap in gaps_by_severity["Medium"]:
                elements.append(Paragraph(f"• {gap.skill}", styles['Normal']))
            elements.append(Spacer(1, 6))
        
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("---", styles['Normal']))
        elements.append(Spacer(1, 12))

    doc.build(elements)
    buffer.seek(0)
    return buffer


# ============================================
# 4. UI LAYOUT
# ============================================

st.title("🎯 Semantic Resume-JD Matcher with Gap Severity")
st.markdown("Rank resumes based on **meaning** and see exactly what's missing with priority levels.")

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
            jd_emb = model.encode(jd_input, convert_to_tensor=True)
            for file in uploaded_files:
                text = extract_text(file)
                
                # 1. Global Similarity
                res_emb = model.encode(text, convert_to_tensor=True)
                raw_sim = util.cos_sim(jd_emb, res_emb).item()
                
                # 2. Skill-Level Audit with Severity
                result = analyze_skills_with_severity(jd_input, text)
                matches = result["matches"]
                gaps_by_severity = result["gaps_by_severity"]
                quality_ratio = result["quality_ratio"]
                
                # 3. Seniority Detection
                bonus = calculate_seniority_bonus(text)

                # 4. Final Differentiated Formula
                final_score = (raw_sim * 25) + (quality_ratio * 65) + bonus
                final_score = float(np.clip(final_score, 0, 100))

                # Count gaps by severity
                critical_count = len(gaps_by_severity.get("Critical", []))
                high_count = len(gaps_by_severity.get("High", []))
                medium_count = len(gaps_by_severity.get("Medium", []))

                results.append({
                    "Candidate": file.name,
                    "Match Score": final_score,
                    "Matches": ", ".join([f"{m[0]} ({m[1]}%)" for m in matches]),
                    "Critical_Gaps": critical_count,
                    "High_Gaps": high_count,
                    "Medium_Gaps": medium_count,
                    "Full_Matches": matches,
                    "Full_Gaps": gaps_by_severity,
                    "Severity_Report": result["gap_summary"]
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

        # 1. High-Level Summary Table with Severity
        st.subheader("📊 Ranking Overview")
        display_df = df[['Candidate', 'Match Score']].copy()
        display_df['🔴 Critical'] = df['Critical_Gaps']
        display_df['🟠 High'] = df['High_Gaps']
        display_df['🟡 Medium'] = df['Medium_Gaps']
        display_df['Total Issues'] = df['Critical_Gaps'] + df['High_Gaps'] + df['Medium_Gaps']

        st.dataframe(
            display_df.style.highlight_max(
                axis=0, 
                subset=['Match Score'], 
                color='lightgreen'
            ).highlight_min(
                axis=0,
                subset=['Total Issues'],
                color='lightcyan'
            ),
            use_container_width=True
        )

        # 2. Deep Dive Expanders
        st.subheader("🔍 Individual Candidate Audit")
        for _, row in df.iterrows():
            with st.expander(f"Audit: {row['Candidate']} ({row['Match Score']:.2f}%)"):
                
                # Quick metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🔴 Critical", row['Critical_Gaps'])
                with col2:
                    st.metric("🟠 High", row['High_Gaps'])
                with col3:
                    st.metric("🟡 Medium", row['Medium_Gaps'])
                with col4:
                    st.metric("✅ Matches", len(row['Full_Matches']))
                
                st.divider()
                
                # Semantic Matches
                st.write("**✅ Semantic Matches**")
                if row['Full_Matches']:
                    for m in row['Full_Matches']:
                        st.write(f"- {m[0]} (Confidence: {m[1]}%)")
                else:
                    st.write("- No strong matches detected.")
                
                st.divider()
                
                # Gaps organized by severity
                st.write("**⚠️ Skill Gaps by Priority**")
                
                gaps_by_severity = row['Full_Gaps']
                
                # CRITICAL GAPS
                critical_gaps = gaps_by_severity.get("Critical", [])
                if critical_gaps:
                    st.markdown("### 🔴 **CRITICAL GAPS** (Deal-breakers)")
                    st.markdown("These are must-have skills or experience levels. Without these, the candidate may not be ready for this role.")
                    for gap in critical_gaps:
                        with st.container():
                            st.error(f"**{gap.skill}**")
                            st.caption(f"Category: {gap.category}")
                            st.caption(f"Reason: {gap.reason}")
                
                # HIGH GAPS
                high_gaps = gaps_by_severity.get("High", [])
                if high_gaps:
                    st.markdown("### 🟠 **HIGH-PRIORITY GAPS** (Strongly Preferred)")
                    st.markdown("Important to have, but candidate could potentially grow into these with training or mentoring.")
                    for gap in high_gaps:
                        with st.container():
                            st.warning(f"**{gap.skill}**")
                            st.caption(f"Category: {gap.category}")
                            st.caption(f"Reason: {gap.reason}")
                
                # MEDIUM GAPS
                medium_gaps = gaps_by_severity.get("Medium", [])
                if medium_gaps:
                    st.markdown("### 🟡 **MEDIUM-PRIORITY GAPS** (Nice-to-Have)")
                    st.markdown("Would be valuable, but not essential. Candidate can likely learn these on the job.")
                    for gap in medium_gaps:
                        with st.container():
                            st.info(f"**{gap.skill}**")
                            st.caption(f"Reason: {gap.reason}")
                
                # LOW GAPS
                low_gaps = gaps_by_severity.get("Low", [])
                if low_gaps and st.checkbox(f"Show Low-Priority gaps for {row['Candidate']}", key=f"low_{row['Candidate']}"):
                    st.markdown("### 🟢 **LOW-PRIORITY GAPS** (Learning Opportunity)")
                    for gap in low_gaps:
                        st.write(f"- {gap.skill}")
                
                st.divider()
                
                # Match Confidence Chart
                if row['Full_Matches']:
                    st.write("**Match Confidence Profile**")
                    
                    match_data = pd.DataFrame({
                        "Requirement": [m[0][:50] for m in row['Full_Matches']],
                        "Confidence": [m[1] for m in row['Full_Matches']]
                    }).set_index("Requirement")
                    
                    st.bar_chart(match_data)