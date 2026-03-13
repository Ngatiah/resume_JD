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
# ENHANCED JD EXTRACTOR (INTEGRATED)
# ============================================

class EnhancedJDExtractor:
    """
    Intelligent job description extractor that handles:
    - Multiple keyword variations
    - Section detection and prioritization
    - Nested sections
    - Various formatting styles
    - FILTERS OUT trigger keywords from actual requirements
    """
    
    def __init__(self):
        """Initialize with comprehensive trigger keyword groups"""
        
        # Group 1: RESPONSIBILITIES/DUTIES
        self.responsibility_keywords = {
            'responsibilities',
            'responsibility',
            'key responsibilities',
            'primary responsibilities',
            'job responsibilities',
            'main responsibilities',
            'core responsibilities',
            'essential responsibilities',
            'job duties',
            'duties',
            'what you will do',
            'what you\'ll do',
            'role responsibilities',
            'key accountabilities',
            'accountabilities',
            'you will',
            'main duties'
        }
        
        # Group 2: REQUIREMENTS/QUALIFICATIONS
        self.requirement_keywords = {
            'requirements',
            'requirement',
            'required',
            'required qualifications',
            'basic requirements',
            'minimum requirements',
            'job requirements',
            'technical requirements',
            'core requirements',
            'essential requirements',
            'must haves',
            'must have',
            'hard requirements',
            'mandatory requirements',
            'what we need'
        }
        
        # Group 3: QUALIFICATIONS/EDUCATION
        self.qualification_keywords = {
            'qualifications',
            'qualification',
            'qualified',
            'education and experience',
            'education',
            'educational background',
            'educational requirements',
            'degree required',
            'required education',
            'academic qualifications',
            'background',
            'experience and education',
            'required qualifications',
            'preferred qualifications'
        }
        
        # Group 4: SKILLS
        self.skill_keywords = {
            'skills',
            'skill',
            'technical skills',
            'required skills',
            'necessary skills',
            'key skills',
            'core skills',
            'competencies',
            'technical competencies',
            'skills and abilities',
            'abilities',
            'expertise',
            'proficiency',
            'what we\'re looking for'
        }
        
        # Group 5: EXPERIENCE
        self.experience_keywords = {
            'experience',
            'required experience',
            'years of experience',
            'professional experience',
            'relevant experience',
            'prior experience',
            'work experience',
            'industry experience',
            'background experience',
            'preferred experience'
        }
        
        # Group 6: PREFERRED/NICE-TO-HAVE
        self.preferred_keywords = {
            'preferred',
            'preferred qualifications',
            'preferred skills',
            'nice to have',
            'nice-to-have',
            'bonus',
            'a plus',
            'extra credit',
            'desirable',
            'would be great',
            'ideally',
            'optional'
        }
        
        # Combine all for quick lookup
        self.all_keywords = (
            self.responsibility_keywords |
            self.requirement_keywords |
            self.qualification_keywords |
            self.skill_keywords |
            self.experience_keywords |
            self.preferred_keywords
        )
        
        # BLACKLIST: Keywords that should never appear in semantic matches
        # These are section headers/trigger words, not actual requirements
        self.blacklist_trigger_words = {
            'responsibilities',
            'responsibility',
            'requirements',
            'requirement',
            'required',
            'qualifications',
            'qualification',
            'qualified',
            'skills',
            'skill',
            'experience',
            'preferred',
            'professional',
            'education',
            'background',
            'duties',
            'accountabilities',
            'competencies',
            'abilities',
            'expertise',
            'what we need',
            'what we\'re looking for',
            'core',
            'essential',
            'technical',
            'soft',
            'hard',
            'basic',
            'advanced'
        }
    
    def find_all_sections(self, jd_text: str) -> List[Tuple[str, int, str]]:
        """Find all section headers and their content"""
        jd_clean = jd_text.lower()
        sections = []
        
        for keyword in self.all_keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            for match in re.finditer(pattern, jd_clean):
                start_pos = match.start()
                
                if keyword in self.responsibility_keywords:
                    section_type = 'responsibility'
                elif keyword in self.requirement_keywords:
                    section_type = 'requirement'
                elif keyword in self.qualification_keywords:
                    section_type = 'qualification'
                elif keyword in self.skill_keywords:
                    section_type = 'skill'
                elif keyword in self.experience_keywords:
                    section_type = 'experience'
                elif keyword in self.preferred_keywords:
                    section_type = 'preferred'
                else:
                    section_type = 'other'
                
                sections.append((section_type, start_pos, keyword))
        
        sections.sort(key=lambda x: x[1])
        return sections
    
    def extract_section_content(self, jd_text: str, start_pos: int, 
                               next_section_pos: int = None) -> str:
        """Extract content between section header and next section"""
        if next_section_pos is None:
            content = jd_text[start_pos:]
        else:
            content = jd_text[start_pos:next_section_pos]
        return content
    
    def parse_bullets_and_lines(self, text: str) -> List[str]:
        """Parse bullet points and numbered lists from text"""
        lines = text.split('\n')
        parsed_items = []
        
        for line in lines:
            clean_line = line.strip()
            
            if not clean_line or len(clean_line) < 8 or clean_line.endswith(':'):
                continue
            
            # Remove common bullet formats
            # cleaned = re.sub(
            #     r'^[\s]*'
            #     r'(?:'
            #     r'[•\-\*\○\●]'
            #     r'|'
            #     r'\d+[.\)]\s*'
            #     r'|'
            #     r'[a-zA-Z]\s*[.\)]\s*'
            #     r'|'
            #     r'\[\s*[xX\-]\s*\]'
            #     r')'
            #     r'\s*',
            #     '',
            #     clean_line
            # )
            
            # cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            pattern = (
                r'^[\s]*'
                r'(?:'
                r'[•\-\*\○\●]'
                r'|'
                r'\d+[.\)]\s*'
                r'|'
                r'[a-zA-Z]\s*[.\)]\s*'
                r'|'
                r'\[\s*[xX\-]\s*\]'
                r')'
                r'\s*'
            )
            cleaned = re.sub(pattern, '', clean_line)
            
            if len(cleaned) > 10:
                parsed_items.append(cleaned)
        
        return parsed_items
    
    def is_trigger_keyword(self, item: str) -> bool:
        """
        Check if item is just a trigger keyword (section header)
        Returns True if should be FILTERED OUT
        """
        item_lower = item.lower().strip()
        
        # If item is exactly one of our trigger words, filter it
        if item_lower in self.blacklist_trigger_words:
            return True
        
        # If item is very short (2-3 words) and consists mostly of trigger words, filter it
        words = item_lower.split()
        if len(words) <= 3:
            trigger_word_count = sum(1 for word in words if word in self.blacklist_trigger_words)
            if trigger_word_count >= len(words) - 1:  # Most/all words are triggers
                return True
        
        return False
    
    def extract_from_sections(self, jd_text: str, 
                            include_preferred: bool = True,
                            max_items: int = 25) -> List[str]:
        """Extract requirements from all identified sections"""
        sections = self.find_all_sections(jd_text)
        
        if not sections:
            requirements = self.parse_bullets_and_lines(jd_text)
            # Filter out trigger keywords
            return [r for r in requirements if not self.is_trigger_keyword(r)][:max_items]
        
        all_requirements = []
        
        for i, (section_type, start_pos, keyword) in enumerate(sections):
            if section_type == 'preferred' and not include_preferred:
                continue
            
            if i < len(sections) - 1:
                end_pos = sections[i + 1][1]
            else:
                end_pos = len(jd_text)
            
            section_content = self.extract_section_content(jd_text, start_pos, end_pos)
            items = self.parse_bullets_and_lines(section_content)
            
            for item in items:
                all_requirements.append(item)
        
        # Remove duplicates and trigger keywords
        seen = set()
        unique_requirements = []
        for item in all_requirements:
            item_lower = item.lower()
            # Filter out trigger keywords AND duplicates
            if item_lower not in seen and len(item) > 10 and not self.is_trigger_keyword(item):
                seen.add(item_lower)
                unique_requirements.append(item)
        
        return unique_requirements[:max_items]
    
    def extract_jd_optimized(self, jd_text: str) -> List[str]:
        """Optimized extraction with fallbacks"""
        requirements = self.extract_from_sections(jd_text, include_preferred=True)
        
        if requirements:
            return requirements
        
        jd_clean = jd_text.lower()
        earliest_pos = len(jd_text)
        
        for keyword in self.all_keywords:
            idx = jd_clean.find(keyword)
            if idx != -1 and idx < earliest_pos:
                earliest_pos = idx
        
        if earliest_pos < len(jd_text):
            remaining_text = jd_text[earliest_pos:]
            requirements = self.parse_bullets_and_lines(remaining_text)
            # Filter out trigger keywords
            return [r for r in requirements if not self.is_trigger_keyword(r)][:25]
        
        return self.parse_bullets_and_lines(jd_text)[:25]


# ============================================
# GAP SEVERITY CLASSES
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
    jd_extractor = EnhancedJDExtractor()
    return model, scaler, severity_analyzer, jd_extractor

model, scaler, severity_analyzer, jd_extractor = load_assets()


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


def analyze_skills_with_severity(jd_text, resume_text):
    """Enhanced analyze_skills function with gap severity"""
    
    # Use enhanced extractor (now filters out trigger keywords!)
    jd_requirements = jd_extractor.extract_jd_optimized(jd_text)
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
        
        if score_val > 60:
            matched.append((skill_name, round(score_val, 2)))
            total_quality_score += 1.0
        elif score_val > 45:
            matched.append((skill_name + " (Partial)", round(score_val, 2)))
            total_quality_score += 0.5
        else:
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
        "gap_summary": severity_analyzer.format_gap_report(severity_organized),
        "extracted_requirements": jd_requirements
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
    
    # Debug option
    if st.checkbox("Debug: Show extracted requirements"):
        if jd_input:
            extracted = jd_extractor.extract_jd_optimized(jd_input)
            st.write(f"**Found {len(extracted)} requirements (trigger keywords filtered):**")
            for i, req in enumerate(extracted, 1):
                st.caption(f"{i}. {req}")

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