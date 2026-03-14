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
# from docx import Document

# ============================================
# SKILL NORMALIZER MAP
# ============================================
SKILL_NORMALIZER = {
    "data integrations": "Data Integration (e.g. Fivetran, Airbyte, Stitch)",
    "master data pipelines": "Pipeline Orchestration (e.g. Airflow, Prefect)",
    "data pipelines": "Pipeline Orchestration (e.g. Airflow, Prefect)",
    "schema mappings": "Schema Design & Mapping",
    "schema mapping": "Schema Design & Mapping",
    "data hierarchies": "Data Modeling & Hierarchy Design",
    "data quality rules": "Data Quality Frameworks (e.g. Great Expectations, dbt tests)",
    "data quality": "Data Quality Frameworks (e.g. Great Expectations, dbt tests)",
    "sync failures": "ETL Troubleshooting & Pipeline Monitoring",
    "data conflicts": "ETL Troubleshooting & Conflict Resolution",
    "source systems": "ERP/CRM Integration (e.g. Stripe, HubSpot, QuickBooks, Gusto)",
    "stripe": "Payment System Integration (Stripe)",
    "quickbooks": "Accounting System Integration (QuickBooks)",
    "hubspot": "CRM Integration (HubSpot)",
    "gusto": "HR/Payroll Integration (Gusto)",
    "normalize data": "Data Normalization & Standardization",
    "unify": "Data Unification & Master Data Management",
    "client-specific data models": "Client Data Modeling & Documentation",
    "document configurations": "Technical Documentation & Configuration Management",
    "sync rules": "Data Sync Rules & Scheduling Logic",
    "templates": "Reusable Pipeline Templating",
    "standardize processes": "Process Standardization & Engineering Best Practices",
    "python": "Python Programming",
    "sql": "SQL & Database Querying",
    "machine learning": "Machine Learning",
    "api": "REST API Design & Integration",
    "docker": "Containerization (Docker)",
    "kubernetes": "Container Orchestration (Kubernetes)",
    "aws": "Cloud Engineering (AWS)",
    "gcp": "Cloud Engineering (GCP)",
    "azure": "Cloud Engineering (Azure)",
    "statistics": "Statistical Analysis & Probability",
    "communication": "Communication & Stakeholder Management",
    "leadership": "Team Leadership & People Management",
    "mentoring": "Mentoring & Knowledge Transfer",
}

SEVERITY_TEMPLATES = {
    "Critical": (
        "Candidate has no demonstrated experience with **{skill}**. "
        "This is a core requirement — recommend direct probing in interview or consider disqualifying."
    ),
    "High": (
        "Candidate shows partial or no exposure to **{skill}**. "
        "Strongly preferred by the JD — could be developed with 3–6 months structured onboarding."
    ),
    "Medium": (
        "**{skill}** is preferred but not blocking. "
        "Candidate can likely acquire this on the job with minimal ramp-up time."
    ),
    "Low": (
        "**{skill}** is a minor nice-to-have. "
        "Not a hiring decision factor — worth noting but should not affect ranking."
    ),
}


def normalize_gap_to_skill(gap_text: str) -> Tuple[str, str]:
    """
    Returns (normalized_skill_label, original_jd_phrase).
    Maps vague JD phrases to concrete, learnable skill names.
    Falls back to original text if no match found.
    """
    gap_lower = gap_text.lower()
    for phrase, skill_label in SKILL_NORMALIZER.items():
        if phrase in gap_lower:
            return skill_label, gap_text
    return gap_text, gap_text


def build_severity_reason(skill_label: str, severity_level: str) -> str:
    """Generate recruiter-facing reason text from template."""
    template = SEVERITY_TEMPLATES.get(
        severity_level,
        "**{skill}** is missing from the candidate profile."
    )
    return template.format(skill=skill_label)


# ============================================
# ENHANCED JD EXTRACTOR
# ============================================
class EnhancedJDExtractor:
    def __init__(self):
        self.responsibility_keywords = {
            'responsibilities', 'responsibility', 'key responsibilities',
            'primary responsibilities', 'job responsibilities', 'main responsibilities',
            'core responsibilities', 'essential responsibilities', 'job duties',
            'duties', 'what you will do', 'what you\'ll do', 'role responsibilities',
            'key accountabilities', 'accountabilities', 'you will', 'main duties'
        }
        self.requirement_keywords = {
            'requirements', 'requirement', 'required', 'required qualifications',
            'basic requirements', 'minimum requirements', 'job requirements',
            'technical requirements', 'core requirements', 'essential requirements',
            'must haves', 'must have', 'hard requirements', 'mandatory requirements',
            'what we need'
        }
        self.qualification_keywords = {
            'qualifications', 'qualification', 'qualified', 'education and experience',
            'education', 'educational background', 'educational requirements',
            'degree required', 'required education', 'academic qualifications',
            'background', 'experience and education', 'required qualifications',
            'preferred qualifications'
        }
        self.skill_keywords = {
            'skills', 'skill', 'technical skills', 'required skills', 'necessary skills',
            'key skills', 'core skills', 'competencies', 'technical competencies',
            'skills and abilities', 'abilities', 'expertise', 'proficiency',
            'what we\'re looking for'
        }
        self.experience_keywords = {
            'experience', 'required experience', 'years of experience',
            'professional experience', 'relevant experience', 'prior experience',
            'work experience', 'industry experience', 'background experience',
            'preferred experience'
        }
        self.preferred_keywords = {
            'preferred', 'preferred qualifications', 'preferred skills',
            'nice to have', 'nice-to-have', 'bonus', 'a plus', 'extra credit',
            'desirable', 'would be great', 'ideally', 'optional'
        }

        self.all_keywords = (
            self.responsibility_keywords | self.requirement_keywords |
            self.qualification_keywords | self.skill_keywords |
            self.experience_keywords | self.preferred_keywords
        )

        # -------------------------------------------------------
        # FIX: Section header blacklist
        # Any extracted item whose full cleaned text matches one
        # of these phrases is discarded as a header artifact.
        # No NLP or retraining required.
        # -------------------------------------------------------
        self.header_blacklist = self.all_keywords | {
            "what you'll do", "what we're looking for", "what we need",
            "who you are", "about you", "about the role", "about us",
            "the role", "your role", "job summary", "overview",
            "position summary", "role overview", "position overview",
            "key responsibilities", "nice to have", "must have",
            "you will", "we are looking for", "we're looking for",
            "join us", "why join us", "benefits", "perks",
            "equal opportunity", "our team", "the team",
        }

    def is_section_header(self, item: str) -> bool:
        """
        Returns True if the item is a section header and should be discarded.
        Three simple rules — no NLP required:
          1. Exact match to a known header phrase
          2. Too short to be a real requirement (<=3 words)
          3. Ends with a colon (formatting artifact)
        """
        item_clean = item.strip().lower().rstrip(':').rstrip('.')
        if item_clean in self.header_blacklist:
            return True
        if len(item_clean.split()) <= 3:
            return True
        if item.strip().endswith(':'):
            return True
        return False

    def find_all_sections(self, jd_text: str) -> List[Tuple[str, int, str]]:
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
        if next_section_pos is None:
            return jd_text[start_pos:]
        return jd_text[start_pos:next_section_pos]

    def parse_bullets_and_lines(self, text: str) -> List[str]:
        lines = text.split('\n')
        parsed_items = []
        for line in lines:
            clean_line = line.strip()
            if not clean_line or len(clean_line) < 8 or clean_line.endswith(':'):
                continue
            cleaned = re.sub(
                r'^[\s]*'
                r'(?:[•\-\*\○\●]|\d+[.\)]\s*|[a-zA-Z]\s*[.\)]\s*|\[\s*[xX\-]\s*\])'
                r'\s*',
                '',
                clean_line
            )
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            if len(cleaned) > 10:
                parsed_items.append(cleaned)
        return parsed_items

    def extract_from_sections(self, jd_text: str,
                              include_preferred: bool = True,
                              max_items: int = 25) -> List[str]:
        sections = self.find_all_sections(jd_text)
        if not sections:
            return [
                item for item in self.parse_bullets_and_lines(jd_text)
                if not self.is_section_header(item)
            ][:max_items]

        all_requirements = []
        for i, (section_type, start_pos, keyword) in enumerate(sections):
            if section_type == 'preferred' and not include_preferred:
                continue
            end_pos = sections[i + 1][1] if i < len(sections) - 1 else len(jd_text)
            section_content = self.extract_section_content(jd_text, start_pos, end_pos)
            items = self.parse_bullets_and_lines(section_content)
            for item in items:
                if not self.is_section_header(item):
                    all_requirements.append(item)

        seen = set()
        unique_requirements = []
        for item in all_requirements:
            item_lower = item.lower()
            if item_lower not in seen and len(item) > 10:
                seen.add(item_lower)
                unique_requirements.append(item)
        return unique_requirements[:max_items]

    def extract_jd_optimized(self, jd_text: str) -> List[str]:
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
            return [
                item for item in self.parse_bullets_and_lines(remaining_text)
                if not self.is_section_header(item)
            ][:25]
        return [
            item for item in self.parse_bullets_and_lines(jd_text)
            if not self.is_section_header(item)
        ][:25]


# ============================================
# GAP SEVERITY CLASSES
# ============================================

class GapSeverity(Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass
class GapItem:
    skill: str        # normalized skill label (e.g. "Pipeline Orchestration")
    raw_skill: str    # original JD phrase kept for recruiter context
    category: str
    severity: GapSeverity
    reason: str       # recruiter-facing explanation
    frequency: int
    position: int


class GapSeverityAnalyzer:
    def __init__(self):
        self.critical_keywords = {
            'must have', 'required', 'essential', 'mandatory',
            '7+', '8+', '10+', 'lead', 'architect', 'design', 'responsible for'
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
            'testing', 'optimization', 'scalability',
            'pipeline', 'etl', 'integration', 'schema', 'sync',
            'stripe', 'hubspot', 'quickbooks', 'gusto'
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
        reasons = []
        severity_score = 0
        req_lower = requirement.lower()

        if position < 3:
            severity_score += 30
            reasons.append("Listed in top 3 requirements")
        elif position < 7:
            severity_score += 20
            reasons.append("Listed in top 7 requirements")
        elif position < 15:
            severity_score += 10
            reasons.append("Listed in requirements")

        if any(kw in req_lower for kw in self.critical_keywords):
            severity_score += 35
            matching_kw = [kw for kw in self.critical_keywords if kw in req_lower][0]
            reasons.append(f"Contains critical keyword: '{matching_kw}'")
        elif any(kw in req_lower for kw in self.high_keywords):
            severity_score += 20
            matching_kw = [kw for kw in self.high_keywords if kw in req_lower][0]
            reasons.append(f"Contains high-priority keyword: '{matching_kw}'")

        if frequency >= 3:
            severity_score += 25
            reasons.append(f"Mentioned {frequency} times in JD")
        elif frequency == 2:
            severity_score += 15
            reasons.append("Mentioned multiple times")

        exp_match = re.search(r'(\d+)\+?\s*(?:years|yrs)', req_lower)
        if exp_match:
            years = int(exp_match.group(1))
            severity_score += 25
            if years >= 10:
                severity_score += 10
                reasons.append(f"Requires {years}+ years (senior-level)")
            else:
                reasons.append(f"Requires {years}+ years")

        jd_lower = jd_text.lower()
        req_section = jd_lower.find('required')
        resp_section = jd_lower.find('responsibility')
        pref_section = jd_lower.find('preferred')

        if req_section != -1:
            severity_score += 15
            reasons.append("In 'Requirements' section")
        elif resp_section != -1:
            severity_score += 10
            reasons.append("In 'Responsibilities' section")
        elif pref_section != -1:
            severity_score -= 10
            reasons.append("In 'Preferred' section (not required)")

        impact_keywords = ['lead', 'architect', 'design', 'manage', 'mentor',
                           'implement', 'optimize', 'scale', 'improve', 'reduce']
        if any(kw in req_lower for kw in impact_keywords):
            severity_score += 15
            reasons.append("High-impact responsibility")

        if severity_score >= 85:
            return GapSeverity.CRITICAL, "; ".join(reasons)
        elif severity_score >= 60:
            return GapSeverity.HIGH, "; ".join(reasons)
        elif severity_score >= 35:
            return GapSeverity.MEDIUM, "; ".join(reasons)
        else:
            return GapSeverity.LOW, "; ".join(reasons)

    def categorize_gap(self, gap: str) -> str:
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
        keywords = re.findall(r'\b\w{4,}\b', requirement.lower())
        if not keywords:
            return 1
        jd_lower = jd_text.lower()
        total_count = sum(
            len(re.findall(r'\b' + re.escape(kw) + r'\b', jd_lower))
            for kw in keywords
        )
        return max(1, total_count // len(keywords))

    def analyze_all_gaps(self, gaps_dict: Dict[str, List[str]],
                         jd_requirements: List[str],
                         jd_text: str) -> Dict[str, List[GapItem]]:
        severity_organized = {"Critical": [], "High": [], "Medium": [], "Low": []}
        all_gaps = [
            (gap, cat)
            for cat, gap_list in gaps_dict.items()
            for gap in gap_list
        ]

        for gap, original_category in all_gaps:
            position = self._find_requirement_position(gap, jd_requirements)
            frequency = self.count_requirement_frequency(gap, jd_text)
            severity, _ = self.get_severity_level(gap, position, frequency, jd_text)
            category = self.categorize_gap(gap)

            # Normalize gap phrase to a concrete skill label
            skill_label, raw_skill = normalize_gap_to_skill(gap)

            # Build recruiter-facing reason from template
            reason = build_severity_reason(skill_label, severity.value)

            gap_item = GapItem(
                skill=skill_label,
                raw_skill=raw_skill,
                category=category,
                severity=severity,
                reason=reason,
                frequency=frequency,
                position=position
            )
            severity_organized[severity.value].append(gap_item)

        return severity_organized

    def _find_requirement_position(self, gap: str, requirements: List[str]) -> int:
        gap_lower = gap.lower()
        for i, req in enumerate(requirements):
            if gap_lower in req.lower() or req.lower() in gap_lower:
                return i
        return len(requirements)

    def format_gap_report(self, severity_organized):
        report = []
        severity_labels = {
            "Critical": "[CRITICAL]",
            "High":     "[HIGH]",
            "Medium":   "[MEDIUM]",
            "Low":      "[LOW]"
        }
        for level in ["Critical", "High", "Medium", "Low"]:
            gaps = severity_organized.get(level, [])
            if not gaps:
                continue
            report.append(f"\n{severity_labels[level]} {level} Gaps ({len(gaps)})")
            report.append("-" * 60)
            for gap in gaps:
                skill     = getattr(gap, 'skill', 'Unknown skill')
                raw_skill = getattr(gap, 'raw_skill', skill)
                category  = getattr(gap, 'category', 'Unknown')
                reason    = getattr(gap, 'reason', 'No additional context available.')
                report.append(f"\n  Skill: {skill}")
                report.append(f"  In context of: {raw_skill}")
                report.append(f"  Category: {category}")
                report.append(f"  Recruiter Note: {reason}")
        return "\n".join(report)


# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(page_title="AI Resume Job Matcher", layout="wide")


# ============================================
# LOAD ASSETS
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
# HELPER FUNCTIONS
# ============================================

# def extract_text(file):
#     """Extract text from PDF."""
#     pdf = PyPDF2.PdfReader(file)
#     return " ".join([page.extract_text() or "" for page in pdf.pages])

def extract_text(file):
    if file.name.endswith('.pdf'):
        pdf = PyPDF2.PdfReader(file)
        return " ".join([page.extract_text() or "" for page in pdf.pages])
    elif file.name.endswith('.docx'):
        from docx import Document
        doc = Document(file)
        return " ".join([p.text for p in doc.paragraphs])
    return ""

def chunk_resume(resume_text):
    """Split resume into meaningful chunks."""
    chunks = re.split(r'[,.\n•●/-]', resume_text)
    return [c.strip() for c in chunks if len(c.strip()) > 3]


def analyze_skills_with_severity(jd_text, resume_text):
    """Enhanced analyze_skills function with gap severity."""
    jd_requirements = jd_extractor.extract_jd_optimized(jd_text)
    resume_chunks = chunk_resume(resume_text)

    if not jd_requirements or not resume_chunks:
        return {
            "matches": [],
            "gaps_by_severity": {"Critical": [], "High": [], "Medium": [], "Low": []},
            "quality_ratio": 0.0,
            "gap_summary": "Unable to extract requirements or resume content"
        }

    jd_embs = model.encode(jd_requirements, convert_to_tensor=True)
    res_embs = model.encode(resume_chunks, convert_to_tensor=True)
    cosine_scores = util.cos_sim(jd_embs, res_embs)
    max_scores, _ = torch.max(cosine_scores, dim=1)

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
            if any(k in low_s for k in ['python', 'sql', 'machine learning', 'code', 'production',
                                         'pipeline', 'etl', 'integration', 'schema', 'sync']):
                gaps_basic["Technical"].append(skill_name)
            elif any(k in low_s for k in ['statistics', 'causal', 'inference', 'probability', 'math']):
                gaps_basic["Stats"].append(skill_name)
            else:
                gaps_basic["Soft Skills"].append(skill_name)

    quality_ratio = total_quality_score / len(jd_requirements) if jd_requirements else 0
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
    """Calculate experience bonus."""
    experience_matches = re.findall(r'(\d+)\+?\s*(?:years|yrs)', text.lower())
    if experience_matches:
        years = max([int(x) for x in experience_matches])
        if years >= 10:
            return 15
        if years >= 3:
            return 10
    return 0


def sanitize_for_pdf(text: str) -> str:
    """
    Escape characters that break ReportLab's XML parser.
    Also strips markdown bold markers (**) since ReportLab
    uses its own tag syntax.
    """
    text = str(text)
    text = text.replace('**', '')        # strip markdown bold
    text = text.replace('&', '&amp;')   # must come first
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    return text


def generate_pdf_report(df):
    """Generate downloadable PDF report."""
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from io import BytesIO

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph(
        "<b>Ranked Candidate Audit Report (With Gap Severity)</b>",
        styles['Heading1']
    ))
    elements.append(Spacer(1, 12))

    for idx, row in df.reset_index(drop=True).iterrows():
        elements.append(Paragraph(
            f"<b>Rank {idx+1}: {sanitize_for_pdf(row['Candidate'])}</b>",
            styles['Heading2']
        ))
        elements.append(Paragraph(
            f"<b>Overall Match Score:</b> {row['Match Score']:.2f}%",
            styles['Normal']
        ))
        elements.append(Paragraph(
            f"<b>Gap Severity Summary:</b> "
            f"[CRITICAL] {row['Critical_Gaps']} | "
            f"[HIGH] {row['High_Gaps']} | "
            f"[MEDIUM] {row['Medium_Gaps']}",
            styles['Normal']
        ))
        elements.append(Spacer(1, 10))

        # Matches
        elements.append(Paragraph("<b>Semantic Matches:</b>", styles['Normal']))
        if row["Full_Matches"]:
            for m in row["Full_Matches"]:
                elements.append(Paragraph(
                    f"- {sanitize_for_pdf(m[0])} (Confidence: {m[1]}%)",
                    styles['Normal']
                ))
        else:
            elements.append(Paragraph("- No strong matches detected.", styles['Normal']))

        elements.append(Spacer(1, 12))
        elements.append(Paragraph("<b>Gaps by Severity:</b>", styles['Normal']))
        elements.append(Spacer(1, 4))

        gaps_by_severity = row["Full_Gaps"]
        severity_config = [
            ("Critical", "<font color=red><b>CRITICAL GAPS (Deal-breakers):</b></font>"),
            ("High",     "<font color=orange><b>HIGH-PRIORITY GAPS (Strongly Preferred):</b></font>"),
            ("Medium",   "<b>MEDIUM-PRIORITY GAPS (Nice-to-Have):</b>"),
        ]

        for level, label in severity_config:
            level_gaps = gaps_by_severity.get(level, [])
            if level_gaps:
                elements.append(Paragraph(label, styles['Normal']))
                elements.append(Spacer(1, 4))
                for gap in level_gaps:
                    # Sanitize each field individually before composing
                    skill_clean     = sanitize_for_pdf(getattr(gap, 'skill', ''))
                    raw_skill_clean = sanitize_for_pdf(getattr(gap, 'raw_skill', skill_clean))
                    reason_clean    = sanitize_for_pdf(getattr(gap, 'reason', ''))
                    category_clean  = sanitize_for_pdf(getattr(gap, 'category', ''))

                    elements.append(Paragraph(
                        f"<b>{skill_clean}</b>",
                        styles['Normal']
                    ))
                    elements.append(Paragraph(
                        f"In context of: {raw_skill_clean}",
                        styles['Normal']
                    ))
                    elements.append(Paragraph(
                        f"Category: {category_clean}",
                        styles['Normal']
                    ))
                    elements.append(Paragraph(
                        f"Recruiter Note: {reason_clean}",
                        styles['Normal']
                    ))
                    elements.append(Spacer(1, 6))
                elements.append(Spacer(1, 4))

        elements.append(Spacer(1, 12))
        elements.append(Paragraph("_" * 60, styles['Normal']))
        elements.append(Spacer(1, 12))

    doc.build(elements)
    buffer.seek(0)
    return buffer


# ============================================
# UI LAYOUT
# ============================================

st.title("🎯 Semantic Resume-JD Matcher with Gap Severity")
st.markdown("Rank resumes based on **meaning** and see exactly what's missing with priority levels.")

with st.sidebar:
    st.header("Step 1: Job Description")
    jd_input = st.text_area("Paste the Job Description here:", height=300)

    if st.checkbox("Debug: Show extracted requirements"):
        if jd_input:
            extracted = jd_extractor.extract_jd_optimized(jd_input)
            st.write(f"**Found {len(extracted)} requirements:**")
            for i, req in enumerate(extracted, 1):
                st.caption(f"{i}. {req}")

st.header("Step 2: Upload Resumes")
uploaded_files = st.file_uploader(
    "Upload candidate PDFs", type="pdf", accept_multiple_files=True
)

if st.button("🚀 Analyze & Rank"):
    if not jd_input or not uploaded_files:
        st.error("Please provide both a JD and at least one resume.")
    else:
        results = []
        with st.spinner("Performing Deep Semantic Audit..."):
            jd_emb = model.encode(jd_input, convert_to_tensor=True)

            for file in uploaded_files:
                text = extract_text(file)

                # Global Similarity
                res_emb = model.encode(text, convert_to_tensor=True)
                raw_sim = util.cos_sim(jd_emb, res_emb).item()

                # Skill-Level Audit with Severity
                result = analyze_skills_with_severity(jd_input, text)
                matches = result["matches"]
                gaps_by_severity = result["gaps_by_severity"]
                quality_ratio = result["quality_ratio"]

                # Seniority Bonus
                bonus = calculate_seniority_bonus(text)

                # Final Score
                final_score = (raw_sim * 25) + (quality_ratio * 65) + bonus
                final_score = float(np.clip(final_score, 0, 100))

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

        df = pd.DataFrame(results).sort_values(by="Match Score", ascending=False)

        st.success(f"✅ Analysis Complete! Top candidate: {df.iloc[0]['Candidate']}")

        pdf_buffer = generate_pdf_report(df)
        st.download_button(
            label="📄 Download Full Ranking Report (PDF)",
            data=pdf_buffer,
            file_name="Ranked_Candidates_Audit.pdf",
            mime="application/pdf"
        )

        st.divider()

        # Summary Table
        st.subheader("📊 Ranking Overview")
        display_df = df[['Candidate', 'Match Score']].copy()
        display_df['🔴 Critical'] = df['Critical_Gaps']
        display_df['🟠 High'] = df['High_Gaps']
        display_df['🟡 Medium'] = df['Medium_Gaps']
        display_df['Total Issues'] = (
            df['Critical_Gaps'] + df['High_Gaps'] + df['Medium_Gaps']
        )

        st.dataframe(
            display_df.style
                .highlight_max(axis=0, subset=['Match Score'], color='lightgreen')
                .highlight_min(axis=0, subset=['Total Issues'], color='lightcyan'),
            use_container_width=True
        )

        # Individual Audits
        st.subheader("🔍 Individual Candidate Audit")
        for _, row in df.iterrows():
            with st.expander(f"Audit: {row['Candidate']} ({row['Match Score']:.2f}%)"):

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

                # Gaps by Severity
                st.write("**⚠️ Skill Gaps by Priority**")
                gaps_by_severity = row['Full_Gaps']

                def render_gap(gap, render_fn):
                    """Safely render a GapItem using getattr fallbacks."""
                    skill     = getattr(gap, 'skill', 'Unknown skill')
                    raw_skill = getattr(gap, 'raw_skill', skill)
                    category  = getattr(gap, 'category', 'Unknown')
                    reason    = getattr(gap, 'reason', 'No additional context available.')
                    render_fn(skill)
                    st.caption(f"In context of: _{raw_skill}_")
                    st.caption(f"Category: {category}")
                    st.caption(f"Recruiter Note: {reason}")

                critical_gaps = gaps_by_severity.get("Critical", [])
                if critical_gaps:
                    st.markdown("### 🔴 CRITICAL GAPS (Deal-breakers)")
                    st.markdown(
                        "These are must-have skills or experience levels. "
                        "Without these, the candidate may not be ready for this role."
                    )
                    for gap in critical_gaps:
                        render_gap(gap, lambda s: st.error(f"**{s}**"))

                high_gaps = gaps_by_severity.get("High", [])
                if high_gaps:
                    st.markdown("### 🟠 HIGH-PRIORITY GAPS (Strongly Preferred)")
                    st.markdown(
                        "Important to have, but candidate could potentially grow into these "
                        "with 3–6 months of structured onboarding."
                    )
                    for gap in high_gaps:
                        render_gap(gap, lambda s: st.warning(f"**{s}**"))

                medium_gaps = gaps_by_severity.get("Medium", [])
                if medium_gaps:
                    st.markdown("### 🟡 MEDIUM-PRIORITY GAPS (Nice-to-Have)")
                    st.markdown(
                        "Would be valuable, but not essential. "
                        "Candidate can likely learn these on the job."
                    )
                    for gap in medium_gaps:
                        render_gap(gap, lambda s: st.info(f"**{s}**"))

                low_gaps = gaps_by_severity.get("Low", [])
                if low_gaps and st.checkbox(
                    f"Show Low-Priority gaps for {row['Candidate']}",
                    key=f"low_{row['Candidate']}"
                ):
                    st.markdown("### 🟢 LOW-PRIORITY GAPS (Learning Opportunity)")
                    for gap in low_gaps:
                        skill     = getattr(gap, 'skill', 'Unknown skill')
                        raw_skill = getattr(gap, 'raw_skill', skill)
                        st.write(f"- **{skill}** _(context: {raw_skill})_")

                st.divider()

                # Match Confidence Chart
                if row['Full_Matches']:
                    st.write("**Match Confidence Profile**")
                    match_data = pd.DataFrame({
                        "Requirement": [m[0][:50] for m in row['Full_Matches']],
                        "Confidence": [m[1] for m in row['Full_Matches']]
                    }).set_index("Requirement")
                    st.bar_chart(match_data)