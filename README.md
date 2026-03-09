# 🎯 Explainable AI Resume-Job Matching System

[](https://resumejd-75ykods9j2snaxgh5jss88.streamlit.app/)
**A Three-Tiered NLP Architecture using Remote Transformer Models and Automated Gap Analysis.**

## 🚀 Key Features

* **Remote Model Integration:** Loads a customized SBERT model (`iwamu/bert-data-analyst-matcher`) directly from **Hugging Face**, ensuring version control and model persistence.
* **Semantic Matching:** Uses Transformer-based embeddings to find contextual matches beyond simple keywords.
* **Three-Tiered Architecture:** A structured pipeline from raw text extraction to calibrated contextual ranking.
* **Explainable AI (XAI):** Identifies **Matched Skills** and **Critical Gaps** using a semantic similarity threshold of `0.65`.
* **Professional Reporting:** Generates downloadable **PDF Audit Reports** using `ReportLab` for recruiter documentation.

---

## 🏗️ System Architecture

1. **Tier 1: Multi-Format Extraction**
* Uses `PyPDF2` and `python-docx` for universal document parsing.


2. **Tier 2: Remote Semantic Vectorization**
* The app fetches the specialized **SBERT** model from Hugging Face.
* Sentences are converted into 384-dimensional dense vectors to capture professional intent.


3. **Tier 3: Calibration & XAI**
* Raw cosine similarity is passed through a `joblib`-loaded **MinMaxScaler** for 0-100% scoring.
* A logical audit layer compares JD requirements against resume chunks.



---

## 🛠️ Tech Stack

* **Model Hosting:** [Hugging Face Hub](https://www.google.com/search?q=https://huggingface.co/iwamu/bert-data-analyst-matcher)
* **NLP Framework:** Sentence-Transformers (SBERT), PyTorch, spaCy
* **Deployment:** Streamlit Community Cloud
* **Libraries:** Pandas, NumPy, Scikit-learn, ReportLab, PyPDF2, python-docx

---

## 📥 Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Ngatiah/Final-year-prj.git
cd Final-year-prj

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Model Configuration:**
The system is configured to load the model automatically:
```python
# In app.py
model = SentenceTransformer('iwamu/bert-data-analyst-matcher')

```


4. **Run the application:**
```bash
streamlit run app.py

```



---

## 📊 Performance Evaluation

* **Precision@K:** Evaluated on top-tier candidate accuracy.
* **Semantic Thresholding:** Optimized at `>0.65` for reliable skill extraction.
* **Cloud Efficiency:** Model caching implemented via `@st.cache_resource` to minimize Hugging Face API calls.

---
