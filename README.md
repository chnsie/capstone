
# Guide Copilot — Two-Source RAG (A/B Output)

Minimal, patient, safety-first guidance for product delivery with a simple two-source RAG:
- **Lifecycle Context** from the DOCX (explain & caution)
- **Recommended Guides** from `guides.json` (resources + links)

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# ensure data/guides.json and data/Delivery of Digital Products.docx exist
streamlit run app.py
```
