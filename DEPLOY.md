# Deploy Checklist (Free Hosting)

## 1. Preflight (Local)

- [ ] Activate venv: `source .venv/bin/activate`
- [ ] Install deps: `pip install -r requirements.txt`
- [ ] Verify app starts: `python app.py`
- [ ] Confirm at least one PDF exists in:
  - [ ] `data/pdfs/`
  - [ ] or `data/professor_materials/`

## 2. GitHub

- [ ] Commit latest changes
- [ ] Push `main` to `https://github.com/soham-padia/rag-algorithms-assistant`
- [ ] Confirm repo contains app code and PDFs you are allowed to publish

## 3. Hugging Face Spaces (Recommended Free)

- [ ] Create new Space
- [ ] SDK: **Gradio**
- [ ] Hardware: **CPU Basic (free)**
- [ ] Import this GitHub repo
- [ ] Deploy

## 4. Runtime Settings

- [ ] Leave `HF_TOKEN` unset for default model (`Qwen/Qwen2.5-1.5B-Instruct`)
- [ ] Set `HF_TOKEN` only if you switch to a gated/private model
- [ ] Do not set Apple-only MPS variables on Spaces

## 5. First Boot Expectations

- [ ] Startup health message appears in UI
- [ ] If vectorstore is missing, app auto-indexes PDFs on first query
- [ ] First response can be slow on free CPU (model load + indexing)

## 6. Post-Deploy Validation

- [ ] Open Space URL
- [ ] Ask a test query (e.g., "Master theorem cases")
- [ ] Verify:
  - [ ] Chat response streams
  - [ ] Math renders in Markdown/LaTeX
  - [ ] Source citations appear
  - [ ] SVG summary card appears

## 7. Safety/Compliance

- [ ] Only host PDFs/content you have rights to share
- [ ] Keep private course files out of public repos unless permission is explicit
