# Phishing‑Triage  (CSC‑7700 Final Project)

Detects **Benign**, **Human‑Phish**, and **AI‑Phish** emails with  
1. **Baseline** – TF‑IDF (5 k uni/bi‑grams) → Logistic Regression  
2. **Proposed** – frozen Sentence‑BERT (`all‑MiniLM‑L6‑v2`) → small MLP  

| Model          | Accuracy | Weighted F1 |
| -------------- | :------: | :---------: |
| TF‑IDF + LR    | 0.91     | 0.91        |
| **SBERT + MLP**| **0.95** | **0.94**    |

---

## Repo Layout

tfidf_logreg.py baseline training / eval
sbert_mlp.py SBERT + MLP training / eval

