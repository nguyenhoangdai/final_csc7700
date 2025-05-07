#!/usr/bin/env python3
"""
tfidf_logreg.py
Baseline three‑class e‑mail classifier:
  • Clean & tokenize text
  • TF‑IDF (5 000 uni/bi‑grams)
  • L2‑regularised Logistic Regression
Outputs per‑fold confusion matrices + metrics CSV.
"""

import re, os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
from sklearn.model_selection import StratifiedKFold

# ---------- helpers ----------
def clean_text(text: str) -> str:
    text = re.sub(r'http[s]?://\S+', '<URL>', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\r?\n', ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

def load_phishing():
    df = pd.read_csv("phishing.csv")
    df = df[df['label'].astype(str) == '1']
    df['text'] = (df['subject'].fillna('') + ' ' + df['body'].fillna('')).apply(clean_text)
    df['y'] = 1
    return df[['text','y']]

def load_benign():
    df = pd.read_csv("benign.csv")
    def extract(msg):
        parts = re.split(r'\r?\n\r?\n', msg, maxsplit=1)
        header, body = parts if len(parts)==2 else ('', parts[0])
        m = re.search(r'(?im)^Subject:\s*(.*)', header)
        subj = m.group(1).strip() if m else ''
        return clean_text(subj+' '+body)
    df['text'] = df['message'].astype(str).apply(extract)
    df['y'] = 0
    return df[['text','y']]

def load_ai():
    df = pd.read_csv("ai_phishing_emails.csv")
    def combine(email):
        parts = email.split('\n',1)
        subj = parts[0][len("Subject:"):].strip() if parts[0].lower().startswith("subject:") else parts[0]
        body = parts[1] if len(parts)>1 else ''
        return clean_text(subj+' '+body)
    df['text'] = df['email'].apply(combine)
    df['y'] = 2
    return df[['text','y']]

# ---------- data ----------
df = pd.concat([load_phishing(), load_benign(), load_ai()], ignore_index=True)
texts, labels = df['text'].tolist(), df['y'].values

# ---------- CV ----------
out_dir = "tfidf_output"; os.makedirs(out_dir, exist_ok=True)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics = []

for fold,(tr,vl) in enumerate(skf.split(texts,labels),1):
    print(f"Fold {fold}")
    X_train, X_val = [texts[i] for i in tr], [texts[i] for i in vl]
    y_train, y_val = labels[tr], labels[vl]

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True)
    Xtr = vec.fit_transform(X_train); Xvl = vec.transform(X_val)

    clf = LogisticRegression(max_iter=50, solver="saga")
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xvl)

    acc = accuracy_score(y_val, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')
    metrics.append({'fold':fold,'accuracy':acc,'precision':prec,'recall':rec,'f1':f1})

    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign','Human‑Phish','AI‑Phish'],
                yticklabels=['Benign','Human‑Phish','AI‑Phish'], cbar=False)
    plt.title(f'TF‑IDF + LR Confusion (Fold {fold})')
    plt.tight_layout(); plt.savefig(f"{out_dir}/conf_fold{fold}.png", dpi=300); plt.close()

    pd.DataFrame(classification_report(y_val, y_pred,
              target_names=['Benign','Human‑Phish','AI‑Phish'], output_dict=True)).T\
              .to_csv(f"{out_dir}/report_fold{fold}.csv")

pd.DataFrame(metrics).to_csv(f"{out_dir}/aggregate_metrics.csv", index=False)
print("Done. Outputs in tfidf_output/")
