#!/usr/bin/env python3
"""
sbert_mlp.py
Three‑class classifier using frozen Sentence‑BERT embeddings + small MLP.
Produces mean loss/accuracy curve and aggregate confusion matrix.
Requires: sentence-transformers, tensorflow, matplotlib, seaborn, pandas, sklearn.
"""

import os, re, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

# ---------- helpers ----------
def clean_text(t:str)->str:
    t = re.sub(r'http[s]?://\S+', '<URL>', t)
    t = re.sub(r'<[^>]+>', ' ', t)
    t = re.sub(r'\r?\n', ' ', t)
    return re.sub(r'\s+',' ',t).strip().lower()

def load_phishing():
    df = pd.read_csv("phishing.csv")
    df=df[df['label'].astype(str)=='1']
    df['text']=(df['subject'].fillna('')+' '+df['body'].fillna('')).apply(clean_text)
    df['y']=1
    return df[['text','y']]

def load_benign():
    df=pd.read_csv("benign.csv")
    def ext(msg):
        parts=re.split(r'\r?\n\r?\n',msg,maxsplit=1)
        header,body=parts if len(parts)==2 else ('',parts[0])
        m=re.search(r'(?im)^Subject:\s*(.*)',header)
        subj=m.group(1).strip() if m else ''
        return clean_text(subj+' '+body)
    df['text']=df['message'].astype(str).apply(ext); df['y']=0
    return df[['text','y']]

def load_ai():
    df=pd.read_csv("ai_phishing_emails.csv")
    def cb(e):
        parts=e.split('\n',1)
        subj=parts[0][len("Subject:"):].strip() if parts[0].lower().startswith("subject:") else parts[0]
        body=parts[1] if len(parts)>1 else ''
        return clean_text(subj+' '+body)
    df['text']=df['email'].apply(cb); df['y']=2
    return df[['text','y']]

# ---------- data ----------
df=pd.concat([load_phishing(),load_benign(),load_ai()],ignore_index=True)
texts,labels=df['text'].tolist(),df['y'].values

# ---------- SBERT embedder ----------
sbert=SentenceTransformer('all-MiniLM-L6-v2')
embeds=sbert.encode(texts,batch_size=32,show_progress_bar=True)
dims=embeds.shape[1]

# ---------- CV ----------
out_dir="sbert_output"; os.makedirs(out_dir,exist_ok=True)
skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
histories=[]; all_cm=np.zeros((3,3),int); fold_metrics=[]

for fold,(tr,vl) in enumerate(skf.split(embeds,labels),1):
    Xtr,Xvl=embeds[tr],embeds[vl]
    ytr,yvl=labels[tr],labels[vl]

    model=tf.keras.Sequential([
        tf.keras.layers.Input(shape=(dims,)),
        tf.keras.layers.Dense(256,activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3,activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    hist=model.fit(Xtr,ytr,validation_data=(Xvl,yvl),
                   epochs=10,batch_size=16,verbose=0)
    histories.append(hist.history)

    y_pred=np.argmax(model.predict(Xvl,verbose=0),axis=1)
    acc=accuracy_score(yvl,y_pred)
    prec,rec,f1,_=precision_recall_fscore_support(yvl,y_pred,average='weighted')
    fold_metrics.append({'fold':fold,'accuracy':acc,'precision':prec,'recall':rec,'f1':f1})

    cm=confusion_matrix(yvl,y_pred); all_cm+=cm
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',
                xticklabels=['Benign','Human‑Phish','AI‑Phish'],
                yticklabels=['Benign','Human‑Phish','AI‑Phish'],cbar=False)
    plt.title(f'SBERT + MLP Confusion (Fold {fold})'); plt.tight_layout()
    plt.savefig(f"{out_dir}/conf_fold{fold}.png",dpi=300); plt.close()

# ---------- aggregate loss curve ----------
mean_hist={k:np.mean([h[k] for h in histories],axis=0) for k in histories[0]}
epochs=np.arange(1,11)
plt.figure(figsize=(6,4))
plt.plot(epochs,mean_hist['loss'],'o-',label='Mean Train Loss')
plt.plot(epochs,mean_hist['val_loss'],'s-',label='Mean Val Loss')
plt.plot(epochs,mean_hist['val_accuracy'],'d--',label='Mean Val Acc')
plt.title('SBERT + MLP – Mean Training Dynamics (5 Folds)')
plt.xlabel('Epoch'); plt.ylabel('Value'); plt.legend(); plt.tight_layout()
plt.savefig(f"{out_dir}/loss_acc_mean.png",dpi=300); plt.close()

# ---------- aggregate confusion ----------
sns.heatmap(all_cm,annot=True,fmt='d',cmap='Blues',
            xticklabels=['Benign','Human‑Phish','AI‑Phish'],
            yticklabels=['Benign','Human‑Phish','AI‑Phish'],cbar=False)
plt.title('SBERT + MLP – Aggregate Confusion Matrix (5 Folds)', pad=14)
plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
plt.savefig(f"{out_dir}/conf_aggregate.png",dpi=300); plt.close()

pd.DataFrame(fold_metrics).to_csv(f"{out_dir}/aggregate_metrics.csv",index=False)
print("Finished. Outputs in sbert_output/")
