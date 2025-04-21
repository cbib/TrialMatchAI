#!/usr/bin/env python3
import os
import json
import re
import warnings
import argparse

import dateutil.parser
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


class SentenceEmbedder:
    def __init__(self, model_name: str = 'BAAI/bge-m3'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Embedding on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def mean_pooling(self, model_output, attention_mask):
        tokens = model_output[0]
        mask = attention_mask.unsqueeze(-1).expand(tokens.size()).float()
        return torch.sum(tokens * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def get_embeddings(self, text: str):
        if not text:
            return None
        enc = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
        emb = self.mean_pooling(out, enc['attention_mask'])
        emb = F.normalize(emb, p=2, dim=1)
        return emb.squeeze().cpu().tolist()

    def preprocess_text(self, t: str) -> str:
        return re.sub(r'\s+', ' ', t).strip() if t else t

    def to_iso(self, s: str):
        try:
            return dateutil.parser.parse(s).date().isoformat() if s else None
        except:
            return None

    def age_to_years(self, s: str):
        if not s:
            return None
        m = re.search(r'([\d\.]+)', s)
        if not m:
            return None
        v = float(m.group(1))
        u = s.lower()
        if 'year' in u:
            y = v
        elif 'month' in u:
            y = v / 12
        elif 'week' in u:
            y = v / 52
        elif 'day' in u:
            y = v / 365
        else:
            return None
        return round(y, 2)


def embed_and_prepare(doc: dict, embedder: SentenceEmbedder):
    out = {'nct_id': doc['nct_id']}
    # text fields → clean + vector
    for field, vec_name in [
        ('brief_title', 'brief_title_vector'),
        ('brief_summary', 'brief_summary_vector'),
        ('condition', 'condition_vector'),
        ('eligibility_criteria', 'eligibility_criteria_vector'),
    ]:
        if field in doc:
            txt = doc[field]
            if isinstance(txt, list):
                txt = " ".join([str(t) for t in txt if isinstance(t, str) and t.strip()])
            txt = embedder.preprocess_text(txt)
            emb = embedder.get_embeddings(txt) or [0.0] * len(embedder.get_embeddings("test"))
            out[field] = txt
            out[vec_name] = emb

    # passthroughs
    for simple in ['overall_status', 'phase', 'study_type', 'gender']:
        if simple in doc:
            out[simple] = doc[simple]

    # dates
    for d in ['start_date', 'completion_date']:
        if d in doc:
            iso = embedder.to_iso(doc[d])
            if iso:
                out[d] = iso

    # ages
    for a in ['minimum_age', 'maximum_age']:
        if a in doc:
            yrs = embedder.age_to_years(doc[a])
            if yrs is not None:
                out[a] = yrs

    # nested passthroughs
    for nest in ['intervention', 'location', 'reference']:
        if nest in doc:
            out[nest] = doc[nest]

    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Prepare & embed clinical trial JSONs")
    p.add_argument('--ids-file',        required=True, help='One NCT ID per line')
    p.add_argument('--source-folder',   required=True, help='Raw JSONs dir')
    p.add_argument('--processed-folder', default='processed_docs',
                   help='Where to write embedded JSONs')
    p.add_argument('--model-name',      default='BAAI/bge-m3',
                   help='Sentence embedding model')
    args = p.parse_args()

    os.makedirs(args.processed_folder, exist_ok=True)
    embedder = SentenceEmbedder(model_name=args.model_name)

    with open(args.ids_file) as f:
        ids = [l.strip() for l in f if l.strip()]

    processed = 0
    skipped = 0

    for nct in ids:
        out_path = os.path.join(args.processed_folder, f"{nct}.json")
        if os.path.exists(out_path):
            print(f"🟡 Skipping {nct}: already processed")
            skipped += 1
            continue

        in_path = os.path.join(args.source_folder, f"{nct}.json")
        if not os.path.exists(in_path):
            print(f"⚠️  Missing raw JSON for {nct}")
            continue

        doc = json.load(open(in_path))
        doc['nct_id'] = nct
        proc = embed_and_prepare(doc, embedder)

        with open(out_path, 'w') as wf:
            json.dump(proc, wf, indent=2)
        processed += 1
        print(f"✅ Processed {nct}")

    print(f"\nSummary: {processed} processed, {skipped} skipped, {len(ids)-processed-skipped} missing.")
