import csv
import ast
import re
from dataclasses import dataclass
from typing import List, Dict, DefaultDict
from collections import defaultdict

from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz

@dataclass
class Tag:
    id: int
    name: str
    keywords: List[str]

def normalize_ws(s: str) -> str:
    cleaned = re.sub(r"[^\w]+", " ", s)
    return re.sub(r"\s+", " ", cleaned).strip().lower()

def load_tags(csv_path: str) -> List[Tag]:
    tags = []
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row["keywords"].strip()
            try:
                kws = ast.literal_eval(raw)
            except:
                inner = raw.strip('"').strip("'")
                kws = [w.strip().strip("'\"") for w in inner.split(",") if w.strip()]
            norm_kws = [normalize_ws(kw) for kw in kws]
            tags.append(Tag(int(row["id"]), row["name"], norm_kws))
    return tags

def build_keyword_index(tags: List[Tag]) -> DefaultDict[str, List[Tag]]:
    idx: DefaultDict[str, List[Tag]] = defaultdict(list)
    for t in tags:
        for kw in t.keywords:
            idx[kw].append(t)
    return idx

def match_multi(tokens: List[str], kw_tokens: List[str]) -> bool:
    if len(kw_tokens) == 2:
        a, b = kw_tokens
        for i in range(len(tokens)-1):
            if tokens[i:i+2] in ([a,b],[b,a]):
                return True
        for i in range(len(tokens)-2):
            if (tokens[i]==a and tokens[i+2]==b) or (tokens[i]==b and tokens[i+2]==a):
                return True
        return False
    return " ".join(kw_tokens) in " ".join(tokens)

def tag_v2_for_sentence(sent: str, keyword_index: Dict[str, List[Tag]]) -> List[str]:
    norm   = normalize_ws(sent)
    tokens = norm.split()
    matched = set()
    for kw, tlist in keyword_index.items():
        if " " not in kw and re.search(rf'\b{re.escape(kw)}\b', norm):
            matched.update(t.name for t in tlist)
    for kw, tlist in keyword_index.items():
        if " " in kw and match_multi(tokens, kw.split()):
            matched.update(t.name for t in tlist)
    return sorted(matched)

def tag_sentences_ml(
    sentences_path: str,
    tags_csv_path: str,
    output_path: str,
    model_name: str = "all-MiniLM-L6-v2",
    semantic_threshold: float = 0.5,
    fuzzy_single_threshold: int = 90,
    fuzzy_multi_threshold: int = 80,
    min_fuzzy_token_len: int = 5,
    per_tag_thresholds: Dict[str, float] = None
):
    tags = load_tags(tags_csv_path)
    keyword_index = build_keyword_index(tags)

    with open(sentences_path, encoding='utf-8') as f:
        sentences = [l.rstrip("\n") for l in f if l.strip()]

    v2_labels = [tag_v2_for_sentence(s, keyword_index) for s in sentences]

    model = SentenceTransformer(model_name)
    tag_names = [t.name for t in tags]
    tag_prompts = [f"{t.name}: {' '.join(t.keywords)}" for t in tags]
    tag_embs = model.encode(tag_prompts, convert_to_tensor=True, show_progress_bar=False)

    per_tag_thresholds = per_tag_thresholds or {}
    stop_words = {"a", "an", "the"}

    with open(output_path, 'w', encoding='utf-8') as out:
        for sent, v2 in zip(sentences, v2_labels):
            if v2:
                final = v2
            else:
                norm   = normalize_ws(sent)
                tokens = norm.split()

                sent_emb = model.encode(norm, convert_to_tensor=True)
                cos_scores = util.cos_sim(sent_emb, tag_embs)[0]
                sem_hits = [
                    tag_names[i]
                    for i, sc in enumerate(cos_scores)
                    if sc.item() >= per_tag_thresholds.get(tag_names[i], semantic_threshold)
                ]

                bow_hits = set()
                for kw, tlist in keyword_index.items():
                    if " " in kw:
                        parts = [w for w in kw.split() if w not in stop_words]
                        if all(p in tokens for p in parts):
                            bow_hits.update(t.name for t in tlist)

                multi_fuzzy_hits = set()
                for kw, tlist in keyword_index.items():
                    if " " in kw:
                        parts = [w for w in kw.split() if w not in stop_words]
                        ok = True
                        for p in parts:
                            if not any(fuzz.ratio(tok, p) >= fuzzy_multi_threshold for tok in tokens):
                                ok = False
                                break
                        if ok:
                            multi_fuzzy_hits.update(t.name for t in tlist)

                fuzzy_hits = set()
                for kw, tlist in keyword_index.items():
                    if " " not in kw:
                        for tok in tokens:
                            if len(tok) >= min_fuzzy_token_len and fuzz.ratio(tok, kw) >= fuzzy_single_threshold:
                                fuzzy_hits.update(t.name for t in tlist)
                                break

                final = sorted(set(sem_hits) | bow_hits | multi_fuzzy_hits | fuzzy_hits)

            out.write(f"{sent}\t{', '.join(final)}\n")

    print(f"✅ Done writing {output_path}")

if __name__ == "__main__":
    per_tag_thresholds = {
        "Marital Status": 0.97,   
        "Online Accounts": 0.4,
        "Personal Loans":   0.5,
        "Home Loans":       0.55,
        "Vehicle Loans":   0.85,
    }

    tag_sentences_ml(
        sentences_path="data/sentences.txt",
        tags_csv_path="data/tags.csv",
        output_path="task_2_output.tsv",
        semantic_threshold=0.7,
        fuzzy_single_threshold=90,
        fuzzy_multi_threshold=85,
        min_fuzzy_token_len=5,
        per_tag_thresholds=per_tag_thresholds
    )
