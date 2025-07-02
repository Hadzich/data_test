import csv
import ast
import re
from dataclasses import dataclass
from typing import List, Dict, DefaultDict, Iterable
from collections import defaultdict

@dataclass
class Tag:
    id: int
    name: str
    keywords: List[str]

def normalize_ws(s: str) -> str:
    """
    Collapse any run of non-word chars into a single space, lowercase.
    E.g. "  Help…Login " -> "help login"
    """
    cleaned = re.sub(r"[^\w]+", " ", s)
    return re.sub(r"\s+", " ", cleaned).strip().lower()

def load_tags(csv_path: str) -> List[Tag]:
    tags: List[Tag] = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_kw = row["keywords"].strip()
            try:
                kw_list = ast.literal_eval(raw_kw)
            except (ValueError, SyntaxError):
                inner = raw_kw.strip('"').strip("'")
                kw_list = [w.strip().strip("'\"") for w in inner.split(",") if w.strip()]
            norm_kws = [normalize_ws(kw) for kw in kw_list]
            tags.append(Tag(
                id=int(row["id"]),
                name=row["name"],
                keywords=norm_kws
            ))
    return tags

def build_keyword_index(tags: List[Tag]) -> DefaultDict[str, List[Tag]]:
    idx: DefaultDict[str, List[Tag]] = defaultdict(list)
    for t in tags:
        for kw in t.keywords:
            idx[kw].append(t)
    return idx

def make_inflectional_pattern(word: str) -> re.Pattern:
    """
    Create a regex that matches the word plus optional 's', 'ed', or 'ing' suffixes.
    E.g. 'unlock' -> r'\bunlock(?:s|ed|ing)?\b'
    """
    w = re.escape(word)
    return re.compile(rf'\b{w}(?:s|ed|ing)?\b', re.IGNORECASE)

def ngrams(tokens: List[str], n: int) -> Iterable[List[str]]:
    for i in range(len(tokens) - n + 1):
        yield tokens[i : i + n]

def match_multi(tokens: List[str], kw_tokens: List[str]) -> bool:
    L = len(kw_tokens)
    if L == 2:
        a, b = kw_tokens
        for i in range(len(tokens) - 1):
            if tokens[i] == a and tokens[i+1] == b:
                return True
            if tokens[i] == b and tokens[i+1] == a:
                return True
        for i in range(len(tokens) - 2):
            if tokens[i] == a and tokens[i+2] == b:
                return True
            if tokens[i] == b and tokens[i+2] == a:
                return True
        return False
    else:
        seq = " ".join(kw_tokens)
        joined = " ".join(tokens)
        return seq in joined

def tag_sentences_v2(
    sentences_path: str,
    tags_csv_path: str,
    output_tsv_path: str
) -> None:
    tags = load_tags(tags_csv_path)
    keyword_index = build_keyword_index(tags)

    single_word_patterns: Dict[str, re.Pattern] = {
        kw: make_inflectional_pattern(kw)
        for kw in keyword_index
        if " " not in kw
    }

    with open(sentences_path, encoding='utf-8') as fin, \
         open(output_tsv_path, 'w', encoding='utf-8') as fout:

        for raw in fin:
            sent = raw.rstrip("\n")
            norm = normalize_ws(sent)
            tokens = norm.split()
            matched = set()

            for kw, patt in single_word_patterns.items():
                if patt.search(norm):
                    matched.update(t.name for t in keyword_index[kw])

            for kw, tag_list in keyword_index.items():
                if " " in kw:
                    kw_toks = kw.split()
                    if match_multi(tokens, kw_toks):
                        matched.update(t.name for t in tag_list)

            tags_str = ", ".join(sorted(matched))
            fout.write(f"{sent}\t{tags_str}\n")

if __name__ == "__main__":
    tag_sentences_v2(
        sentences_path="data/sentences.txt",
        tags_csv_path="data/tags.csv",
        output_tsv_path="task_1_output_v2.tsv"
    )
    print("✅ Done writing task_1_output_v2.tsv")
