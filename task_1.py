import csv
import ast
import re
from dataclasses import dataclass
from typing import List, Dict, DefaultDict
from collections import defaultdict

@dataclass
class Tag:
    id: int
    name: str
    keywords: List[str]

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
            tags.append(Tag(
                id=int(row["id"]),
                name=row["name"],
                keywords=[kw.lower() for kw in kw_list]
            ))
    return tags

def build_keyword_index(tags: List[Tag]) -> DefaultDict[str, List[Tag]]:
    idx: DefaultDict[str, List[Tag]] = defaultdict(list)
    for t in tags:
        for kw in t.keywords:
            idx[kw].append(t)
    return idx

def load_sentences(path: str) -> List[str]:
    with open(path, encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]
def tag_sentences(
    sentences_path: str,
    tags_csv_path: str,
    output_tsv_path: str
) -> None:
    tags = load_tags(tags_csv_path)
    keyword_index = build_keyword_index(tags)

    single_word_patterns: Dict[str, re.Pattern] = {}
    for kw in keyword_index:
        if ' ' not in kw:
            single_word_patterns[kw] = re.compile(rf'\b{re.escape(kw)}\b', re.IGNORECASE)

    sentences = load_sentences(sentences_path)
    with open(output_tsv_path, 'w', encoding='utf-8') as out:
        for sent in sentences:
            sent_lower = sent.lower()
            matched_tags = set()

            for kw, tag_list in keyword_index.items():
                if ' ' in kw and kw in sent_lower:
                    matched_tags.update(t.name for t in tag_list)

            for kw, pattern in single_word_patterns.items():
                if pattern.search(sent):
                    matched_tags.update(t.name for t in keyword_index[kw])

            tags_str = ", ".join(sorted(matched_tags))
            out.write(f"{sent}\t{tags_str}\n")


if __name__ == "__main__":
    tag_sentences(
        sentences_path="data/sentences.txt",
        tags_csv_path="data/tags.csv",
        output_tsv_path="task_1_output.tsv"
    )
    print("✅ Done writing task_1_output.tsv")
