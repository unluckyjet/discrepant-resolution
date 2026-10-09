"""Write a compact display bibliography (paper/refs.bib) from the merged verified entries (paper/refs_full.bib).

Drops fields that lengthen the printed references (url, doi, issn, month, publisher of journals, collection,
series, abstract, keywords, language) and turns arXiv e-print fields into a note. Authors, titles, venues,
volumes, pages and years are untouched.
"""
import re
from pathlib import Path

DROP = {"url", "doi", "issn", "isbn", "month", "collection", "series", "abstract", "keywords", "language",
        "copyright", "urldate", "file", "archiveprefix", "primaryclass", "eprinttype"}


def split_fields(body):
    fields, depth, cur, inq = [], 0, "", False
    for ch in body:
        if ch == "{": depth += 1
        elif ch == "}": depth -= 1
        elif ch == '"' and depth == 0: inq = not inq          # quoted BibTeX values may contain commas
        if ch == "," and depth == 0 and not inq:
            fields.append(cur); cur = ""
        else:
            cur += ch
    if cur.strip(): fields.append(cur)
    return [f.strip() for f in fields if f.strip()]


out = []
for chunk in re.split(r"\n(?=@)", Path("paper/refs_full.bib").read_text()):
    m = re.search(r"^@(\w+)\s*\{\s*([^,\s]+)\s*,(.*)\}\s*$", chunk, re.S | re.M)
    if not m:
        continue
    typ, key, body = m.group(1).lower(), m.group(2), m.group(3)
    kept, eprint = [], None
    for f in split_fields(body):
        if "=" not in f:
            continue
        name, val = f.split("=", 1)
        name = name.strip().lower()
        if name == "eprint":
            eprint = val.strip().strip("{}\"")
            continue
        if name in DROP or (name == "publisher" and typ == "article"):
            continue
        kept.append(f"  {name} = {val.strip()}")
    if eprint and not any(k.strip().startswith(("journal", "booktitle", "howpublished")) for k in kept):
        kept.append(f"  howpublished = {{arXiv:{eprint}}}")
    out.append(f"@{typ}{{{key},\n" + ",\n".join(kept) + "\n}")
Path("paper/refs.bib").write_text("\n\n".join(out) + "\n")
print(len(out), "entries")
