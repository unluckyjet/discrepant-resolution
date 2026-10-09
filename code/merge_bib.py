"""Merge verified bib files (methods first, then literature), dropping duplicate keys."""
import re
from pathlib import Path
out, seen = [], set()
for f in ["paper/refs_methods.bib", "data/lit/lit.bib"]:
    txt = Path(f).read_text()
    for chunk in re.split(r"\n(?=@)", "\n" + txt):
        m = re.search(r"^@\w+\s*\{\s*([^,\s]+)", chunk, re.M)
        if not m:
            continue
        k = m.group(1)
        if k in seen:
            continue
        seen.add(k); out.append(chunk.strip())
Path("paper/refs_full.bib").write_text("\n\n".join(out) + "\n")
print(len(out), "entries")
