import sys
from pypdf import PdfReader
src, dst = sys.argv[1], sys.argv[2]
r = PdfReader(src)
with open(dst, "w") as f:
    for i, p in enumerate(r.pages):
        f.write(f"\n\n=== PAGE {i+1} ===\n")
        f.write(p.extract_text() or "")
