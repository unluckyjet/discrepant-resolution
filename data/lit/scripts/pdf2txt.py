import sys
from pypdf import PdfReader
for p in sys.argv[2:]:
    r=PdfReader(p); t="\n".join((pg.extract_text() or "") for pg in r.pages)
    open(sys.argv[1]+"/"+p.split("/")[-1].replace(".pdf",".txt"),"w").write(t)
    print(p,len(t))
