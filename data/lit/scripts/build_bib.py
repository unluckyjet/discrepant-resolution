"""Assemble lit.bib from programmatically downloaded BibTeX, re-keyed to firstauthorYEARfirstword.
SWE-bench Verified has no DOI/arXiv entry; its @misc is generated from metadata parsed from the
fetched (Wayback) page text, not typed from memory."""
import re, sys, pathlib
L = pathlib.Path(sys.argv[1]); B = L / "web" / "bib"
files = {  # new key -> downloaded file (file stem = requested key; actual key may differ)
 "northcutt2021pervasive": "northcutt2021pervasive", "nahum2025are": "nahum2025are", "vasudevan2022when": "vasudevan2022when",
 "beyer2020are": "beyer2020are", "shankar2020evaluating": "shankar2020evaluating", "gema2025are": "gema2025are",
 "vendrow2025do": "vendrow2025do", "wang2024mmlupro": "wang2024mmlupro", "chong2022detecting": "chong2022detecting",
 "klie2023annotation": "klie2023annotation", "wang2019crossweigh": "wang2019crossweigh", "reiss2020identifying": "reiss2020identifying",
 "alt2020tacred": "alt2020tacred", "stoica2021retacred": "stoica2021retacred", "rucker2023cleanconll": "rucker2023cleanconll",
 "zhai2026hleverified": "zhai2026hleverified", "ye2026scalable": "ye2025scalable", "land2026auditing": "land2026auditing",
 "ansari2026how": "ansari2026how", "brunello2026fixing": "brunello2026fixing", "sylvestre2026gold": "sylvestre2026gold",
 "aleithan2024swebench": "aleithan2024swebench",
}
out = []
for key, stem in files.items():
    t = (B / f"{stem}.bib").read_text().strip()
    assert t.startswith("@"), (key, t[:80])
    t = re.sub(r"^@(\w+)\{\s*[^,]+,", lambda m: f"@{m.group(1)}{{{key},", t, count=1)
    t = re.sub(r"\n\s*abstract\s*=\s*\{.*?\}\n\}$", "\n}", t, flags=re.S)  # drop PMLR abstract
    t = t.replace("–", "--")
    out.append(t)
# SWE-bench Verified from page text
txt = (L / "web" / "swebv_archive.txt").read_text()
authors = re.search(r"Authors (.*?) NC, JA", txt).group(1)
authors = " and ".join(a.strip() for a in authors.split(" , ") if a.strip())
assert "Neil Chowdhury" in authors and re.search(r"August 13, 2024 Introducing SWE-bench Verified", txt)
out.append("@misc{chowdhury2024introducing,\n  title = {Introducing {SWE}-bench Verified},\n"
           f"  author = {{{authors}}},\n  howpublished = {{OpenAI blog, \\url{{https://openai.com/index/introducing-swe-bench-verified/}}}},\n"
           "  year = {2024},\n  month = aug,\n  note = {Published August 13, 2024. Metadata parsed from the Wayback Machine copy of the page.}\n}")
(L / "lit.bib").write_text("\n\n".join(out) + "\n")
print(len(out), "entries")
