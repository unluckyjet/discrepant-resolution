#!/bin/sh
# Build paper/main.pdf (pdfTeX via TinyTeX).
set -e
cd "$(dirname "$0")"
TEX=~/Library/TinyTeX/bin/universal-darwin
$TEX/pdflatex -interaction=nonstopmode -halt-on-error main.tex > build.log
$TEX/bibtex main >> build.log
$TEX/pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build.log
$TEX/pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build.log
grep -E "Warning|undefined" main.log | grep -v "Font shape\|pdfTeX warning" || true
$TEX/pdflatex -interaction=nonstopmode -halt-on-error online_appendix.tex > build_oa.log
$TEX/pdflatex -interaction=nonstopmode -halt-on-error online_appendix.tex >> build_oa.log
grep -o "Output written on online_appendix.pdf ([0-9]* pages" online_appendix.log
grep -o "Output written on main.pdf ([0-9]* pages" main.log
