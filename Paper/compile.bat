@echo off
REM ── MixTrace Paper — Compile Script ───────────────────────────
REM Requires: MiKTeX or TeX Live with svjour3 class installed
REM Download svjour3 from: https://www.springer.com/gp/authors-editors/book-authors-editors/manuscript-preparation/5636
REM Or open main.tex on Overleaf — svjour3 is pre-installed there.

cd /d "%~dp0"

echo [1/4] pdflatex pass 1 ...
pdflatex -interaction=nonstopmode main.tex

echo [2/4] bibtex ...
bibtex main

echo [3/4] pdflatex pass 2 ...
pdflatex -interaction=nonstopmode main.tex

echo [4/4] pdflatex pass 3 (final) ...
pdflatex -interaction=nonstopmode main.tex

echo.
echo Done.  Output: main.pdf
pause
