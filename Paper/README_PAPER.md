# Paper — LaTeX Source

## Files

| File | Purpose |
|------|---------|
| `main.tex` | Full manuscript (svjour3 class, Crime Law Social Change) |
| `references.bib` | BibTeX bibliography |
| `compile.bat` | Local compile script (Windows) |
| `figures/` | Place all figure files here (PDF + PNG) |

## Compilation

### Option A — Overleaf (Recommended)
1. Create a new project on [overleaf.com](https://www.overleaf.com)
2. Upload `main.tex`, `references.bib`, and all figures
3. Overleaf has `svjour3` pre-installed — compile immediately

### Option B — Local (MiKTeX / TeX Live)
1. Download the `svjour3` package from Springer:  
   https://www.springer.com/gp/authors-editors/book-authors-editors/manuscript-preparation/5636
2. Place `svjour3.cls`, `svjour3.clo`, `svind.clo`, `natbib.sty`, `spbasic.bst` in this folder
3. Run `compile.bat`

### Compile order
```
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Submission Target

**Journal**: Crime, Law and Social Change (Springer, ISSN 0925-4994)  
**Special Issue**: AI Enabled Security Frameworks for Emerging Cybercrime and Digital Threats  
**Submission portal**: https://www.editorialmanager.com/crim/

## Manuscript Checklist

- [ ] Abstract: 150–250 words
- [ ] Keywords: 4–8 terms
- [ ] All tables populated with actual results (run Steps 5, 9–13)
- [ ] All figures generated (run Step 14) and placed in `figures/`
- [ ] Declarations section complete
- [ ] Bibliography cross-checked
- [ ] Word count ≤ 10,000 (body text)
