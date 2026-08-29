#!/bin/bash
#SBATCH --job-name=sw-latex
#SBATCH --output=slurm/logs/latex_%j.out
#SBATCH --error=slurm/logs/latex_%j.err
#SBATCH --time=00:06:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
cd /work/pi_sniekum_umass_edu/kn/ShadowWeave/paper
for f in main main_short; do
  pdflatex -interaction=nonstopmode "$f.tex" >/dev/null 2>&1
  bibtex "$f" >/dev/null 2>&1
  pdflatex -interaction=nonstopmode "$f.tex" >/dev/null 2>&1
  pdflatex -interaction=nonstopmode "$f.tex" >/dev/null 2>&1
  echo "########## $f ##########"
  echo "PAGES: $(grep -oE "Output written on $f.pdf \([0-9]+ pages" "$f.log" | tail -1)"
  echo "ERRORS: $(grep -cE '^! |LaTeX Error|Fatal' "$f.log")"
  echo "UNDEFINED refs/cites: $(grep -ciE 'undefined (reference|citation)' "$f.log")"
  echo "OVERFULL hbox: $(grep -c 'Overfull \\hbox' "$f.log")   UNDERFULL hbox: $(grep -c 'Underfull \\hbox' "$f.log")"
  echo "MISSING files/chars: $(grep -ciE 'file.*not found|No file (\w+\.png)|Missing character' "$f.log")"
  echo "bibtex warnings: $(grep -ciE 'warning' "$f.blg")"
  echo "last body section page:"; grep -E "newlabel\{sec:conclusion\}" "$f.aux" 2>/dev/null | sed -E 's/.*\{\{[0-9.]+\}\{([0-9]+)\}.*/  conclusion p\1/'
done
echo "=== DONE ==="
