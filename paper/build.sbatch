#!/bin/bash
#SBATCH --job-name=sw-latex
#SBATCH --output=slurm/logs/latex_%j.out
#SBATCH --error=slurm/logs/latex_%j.err
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
cd /work/pi_sniekum_umass_edu/kn/ShadowWeave/paper
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
bibtex main >/dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
echo "=== TOTAL PAGES ==="; grep -oE "main.pdf \([0-9]+ pages" main.log | tail -1
echo "=== last body section page (Conclusion) & refs/appendix ==="
grep -E "newlabel\{(sec:conclusion|app:gains)\}" main.aux
echo "=== undefined refs/citations (should be empty) ==="
grep -iE "undefined (reference|citation)" main.log | head
echo "=== overfull hboxes (count) ==="; grep -c "Overfull \\\\hbox" main.log
echo "=== build exit OK ==="
