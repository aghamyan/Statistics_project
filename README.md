# Armenia LFS Statistical Project (2021-2024)

## Project outputs
- `src/lfs_analysis.py`: full data cleaning + analysis pipeline.
- `outputs/data/`: cleaned dataset and metadata.
- `outputs/tables/`: statistical result tables (CSV/TXT).
- `outputs/figures/`: saved charts.
- `report.tex`: final LaTeX report template.

## How to run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute analysis:
   ```bash
   python src/lfs_analysis.py
   ```
3. Compile report:
   ```bash
   pdflatex report.tex
   ```

## Important setup note
Because Armstat LFS variable names can differ by year, update the `VARMAP` and category maps in `src/lfs_analysis.py` using the official questionnaire/codebook before final inference.
