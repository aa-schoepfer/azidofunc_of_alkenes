# Supplementary materials for: _Universal Photocatalyzed Azidofunctionalization of Alkenes via Radical-Polar Crossover_

This repository contains all codes to reproduce the substrate selection for _"Universal Photocatalyzed Azidofunctionalization of Alkenes via Radical-Polar Crossover"_ (DOI).

## Requirements
- Python (3.10.14 tested)
- Pandas (2.2.2 tested)
- Numpy (1.26.4 tested)
- Matplotlib (3.9.0 tested)
- Rdkit (2023.9.6 tested)
- Scikit-learn (1.4.2 tested)
- Umap-learn (0.5.6 tested)

## Run the code

```bash
python get_results.py
```

This process takes around 6h on an Intel® Core™ i7-9700K CPU @ 3.60GHz × 8. 

## Check for reproducibility

```bash
python test/test_reproducibilty.py
```
If "Same results." is returned, the selected exerpiments match the one from the publication.  