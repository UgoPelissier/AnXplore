AnXplore analysis framework
=======================================

### Incidactors
- `WSS`: Wall Shear Stress
- `OSI`: Oscillatory Shear Index
- `TAWSS`: Time Average Wall Shear Stress
- `KER`: Kinetic Energy Ratio
- `VDR`: Viscous Dissipation Ratio
- `LSA`: Low Shear Area
- `HSA`: High Shear Area
- `SCI`: Shear Concentration Index
- `ICI`: Inflow Concentration Index

Source: [Computational Hemodynamics Framework for the Analysis of Cerebral Aneurysms](https://doi.org/10.1002/cnm.1424)

### Run
- git pull
- conda env create -f environment.yml
- conda activate anxplore
- Rename `ANEXPLORE_RIG_results` -> `rigid`  et `ANEXPLORE_FSI_results` -> `fsi`
- Update `data_dir` (line `63`) in `AnXplore_analysis_framework.py` with the path to the `rigid` and `fsi` folders
- bash run.sh