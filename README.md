AnXplore analysis framework
=======================================

### Incidactors
- `WSS`: Wall Shear Stress
- `OSI`: Oscillatory Shear Index
- `KER`: Kinetic Energy Ratio
- `VDR`: Viscous Dissipation Ratio
- `LSA`: Low Shear Area
- `HSA`: High Shear Area
- `SCI`: Shear Concentration Index
- `ICI`: Inflow Concentration Index

Source: [Computational Hemodynamics Framework for the Analysis of Cerebral Aneurysms](https://doi.org/10.1002/cnm.1424)

### Folder structure
```
└── data
    ├── AnXplore178_FSI.h5
    └── AnXplore178_FSI.xdmf
├── utils/
├── .gitignore
├── AnXplore_analysis_framework.py
└── README.md
```

### Run the code
```
python AnXplore_analysis_framework.py
```

Generates csv file in the folder `data/csv/` containing the values of the indicators at each time step of a cardiac cycle.

The last 4 lines are the min, max, mean and std of the indicators over the whole cardiac cycle.