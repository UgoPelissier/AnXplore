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

### Folder structure
```
└── data
    ├── fsi
        ├── *_1.h5
        ├── *_1.xdmf
        .
        .
        .
        ├── *_101.h5
        └── *_101.xdmf
    └── rigid
        ├── *_1.h5
        ├── *_1.xdmf
        .
        .
        .
        ├── *_101.h5
        └── *_101.xdmf
├── utils
├── .gitignore
├── AnXplore_analysis_framework.py
├── AnXplore_stat_post_process.py
└── README.md
```

### Run the code
```
python AnXplore_analysis_framework.py
```

Generates csv file in the folder `res/csv/` containing the values of the indicators at each time step of a cardiac cycle.
The last 4 lines are the min, max, mean and std of the indicators over the whole cardiac cycle.

```
python AnXplore_stat_post_process.py
```

Generates violin plots of the indicators over a cardiac cycle in the folder `res/violin/`.