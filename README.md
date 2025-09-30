# EE641 Homework 2

**Full Name:** Sungmin Kang  
**USC Email:** kangsung@usc.edu  

---

## Instructions

### Problem 1: Font Generation GAN – Understanding Mode Collapse
To train and evaluate the GAN models:

```bash
cd problem1
python train.py
python evaluate.py
```


### Problem 2: Hierarchical VAE for Drum Pattern Generation
train, evaluation, visualization code in ***experiments.ipynb***

---

## Implementation Notes

- Problem 1:
Implemented a simplified GAN for font image generation, analyzed mode collapse, and applied feature matching as a stabilization technique. Results include quantitative mode coverage scores and qualitative visualizations.
## Saved models, visualizations for each models - vanilla and fixed ##
- Problem 2:
Implemented a hierarchical VAE with low- and high-level latent variables for drum pattern generation. Used KL annealing to mitigate posterior collapse. Explored creative experiments such as genre blending, complexity control, humanization, and style consistency.
Results, models, and visualizations are saved under each problem’s results/ directory.


Folder Tree: 

ee641-hw2-sungminkg/
├── problem1/
│   ├── models.py
│   ├── dataset.py
│   ├── models.py
│   ├── training_dynamics.py
│   ├── fixes.py
│   ├── train.py
│   ├── evaluate.py
│   └── results/
│       ├── training_log.json
│       ├── best_generator.pth
│       ├── mode_collapse_analysis.png
│       └── visualizations/
├── problem2/
│   ├── dataset.py
│   ├── hierarchical_vae.py
│   ├── training_utils.py
│   ├── train.py
│   ├── analyze_latent.py
│   └── results/
│       ├── training_log.json
│       ├── best_model.pth
│       ├── generated_patterns/
│       └── latent_analysis/
├── report.pdf
└── README.md
