# Reconstruction and ablation study of the AugMix paper
**(Reproducibility Challenge, Fundamentals of Inference and Learning EE-411, EPFL)**



## Structure of the project

```
augmix/
    |
    |-- src/
    |   |
    |   |-- scripts/
    |   |   |-- run_params_ablation.sh
    |   |   |-- stage_logs.sh
    |   |
    |   |-- snapshots/
    |   |   |-- {dataset}_{model}_{width}_{depth}
    |   |
    |   |-- main.py
    |   |-- models.py
    |   |-- utils.py
    |   |-- results_visualization.xlsx
    |   
    |-- report.pdf
    |
    |-- README.md
```

* ```main.py```
    * script containing the main pipeline of the project - training, testing, checkpointing, ...
    * *DISCLAIMER* - most of the code in this script was borrowed from the official 
    implementation of Augmix by Google (https://github.com/google-research/augmix)
* ```models.py```
    * python file containing implementations of the architectures used in this project



## Running the code

**TODO**


## Team info

* Team members:
    * Gojko Cutura - gojko.cutura@epfl.ch
    

