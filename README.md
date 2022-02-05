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
    |-- environment.yml
    |
    |-- report.pdf
    |
    |-- README.md
```

* ```main.py```
    * script containing the main pipeline of the project - training, testing, checkpointing, ...
    * *DISCLAIMER* - most of the code in this script was borrowed from the official 
    implementation of Augmix by Google (https://github.com/google-research/augmix)
* ```scripts/run_params_ablation.sh```
    * bash script used for running all the experiments in our project
    * runs all desired combinations of AugMix hyperparameters for all the models of interest and datasets
* ```models.py```
    * python file containing implementations of the architectures used in this project
    * some of the models are inherited from the original papers, 
    and some were added by us in order to test the method more broadly (Wide ResNet and AlexNet)
* ```utils.py```
    * python file containing functions that are used all over the project 
    (mostly AugMix-related transformations)
* ```results_visualization.xlsx```
    * Excell file containing the visualization process of our results; used to generate
    all the figures in our report
* ```snapshots``` 
    * directory containing the results (training and testing logs) of all the experiments
    * it's generated during training and later used for the visualization process


## Running the code

*DISCLAIMER* - the good old 'it worked on my machine' ... 
We hope it works on your as well, but can't guarantee :) 

* Set up the conda environment (GPU required!)

```
conda env create -f environment.yml
conda activate augmix
```

* Run the ablation study bash script

```
cd src/scripts
chmod 777 run_params_ablation.sh
./run_params_ablation.sh
```




## Team info

* Team members:
    * Gojko Cutura - gojko.cutura@epfl.ch
    * Soroush Mehdi - soroush.mehdi@epfl.ch
    * Khashayar Najafi - khashayar.najafi@epfl.ch
    

