[English](./keentune-brain/README.md)| [简体中文](./keentune-brain/README_cn.md)

# KeenTune Brain  
## Introduction
---  
KeenTune-brain is an AI tuning Engine of 'KeenTune' system parameter optimization system. KeenTune-brain implements a variety of intelligent tuning algorithms. It generates a candidate configuration for KeenTune system, obtains an evaluation from Keentune-bench, and gives the optimal parameter configuration.

## Installation
---  
### 1. install python-setuptools  
```sh
$ sudo apt-get install python-setuptools
or
$ sudo yum install python-setuptools
```

### 2. install keentune-brain  
```shell
$ sudo python3 setup.py install
```

### 3. install requirements  
```shell
$ pip3 install -r requirements.txt
```

### 4. run  
```shell
$ keentune-brain
```

## Algorithm
---   
### Sensitive Parameter Detection Algorithm
### Linear regression model - ElasticNet
Used to capture the apparent linear correlation between the parameters and the tuning results.ElasticnNet can be trained quickly, and combined with the introduction of the multi-round identification section, the selection of linear models can ensure that the overall algorithm execution efficiency is still high in the case of multiple rounds of identification.
### Univariate mutual information - Mutual Information
Use to capture linear / nonlinear correlations between single parameters and tuning results, to avoid underreporting of sensitive parameters due to parameter redundancy.
### Non-linear model-XGBoost + Explainable AI algorithm-SHAP
It is used to capture the complex nonlinear relationship between parameters and tuning results, and to quantify the correlations captured by the black-box nonlinear model through an interpretable AI algorithm.

### Tuning Algorithm
#### TPE(Tree-structured Parzen Estimator)
Parameter tuning algorithm implemented based on the GP agent model and the SMBO framework，KeenTune use[hyperopt](https://github.com/hyperopt/hyperopt) implement the TPE algorithm
[Algorithms for Hyper-Parameter Optimization](https://proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf)

#### HORD(Radial Basis Function and Dynamic coordinate search)
Parameter tuning algorithm based on the RBF agent model and Dycors，KeenTune use[pySOT](https://github.com/dme65/pySOT) implement the HORD algorithm
[HORD](https://github.com/ilija139/HORD)

## Code structure
---  
+ algorithm: algorithm module，包括Tuning Algorithm和Sensitive Parameter Detection Algorithm
+ common: common methods
+ controller: Web communication module
+ visualization: Visualization module

## Documentation
---
