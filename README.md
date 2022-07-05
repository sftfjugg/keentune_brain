# KeenTune Brain  
## Introduction
---  
KeenTune-brain is an AI tuning Engine of 'KeenTune' system parameter optimization system. KeenTune-brain implements a variety of intelligent tuning algorithms. It generates a candidate configuration for KeenTune system, obtains an evaluation from Keentune-brain, and gives the optimal parameter configuration.

## Build & Install
### By setuptools
Setuptools can build KeenTune-brain as a python lib. We can run setuptools as  
```s
>> pip3 install setuptools
>> python3 setup.py install
```

### By pyInstaller
pyInstaller can build KeenTune-brain as a binary file. We can run pyInstaller as  
```s
>> pip3 install pyInstaller
>> make
>> make install
```

### Configuration
After install KeenTune-brain by setuptools or pyInstaller, we can find configuration file in **/etc/keentune/conf/brain.conf**
```conf
[brain]
KEENTUNE_HOME = /etc/keentune/                  # KeenTune-brain install path.
KEENTUNE_WORKSPACE = /var/keentune/             # KeenTune-brain user file workspace.
BRAIN_PORT = 9872                               # KeenTune-brain listening port.

[log]
CONSOLE_LEVEL = ERROR                           # Log level of console.
LOGFILE_LEVEL = DEBUG                           # Log level of logfile.
LOGFILE_PATH  = /var/log/keentune/brain.log     # Logfile saving path.
LOGFILE_INTERVAL = 1                            
LOGFILE_BACKUP_COUNT = 14

[tune]
MAX_SEARCH_SPACE = 1000
SURROGATE = RBFInterpolant
STRATEGY  = DYCORSStrategy

[sensitize]
EPOCH = 5
TOPN = 10
THRESHOLD = 0.9
```

### Run
After modify KeenTune-brain configuration file, we can deploy KeenTune-brain and listening to requests as 
```s
>> keentune-brain
```
or depoly KeenTune-brain by systemctl  
```s
>> systemctl start keentune-brain
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