# KeenTune Brain  
---  
### Introduction
KeenTune-brain is an AI tuning Engine of 'KeenTune' system parameter optimization system. KeenTune-brain implements a variety of intelligent tuning algorithms. It generates a candidate configuration for KeenTune system, obtains an evaluation from Keentune-brain, and gives the optimal parameter configuration.

### Build & Install
Setuptools can build KeenTune-brain as a python lib. We can run setuptools as  
```s
>> pip3 install setuptools
>> python3 setup.py install
>> pip3 install -r requirements.txt
```
See More Details about the dependencies of keentune-brain in [<Dependencies of KeenTune>](https://gitee.com/anolis/keentuned/blob/doc-0704/docs/install/Dependencies_cn.md)

### Configuration
After install KeenTune-brain by setuptools or pyInstaller, we can find configuration file in **/etc/keentune/conf/brain.conf**
```conf
[brain]
# Basic Configuration
KeenTune_HOME       = /etc/keentune/    ; KeenTune-brain install path.
KeenTune_WORKSPACE  = /var/keentune/    ; KeenTune-brain workspace.
BRAIN_PORT          = 9872              ; KeenTune-brain service port

[tuning]
# Auto-tuning Algorithm Configuration.
MAX_SEARCH_SPACE    = 1000              ; Limitation of the Max-number of available value of a single knob to avoid dimension explosion.
SURROGATE           = RBFInterpolant    ; Surrogate in tuning algorithm - HORD 
STRATEGY            = DYCORSStrategy    ; Strategy in tuning algorithm - HORD 

[sensitize]
# Sensitization Algorithm Configuration.
EPOCH       = 5         ; Modle train epoch in Sensitization Algorithm, improve the accuracy and running time
TOPN        = 10        ; The top number to select sensitive knobs.
THRESHOLD   = 0.9       ; The sensitivity threshold to select sensitive knobs.

[log]
# Configuration about log
LOGFILE_PATH        = /var/log/keentune/brain.log   ; Log file of brain
CONSOLE_LEVEL       = INFO                          ; Console Log level
LOGFILE_LEVEL       = DEBUG                         ; File saved log level
LOGFILE_INTERVAL    = 1                             ; The interval of log file replacing
LOGFILE_BACKUP_COUNT= 14                            ; The count of backup log file  
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
**NOTE**: You need copy the file 'keentune-brain.service' to '/usr/lib/systemd/system' manually, if you installed the keentune-brain by 'setuptools' rather then 'yum install'.

---
## Algorithm
### Sensitive Parameter Detection Algorithm
#### Linear regression model - ElasticNet
Used to capture the apparent linear correlation between the parameters and the tuning results.ElasticnNet can be trained quickly, and combined with the introduction of the multi-round identification section, the selection of linear models can ensure that the overall algorithm execution efficiency is still high in the case of multiple rounds of identification.  

#### Univariate mutual information - Mutual Information
Use to capture linear / nonlinear correlations between single parameters and tuning results, to avoid underreporting of sensitive parameters due to parameter redundancy.  

#### Non-linear model-XGBoost + Explainable AI algorithm-SHAP
It is used to capture the complex nonlinear relationship between parameters and tuning results, and to quantify the correlations captured by the black-box nonlinear model through an interpretable AI algorithm.  

### Tuning Algorithm
#### TPE(Tree-structured Parzen Estimator)
Parameter tuning algorithm implemented based on the GP agent model and the SMBO framework，KeenTune use[hyperopt](https://github.com/hyperopt/hyperopt) implement the TPE algorithm
[Algorithms for Hyper-Parameter Optimization](https://proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf)  

#### HORD(Radial Basis Function and Dynamic coordinate search)
Parameter tuning algorithm based on the RBF agent model and Dycors，KeenTune use[pySOT](https://github.com/dme65/pySOT) implement the HORD algorithm
[HORD](https://github.com/ilija139/HORD)  

---
## Code Structure
```
brain/
├── algorithm           # Algorithm module
│   ├── __init__.py
│   ├── sensitize           # Sensitization Algorithm
│   │   ├── __init__.py
│   │   ├── sensitize.py
│   │   └── sensitizer.py
│   └── tunning             # Auto-Tuning Algorithm
│       ├── base.py
│       ├── hord.py
│       ├── __init__.py
│       ├── random.py
│       └── tpe.py
├── common              # Common module, includes log, config and tools.
│   ├── config.py
│   ├── dataset.py
│   ├── __init__.py
│   ├── pylog.py
│   ├── system.py
│   └── tools.py
├── controller          # Service response module.
│   ├── __init__.py
│   ├── sensi.py
│   ├── system.py
│   └── tunning.py
├── brain.conf          # Configuration file
├── brain.py            # Entrance of keentune-brain
└── __init__.py
5 directories, 22 files
```