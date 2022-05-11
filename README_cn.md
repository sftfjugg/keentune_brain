[English](./keentune-brain/README.md)| [简体中文](./keentune-brain/README_cn.md) 

# KeenTune Brain  
## 简介
---  
KeenTune-brain是“KeenTune”系统参数优化系统的人工智能调优引擎。KeenTune-brain实现了多种智能调优算法。它为KeenTune系统生成了一个候选配置，从Keentune-工作台上获得了一个评估，并给出了最优的参数配置。

## 安装
---  
### 1. 安装 python-setuptools  
```sh
$ sudo apt-get install python-setuptools
or
$ sudo yum install python-setuptools
```

### 2. 安装 keentune-brain  
```shell
$ sudo python3 setup.py install
```

### 3. 安装 requirements  
```shell
$ pip3 install -r requirements.txt
```

### 4. 运行 keentune-brain  
```shell
$ keentune-brain
```

## 算法
---   
### Sensitive Parameter Detection 算法
### 线性回归模型 ElasticNet
用于捕捉参数与调优结果之间明显的线性相关性。ElasticnNet可以快速训练完成，结合多轮次识别部分介绍，选择线性模型可以保障在多轮次识别的情况下，整体算法执行效率仍然较高。
### 单变量互信息 Mutual Information
用于捕捉单一参数和调优结果之间的线性/非线性相关性，避免由于参数冗余造成敏感参数的漏报。
### 非线性模型XGBoost+可解释AI算法-SHAP
用于捕捉参数与调有结果之间复杂的非线性关系，并通过可解释AI算法量化黑盒非线性模型捕捉到的相关性。

### Tuning 算法
#### TPE(Tree-structured Parzen Estimator)
基于GP代理模型和SMBO框架实现的参数调优算法，KeenTune中使用[hyperopt](https://github.com/hyperopt/hyperopt)实现TPE算法   
[Algorithms for Hyper-Parameter Optimization](https://proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf)

#### HORD(Radial Basis Function and Dynamic coordinate search)
基于RBF代理模型和Dycors的参数调优算法，KeenTune中使用[pySOT](https://github.com/dme65/pySOT)实现了HORD算法  
[HORD](https://github.com/ilija139/HORD)

## 代码结构
---  
+ algorithm: 算法模块，包括Tuning Algorithm和Sensitive Parameter Detection Algorithm  
+ common: 通用方法模块
+ controller: Web通信模块
+ visualization: 可视化模块

## Documentation
---  
