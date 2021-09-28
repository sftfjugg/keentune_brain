# Restful API
## /init
### Get
查看调优器激活情况
### Post
初始化调优器
#### 输入格式
```json
{
    "algorithm":"tpe",
    "iteration":500,
    "parameters":[{
            "name":"kernel.sched_migration_cost_ns",
            "range": [100000, 5000000],
            "dtype": "int",
            "step":1,
            "domain":"sysctl"
        },
        {
            "name":"kernel.randomize_va_space",
            "options": ["0", "1"],
            "dtype": "string",
            "domain":"sysctl"
        },
        {
            "name":"kernel.ipv4.udp_mem",
            "options": ["16000 512000000 256 16000", "32000 1024000000 500 32000", "64000 2048000000 1000 64000"],
            "dtype": "string",
            "domain":"sysctl"
        }
    ]
}
```
## /acquire
### Get
#### 返回值
```json
{
    "iteration":13,
    "candidate":[
        {
            "name":"kernel.sched_migration_cost_ns",
            "range": [100000, 5000000],
            "dtype":"int",
            "value":4419533,
            "domain":"sysctl"
        },
        {
            "name":"kernel.randomize_va_space",
            "options": ["0", "1"],
            "dtype":"string",
            "value":0,
            "domain":"sysctl"
        },
        {
            "name":"kernel.ipv4.udp_mem",
            "options": ["16000 512000000 256 16000", "32000 1024000000 500 32000", "64000 2048000000 1000 64000"],
            "dtype":"string",
            "value":"64000 2048000000 1000 64000",
            "domain":"sysctl"
        }
    ]
    "budget": 1
}
```
## /feedback
### Post
```json
{
    "iteration":13,
    "candidate":[
        {
            "name":"kernel.sched_migration_cost_ns",
            "range": [100000, 5000000],
            "dtype":"int",
            "value":4419533,
            "domain":"sysctl"
        },
        {
            "name":"kernel.randomize_va_space",
            "options": ["0", "1"],
            "dtype":"string",
            "value":"0",
            "domain":"sysctl"
        },
        {
            "name":"kernel.ipv4.udp_mem",
            "options": ["16000 512000000 256 16000", "32000 1024000000 500 32000", "64000 2048000000 1000 64000"],
            "dtype":"string",
            "value":"64000 2048000000 1000 64000",
            "domain":"sysctl"
        }
    ],
    "score":"100"
}
```
## /feedback/v2
```json
{
    "iteration":13,
    "candidate":[
        {
            "name":"kernel.sched_migration_cost_ns",
            "range": [100000, 5000000],
            "dtype":"int",
            "value":4419533,
            "domain":"sysctl"
        },
        {
            "name":"kernel.randomize_va_space",
            "options": ["0", "1"],
            "dtype":"string",
            "value":"0",
            "domain":"sysctl"
        },
        {
            "name":"kernel.ipv4.udp_mem",
            "options": ["16000 512000000 256 16000", "32000 1024000000 500 32000", "64000 2048000000 1000 64000"],
            "dtype":"string",
            "value":"64000 2048000000 1000 64000",
            "domain":"sysctl"
        }
    ],
    "score":{
        "Throughput": {
            "value":45000,
            "negative": false,
            "weight": 1,
            "strict":false,
            "baseline":44000
        },
        "Latency": {
            "value":10,
            "negative": true,
            "weight": 0.1,
            "strict":true,
            "baseline":12
        }
    }
}
```

## /best
```json
{
    "iteration":13,
    "candidate":[
        {
            "name":"kernel.sched_migration_cost_ns",
            "range": [100000, 5000000],
            "dtype":"int",
            "value":4419533,
            "domain":"sysctl"
        },
        {
            "name":"kernel.randomize_va_space",
            "options": ["0", "1"],
            "dtype":"string",
            "value":"0",
            "domain":"sysctl"
        },
        {
            "name":"kernel.ipv4.udp_mem",
            "options": ["16000 512000000 256 16000", "32000 1024000000 500 32000", "64000 2048000000 1000 64000"],
            "dtype":"string",
            "value":"64000 2048000000 1000 64000",
            "domain":"sysctl"
        }
    ],
    "score":{
        "Throughput": {
            "value":45000,
            "negative": false,
            "weight": 1,
            "strict":false,
            "baseline":44000
        },
        "Latency": {
            "value":10,
            "negative": true,
            "weight": 0.1,
            "strict":true,
            "baseline":12
        }
    }
}
```
## /end
### Get
终止调优, 删除调优器实例

# 算法性能指标
1. init 响应时间不超过10s
1. acquire 响应时间不超过11s 
2. feedback 响应时间不超过0.2s
3. end 响应时间不超过20s(保存和分析数据)
300轮调优算法占用时间不超过(10s + (10s + 0.2s) * 300 + 20s) = 3390s (约56min)

# SMBO算法族
## TPE
## SMAC
## Metis
## GP
## HORD

# 其他
## PBTTuner
## Hyperband
## BOHB(Bayesian Optimization Hyperband)