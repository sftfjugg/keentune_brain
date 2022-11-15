import json
import requests
import unittest


class TestBrainInit(unittest.TestCase):
    def setUp(self) -> None:
        self.proxies={"http": None, "https": None}
        url = "http://{}:{}/sensitize_list".format("localhost", "9872")
        re = requests.get(url, proxies=self.proxies)
        if re.status_code != 200:
            print("ERROR: Can't reach KeenTune-Brain.", re.status_code)
            exit()

    def tearDown(self) -> None:
        url = "http://{}:{}/{}".format("localhost", "9872", "end")

        headers = {"Content-Type": "application/json"}
        
        result = requests.get(url, headers=headers, proxies=self.proxies)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.text, '{"suc": true, "msg": ""}')

    def test_brain_server_FUN_init(self):
        url = "http://{}:{}/{}".format("localhost", "9872", "init")
        data = {
            "algorithm": "tpe",
            "iteration": 500,
            "name": "test",
            "type": "tunning",
            "parameters": [
                {
                    "name":"kernel.sched_migration_cost_ns",
                    "range": [100000, 5000000],
                    "dtype": "int",
                    "step": 1,
                    "domain": "sysctl",
                    "base": 100000
                    },
                {
                    "name": "kernel.randomize_va_space",
                    "options": ["0", "1"],
                    "dtype": "string",
                    "domain": "sysctl",
                    "base": "0"
                    },
                {
                    "name": "kernel.ipv4.udp_mem",
                    "options": ["16000 512000000 256 16000", "32000 1024000000 500 32000", "64000 2048000000 1000 64000"],
                    "dtype": "string",
                    "domain": "sysctl",
                    "base": "16000 512000000 256 16000"
                    }
            ],
            "baseline_score":{
                "Requests_sec":{
                    "base": [23047.33, 23754.47],
                    "negative": False,
                    "weight": 100,
                    "strict": False
                }
            }
        }

        headers = {"Content-Type": "application/json"}
        
        result = requests.post(url, data=json.dumps(data), headers=headers, proxies=self.proxies)
        self.assertEqual(result.status_code, 200)
        self.assertIn('"suc": true', result.text)
