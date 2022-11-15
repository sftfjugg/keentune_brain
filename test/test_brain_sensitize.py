import json
import requests
import unittest


class TestBrainSensitize(unittest.TestCase):
    def setUp(self) -> None:
        self.proxies={"http": None, "https": None}
        url = "http://{}:{}/sensitize_list".format("localhost", "9872")
        re = requests.get(url, proxies=self.proxies)
        if re.status_code != 200:
            print("ERROR: Can't reach KeenTune-Brain.")
            exit()

    def tearDown(self) -> None:
        pass

    def test_brain_server_FUN_sensitize(self):
        url = "http://{}:{}/{}".format("localhost", "9872", "sensitize")
        data = {
                    "data": "",
                    "trials": 1,
                    "resp_ip": "localhost",
                    "resp_port": "9871",
                    "explainer": "shap"
                }

        headers = {"Content-Type": "application/json"}
        
        result = requests.post(url, data=json.dumps(data), headers=headers, proxies=self.proxies)
        self.assertEqual(result.status_code, 200)
        self.assertIn('"suc": true', result.text)
        
