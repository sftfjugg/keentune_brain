import json
import requests
import unittest


class TestBrainSensitizeList(unittest.TestCase):
    def setUp(self) -> None:
        self.proxies={"http": None, "https": None}
        url = "http://{}:{}/sensitize_list".format("localhost", "9872")
        re = requests.get(url, proxies=self.proxies)
        if re.status_code != 200:
            print("ERROR: Can't reach KeenTune-Brain.")
            exit()

    def tearDown(self) -> None:
        pass

    def test_brain_server_FUN_sensitize_list(self):
        url = "http://{}:{}/{}".format("localhost", "9872", "sensitize_list")
        result = requests.get(url, proxies=self.proxies)
        self.assertEqual(result.status_code, 200)
        self.assertIn('"suc": true, "data":', result.text)
