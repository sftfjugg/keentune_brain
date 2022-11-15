import json
import requests
import unittest


class TestBrainAvailable(unittest.TestCase):
    def setUp(self) -> None:
        self.proxies={"http": None, "https": None}
        url = "http://{}:{}/sensitize_list".format("localhost", "9872")
        re = requests.get(url, proxies=self.proxies)
        if re.status_code != 200:
            print("ERROR: Can't reach KeenTune-Brain.")
            exit()

    def tearDown(self) -> None:
        pass

    def test_brain_server_FUN_available(self):
        url = "http://{}:{}/{}".format("localhost", "9872", "avaliable")
        result = requests.get(url, proxies=self.proxies)
        self.assertEqual(result.status_code, 200)
        self.assertIn('{"suc": true', result.text)
