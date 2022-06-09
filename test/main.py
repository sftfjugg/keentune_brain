import os
import sys
import unittest

from test_brain_init import TestBrainInit
from test_brain_acquire import TestBrainAcquire
from test_brain_feedback import TestBrainFeedback
from test_brain_best import TestBrainBest
from test_brain_end import TestBrainEnd
from test_brain_sensitize import TestBrainSensitize


def RunModelCase():
    suite = unittest.TestSuite()
    suite.addTest(TestBrainInit('test_brain_server_FUN_init'))
    suite.addTest(TestBrainAcquire('test_brain_server_FUN_acquire'))
    suite.addTest(TestBrainFeedback('test_brain_server_FUN_feedback'))
    suite.addTest(TestBrainBest('test_brain_server_FUN_best'))
    suite.addTest(TestBrainEnd('test_brain_server_FUN_end'))
    suite.addTest(TestBrainSensitize('test_brain_server_FUN_sensitize'))
    return suite


if __name__ == '__main__':
    print("--------------- start to run test cases ---------------")
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(RunModelCase())
    print("--------------- run test cases end ---------------")
