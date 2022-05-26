import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from MT_restful.test_target_configure import TestTargetConfigure
from MT_restful.test_target_backup import TestTargetBackup
from MT_restful.test_target_rollback import TestTargetRollback
from MT_restful.test_target_status import TestTargetStatus
from MT_restful.test_bench_sendfile import TestBenchSendfile
from MT_restful.test_bench_benchmark import TestBenchBenchmark
from MT_restful.test_bench_status import TestBenchStatus
from MT_restful.test_brain_init import TestBrainInit
from MT_restful.test_brain_acquire import TestBrainAcquire
from MT_restful.test_brain_feedback import TestBrainFeedback
from MT_restful.test_brain_best import TestBrainBest
from MT_restful.test_brain_end import TestBrainEnd
from MT_restful.test_brain_sensitize import TestBrainSensitize
from MT_restful.test_keentuned_apply_result import TestKeentunedApplyResult
from MT_restful.test_keentuned_benchmark_result import TestKeentunedBenchmarkResult
from MT_restful.test_keentuned_sensitize_result import TestKeentunedSensitizeResult
from MT_restful.test_brain_sensitize_list import TestKeentuneSensitizeList
from MT_restful.test_brain_sensitize_delete import TestKeentuneSensitizeDelete


def RunModelCase():
    suite = unittest.TestSuite()

    suite.addTest(TestTargetConfigure('test_target_server_FUN_configure'))
    suite.addTest(TestTargetBackup('test_target_server_FUN_backup'))
    suite.addTest(TestTargetRollback('test_target_server_FUN_rollback'))
    suite.addTest(TestTargetStatus('test_target_server_FUN_status'))

    suite.addTest(TestBenchSendfile('test_bench_server_FUN_sendfile'))
    suite.addTest(TestBenchBenchmark('test_bench_server_FUN_benchmark'))
    suite.addTest(TestBenchStatus('test_bench_server_FUN_status'))

    suite.addTest(TestBrainInit('test_brain_server_FUN_init'))
    suite.addTest(TestBrainAcquire('test_brain_server_FUN_acquire'))
    suite.addTest(TestBrainFeedback('test_brain_server_FUN_feedback'))
    suite.addTest(TestBrainBest('test_brain_server_FUN_best'))
    suite.addTest(TestBrainEnd('test_brain_server_FUN_end'))
    suite.addTest(TestBrainSensitize('test_brain_server_FUN_sensitize'))

    suite.addTest(TestKeentunedApplyResult('test_keentuned_server_FUN_apply_result'))
    suite.addTest(TestKeentunedBenchmarkResult('test_keentuned_server_FUN_benchmark_result'))
    suite.addTest(TestKeentunedSensitizeResult('test_keentuned_server_FUN_sensitize_result'))
    suite.addTest(TestKeentuneSensitizeList('test_brain_server_FUN_sensitize_list'))
    suite.addTest(TestKeentuneSensitizeDelete('test_brain_server_FUN_sensitize_delete'))

    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(RunModelCase())
