"""
一键执行全套单元测试与端到端系统测试
"""
import sys
import unittest
from pathlib import Path

# 将项目根目录加入模块搜索路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_suite():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    from tests.test_tools import TestTools
    from tests.test_session import TestSessionIsolation
    from tests.test_context import TestContextManagement
    from tests.test_runtime_loop import TestRuntimeLoop

    suite.addTests(loader.loadTestsFromTestCase(TestTools))
    suite.addTests(loader.loadTestsFromTestCase(TestSessionIsolation))
    suite.addTests(loader.loadTestsFromTestCase(TestContextManagement))
    suite.addTests(loader.loadTestsFromTestCase(TestRuntimeLoop))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print(f"📊 测试运行统计: 总运行用例数: {result.testsRun}")
    print(f"✅ 成功用例数: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ 失败用例数: {len(result.failures)}")
    print(f"⚠️ 异常用例数: {len(result.errors)}")
    print("=" * 60)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_suite())
