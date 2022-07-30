import unittest
from aggregators_test import AggregatorsTest
from local_attacks_test import LocalAttacksTest
from preprocess_test import PreprocessTest
from run_experiment_test import RunExperimentTest


if __name__ == '__main__':
  test_classes = [AggregatorsTest, LocalAttacksTest, PreprocessTest, RunExperimentTest]
  loader = unittest.TestLoader()
  suites = [loader.loadTestsFromTestCase(test_class) for test_class in test_classes]
  suite = unittest.TestSuite(suites)
  runner = unittest.TextTestRunner()
  results = runner.run(suite)
