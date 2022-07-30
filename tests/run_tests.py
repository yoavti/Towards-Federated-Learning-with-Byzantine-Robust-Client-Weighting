from unittest import TextTestRunner, TestSuite, TestLoader


if __name__ == '__main__':
  TextTestRunner().run(TestSuite(TestLoader().discover('test_cases')))
