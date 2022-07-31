import unittest

from shared.aggregators import mean, median, trimmed_mean


class AggregatorsTest(unittest.TestCase):
  def test_mean(self):
    self.assertEqual(mean([80, 90], [20, 30]), 86)

  def test_median(self):
    self.assertEqual(median(list(range(1, 5)), [0.25] * 4), 2.5)
    self.assertEqual(median([1, 2, 2.5, 3, 4], [0.25, 0.25, 0, 0.25, 0.25]), 2.5)

  def test_trimmed_mean(self):
    self.assertEqual(trimmed_mean(list(range(1, 11)), None, 0.1), sum([2, 2, 3, 4, 5, 6, 7, 8, 9, 9]) / 10)


if __name__ == '__main__':
  unittest.main()
