# Copyright 2022, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from aggregators import mean, median, trimmed_mean


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
