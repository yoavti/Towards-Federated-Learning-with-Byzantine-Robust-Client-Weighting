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

import tempfile
import unittest

from itertools import product

from absl import app
from absl import flags

from experiments import run_experiment
from experiments.run_experiment import SUPPORTED_TASKS, CLIENT_WEIGHTING, PREPROC_FUNCS, AGGREGATORS, ATTACKS

FLAGS = flags.FLAGS

FLAGS.experiment_name = 'test_experiment'
FLAGS.total_rounds = 2
FLAGS.client_optimizer = 'sgd'
FLAGS.client_learning_rate = 0.01
FLAGS.server_optimizer = 'sgd'
FLAGS.server_learning_rate = 1.0
FLAGS.server_sgd_momentum = 0.0


def test_run_experiment(task, weight_preproc='num_examples', aggregation='mean', attack='none'):
  print(task, weight_preproc, aggregation, attack)
  FLAGS.task = task
  FLAGS.weight_preproc = weight_preproc
  FLAGS.aggregation = aggregation
  FLAGS.attack = attack
  FLAGS.root_output_dir = tempfile.mkdtemp()
  try:
    app.run(run_experiment.main)
  except SystemExit as system_exit:
    print(system_exit)


class TrainerTest(unittest.TestCase):
  def test_shakespeare_no_attack(self):
    test_run_experiment('shakespeare')

  def test_all_configurations(self):
    for task, weight_preproc, aggregation, attack in product(SUPPORTED_TASKS,
                                                             list(CLIENT_WEIGHTING) + list(PREPROC_FUNCS),
                                                             AGGREGATORS,
                                                             list(ATTACKS)):
      test_run_experiment(task, weight_preproc, aggregation, attack)


if __name__ == '__main__':
  unittest.main()
