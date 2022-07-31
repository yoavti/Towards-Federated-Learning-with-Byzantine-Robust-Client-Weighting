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

from absl import app, flags

from shared.client_weighting import CLIENT_WEIGHTING
from shared.preprocess import PREPROC_TRANSFORMS
from shared.aggregators import AGGREGATORS
from shared.attacks import ALL_ATTACKS

from experiments.training import run_experiment

FLAGS = flags.FLAGS

FLAGS.experiment_name = 'test_experiment'
FLAGS.total_rounds = 2
FLAGS.client_optimizer = 'sgd'
FLAGS.client_learning_rate = 0.01
FLAGS.server_optimizer = 'sgd'
FLAGS.server_learning_rate = 1.0
FLAGS.server_sgd_momentum = 0.0


def test_run_experiment(task, client_weighting='num_examples', weight_preproc='none', aggregation='mean',
                        attack='sign_flip', num_byzantine=0.1):
  print(task, client_weighting, weight_preproc, aggregation, attack, num_byzantine)
  FLAGS.task = task
  FLAGS.client_weighting = client_weighting
  FLAGS.weight_preproc = weight_preproc
  FLAGS.aggregation = aggregation
  FLAGS.attack = attack
  FLAGS.num_byzantine = num_byzantine
  FLAGS.root_output_dir = tempfile.mkdtemp()
  try:
    app.run(run_experiment.main)
  except SystemExit as system_exit:
    print(system_exit)


class RunExperimentTest(unittest.TestCase):
  def test_all(self):
    tasks = ['cifar100_image', 'emnist_autoencoder', 'emnist_character', 'shakespeare_character']
    num_byzantines = [0.1, 2]
    for task in tasks:
      for client_weighting in CLIENT_WEIGHTING:
        for weight_preproc in list(PREPROC_TRANSFORMS) + ['none']:
          for aggregation in AGGREGATORS:
            for attack in ALL_ATTACKS:
              for num_byzantine in num_byzantines:
                test_run_experiment(task, client_weighting, weight_preproc, aggregation, attack, num_byzantine)


if __name__ == '__main__':
  unittest.main()
