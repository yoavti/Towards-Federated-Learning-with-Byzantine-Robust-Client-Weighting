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

from shared.google_tff_research.utils.task_utils import SUPPORTED_TASKS
from shared.preprocess import PREPROC_FUNCS
from experiments.training import run_experiment
from experiments.training.attacks.local import ATTACKS
from experiments.training.run_experiment import CLIENT_WEIGHTING, AGGREGATORS, BYZANTINES_PART_OF

FLAGS = flags.FLAGS

FLAGS.experiment_name = 'test_experiment'
FLAGS.total_rounds = 2
FLAGS.client_optimizer = 'sgd'
FLAGS.client_learning_rate = 0.01
FLAGS.server_optimizer = 'sgd'
FLAGS.server_learning_rate = 1.0
FLAGS.server_sgd_momentum = 0.0


def test_run_experiment(task, weight_preproc='num_examples', aggregation='mean', attack='sign_flip', num_byzantine=0.1,
                        byzantines_part_of='total'):
  print(task, weight_preproc, aggregation, attack)
  FLAGS.task = task
  FLAGS.weight_preproc = weight_preproc
  FLAGS.aggregation = aggregation
  FLAGS.attack = attack
  FLAGS.num_byzantine = num_byzantine
  FLAGS.byzantines_part_of = byzantines_part_of
  FLAGS.root_output_dir = tempfile.mkdtemp()
  try:
    app.run(run_experiment.main)
  except SystemExit as system_exit:
    print(system_exit)


class RunExperimentTest(unittest.TestCase):
  def test_shakespeare(self):
    test_run_experiment('shakespeare_character')

  def test_all_configurations(self):
    for task in SUPPORTED_TASKS:
      for weight_preproc in list(CLIENT_WEIGHTING) + list(PREPROC_FUNCS):
        for aggregation in AGGREGATORS:
          for attack in list(ATTACKS):
            for num_byzantine in [0.1, 2.]:
              for byzantines_part_of in BYZANTINES_PART_OF:
                test_run_experiment(task, weight_preproc, aggregation, attack, num_byzantine, byzantines_part_of)


if __name__ == '__main__':
  unittest.main()
