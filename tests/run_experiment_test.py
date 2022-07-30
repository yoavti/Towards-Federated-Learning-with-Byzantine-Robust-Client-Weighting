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

from experiments.training import run_experiment

FLAGS = flags.FLAGS

FLAGS.experiment_name = 'test_experiment'
FLAGS.total_rounds = 2
FLAGS.client_optimizer = 'sgd'
FLAGS.client_learning_rate = 0.01
FLAGS.server_optimizer = 'sgd'
FLAGS.server_learning_rate = 1.0
FLAGS.server_sgd_momentum = 0.0


def test_run_experiment(task, weight_preproc='num_examples', aggregation='mean', attack='sign_flip', num_byzantine=0.1,
                        byzantines_part_of='round'):
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


if __name__ == '__main__':
  unittest.main()
