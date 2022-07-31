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
  def test_shakespeare(self):
    test_run_experiment('shakespeare_character')


if __name__ == '__main__':
  unittest.main()
