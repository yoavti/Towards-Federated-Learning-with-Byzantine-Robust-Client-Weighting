# Evaluation: Training

This submodule contains the training experiments.

## Training

In order to reproduce the paper's results, execute `experiments/training/run_experiment.py` (the current working directory should be the repo's root).

You can view the documentation for every command line parameter using `experiments/training/run_experiment.py --help`.

## Results

You can monitor the progress using TensorBoard:

```setup
tensorboard --logdir <root_output_dir>/logdir
```