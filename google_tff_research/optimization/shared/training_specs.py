# Copyright 2019, Google LLC.
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
"""Configuration classes for creating and running federated training tasks."""

from typing import Any, Callable, Dict, List, Optional

import attr
import tensorflow as tf
import tensorflow_federated as tff

ModelFnType = Callable[[], tff.learning.Model]
ValidationFnType = Optional[Callable[[Any, int], Dict[str, float]]]
TestFnType = EvaluationFnType = Optional[Callable[[Any], Dict[str, float]]]


BYZANTINES_PART_OF = ['total', 'round']


def _check_positive(instance, attribute, value):
  if value <= 0:
    raise ValueError(f'{attribute.name} must be positive. Found {value}.')


def _int_or_proportion(instance, attribute, value: float):
  if value >= 1. and not value.is_integer():
    raise ValueError(f'{attribute.name} must either be a proportion (i.e. [0, 1)) or a full number. Found {value}')


def _check_in_arr(arr: List[Any]):
  def validator(instance, attribute, value):
    if value not in arr:
      raise ValueError(f'{attribute.name} must be in {arr}. Found {value}.')
  return validator


@attr.s(eq=False, order=False, frozen=True)
class TaskSpec(object):
  """Contains information for creating a federated training task.

  This class contains a callable `iterative_process_builder` for building a
  `tff.templates.IterativeProcess`, as well as hyperparameters governing
  how to perform federated training using the resulting iterative process.

  We assume that the iterative process has the following functional type
  signatures:

    *   `initialize`: `( -> S@SERVER)` where `S` represents the server state.
    *   `next`: `<S@SERVER, {B*}@CLIENTS> -> <S@SERVER, T@SERVER>` where `S`
        represents the server state, `{B*}` represents the client datasets,
        and `T` represents a python `Mapping` object.

  The iterative process must also have a callable attribute `get_model_weights`
  that takes as input the state of the iterative process, and returns a
  `tff.learning.ModelWeights` object.
  """
  iterative_process_builder: Callable[
      [ModelFnType], tff.templates.IterativeProcess] = attr.ib()
  """A function that accepts a no-arg `model_fn`,
  and returns a `tff.templates.IterativeProcess`.
  The `model_fn` must return a `tff.learning.Model`.
  """
  client_epochs_per_round: int = attr.ib(
      validator=[attr.validators.instance_of(int), _check_positive],
      converter=int)
  """An integer representing the number of epochs of training performed per client in each training round."""
  client_batch_size: int = attr.ib(
      validator=[attr.validators.instance_of(int), _check_positive],
      converter=int)
  """An integer representing the batch size used on clients."""
  clients_per_round: int = attr.ib(
      validator=[attr.validators.instance_of(int), _check_positive],
      converter=int)
  """An integer representing the number of clients participating in each round."""
  client_datasets_random_seed: Optional[int] = attr.ib(
      default=None,
      validator=attr.validators.optional(attr.validators.instance_of(int)),
      converter=attr.converters.optional(int))
  """An optional int used to seed which clients are sampled at each round.
  If `None`,
  no seed is used."""
  num_byzantine: float = attr.ib(
    default=0.1,
    validator=[attr.validators.instance_of(float), _int_or_proportion])
  """A float representing how many Byzantine clients are active."""
  byzantines_part_of: str = attr.ib(
    default='total',
    validator=[attr.validators.instance_of(str), _check_in_arr(BYZANTINES_PART_OF)])
  """A string representing whether `num_byzantine` is taken as part of the total amount of clients or in each round."""


@attr.s(eq=False, order=False, frozen=True)
class RunnerSpec(object):
  """Contains information for running a federated training task.

  This class contains a `tff.templates.IterativeProcess`, as well as auxiliary
  utilities for running rounds of the iterative process, and evaluating its
  progress.

  We assume that the iterative process has the following functional type
  signatures:

    *   `initialize`: `( -> S@SERVER)` where `S` represents the server state.
    *   `next`: `<S@SERVER, {B*}@CLIENTS> -> <S@SERVER, T@SERVER>` where `S`
        represents the server state, `{B*}` represents the client datasets,
        and `T` represents a python `Mapping` object.

  The iterative process must also have a callable attribute `get_model_weights`
  that takes as input the state of the iterative process, and returns a
  `tff.learning.ModelWeights` object, which can then be used as input to the
  `validation_fn` and `test_fn` (if provided).
  """
  iterative_process: tff.templates.IterativeProcess = attr.ib()
  """A `tff.templates.IterativeProcess` instance to run."""
  client_datasets_fn: Callable[[int], List[Any]] = attr.ib(
      validator=attr.validators.is_callable())
  """Function accepting an integer argument (the round number)
  and returning a list of client datasets to use as federated data for that round.
  """
  validation_fn: ValidationFnType = attr.ib(
      default=None,
      validator=attr.validators.optional(attr.validators.is_callable()))
  """An optional callable accepting used to compute validation metrics during training."""
  test_fn: TestFnType = attr.ib(
      default=None,
      validator=attr.validators.optional(attr.validators.is_callable()))
  """An optional callable accepting used to compute test metrics during training."""
