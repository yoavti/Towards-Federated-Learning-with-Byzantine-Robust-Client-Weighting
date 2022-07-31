# Shared Utilities

This folder contains modules used throughout the code.

## Folder Structure

* `aggregators` includes different aggregation methods.
* `attacks` includes implementation of various Byzantine attacks.
* `client_weighting` provides an interface to convert a string to a `ClientWeighting` value.
* `google_tff_research` includes files taken from the `google-research/federated` repository.
* `preprocess` includes different preprocessing methods.
* `tff_patch` includes slightly edited scripts taken from the Tensorflow Federated library.
* `byzantines_part_of` contain all possibilities of the argument with the same name.
* `extract_client_weights` provides a way to extract the amount of data each client possesses in a federated dataset.
* `flags_validators` includes different validators for command-line arguments.

## Modules

The modules `aggregators`, `attacks`, `client_weighting`, and `preprocess` include the following files:
* `options`/`dict`: This file contains the different options of some argument relevant to the module.
* `spec`: This file provides a dataclass which is needed in order to pass arguments to the module facade.
* `configure`: This file contains a method which serves as a facade for the module.