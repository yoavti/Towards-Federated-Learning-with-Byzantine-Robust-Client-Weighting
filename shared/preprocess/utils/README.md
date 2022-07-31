# Preprocess Utilities

This submodule includes base classes and helper functions.

## Folder Structure

* `transform` includes base classes for preprocessing methods.
* `helpers` includes general helper methods.
* `preprocess_func_creators` includes functions that transform a given preprocess transformer into a function based on whether preprocess is meant to look at all clients, or only at clients in the current round.