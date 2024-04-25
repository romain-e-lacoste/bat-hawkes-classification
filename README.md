# A Classification Procedure of Bats Behaviors based on Hawkes Processes

Source code for the article [Bats Monitoring: A Classification Procedure of Bats Behaviors based on Hawkes Processes](https://hal.science/hal-04345822).

## Data

Loading with `torch.load`
- X : sequence of start times of echolocation calls emitted by bats for each labeled sites
- Y : associated labels : 0 if the majority behavior at a site is commuting, 1 if the majority behavior is foraging
- X_unlabeled : sequence of start times of echolocation calls emitted by bats for unlabeled sites

CSV format
- data_set_labeled.csv : labeled data set (the first column corresponds to the label values, the others to event times, with zero padding to match the maximum length)
- data_set_unlabeled.csv : unlabeled data set (event times with zero padding to match the maximum length)
## Functions 

- classification.py : functions used in the classification procedure
- goodness_of_fit_test.py : functions used to perform a goodness-of-fit test
- simulation.py : function to simulate a linear exponential Hawkes process using cluster representation

## Examples

- example_classification_synthetic_data.py : example of classification and test on synthetic data
- example_classification_real_data.py : example of classification and test on real data of bats echolocation calls

## Installation 

Copy all files in the current working directory.
Packages needed to run functions : PyTorch, NumPy, SciPy.

## Author

Romain E. LACOSTE
