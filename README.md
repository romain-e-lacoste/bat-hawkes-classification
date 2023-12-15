# A Classification Procedure of Bats Behaviors based on Hawkes Processes

Source code for the article *Bats Monitoring: A Classification Procedure of Bats Behaviors based on Hawkes Processes*.

## Data

- X : sequence of start times of echolocation calls emitted by bats for each sites
- Y : associated labels : 0 if the majority behavior at a site is commuting, 1 if the majority behavior is foraging

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
