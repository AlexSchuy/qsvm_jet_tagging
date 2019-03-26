# qsvm-jet-tagging
Experiments with using quantum machine learning algorithms to solve Higgs boson tagging problems in particle physics.

## Setup
This project uses [pipenv](https://pipenv.readthedocs.io/en/latest/) to manage dependencies. Run `pipenv install` followed by `pipenv shell` to install and enter the required virtual environment.

## Samples
The samples represent a binary classification problem. The two classes are 'higgs' and 'qcd', representing jets from either a Higgs decay or QCD background. The feature set is kinematic data from the jet (specifically: pt, eta, phi, mass, ee2, ee3, and d2 (see [energy correlators](https://arxiv.org/pdf/1411.0665.pdf) for more information on these last few variables)). 
