TransBEAM: Transfer Learning for Building Energy Modeling
=======================================================

TransBEAM contains a set of classes and methods to analyze and compare different state-of-the-art data-driven building energy modeling techiques. The module also implements two ways to implement transfer learning: data transfer and model transfer, to make use of both the simulation data and the field data.  The current version is still in development stage with two models being implemented to predict building energy consumption: Random Forest and Feed-Forward Network. In the current version, the library can only handle timeseries data in tabular format. In the future versions, we plan to support much more advanced techniques, including other data formats. 

Installation Instructions
-------------------------

### Setup the Tools
- Setup the conda environment in your system:
	- [Miniconda](https://docs.conda.io/en/latest/miniconda.html), or 
	- [Anaconda](https://docs.anaconda.com/anaconda/install/).
- Once conda is installed, open Miniconda/Anaconda shell and go to the installation directory. 

### Build the Code
- Git clone the repository: `git clone https://stash.pnnl.gov/scm/~jain432/be_modeler.git`
- Go to the installed module folder: `cd <installation_folder>\be_modeler`
- Create conda environment using `environment.yml`: `conda env create -f environment.yml`
- Activate the conda environment: `conda activate bem`

### Test the Code
- Go to the examples folder: `cd <installation_folder>\be_modeler\examples`
- Run the jupyter notebook: `jupyter notebook`

Published Research Work
-----------------------

Jain, Milan, Khushboo Gupta, Arun Sathanur, Vikas Chandan, and Mahantesh M Halappanavar. "Transfer-Learnt Energy Models to Assist Buildings Control with Sparse Field Data." 2021 American Control Conference. IEEE, 2021.

License
-------

Released under the 3-Clause BSD license (see License)
