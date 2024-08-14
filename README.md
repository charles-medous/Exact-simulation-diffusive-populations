# ExactSimulation
This repository gives the simulation algorithms for the 
models of interacting populations with diffusive trait introduced in the article
'Exact simulation algorithm for individual-based population models', available on arxiv:
https://arxiv.org/pdf/#####.pdf


The `python` implementation uses the classical libraries
`numpy`, `matplotlib`, `scipy` to be able 
to run on every setup. However an optimized version using 
parallel programming and more fancy packages like 
`multiprocessing` and `oml` will be soon proposed.

# Installation

Download/Clone the repository, and acces it.

```sh
git clone https://github.com/charles-medous/Exact-simulation-diffusive-populations
cd Exact-simulation-diffusive-populations

```

Please ensure that the required packages are locally 
installed on your python version.

On windows, type `cmd` in the search bar and hit `Enter`
to open the command line. On linux, open your terminal or
shell. If you installed python via Anacomba, directly 
use the conda shell. Then type

```sh
# Create a virtual environnement
python3 -m venv venv

# Activate the virtual environnement
source venv/bin/activate

# Install the mandatory modules
pip install -r requirements.txt
```
