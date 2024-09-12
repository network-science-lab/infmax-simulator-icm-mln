# Inf. Max. Simulator for Multilayer Networks under ICM 

A repository to generate dataset with marginal efficiency for each actor from the evaluated network.

* Authors: Piotr Bródka, Michał Czuba, Adam Piróg, Mateusz Stolarski
* Affiliation: WUST, Wrocław, Lower Silesia, Poland

## Configuration of the runtime

First, initialise the enviornment:

```bash
conda env create -f env/conda.yaml
conda activate infmax-simulator-icm-mln
```

Then, pull the submodule and install its code:

```bash
git submodule init && git submodule update
pip install -e _dataset/infmax_data_utils
```

## Data

Dataset is stored in a separate repository bounded with this project as a git submodule. Thus, to
obtain it you have to pull the data from the DVC remote. In order to access it, please sent a
request to get  an access via  e-mail (michal.czuba@pwr.edu.pl). Then, simply execute in a shell:
* `cd _data_set && dvc pull nsl_data_sources/raw/multi_layer_networks/*.dvc && cd ..`
* `cd _data_set && dvc pull nsl_data_sources/spreading_potentials/multi_layer_networks/*.dvc && cd ..`

## Structure of the repository
```
.
├── _configs                -> def. of the spreading regimes under which do computations
├── _data_set               -> networks to compute actors' marginal efficiency for
├── _test_data              -> examplary outputs of the dataset generator used in the E2E test
├── _output                 -> a directory where we recommend to save results
├── env                     -> a definition of the runtime environment
├── runners                 -> scripts to execute experiments according to provided configs
├── README.md          
├── run_experiments.py      -> main entrypoint to trigger the pipeline
└── test_reproducibility.py -> E2E test to prove that results can be repeated
```

## Running the pipeline

To run experiments execute: `run_experiments.py` and provide proper CLI arguments, i.e. a path to 
the configuration file. See examples in `_config/examples` for inspirations. As a result, for each
evaluated spreading case, a csv file will be obtained with a folllowing data regarding each actor of
the network:

```python
actor: int              # actor's id
simulation_length: int  # nb. of simulation steps
exposed: int            # nb. of infected actors
not_exposed: int        # nb. of not infected actors
peak_infected: int      # maximal nb. of infected actors in a single sim. step
peak_iteration: int     # a sim. step when the peak occured
```

Selecting GPU (for a `tensor` runner) is possible only by setting an env variable before executing 
the Python code, e.g. `export CUDA_VISIBLE_DEVICES=3`

For instance:

```bash
conda activate infmax-simulator-icm-mln
export CUDA_VISIBLE_DEVICES=2
python run_experiments.py _configs/example_tensor.yaml
```

## Results reproducibility

Results are supposed to be fully reproducable. There is a test for that: `test_reproducibility.py`.
