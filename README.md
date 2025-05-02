# Inf. Max. Simulator for Multilayer Networks under ICM 

A repository for generating datasets with marginal efficiency for each actor from the evaluated 
network and evaluating various influence maximisation methods.

* Authors: Piotr Bródka, Michał Czuba, Adam Piróg, Mateusz Stolarski
* Affiliation: WUST, Wrocław, Lower Silesia, Poland

## Configuration of the runtime

First, initialise the environment:

```bash
conda env create -f env/conda.yaml
conda activate infmax-simulator-icm-mln
```

Then, pull the submodule with data loaders and install its code:

```bash
git submodule init && git submodule update
pip install -e data/nslds
```

To use scripts that produce analysis, install the source code:

```bash
pip install -e .
```

The final step is to install wrappers for influence-maximisation methods into the conda environment.
We recommend linking them in editable mode, so after cloning a particular method, simply install it 
with: `pip install -e ../path/to/infmax/method`.

## Data

The dataset is stored in a separate repository linked to this project as a Git submodule. To obtain
it, you must pull the data from the DVC remote. To access it, please send a request via email
(michal.czuba@pwr.edu.pl). Then, simply execute the following commands in a shell:

```bash
cd data/nslds && dvc pull nsl_data_sources/raw/multi_layer_networks/*.dvc && cd ..
cd data/nslds && dvc pull nsl_data_sources/spreading_potentials/multi_layer_networks/*.dvc && cd ..
cd data/nslds && dvc pull nsl_data_sources/centralities/multi_layer_networks/*.dvc && cd ..
```

## Structure of the repository TODO - update it!

```bash
.
├── README.md 
├── data                     -> use DVC to fetch this folder
│   ├── iou_curves           -> results of the evaluation
│   ├── nslds                -> data set with aux. library providing loaders
│   ├── networks             -> real networks used in experiments
│   └── test                 -> data used in the E2E test
├── env                      -> definition of the runtime environment
├── scripts                  -> call these to process `data` with `src`
│   ├── analysis
│   └── configs
├── src
│   ├── evaluators          -> scripts to evaluate performance of infmax methods
│   ├── generators          -> scripts to generate SPs according to provided configs
│   └── icm                 -> implementations of ICM adapted to multilayer networks
├── run_experiments.py      -> main entry point to trigger the pipeline
└── test_reproducibility.py -> E2E test to verify reproducibility of results
```

## Running the pipeline

To run experiments, execute `run_experiments.py` and provide the appropriate CLI arguments, such as
a path to the configuration file. See examples in `scripts/configs` for inspiration. The pipeline
has two modes defined under the `run:experiment_type` field.

### Generating a dataset

The first mode (`"generate"`) produces a CSV file for each evaluated ICM case, containing the
following data for each actor in the network:

```python
actor: int              # actor's ID
simulation_length: int  # number of simulation steps
exposed: int            # number of infected actors
not_exposed: int        # number of non-infected actors
peak_infected: int      # maximum number of infected actors in a single simulation step
peak_iteration: int     # simulation step when the peak occurred
```

### Evaluating seed selection methods with ICM

The second mode (`"evaluate"`) serves as an evaluation pipeline for various seed selection methods
defined in the study. For each evaluated case, the seed will be obtained, and ICM spreading will be
executed to produce the following CSV:

```python
infmax_model: str       # name of the model used in the evaluation
seed_set: str           # IDs of seed actors aggregated into a string (separated by ;)
gain: float             # gain obtained using this seed set
simulation_length: int  # number of simulation steps
exposed: int            # number of active actors at the end of the simulation
not_exposed: int        # number of actors that remained inactive
peak_infected: int      # maximum number of infected actors in a single simulation step
peak_iteration: int     # simulation step when the peak occurred
expositions_rec: str    # record of new activations aggregated into a string (separated by ;)
```

### Evaluating seed selection methods with GT

Another option for evaluating seed selection methods is to compare seed sets with ground truth from
the dataset. To do so, first run `evaluate_gt` to obtain a complete ranking of actors sorted by their
spreading potential score. These rankings can then be compared with each other (`scripts/analysis`).

Here is an order of execution for evaluation scripts:

```bash
scripts
├── analysis
│   ├── distribution_plots.py
│   ├── iou_ranking_plots.py
│   └── rel_score_plots.py
└── configs
```

## GPU acceleration for computations

Selecting a GPU (for a `tensor` runner) is possible either by setting an environment variable before
executing the Python code or by modifying `env/variables.env`.

For example:

```bash
conda activate infmax-simulator-icm-mln
export CUDA_VISIBLE_DEVICES=2
python generate_sp.py _configs/example_tensor.yaml
```

## Reproducibility of results

Results are expected to be fully reproducible. A test is available for verification:
`test_reproducibility.py`.


# TODO: 
replace 821 with 832 in all files from final_artificial and final_real