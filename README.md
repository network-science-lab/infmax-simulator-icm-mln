# Inf. Max. Simulator / Evaluator for Multilayer Networks under ICM 

A repository for (1) computing actor spreading potentials in multilayer networks, and (2) evaluating
various methods for the super-spreader identification problem. This code was used in the preparation
of the paper [Identifying Super Spreaders in Multilayer Networks](https://arxiv.org/abs/2505.20980).

* Authors: Michał Czuba, Mateusz Stolarski, Adam Piróg, Piotr Bielak, Piotr Bródka
* Affiliation: WUST, Wrocław, Lower Silesia, Poland

## Functionality

The project has three main functions:

1. Generate spreading potentials for actors of a given multilayer network according to the provided
configuration. This moed was used to generate the
[TopSpreadersDataset](https://github.com/network-science-lab/top-spreaders-dataset).

2. Evaluate given methods for super-spreaders identification according to the TopSpreadersDataset,
which is use as a ground truth. Then with proper scripts from `scripts/analysis` one can compare
the obtained results.  

3. Evaluate given methods for super-spreaders identification according to the Multilayer 
Inddepenent Cascade Model. This mode was not used in experiments due to impossibility to use it as
a ground truth, but can be utulised to evaluate e.g. seed sets.

## Structure of the repository

```bash
.
├── README.md 
├── data                      -> use DVC to fetch this folder
│   ├── iou_curves            -> results of the evaluation
│   ├── top_spreaders_dataset -> data set with aux. library providing loaders
│   └── test                  -> data used in the E2E test
├── env                       -> definition of the runtime environment
├── scripts                   -> call these to process `data` with `src`
│   ├── analysis
│   └── configs
├── src
│   ├── evaluators            -> evaluate performance of infmax methods
│   ├── generators            -> generate SPs according to provided configs
│   └── icm                   -> ICM adapted to multilayer networks
├── run_experiments.py        -> main entry point to trigger the pipeline
└── test_reproducibility.py   -> E2E test to verify reproducibility of results
```

## Configuration of the runtime

I First, initialise the environment:

```bash
conda env create -f env/conda.yaml
conda activate infmax-simulator-icm-mln
```

II Then, pull the submodule with data loaders and install its code:

```bash
git submodule init && git submodule update
pip install -e data/top_spreaders_dataset 
```

III Install the source code as an editable package:

```bash
pip install -e .
```

IV The dataset is stored in a separate repository linked to this project as a Git submodule. To
obtain it, follow the manual from the [README.md](data/top_spreaders_dataset/README.md).

V The final step is to install wrappers for influence-maximisation methods into the conda
environment. We recommend linking them in editable mode, so after cloning a particular method,
simply install it with: `pip install -e ../path/to/infmax/method`. The wrappers for competitive
approaches are available upon request.

## Using the package

To run experiments, execute `run_experiments.py` and provide the appropriate CLI arguments, such as
a path to the configuration file. See examples in `scripts/configs` for inspiration. The pipeline
has three modes defined under the `run:experiment_type` field.

### MODE 1: Generating a dataset

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

Results are expected to be reproducible. A test is available: `test_reproducibility.py`.

### MODE 2: Evaluating seed selection methods with GT

Another option for evaluating seed selection methods is to compare seed sets with ground truth from
the dataset. To do so, first run `evaluate_gt` to obtain a complete ranking of actors sorted by their
spreading potential score. These rankings can then be compared with each other (`scripts/analysis`).

Here is an order of execution for evaluation scripts:

```bash
scripts
├── analysis
│   ├── distribution_plots.py
│   ├── iou_ranking_plots.py
│   ├── rel_score_plots.py
│   └── generate_tables.py
└── configs
```

### MODE 3: Evaluating seed selection methods with ICM

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
