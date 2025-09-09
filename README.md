# Inf. Max. Simulator / Evaluator for Multilayer Networks under ICM

A repository for (1) computing actor spreading potentials in multilayer networks, and (2) evaluating
various methods for the super-spreader identification problem. This code was used in the preparation
of the paper [Identifying Super Spreaders in Multilayer Networks](https://arxiv.org/abs/2505.20980).

* Authors: Michał Czuba, Mateusz Stolarski, Adam Piróg, Piotr Bielak, Piotr Bródka  
* Affiliation: WUST, Wrocław, Lower Silesia, Poland

## Functionality

The project has three main functions:

1. Generate spreading potentials for actors of a given multilayer network according to the provided
configuration. This mode was used to generate the
[TopSpreadersDataset](https://github.com/network-science-lab/top-spreaders-dataset).

2. Evaluate given methods for super-spreader identification using the TopSpreadersDataset,
which serves as ground truth. With the scripts in `scripts/analysis`, the obtained results
can then be compared.

3. Evaluate given methods for super-spreader identification under the Multilayer 
Independent Cascade Model. This mode was not used in experiments, as it cannot serve as 
ground truth, but can be utilised to evaluate e.g. larger seed sets.

## Structure of the repository

```bash
.
├── README.md 
├── data                      -> use DVC to fetch this folder
│   ├── iou_curves            -> results of the evaluation
│   ├── top_spreaders_dataset -> dataset with aux. library providing loaders
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
└── test_reproducibility.py   -> E2E test of simulations reproducibility
```

## Runtime configuration

I. First, initialise the environment:

```bash
conda env create -f env/conda.yaml
conda activate infmax-simulator-icm-mln
```

II. Then, pull the Git submodule with data loaders and install its code:

```bash
git submodule init && git submodule update
pip install -e data/top_spreaders_dataset 
```

III. Install the source code as an editable package:

```bash
pip install -e .
```

IV. Finally, (optionally) install wrappers for the influence maximisation methods into the conda
environment. We recommend linking them in editable mode: after cloning a particular method,
install it using `pip install -e ../path/to/infmax/method`. Wrappers for the competitive evaluated
methods are available upon request.

## Source Data Files

The _TopSpreadersDataset_ is managed using DVC. To fetch it, follow the instructions in
[README.md](data/top_spreaders_dataset/README.md). Additionally, most of results obtained with this 
repository is also stored with DVC - below we describe how to fetch them.

### Full Access

To download the result files, you must authenticate with a Google account that has access to the
shared Google Drive storage:
https://drive.google.com/drive/u/1/folders/1pLWobDjds8SF5rh9_HlvSd9B3ja8QdqN. If you
need access, please contact one of the contributors. Then, to fetch the data, run `dvc pull`.

### Paper Version

A public DVC configuration for the result files in a version used in the paper is available at:
https://drive.google.com/file/d/14p4EDGq4acUOVnqenDyBSbtI9bPF7Jta. To use it, unpack the archive,
and move its contents into the `.dvc` directory of this project. Then, execute: `dvc checkout`.

## Using the package

To run experiments, execute `run_experiments.py` and provide the appropriate CLI arguments, such as
a path to the configuration file. See examples in `scripts/configs` for reference. The pipeline
has three modes, defined under the `run:experiment_type` field.

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

This mode compares seed sets with ground truth from the dataset. First, run `evaluate_gt` to obtain
a complete ranking of actors sorted by their spreading potential score. These rankings can then be 
compared using the scripts in `scripts/analysis`.

An order of execution for evaluation scripts:

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

The third mode (`evaluate`) serves as an evaluation pipeline for various seed selection methods
defined in the study. For each evaluated case, a seed set is selected, and ICM is used to simulate
spreading. The output CSV contains:

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

## Acknowledgment

This work was supported by the National Science Centre, Poland [grant no. 2022/45/B/ST6/04145]
(www.multispread.pwr.edu.pl); the Polish Ministry of Science and Higher Education programme
“International Projects Co-Funded”; and the EU under the Horizon Europe [grant no. 101086321].
Views and opinions expressed are those of the authors and do not necessarily reflect those of
the funding agencies.
