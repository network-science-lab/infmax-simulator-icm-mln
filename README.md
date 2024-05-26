# Inluence Maximisation Simulator for Multilayer Networks under Independent Cascade Model 

A repository to generate dataset with marginal efficiency for each actor from the evaluated network.

Author: Michał Czuba + Network Science Lab
Affiliation: WUST, Wrocław, Lower Silesia, Poland

## Data

TODO

## Configuration of the runtime

```bash
conda env create -f env/conda.yaml
conda activate infmax-simulator-icm-mln
```

## Structure of the repository
```
.
├── _configs           -> definitions of the spreading regime under which computations are performed
├── _data_set          -> networks to compute actors' marginal efficiency for
├── _output            -> a directory where we recommend to save results
├── env                -> a definition of the runtime environment
├── misc               -> miscellaneous scripts helping in simulations
├── runners            -> scripts to execute experiments according to provided configs
├── README.md          
└── run_experiments.py -> main entrypoint to trigger the pipeline
```

## Running the pipeline

To run experiments execute: `run_experiments.py` and provide proper CLI arguments, i.e. a path to
configuration file and a runner type. See examples in `_config/examples` for inspirations. As a
result, for each evaluated spreading case, a csv file will be obtained with a folllowing data 
regarding each actor of the network:

```python
actor_id: int
simulation_length: int
actors_infected: int
actors_not_infected: int
peak_infections_nb: int
peak_iteration_nb: int
```
