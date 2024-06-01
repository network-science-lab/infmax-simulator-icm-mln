# Inf. Max. Simulator for Multilayer Networks under ICM 

A repository to generate dataset with marginal efficiency for each actor from the evaluated network.

* Author: Michał Czuba + Network Science Lab
* Affiliation: WUST, Wrocław, Lower Silesia, Poland

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
├── _configs                -> def. of the spreading regimes under which do computations
├── _data_set               -> networks to compute actors' marginal efficiency for
├── _test_data              -> examplary outputs of the dataset generator used in the E2E test
├── _output                 -> a directory where we recommend to save results
├── env                     -> a definition of the runtime environment
├── misc                    -> miscellaneous scripts helping in simulations
├── runners                 -> scripts to execute experiments according to provided configs
├── README.md          
├── run_experiments.py      -> main entrypoint to trigger the pipeline
└── test_reproducibility.py -> E2E test to prove that results can be repeated
```

## Running the pipeline

To run experiments execute: `run_experiments.py` and provide proper CLI arguments, i.e. a path to
configuration file and a runner type. See examples in `_config/examples` for inspirations. As a
result, for each evaluated spreading case, a csv file will be obtained with a folllowing data 
regarding each actor of the network:

```python
actor: int              # actor's id
simulation_length: int  # nb. of simulation steps
exposed: int            # nb. of infected actors
not_exposed: int        # nb. of not infected actors
peak_infected: int      # maximal nb. of infected actors in a single sim. step
peak_iteration: int     # a sim. step when the peak occured
```

## Results reproducibility

Results are supposed to be fully reproducable. There is a test for that: `test_reproducibility.py`.
