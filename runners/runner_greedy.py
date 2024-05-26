from dataclasses import dataclass
from itertools import product
from pathlib import Path
import time
import yaml

from misc.net_loader import load_network
from misc.utils import *
from tqdm import tqdm
import network_diffusion as nd
import pandas as pd


@dataclass
class Network:
    name: str
    graph: nd.MultilayerNetwork


@dataclass
class SimulationResult:
    actor: int
    simulation_length: int
    actors_infected: int
    actors_not_infected: int
    peak_infections_nb: int
    peak_iteration_nb: int


def parameter_space(protocols: list[str], p_values: list[float], networks: list[str]) -> product:
    nets = [Network(n, load_network(n)) for n in networks]
    return product(protocols, p_values, nets)


def get_ranking(
    actor: nd.MLNetworkActor, actors: list[nd.MLNetworkActor]
) -> nd.seeding.MockyActorSelector:
    ranking_list = [actor, *set(actors).difference({actor})]
    return nd.seeding.MockyActorSelector(ranking_list)


def run_experiments(config):

    # get parameter space and experiment's hyperparams
    p_space = parameter_space(
        protocols=config["model"]["parameters"]["protocols"],
        p_values=config["model"]["parameters"]["p_values"],
        networks=config["networks"],
    )
    max_epochs_num = config["run"]["max_epochs_num"]
    patience = config["run"]["patience"]
    repetitions = config["run"]["repetitions"]

    # prepare output directory
    out_dir = Path(config["logging"]["out_dir"]) / config["logging"]["name"]
    out_dir.mkdir(exist_ok=True, parents=True)

    # save config
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # init containers for results
    global_stats_handler = pd.DataFrame(data={})
    detailed_stats_dirs: list[Path] = []
    print(f"Experiments started at {get_current_time()}")
    
    p_bar = tqdm(list(p_space), desc="main loop", leave=False, colour="green")
    for idx, investigated_case in enumerate(p_bar):

        # obtain parameters of the propagation scenario
        protocol = investigated_case[0]
        p = investigated_case[1]
        net_name = investigated_case[2].name
        net: nd.MultilayerNetwork = investigated_case[2].graph
        actors = net.get_actors()

        greedy_ranking: list[nd.MLNetworkActor] = []
        actors_num = net.get_actors_num()
        eval_seed_budget = 1

        for rep in range(repetitions):

            # update progress_bar
            case_name = (f"proto_{protocol}--mi_{round(p, 3)}--net_{net_name}--repetition_{rep}")
            p_bar.set_description_str(desc=str(case_name))
            time.sleep(0.01)

            # iterate through all available actors and check their influencial power
            for actor in actors:

                    # initialise model with "ranking" that prioritises current actor
                    apriori_ranking = get_ranking(actor, actors)
                    print([a.actor_id for a in apriori_ranking.preselected_actors])
            
                    micm = nd.models.MICModel(
                        seeding_budget=(100 - eval_seed_budget, eval_seed_budget),
                        seed_selector=apriori_ranking,
                        protocol=protocol,
                        probability=p,
                    )

        #         # run experiment on a deep copy of the network!
        #         experiment = nd.Simulator(model=mltm, network=net.copy())
        #         logs = experiment.perform_propagation(max_epochs_num, patience)

        #         # compute boost that current actor provides
        #         diffusion_len, active_actors, _ = extract_basic_stats(
        #             detailed_logs=logs._local_stats
        #         )
        #         coverage = active_actors / actors_num * 100

        #         # if gain is relevant update the best currently actor
        #         if (
        #             coverage > best_coverage or 
        #             (
        #                 coverage == best_coverage and
        #                 diffusion_len < best_diffusion_len
        #             )
        #         ):
        #             best_actor = actor
        #             best_diffusion_len = diffusion_len
        #             best_coverage = coverage
        #             best_logs = logs
        #             print(
        #                 f"\n\tcurrently best actor '{best_actor.actor_id}' for "
        #                 f"greedy list: {[i.actor_id for i in greedy_ranking]}, "
        #                 f"coverage: {round(best_coverage, 2)}"
        #             )

        #         # when the best combination is found update greedy ranking
        #         greedy_ranking.append(best_actor)

        #         # save logs for further analysis
        #         case_dir = out_dir.joinpath(f"{idx}-{case_name}")
        #         case_dir.mkdir(exist_ok=True, parents=True)
        #         detailed_stats_dirs.append(case_dir)
        #         best_logs.report(path=str(case_dir))

        #         # update global logs
        #         case = {
        #             "network": net_name,
        #             "protocol": protocol,
        #             "seeding_budget": eval_seed_budget,
        #             "mi_value": mi_value,
        #             "repetition_run": 1,
        #             "diffusion_len": best_diffusion_len,
        #             "active_actors_prct": best_coverage,
        #             "seed_actors_prct": eval_seed_budget,
        #             "gain": compute_gain(eval_seed_budget, best_coverage),
        #         }
        #         global_stats_handler = pd.concat(
        #             [global_stats_handler, pd.DataFrame.from_records([case])],
        #             ignore_index=True,
        #             axis=0,
        #         )

    # # save global logs and config
    # global_stats_handler.to_csv(out_dir.joinpath("results.csv"))
    # zip_detailed_logs(detailed_stats_dirs, rm_logged_dirs=True)

    print(f"Experiments finished at {get_current_time()}")
