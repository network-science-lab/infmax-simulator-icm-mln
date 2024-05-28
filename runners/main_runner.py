from dataclasses import dataclass
from itertools import product
from pathlib import Path
import warnings
import yaml

from misc.net_loader import load_network
from misc.utils import *
from tqdm import tqdm
import network_diffusion as nd

warnings.filterwarnings(action="ignore", category=FutureWarning)


@dataclass(frozen=True)
class Network:
    name: str
    graph: nd.MultilayerNetwork


def get_parameter_space(protocols: list[str], p_values: list[float], networks: list[str]) -> product:
    nets = [Network(n, load_network(n)) for n in networks]
    return product(protocols, p_values, nets)


def get_ranking(
    actor: nd.MLNetworkActor, actors: list[nd.MLNetworkActor]
) -> nd.seeding.MockyActorSelector:
    ranking_list = [actor, *set(actors).difference({actor})]
    return nd.seeding.MockyActorSelector(ranking_list)


def run_experiments(config):

    # get parameter space and experiment's hyperparams
    p_space = get_parameter_space(
        protocols=config["model"]["parameters"]["protocols"],
        p_values=config["model"]["parameters"]["p_values"],
        networks=config["networks"],
    )
    repetitions = config["run"]["repetitions"]

    # prepare output directory and deterimne how to store results
    out_dir = Path(config["logging"]["out_dir"])
    out_dir.mkdir(exist_ok=True, parents=True)
    compress_to_zip = config["logging"]["compress_to_zip"]
    average_results = config["run"]["average_results"]

    # save config
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    # get a start time
    start_time = get_current_time()
    print(f"Experiments started at {start_time}")

    # main loop
    p_bar = tqdm(list(p_space), desc="main loop", leave=False, colour="green")
    for idx, investigated_case in enumerate(p_bar):

        # obtain parameters of the propagation scenario
        protocol = investigated_case[0]
        p = investigated_case[1]
        net_name = investigated_case[2].name
        net: nd.MultilayerNetwork = investigated_case[2].graph
        actors = net.get_actors()

        # obtain pool of actors and a budget in the run
        actors_num = net.get_actors_num()
        active_fraction = 100 * (1 / actors_num)
        seeding_budget = (100 - active_fraction, active_fraction, 0)

        # init a container for computed marginal efficiency of each actor
        marginal_efficiencies = []

        # iterate through all available actors and check their influencial power
        for actor in actors:

            # initialise model with "ranking" that prioritises current actor
            apriori_ranking = get_ranking(actor, actors)
            micm = nd.models.MICModel(
                seeding_budget=seeding_budget,
                seed_selector=apriori_ranking,
                protocol=protocol,
                probability=p,
            )

            # repeat the simulation to get mean results
            repeated_results = []
            for rep in range(repetitions):

                # update progress_bar
                case_name = (f"{idx}/{len(p_bar)}--proto-{protocol}--p-{round(p, 3)}--net-{net_name}--repetition-{rep}")
                p_bar.set_description_str(desc=str(case_name))

                # run experiment on a deep copy of the network!
                experiment = nd.Simulator(model=micm, network=net.copy())
                logs = experiment.perform_propagation(
                    n_epochs=actors_num * 2,  # this value is an "overkill"
                    patience=1,
                )

                # compute boost that current actor provides
                simulation_result = extract_simulation_result(logs._local_stats, net, actor)
                repeated_results.append(simulation_result)
            
            # get mean value for each result
            if average_results:
                marginal_efficiencies.append(mean_repeated_results(repeated_results))
            else:
                marginal_efficiencies.extend(repeated_results)
        
        # save efficiences obtained for this case
        investigated_case_file_path = out_dir / f"proto-{protocol}--p-{round(p, 3)}--net-{net_name}.csv"
        save_magrinal_efficiences(marginal_efficiencies, investigated_case_file_path)

    # save global logs and config
    if compress_to_zip:
        zip_detailed_logs([out_dir], rm_logged_dirs=True)

    finish_time = get_current_time()
    print(f"Experiments finished at {finish_time}")
    print(f"Experiments lasted {get_diff_of_times(start_time, finish_time)} minutes")
