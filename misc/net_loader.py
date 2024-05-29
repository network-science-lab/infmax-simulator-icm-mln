#  TODO: clean loaders for functions we're not using

from functools import wraps
from pathlib import Path

from network_diffusion.mln.functions import get_toy_network
import pandas as pd
import network_diffusion as nd
import networkx as nx


def _network_from_pandas(path):
    df = pd.read_csv(path, names=["node_1", "node_2", "layer"])
    net_dict = {l_name: nx.Graph() for l_name in df["layer"].unique()}
    for _, row in df.iterrows():  # TODO: consider changing the method of iterating
        net_dict[row["layer"]].add_edge(row["node_1"], row["node_2"])
    return nd.MultilayerNetwork.from_nx_layers(
        layer_names=list(net_dict.keys()), network_list=list(net_dict.values())
    )


def return_some_layers(get_network_func):
    @wraps(get_network_func)
    def wrapper(layer_slice = None):
        net = get_network_func()
        if not layer_slice:
            return net
        l_graphs = [net.layers[layer] for layer in layer_slice]
        return nd.MultilayerNetwork.from_nx_layers(l_graphs, layer_slice)
    return wrapper


def get_aucs_network():
    return nd.MultilayerNetwork.from_mpx(file_path="_data_set/small_real/aucs.mpx")


def get_ckm_physicians_network():
    return _network_from_pandas("_data_set/small_real/CKM-Physicians-Innovation_4NoNature.edges")


@return_some_layers
def get_eu_transportation_network():
    return _network_from_pandas("_data_set/small_real/EUAirTransportation_multiplex_4NoNature.edges")


def get_lazega_network():
    return _network_from_pandas("_data_set/small_real/Lazega-Law-Firm_4NoNatureNoLoops.edges")


def get_er2_network():
    return nd.MultilayerNetwork.from_mpx(file_path="_data_set/small_artificial/er_2.mpx")


def get_er3_network():
    return nd.MultilayerNetwork.from_mpx(file_path="_data_set/small_artificial/er_3.mpx")


@return_some_layers
def get_er5_network():
    return nd.MultilayerNetwork.from_mpx(file_path="_data_set/small_artificial/er_5.mpx")


def get_sf2_network():
    return nd.MultilayerNetwork.from_mpx(file_path="_data_set/small_artificial/sf_2.mpx")


def get_sf3_network():
    return nd.MultilayerNetwork.from_mpx(file_path="_data_set/small_artificial/sf_3.mpx")


@return_some_layers
def get_sf5_network():
    return nd.MultilayerNetwork.from_mpx(file_path="_data_set/small_artificial/sf_5.mpx")


def get_ddm_network(layernames_path, edgelist_path, weighted, digraph):
    # read mapping of layer IDs to their names
    with open(layernames_path, encoding="utf-8") as file:
        layer_names = file.readlines()
    layer_names = [ln.rstrip('\n').split(" ") for ln in layer_names]
    layer_names = {ln[0]: ln[1] for ln in layer_names}
    
    # read the edgelist and create containers for the layers
    df = pd.read_csv(
        edgelist_path,
        names=["layer_id", "node_1", "node_2", "weight"],
        sep=" "
    )
    net_ids_dict = {
        l_name: nx.DiGraph() if digraph else nx.Graph()
        for l_name in list(df["layer_id"].unique())
    }

    # populate network with edges
    for _, row in df.iterrows():  # TODO: consider changing the method of iterating
        if weighted:
            attrs = {"weight": row["weight"]}
        else:
            attrs = {}
        net_ids_dict[row["layer_id"]].add_edge(row["node_1"], row["node_2"], **attrs)
    
    # rename layers
    net_names_dict = {
        layer_names[str(layer_id)]: layer_graph
        for layer_id, layer_graph in net_ids_dict.items()
    }

    # create multilater network from edges
    return nd.MultilayerNetwork.from_nx_layers(
        layer_names=list(net_names_dict.keys()), network_list=list(net_names_dict.values())
    )


def get_arxiv_network():
    root_path = Path("_data_set/arxiv_netscience_coauthorship/Dataset")
    net = get_ddm_network(
        layernames_path= root_path / "arxiv_netscience_layers.txt",
        edgelist_path=root_path / "arxiv_netscience_multiplex.edges",
        weighted=False,
        digraph=False,
    )
    return net


def get_cannes_network():
    root_path = Path("_data_set/cannes_2013_social/Dataset")
    net = get_ddm_network(
        layernames_path= root_path / "Cannes2013_layers.txt",
        edgelist_path=root_path / "Cannes2013_multiplex.edges",
        weighted=False,
        digraph=False,
    )
    return net


def get_timik1q2009_network():
    layer_graphs = []
    layer_names = []
    for i in Path("_data_set/timik1q2009").glob("*.csv"):
        layer_names.append(i.stem)
        layer_graphs.append(nx.from_pandas_edgelist(pd.read_csv(i)))
    return nd.MultilayerNetwork.from_nx_layers(layer_graphs, layer_names)


def load_network(net_name: str) -> nd.MultilayerNetwork:
    if net_name == "arxiv":
        return get_arxiv_network()
    elif net_name == "aucs":
        return get_aucs_network()
    elif net_name == "cannes":
        return get_cannes_network()
    elif net_name == "ckm_physicians":
        return get_ckm_physicians_network()
    elif net_name == "eu_transportation":
        return get_eu_transportation_network()
    elif net_name == "eu_transport_klm":
        return get_eu_transportation_network(["KLM"])
    elif net_name == "lazega":
        return get_lazega_network()
    elif net_name == "er1":
        return get_er5_network(["l2"])
    elif net_name == "er2":
        return get_er2_network()
    elif net_name == "er3":
        return get_er3_network()
    elif net_name == "er5":
        return get_er5_network()
    elif net_name == "sf1":
        return get_sf5_network(["l3"])
    elif net_name == "sf2":
        return get_sf2_network()
    elif net_name == "sf3":
        return get_sf3_network()
    elif net_name == "sf5":
        return get_sf5_network()
    elif net_name == "timik1q2009":
        return get_timik1q2009_network()
    elif net_name == "toy_network":
        return get_toy_network()
    raise AttributeError(f"Unknown network: {net_name}")
