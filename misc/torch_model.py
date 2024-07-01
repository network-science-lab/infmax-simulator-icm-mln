from dataclasses import dataclass
from typing import Any, Callable

import networkx as nx
import network_diffusion as nd
import numpy as np
import torch


import network_diffusion.mln.mlnetwork_torch as mlnt


def create_states_tensor(mln_torch: mlnt.MultilayerNetworkTorch, seed_set: set[Any]) -> torch.Tensor:
    """
    Create tensor of states

    :param mln_torch: a network (in tensor representation) to create a states tensor for
    :param seed_set: a set of initially active actors (ids of actors given in the original form)
    :return: a tensor shaped as [number_of_layers x number_of_actors] with 1. marked for seed nodes
        and -inf for nodes that were artifically added during converting the network to the tensor
        representation
    """
    seed_set_mapped = [mln_torch.actors_map[seed] for seed in seed_set]
    print(f"{seed_set} -> {seed_set_mapped}")
    states_raw = torch.clone(mln_torch.nodes_mask)
    states_raw[states_raw == 1.] = -1 * float("inf")
    states_raw[:, seed_set_mapped] += 1
    return states_raw


def draw_live_edges(A: torch.Tensor, p: float) -> torch.Tensor:
    """Draw eges which transmit the state (i.e. their random weight < p)."""
    raw_signals = torch.rand_like(A.values(), dtype=float)
    thre_signals = (raw_signals < p).to(float)
    T = torch.sparse_coo_tensor(indices=A.indices(), values=thre_signals)
    assert A.shape == T.shape
    assert ((A - T).to_dense() < 0).sum() == 0
    return T


def mask_S_from(S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Create mask for T which discards signals from nodes which state != 1."""
    return (S > 0).to(torch.int).unsqueeze(-1).repeat(1, 1, S.shape[1]).to_sparse_coo()


def mask_S_to(S:torch.Tensor) -> torch.Tensor:
    """Create mask for T which discards signals to nodes which state != 0."""
    return torch.abs(torch.abs(S) - 1).to_sparse_coo()


def get_active_nodes(T: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Obtain newly active nodes (0 -> 1) in the current simulation step."""
    S_f = mask_S_from(S)
    S_t = mask_S_to(S)
    S_new = ((T * S_f).sum(dim=1) * S_t).to_dense()
    assert torch.all(S[S_new.to(torch.int).to(bool)] == 0) == torch.Tensor([True]), \
        "Some nodes were activated against rules (i.e. only these with state 0 can be activated)!"
    return S_new


def protocol_AND(S_raw: torch.Tensor, net: mlnt.MultilayerNetworkTorch) -> torch.Tensor:
    """
    Aggregate positive impulses from the layers using AND strategy.

    :param S_raw: raw impulses obtained by the nodes
    :param net: a network which is a medium for the diffusion
    :return: a tensor shaped as [1 x number of actors] with 1. denoting actors that were activated 
        in this simulation step and 0. denoting actors that weren't activated
    """
    return (S_raw + net.nodes_mask > 0).all(dim=0).to(torch.float)


def protocol_OR(S_raw: torch.Tensor, net: mlnt.MultilayerNetworkTorch) -> torch.Tensor:
    """
    Aggregate positive impulses from the layers using AND strategy.

    :param S_raw: raw impulses obtained by the nodes
    :param net: a network which is a medium for the diffusion
    :return: a tensor shaped as [1 x number of actors] with 1. denoting actors that were activated 
        in this simulation step and 0. denoting actors that weren't activated
    """
    return (S_raw > 0).any(dim=0).to(torch.float)


def decay_active_nodes(S: torch.Tensor) -> torch.Tensor:
    """Change states of nodes that are active to become activated (1 -> -1)."""
    decayed_S = -1. * torch.abs(S)
    decayed_S[decayed_S == -0.] = 0.
    return decayed_S


def simulation_step(net: mlnt.MultilayerNetworkTorch, p: float, protocol: Callable, S0: torch.Tensor) -> torch.Tensor:
    """
    Make a single simulation step.
    
    1. determine which edges drawn value below p
    2. transfer state from active (1.) nodes to their inactive (0.) neighbours only if egdes were preserved at step 1.
    3. aggregate positive impulses from the layers to determine actors that got activated during this simulation step 
    4. decay activation potential for actors that were acting as the active in the current simulation step
    5. obtain the final tensor of states after this simulation step 

    :param net: a network wtihch is a medium of the diffusion
    :param p: a probability of activation between active and inactive node
    :param protocol: a function that aggregates positive impulses from the network's layers
    :param S0: initial tensor of nodes' states (0 - inactive, 1 - active, -1 - activated, -inf - node does not exist)
    :return: updated tensor with nodes' states
    """
    T = draw_live_edges(net.adjacency_tensor, p)
    S1_raw = get_active_nodes(T, S0)
    S1_aggregated = protocol(S_raw=S1_raw, net=net)
    S0_decayed = decay_active_nodes(S0)
    return S1_aggregated + S0_decayed


def S_nodes_to_actors(S: torch.Tensor) -> torch.Tensor:
    """Convert tensor of nodes' states to a vector of actors' states."""
    _S = torch.clone(S)
    _S[_S == -1 * float("inf")] = 0.
    return _S.sum(dim=0).clamp(-1, 1)