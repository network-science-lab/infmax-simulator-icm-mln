import network_diffusion as nd


def _get_seeding_budget_for_network(
        net: nd.MultilayerNetwork, actorwise: bool = False
) -> dict[str, dict[str, int]]:
    """
    Return initial number of actors in compartmental graph proper to ActorTieredMICModel.
     
    This method patches nd.models.CompartmentalGraph.get_seeding_budget_for_network. It is fixed for
    the case that a single actor is a source of the diffusion. Hence, the seeding budget (according
    to the way it is used in network_diffusion) always equals to 1 / number of actors.

    :param net: network to compute initial compartment shares.
    :param actorwise: a field required by the original get_seeding_budget_for_network

    :return: a dictionary with the initial number of actors in each compartment 
    """
    return {
        FixedBudgetMICModel.PROCESS_NAME: {
            FixedBudgetMICModel.INACTIVE_NODE: net.get_actors_num() - 1,
            FixedBudgetMICModel.ACTIVE_NODE: 1,
            FixedBudgetMICModel.ACTIVATED_NODE: 0,
        }
    }


class FixedBudgetMICModel(nd.models.MICModel):
    
    def __init__(
            self, seed_selector: nd.seeding.MockingActorSelector, protocol: str, probability: float
    ) -> None:
        """
        Create the object.

        :param protocol: logical operator that determines how to activate actor can be OR (then 
            actor gets activated if it gets positive input in one layer) or AND (then actor gets 
            activated if it gets positive input in all layers)
        :param probability: threshold parameter which activate actor (a random variable must be 
            greater than this param to result in activation)
        """
        assert 0 <= probability <= 1, f"incorrect probability: {probability}!"
        self.probability = probability
        self.__comp_graph = self._create_compartments()
        self.__seed_selector = seed_selector
        if protocol == "AND":
            self.protocol = self._protocol_and
        elif protocol == "OR":
            self.protocol = self._protocol_or
        else:
            raise ValueError("Only AND & OR value are allowed!")
    
    @property
    def _compartmental_graph(self) -> nd.models.CompartmentalGraph:
        """Compartmental model that defines allowed transitions and states."""
        return self.__comp_graph

    @property
    def _seed_selector(self) -> nd.seeding.MockingActorSelector:
        """A method of selecting seed agents."""
        return self.__seed_selector

    def _create_compartments(self) -> nd.models.CompartmentalGraph:
        """
        Create compartmental graph for the model.

        Create one process with three states: 0, 1, -1 and assign transition weights for transition 
            0 -> 1 (self.probability) and 1 -> -1 (1.0). In this case we're using a fixed budget,
            i.e. only one actor is a source of the diffusion.
        """
        compart_graph = super()._create_compartments([100, 0, 0]) # this is a dummy budget
        compart_graph.get_seeding_budget_for_network = _get_seeding_budget_for_network
        return compart_graph
