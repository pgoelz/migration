from math import inf
from random import randrange

from gurobipy import Model as GurobiModel, GRB, quicksum


def greedy_algorithm(model):
    """The greedy algorithm for maximizing an (approximately) submodular
    utility function.

    Args:
        model (models.Model): The submodular model to use

    Returns:
        pair (locality_per_agent,best_value) of type (list of int/None, float).
        The first component is the matching, the second its queried value in
        the model.
    """
    locality_per_agent = [None for _ in range(model.num_agents)]
    caps_remaining = [cap for cap in model.locality_caps]

    for _ in range(min(model.num_agents, sum(caps_remaining))):
        best_pair = None
        best_value = -inf
        for i, match in enumerate(locality_per_agent):
            if match is not None:
                continue

            for l, spaces in enumerate(caps_remaining):
                if spaces <= 0:
                    continue

                locality_per_agent[i] = l
                utility = model.utility_for_matching(locality_per_agent)
                locality_per_agent[i] = None

                if utility > best_value:
                    best_pair = (i, l)
                    best_value = utility

        assert best_pair is not None
        i, l = best_pair
        locality_per_agent[i] = l
        caps_remaining[l] -= 1

    return locality_per_agent, model.utility_for_matching(locality_per_agent,
                                                          False)


def additive_optimization(model):
    """Optimize the model exactly, but just based on marginal utilities of
    individual migrant-locality pairs and assuming additivity.

    Args:
        model (models.Model): The submodular model to use

    Returns:
        pair (locality_per_agent,best_value) of type (list of int/None, float).
        The first component is the matching, the second its queried value in
        the model.
    """
    gm = GurobiModel()
    gm.setParam("OutputFlag", False)

    variables = []
    matching = [None for _ in range(model.num_agents)]
    objective = 0
    for i in range(model.num_agents):
        agent_vars = []
        for l in range(len(model.locality_caps)):
            matching[i] = l
            utility = model.utility_for_matching(matching)
            matching[i] = None

            v = gm.addVar(vtype=GRB.INTEGER, name=f"m_{i}_{l}")
            gm.addConstr(v >= 0)
            gm.addConstr(v <= 1)
            agent_vars.append(v)

            objective += utility * v

        variables.append(agent_vars)
        gm.addConstr(quicksum(agent_vars) <= 1)

    for l in range(len(model.locality_caps)):
        gm.addConstr(quicksum(variables[i][l] for i in range(model.num_agents))
                     <= model.locality_caps[l])

    gm.setObjective(objective, GRB.MAXIMIZE)
    gm.optimize()

    assert gm.status == GRB.OPTIMAL
    for i in range(model.num_agents):
        for l in range(len(model.locality_caps)):
            if variables[i][l].X > 0.5:
                matching[i] = l
                break

    return matching, model.utility_for_matching(matching, False)


def best_of_random(model, trials):
    best_matching = None
    best_utility = -inf

    for _ in range(trials):
        matching = [None] * model.num_agents
        caps_remaining = [cap for cap in model.locality_caps]
        pairs_remaining = min(model.num_agents, sum(model.locality_caps))

        while pairs_remaining > 0:
            i = randrange(model.num_agents)
            if matching[i] is not None:
                continue
            while True:
                l = randrange(len(caps_remaining))
                if caps_remaining[l] > 0:
                    break
            matching[i] = l
            caps_remaining[l] -= 1
            pairs_remaining -= 1

        utility = model.utility_for_matching(matching)
        if utility > best_utility:
            best_utility = utility
            best_matching = matching

    return best_matching, model.utility_for_matching(best_matching, False)
