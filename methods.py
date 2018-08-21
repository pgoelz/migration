from math import inf


def greedy_algorithm(model):
    """

    Args:
        model (models.Matching)

    Returns:
        a pair (locality_per_agent, best_value) of type (list of int/None, float). The first component is the matching,
        the second its queried value in the model.
    """
    locality_per_agent = [None for _ in range(model.num_agents)]
    caps_remaining = [cap for cap in model.locality_caps]
    best_value = 0

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

    return locality_per_agent, best_value
