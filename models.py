from random import random

from networkx import Graph, max_weight_matching


class Model:
    """A (submodular) model for the utility of matchings.

    Attributes:
        num_agents (int): number of simulated agents, named i = 0, …, num_agents-1
        locality_caps (list of int): for each locality l = 0, …, len(locality_caps), its maximum capacity
    """

    def check_valid_matching(self, matching):
        """Raises an appropriate exception if argument is no valid matching.

        Dimensions might be wrong, indices out of bound or matching constraints violated.

        Args:
            matching (list of (int / None)): for each agent, her locality or None if she remains unmatched
        Raises:
            ValueError: ``matching`` was no real matching
        """
        if len(matching) != self.num_agents:
            raise ValueError(f"Argument matching has {len(matching)} values, but there are {self.num_agents} agents.")
        if any((l is not None) and not (l >= 0 or l < len(self.locality_caps)) for l in matching):
            raise ValueError("Some element of argument matching is not a valid locality index.")
        locality_usage = [0 for _ in self.locality_caps]
        for l in matching:
            if l is not None:
                locality_usage[l] += 1
        for l, cap in enumerate(self.locality_caps):
            if locality_usage[l] > cap:
                raise ValueError(f"Matching places {locality_usage[l]} agents in locality {l}, but cap is {cap}.")

    def utility_for_matching(self, matching):
        """Computes the utility of a matching.

        Args:
            matching (list of (int / None)): for each agent, her locality or None if she remains unmatched
        Returns:
            a nonnegative float
        Raises:
            ValueError: ``matching`` was no real matching
        """
        raise NotImplementedError


class AdditiveModel(Model):
    """Example model with additive utilities."""

    def __init__(self, num_agents, locality_caps, utility_matrix):
        """Initializes the additive model.

        Args:
            num_agents (int): number of simulated agents, named i = 0, …, num_agents-1
            locality_caps (list of int): for each locality l = 0, …, len(locality_caps), its maximum capacity
            utility_matrix (list of list of float): utility_matrix[i][l] is the utility of matching i to l
        """
        self.num_agents = num_agents
        self.locality_caps = locality_caps
        assert len(utility_matrix) == num_agents
        assert num_agents == 0 or len(utility_matrix[0]) == len(locality_caps)
        self.utility_matrix = utility_matrix

    def utility_for_matching(self, matching):
        self.check_valid_matching(matching)
        utility = 0
        for i, l in enumerate(matching):
            utility += self.utility_matrix[i][l]
        return utility


class CoordinationModel(Model):
    """Model that randomly determines compatibilities between agents and jobs, then matches optimally.

    More precisely, each locality has a certain number of jobs. Each agent and each job have a certain probability of
    being compatible, and all these decisions are independent. When all compatibilities in a locality are determined,
    the utility at this locality is the cardinality of a maximum matching in the induced bipartite graph between agents
    and jobs. The total utility is the estimated expectation over possible compatibility resolutions, summed up over all
    localities.
    Note that the number of jobs and the cap do not have to coincide. It can be reasonable to match more agents to a
    locality than the number of jobs if it is likely that quite a few people cannot be matched. Similarly, a cap might
    be smaller than the demands of the job market.
    """

    def __init__(self, num_agents, locality_caps, locality_num_jobs, compatibility_probabilities, random_samples):
        """Initializes the coordination model.

        Args:
            num_agents (int): number of simulated agents, named i = 0, …, num_agents-1
            locality_caps (list of int): for each locality l = 0, …, len(locality_caps), its maximum capacity
            locality_num_jobs (list of int): for each locality l, its number of jobs j = 0, …, locality_num_jobs[l]-1
            compatibility_probabilities (list of list of list of float): compatibility_probabilities[i][l][j] is the
                                                                         probability that agent i is compatible with job
                                                                         j at location l
            random_samples (int): number of random experiments to estimate expectation
            """
        self.num_agents = num_agents
        assert len(locality_caps) == len(locality_num_jobs)
        self.locality_caps = locality_caps
        self.locality_num_jobs = locality_num_jobs
        assert len(compatibility_probabilities) == num_agents
        assert num_agents == 0 or len(compatibility_probabilities[0]) == len(locality_caps)
        assert (num_agents == 0 or len(locality_caps) == 0
                or len(compatibility_probabilities[0][0]) == locality_num_jobs[0])
        self.compatibility_probabilities = compatibility_probabilities
        self.random_samples = random_samples

    def utility_for_matching(self, matching):
        self.check_valid_matching(matching)

        agents_per_locality = [[] for _ in self.locality_caps]
        for i, l in enumerate(matching):
            if l is not None:
                agents_per_locality[l].append(i)

        sum_utilities = 0
        for _ in range(self.random_samples):
            for l, num_jobs in enumerate(self.locality_num_jobs):
                graph = Graph()
                graph.add_nodes_from([("a", i) for i in range(self.num_agents)])
                graph.add_nodes_from([("j", i) for i in range(num_jobs)])
                for i in agents_per_locality[l]:
                    for j in range(num_jobs):
                        probability = self.compatibility_probabilities[i][l][j]
                        if random() < probability:
                            graph.add_edge(("a", i), ("j", j))
                            print(i, j, "compatible")
                sum_utilities += len(max_weight_matching(graph))
                print(max_weight_matching(graph))

        return sum_utilities / self.random_samples
