from random import random, shuffle

from igraph import Graph


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
            if l is None:
                continue

            utility += self.utility_matrix[i][l]
        return utility


class RetroactiveCorrectionModel(Model):
    """Model in which people randomly qualitfy for employment and that number is corrected by a concave function."""

    def __init__(self, num_agents, locality_caps, num_professions, professions, qualification_probabilities,
                 correction_functions, random_samples):
        """Initializes the retroactive correction model.

        Args:
            num_agents (int): number of simulated agents, named i = 0, …, num_agents-1
            locality_caps (list of int): for each locality l = 0, …, len(locality_caps), its maximum capacity
            num_professions (int): number of different professions p = 0, …, num_professions-1
            professions (list of int): for each agent, their profession
            qualification_probabilities (list of list of float): qualification_probabilities[i][l] is the probability
                                                                 of agent i qualifying for employment when in locality
                                                                 l
            correction_functions (list of list of (int → float)): correction_functions[l][p] is the correction function
                                                                  for locality l and profession p
            random_samples (int): number of random experiments to estimate expectation
        """
        self.num_agents = num_agents
        self.locality_caps = locality_caps
        self.num_professions = num_professions
        assert len(professions) == num_agents
        self.professions = professions
        assert len(qualification_probabilities) == num_agents
        assert num_agents == 0 or len(qualification_probabilities[0]) == len(locality_caps)
        self.qualification_probabilities = qualification_probabilities
        assert len(correction_functions) == len(locality_caps)
        assert len(locality_caps) == 0 or len(correction_functions[0]) == num_professions
        self.correction_functions = correction_functions
        assert random_samples > 0
        self.random_samples = random_samples

        self._memoization = [[{} for _ in range(num_professions)] for _ in locality_caps]

    def _utility_at_locality_profession(self, l, p, agents):
        probs = tuple(sorted(self.qualification_probabilities[i][l] for i in agents))
        if probs in self._memoization[l][p]:
            return self._memoization[l][p][probs]

        sum_utilities = 0
        for _ in range(self.random_samples):
            num_qualified = 0
            for prob in probs:
                if random() < prob:
                    num_qualified += 1
            sum_utilities += self.correction_functions[l][p](num_qualified)
        utility = sum_utilities / self.random_samples
        self._memoization[l][p][probs] = utility
        return utility

    def utility_for_matching(self, matching):
        self.check_valid_matching(matching)

        agents_per_locality_profession = [[[] for _ in range(self.num_professions)] for _ in self.locality_caps]
        for i, l in enumerate(matching):
            if l is not None:
                p = self.professions[i]
                agents_per_locality_profession[l][p].append(i)

        utility = 0
        for l in range(len(self.locality_caps)):
            for p in range(self.num_professions):
                utility += self._utility_at_locality_profession(l, p, agents_per_locality_profession[l][p])
        return utility


class InterviewModel(Model):
    """Model in which agents apply for jobs in a random sequential order.
    """

    def __init__(self, num_agents, locality_caps, num_professions, professions, job_numbers,
                 compatibility_probabilities, random_samples):
        """Initializes the interview model.

        Args:
            num_agents (int): number of simulated agents, named i = 0, …, num_agents-1
            locality_caps (list of int): for each locality l = 0, …, len(locality_caps), its maximum capacity
            num_professions (int): number of different professions p = 0, …, num_professions-1
            professions (list of int): for each agent, their profession
            job_numbers(list of list of int): job_numbers[l][p] is the number of available jobs at locality l for
                                              profession p
            compatibility_probabilities(list of float): for each agent, their probability p_i of getting a job of her
                                                      profession
            random_samples (int): number of random experiments to estimate expectation
        """
        self.num_agents = num_agents
        self.locality_caps = locality_caps
        self.num_professions = num_professions
        assert len(professions) == num_agents
        self.professions = professions
        assert len(job_numbers) == len(locality_caps)
        assert len(job_numbers) == 0 or len(job_numbers[0]) == num_professions
        self.job_numbers = job_numbers
        assert len(compatibility_probabilities) == num_agents
        self.compatibility_probabilities = compatibility_probabilities
        assert random_samples > 0
        self.random_samples = random_samples

        self._memoization = [[{} for _ in range(num_professions)] for _ in locality_caps]

    def _utility_at_locality_profession(self, l, p, agents):
        probs = tuple(sorted(self.compatibility_probabilities[i] for i in agents))
        if probs in self._memoization[l][p]:
            return self._memoization[l][p][probs]

        mutable_probs = list(probs)
        sum_utilities = 0
        for _ in range(self.random_samples):
            num_jobs = self.job_numbers[l][p]
            shuffle(mutable_probs)
            for prob in mutable_probs:
                for _ in range(num_jobs):
                    if random() < prob:
                        sum_utilities += 1
                        num_jobs -= 1
                        break
        utility = sum_utilities / self.random_samples
        self._memoization[l][p][probs] = utility
        return utility

    def utility_for_matching(self, matching):
        self.check_valid_matching(matching)

        agents_per_locality_profession = [[[] for _ in range(self.num_professions)] for _ in self.locality_caps]
        for i, l in enumerate(matching):
            if l is not None:
                p = self.professions[i]
                agents_per_locality_profession[l][p].append(i)

        utility = 0
        for l in range(len(self.locality_caps)):
            for p in range(self.num_professions):
                utility += self._utility_at_locality_profession(l, p, agents_per_locality_profession[l][p])
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
                                                                         j at locality l
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
        assert random_samples > 0
        self.random_samples = random_samples

        self._memoization = [{} for _ in locality_caps]

    def _utility_at_locality(self, l, agents):
        agents = tuple(sorted(agents))
        if agents in self._memoization[l]:
            return self._memoization[l][agents]

        sum_utilities = 0
        for _ in range(self.random_samples):
            num_jobs = self.locality_num_jobs[l]

            offset = self.num_agents  # agent i has node id `i`, job j has node id `offset + j`
            edges = []

            for i in agents:
                for j in range(num_jobs):
                    probability = self.compatibility_probabilities[i][l][j]
                    if random() < probability:
                        edges.append((i, offset + j))

            graph = Graph.Bipartite([0] * self.num_agents + [1] * num_jobs, edges)
            matching = graph.maximum_bipartite_matching()

            sum_utilities += len(matching)
        utility = sum_utilities / self.random_samples
        self._memoization[l][agents] = utility
        return utility


    def utility_for_matching(self, matching):
        self.check_valid_matching(matching)

        agents_per_locality = [[] for _ in self.locality_caps]
        for i, l in enumerate(matching):
            if l is not None:
                agents_per_locality[l].append(i)

        utility = 0
        for l in range(len(self.locality_caps)):
            for _ in range(self.random_samples):
                utility += self._utility_at_locality(l, agents_per_locality[l])
        return utility
