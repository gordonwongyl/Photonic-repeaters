import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, brute, dual_annealing
import pygad
from importlib import reload
from IPython.display import clear_output
import os

import tgs 
import rgs 
import delay_feedback_line_cal as line

reload(tgs)
reload(rgs)
reload(line)

L_ATT = 20e3

class Routine:
    def __init__(self, L: float, GAMMA: float, T_SPIN_COH: float) -> None:
        self.L = L
        self.gamma = GAMMA
        self.t_spin_coh = T_SPIN_COH

        self.ga_instance: pygad.GA = None
        self._time: tgs.Time = tgs.Time(gamma=self.gamma)
        self.lookup_for_m_optimization: dict = dict()

    @property
    def time(self) -> tgs.Time:
        if self._time.gamma != self.gamma:
            # print("Updated gamma")
            self._time = tgs.Time(gamma=self.gamma)
        return self._time
    
    @property
    def params(self) -> tuple:
        return self.L, self.time, self.t_spin_coh
    
    # depends on the scheme!
    def key_rate_unpack(self, solution):
        pass

    def fitness_function_GA(self, ga_instance: pygad.GA, solution, solution_idx) -> float:
        return self.key_rate_unpack(solution)

    def negative_key_rate_given_m(self, m, *solution):
        solution = solution + (m,)
        return -self.key_rate_unpack(solution)
    
    def optimized_key_rate_wrt_m(self, branch_param, m0):
        branch_param = tuple(branch_param)
        if branch_param not in self.lookup_for_m_optimization.keys():
            res = minimize_scalar(self.negative_key_rate_given_m, bounds=(max(m0-800, 0.), min(m0+800, 2000.)), args=branch_param, method="Bounded")
            self.lookup_for_m_optimization[branch_param] = float(-res.fun), res.x 
        return self.lookup_for_m_optimization[branch_param]

    
    def on_gen(self, ga_instance:pygad.GA):
        if int(ga_instance.generations_completed) % 2 == 0:
            os.system("clear")
        clear_output(wait=True) 
        print("Generation : ", ga_instance.generations_completed, flush=True)
        print("Current best solution:", ga_instance.best_solution()[0])

    def diversity(self, ga_instance: pygad.GA, K=4):
        index_k_best_list = sorted(range(ga_instance.pop_size[0]), key = lambda sub: ga_instance.last_generation_fitness[sub])[-K:]
        worst_solution = ga_instance.population[index_k_best_list[0]][:-1] # compare the branch paramaters
        best_solution = ga_instance.population[index_k_best_list[-1]][:-1]

        sum_diff = np.sum(np.abs(best_solution-worst_solution))
        return sum_diff

    def on_fitness(self, ga_instance: pygad.GA, population_fitness):
        # find indices of k_best population_fitness solutions
        K=4

        index_k_best_list = sorted(range(ga_instance.pop_size[0]), key = lambda sub: population_fitness[sub])[-K:]
        div = self.diversity(ga_instance, K)

        if ga_instance.generations_completed > 10 or div<=3: 
            for i in index_k_best_list:
                # Refine the key rates and m for these good candidates
                branch_param = ga_instance.population[i][:-1]
                m = ga_instance.population[i][-1]
                print(f"Branch paramaters: {branch_param}")
                print(f"Before: m={m}, key rate={ga_instance.last_generation_fitness[i]}")
                ga_instance.last_generation_fitness[i], ga_instance.population[i][-1] = self.optimized_key_rate_wrt_m(branch_param, m)
                m = ga_instance.population[i][-1]
                print(f"After:  m={m}, key rate={ga_instance.last_generation_fitness[i]}")
        
        div = self.diversity(ga_instance, K)
        print(f"Diversity of this generation: {div}")
        if div == 0 and ga_instance.mutation_percent_genes <= 40.:
            ga_instance.mutation_percent_genes += 1.5
        if div >= 2 and ga_instance.generations_completed > 5:
            ga_instance.mutation_percent_genes -= 2.
        print("Mutation_percent_genes for this generation = ", ga_instance.mutation_percent_genes)

class tgs_a_routine(Routine):
    def __init__(self, L: float, GAMMA: float, T_SPIN_COH: float) -> None:
        super().__init__(L, GAMMA, T_SPIN_COH)
    
    def key_rate_unpack(self, solution):
        m = solution[-1]
        branch_param = np.array(solution[:-1])
        L, time, T_SPIN_COH = self.params
        L_delay = line.delay_line_tgs_ancilla(branch_param=branch_param, time=time)
        miu = tgs.Miu(L, L_ATT, L_delay=L_delay, m=m)
        tree = tgs.Tree_ancilla(branch_param=branch_param, miu=miu)
        error = tgs.Error(tree, time, t_spin_coherence=T_SPIN_COH)
       
        result = tgs.effective_key_rate(tree, time, error, m, 2)
        return max(float(np.nan_to_num(result)), 0.)
    
class tgs_f_routine(Routine):
    def __init__(self, L: float, GAMMA: float, T_SPIN_COH: float) -> None:
        super().__init__(L, GAMMA, T_SPIN_COH)

    def key_rate_unpack(self, solution):
        m = solution[-1]
        branch_param = np.array(solution[:-1])
        L, time, T_SPIN_COH = self.params
        L_delay = line.delay_line_tgs_feedback(branch_param=branch_param, time=time)
        L_feedback = line.feedback_line_tgs_feedback(branch_param=branch_param, time=time)
        miu = tgs.Miu(L, L_ATT, L_delay=L_delay, m=m, L_feedback=L_feedback)
        tree = tgs.Tree_feedback(branch_param, miu=miu)
        error = tgs.Error(tree, time, t_spin_coherence=T_SPIN_COH)
        
        result = tgs.effective_key_rate(tree, time, error, m, 1)
        return max(float(np.nan_to_num(result)), 0.)
    
class rgs_a_routine(Routine):
    def __init__(self, L: float, GAMMA: float, T_SPIN_COH: float) -> None:
        super().__init__(L, GAMMA, T_SPIN_COH)

    def key_rate_unpack(self, solution):
        m = solution[-1]
        branch_param = np.array(solution[:-2])
        n = solution[-2]
        L, time, T_SPIN_COH = self.params
        L_delay = line.delay_line_rgs_ancilla(branch_param=branch_param, n=n, time=time) 
        miu = tgs.Miu(L/2, L_ATT, L_delay=L_delay, m=m)
        rgs_ancilla = rgs.RGS_ancilla(branch_param=branch_param, N=n, miu=miu)
        error = rgs.RGS_Error(rgs_ancilla, time, T_SPIN_COH)

        result = rgs.effective_key_rate(rgs_ancilla, time, error, m, 3, L, L_ATT)
        return max(float(np.nan_to_num(result)), 0.)
        


class rgs_f_routine(Routine):
    def __init__(self, L: float, GAMMA: float, T_SPIN_COH: float) -> None:
        super().__init__(L, GAMMA, T_SPIN_COH)

    def key_rate_unpack(self, solution):
        m = solution[-1]
        branch_param = np.array(solution[:-2])
        n = solution[-2]
        L, time, T_SPIN_COH = self.params
        L_delay = line.delay_line_rgs_feedback(branch_param=branch_param, n=n, time=time) 
        L_feedback = line.feedback_line_rgs_feedback(branch_param=branch_param, n=n, time=time)
        miu = tgs.Miu(L/2, L_ATT, L_delay=L_delay, m=m, L_feedback=L_feedback)
        rgs_feedback = rgs.RGS_feedback(branch_param=branch_param, N=n, miu=miu)
        error = rgs.RGS_Error(rgs_feedback, time, T_SPIN_COH)

        result = rgs.effective_key_rate(rgs_feedback, time, error, m, 1, L, L_ATT)
        return max(float(np.nan_to_num(result)), 0.)

if __name__ == "__main__":
    GAMMA = np.array([2e9, 100e9, 170e6, 100e9]) * 2 * np.pi
    T_SPIN_COHERENCE = [13e-3, 4e-6, 1., 1.]

    rout = tgs_a_routine(1000e3, GAMMA[0], T_SPIN_COHERENCE[0])
    # rout = rgs_a_routine(1000e3, GAMMA[0], T_SPIN_COHERENCE[0])
    num_generations = 20
    num_parents_mating = 4

    fitness_function = rout.fitness_function_GA

    sol_per_pop = 60 # number of solutions within the population (initially)
    num_genes = 4 # the function inputs
    gene_type = [int, int, int, float] # input type
    gene_space = [range(1, 11), range(1, 30), range(1, 30), {"low": 1., "high": 2000.}]  # specify the possible values for each gene
    # gene_space = [range(1, 30), range(1, 20), range(1, 50), {"low": 1., "high": 2000.}]  # specify the possible values for each gene

    init_range_low = 1
    init_range_high = 10

    parent_selection_type = "sss"
    keep_parents = -1 # Number of parents to keep in the current population
    keep_elitism = 1 # only the best solution in the current generation is kept in the next generation.

    crossover_type = "single_point" 

    mutation_type = "random"

    mutation_percent_genes = 25.

    parallel_processing = ["process", 0]


    rout.ga_instance = pygad.GA(num_generations=num_generations,
                            num_parents_mating=num_parents_mating,
                            fitness_func=fitness_function,
                            sol_per_pop=sol_per_pop,
                            num_genes=num_genes,
                            gene_type=gene_type,
                            gene_space=gene_space,
                            init_range_low=init_range_low,
                            init_range_high=init_range_high,
                            parent_selection_type=parent_selection_type,
                            keep_parents=keep_parents,
                            keep_elitism=keep_elitism,
                            crossover_type=crossover_type,
                            mutation_type=mutation_type,
                            mutation_percent_genes=mutation_percent_genes, 
                            on_generation=rout.on_gen, 
                            on_fitness=rout.on_fitness, parallel_processing=parallel_processing, save_solutions=True, stop_criteria="saturate_5")
    rout.ga_instance.run()
    rout.ga_instance.plot_fitness()
