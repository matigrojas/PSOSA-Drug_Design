import copy, random
from algorithm.algorithm import Algorithm
from operators.mutation import Mutation
from problem.drug_likeness import DrugLikeness
from solution.solution import Solution

from rdkit import Chem
import numpy as np


class SimulatedAnnealing(Algorithm):

    def __init__(self,
                 problem: DrugLikeness,
                 init_solution: Solution,
                 max_evaluations: int,
                 min_fitness: float) -> None:
        super().__init__()

        self.max_evaluations = max_evaluations
        self.init_solution = init_solution
        self.problem = problem
        self.min_fitness = min_fitness

        self.temperature = 1.0
        self.minimum_temperature = 1e-6
        self.alpha = 0.95

        self.register = []
    
    def init_progress(self) -> None:
        self.evaluations = 1
        #print(self.init_solution.objectives[0])
    
    def create_initial_solutions(self):
        return [self.init_solution]

    def step(self) -> None:
        mutation_operator = self.random_shuffle#random.choice([self.cut_atom, self.random_shuffle])
        mutated_solution = copy.deepcopy(self.solutions[0])
        mutated_solution.variables = mutation_operator(mutated_solution.variables)
        mutated_solution.attributes['mol'] = Chem.MolFromSmiles(mutated_solution.variables)
        mutated_solution = self.evaluate([mutated_solution])[0]

        
        acceptance_probability = self.compute_acceptance_proability(
            self.solutions[0].objectives[0],
            mutated_solution.objectives[0]
        ) if mutated_solution.attributes['mol'] else 0.0

        if acceptance_probability > random.random():
            self.solutions[0] = mutated_solution

        if mutated_solution.attributes['mol'] and mutated_solution.objectives[0]>self.min_fitness:
            self.register.append(mutated_solution)

        self.temperature *= self.alpha if self.temperature > self.minimum_temperature else 1.0

        self.update_progress()

    def insert_c(self, smile):
        new_smile = copy.copy(smile)
        random_pos = random.randint(0,len(smile)-1)
        new_smile = new_smile[:random_pos] + 'C' + new_smile[random_pos:]
        return new_smile

    def cut_atom(self, smile):
        new_smile = copy.copy(smile)
        c_pos = [i for i in range(len(smile)) if smile[i]=='c' or smile[i] == 'C']
        random_pos = random.choice(c_pos)
        new_smile = smile[:random_pos-1]
        new_smile += smile[(random_pos+1):] if random_pos < len(smile)-1 else ""
        return new_smile

    def random_shuffle(self, smile: str)->str:
        mol = Chem.MolFromSmiles(smile)
        new_atom_order = list(range(mol.GetNumAtoms()))
        random.shuffle(new_atom_order)
        random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
        variables = Chem.MolToSmiles(random_mol, canonical=False, isomericSmiles=False)

        return variables
    
    def get_name(self)->str:
        return "Random Swap Mutation"

    def move_group(self, smile):
        new_smile = copy.copy(smile)

        return new_smile

    def change_bond(self, smile):
        new_smile = copy.copy(smile)

        return new_smile

    def compute_acceptance_proability(self, current: float, new: float):
        if new>current:
            return 1.0
        else:
            value = abs(new - current)/self.temperature
            return np.exp(-1.0 * value)
    
    def update_progress(self) -> None:
        self.evaluations += 1
    
    def evaluate(self, solution_list):
        return [self.problem.evaluate(sol) for sol in solution_list]
    
    def stopping_condition_is_met(self) -> bool:
        return self.evaluations > self.max_evaluations
    
    def get_result(self):
        return self.solutions[0]
    
    def get_register(self):
        return self.register

    def get_name(self) -> str:
        return "Simulated Annealing"