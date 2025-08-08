import copy
import os
import random
import numpy as np

from algorithm.algorithm import Algorithm
from rdkit import Chem
from algorithm.modified_simulated_annealing import SimulatedAnnealing
from operators.crossover import Crossover
from operators.mutation import Mutation
from problem.drug_likeness import DrugLikeness

from solution.solution import Solution
from typing import List



class ParticleSwarmSA(Algorithm):

    def __init__(self,
                 problem: DrugLikeness,
                 max_evaluations: int,
                 swarm_size: int,
                 replace_mutation: Mutation,
                 add_mutation: Mutation,
                 remove_mutation: Mutation,
                 crossover: Crossover,
                 save_smiles_dir: str,
                 bank_size = 1000) -> None:
        super(ParticleSwarmSA, self).__init__()
        self.problem = problem
        self.max_evaluations = max_evaluations
        self.swarm_size = swarm_size
        self.smiles_dir = save_smiles_dir

        self.bank = []
        self.bank_size = self.swarm_size*3
        
        self.replace_mutation = replace_mutation
        self.add_mutation = add_mutation
        self.remove_mutation = remove_mutation
        self.crossover_operator = crossover

        self.global_best = None
        self.local_best = [None] * self.swarm_size

        self.minimum_fit = 0.0

        self.evaluations = 0
        self.iterations = 0
        self.convergence_curve = []

        self.seen_inchikey = set()

        self.ls_init_cc = []
        self.ls_cc = []

    def create_initial_solutions(self)->List[Solution]:
        self.bank = self.problem.load_bank(max_molecules=self.bank_size)
        return self.bank
    
    def evaluate(self, solution_list: List[Solution])->List[Solution]:
        return [self.problem.evaluate(sol) for sol in solution_list]
    
    def related_molecules(self):
        print("Computing Related Drugs")
        dsk = 'OC[CH](O)[CH](O)[CH](O)[CH](O)[CH](O)C[S+]1C[CH](O)[CH](O)[CH]1CO'
        acarbose = '[H]C(=O)[C@H](O)[C@@H](O)[C@]([H])(O[C@@]1([H])O[C@H](CO)[C@@]([H])(O[C@H]2O[C@H](C)[C@@H](N[C@@]3([H])C=C(CO)[C@@H](O)[C@H](O)[C@H]3O)[C@H](O)[C@H]2O)[C@H](O)[C@H]1O)[C@H](O)CO'
        miglitol = 'OCCN1C[C@H](O)[C@@H](O)[C@H](O)[C@H]1CO'

        return [self.problem.evaluate(self.problem.create_solution(rel_mol))
                for rel_mol in [dsk, acarbose, miglitol]]

    
    def init_progress(self) -> None:
        self.evaluations = self.bank_size
        self.init_global_best()
        self.init_particle_best(self.solutions)
        self.solutions = self.solutions[:self.swarm_size]
        self.solutions.extend(self.related_molecules())
        self.solutions = sorted(self.solutions,key=lambda x:x.objectives[0], reverse=True)

        for sol in self.solutions:
            self.seen_inchikey.add(Chem.inchi.MolToInchiKey(sol.attributes['mol']))

        self.convergence_curve.append(self.global_best.objectives[0])
        print(f"Evaluations: {self.evaluations}, Best Fitness Value: {self.global_best.objectives[0]}, QED: {self.global_best.attributes['QED']}, SAS: {self.global_best.attributes['SAS']}, AFF: {self.global_best.attributes['Affinity']}")
        
        self.minimum_fit = self.global_best.objectives[0] * 0.5

        try:
            os.makedirs(os.path.dirname(self.smiles_dir), exist_ok=True)
        except FileNotFoundError:
            pass

        with open(self.smiles_dir,'w+') as f:
            f.write('smiles;fitness;QED;SAS;Affinity\n')

    def update_progress(self) -> None:
        self.iterations += 1
        self.convergence_curve.append(self.global_best.objectives[0])
        self.minimum_fit *= 1.1 if self.minimum_fit < (self.global_best.objectives[0] * 0.9) else 1
        print(f"Evaluations: {self.evaluations}, Best Fitness Value: {self.global_best.objectives[0]}, QED: {self.global_best.attributes['QED']}, SAS: {self.global_best.attributes['SAS']}, AFF: {self.global_best.attributes['Affinity']}")

        self.ls_init_cc.append(self.global_best.objectives[0])
        

    def local_search(self, solution):
        ls_alg = SimulatedAnnealing(problem = self.problem, init_solution=solution, max_evaluations=10, min_fitness=self.minimum_fit)
        ls_alg.run()
        self.evaluations += ls_alg.evaluations
        self.ls_cc.append(ls_alg.get_result().objectives[0])

        return ls_alg.get_result()

    def init_particle_best(self, swarm: List[Solution])-> None:
        self.local_best = copy.deepcopy(swarm)

    def init_global_best(self)->None:
        self.global_best = sorted(self.solutions, key=lambda x:x.objectives[0], reverse=True)[0]

    def step(self) -> None:
        for i in range(self.swarm_size):
            self.solutions[i] = self.local_search(self.solutions[i])
            new_particles = self.update_position(self.solutions[i], self.local_best[i])
            new_particles = self.evaluate(new_particles)
            self.evaluations += len(new_particles)
            self.replacement(new_particles, i)
            self.update_global_best()
    
    def update_position (self, solution, local_best):
        #Crossover        
        offspring_population = self.crossover([(local_best, self.global_best),
                        (local_best, solution),
                        (solution, self.global_best)])
        
        #Mutacion
        offspring_population.extend(self.mutation(solution))

        return offspring_population
    
    def mutation(self, solution)->List[Solution]:
        mutation_offspring = []
        generated = 0
        generation_error = 0
        mol = None
        while generated < 10:
            if generation_error == 180:
                #Logger.warning("Problems in mutating solution by replace mutation.")
                break
            try:
                mut_op = random.choice([self.replace_mutation,
                                        self.remove_mutation,
                                        self.add_mutation])
                offspring, mol = mut_op.execute(solution)
                if mol and Chem.inchi.MolToInchiKey(mol) not in self.seen_inchikey:
                    generated += 1
                    offspring.attributes['mol'] = mol
                    mutation_offspring.append(offspring)
                    self.seen_inchikey.add(Chem.inchi.MolToInchiKey(mol))
                else:
                    generation_error += 1
            except PermissionError:
                generation_error += 1
        return mutation_offspring
    
    def crossover(self, parents_set)->List[Solution]:
        crossover_offspring = []
        for parents in parents_set:
            generation_error = 0
            mol = None
            while generation_error < 10 and not mol:    
                for ring_bool in [True, False]:
                    try:                
                        offspring, mol = self.crossover_operator.execute(parents, 
                                                                            ring_bool= ring_bool)
                        if mol and Chem.inchi.MolToInchiKey(mol) not in self.seen_inchikey:
                            offspring.attributes['mol'] = mol
                            crossover_offspring.append(offspring)
                            self.seen_inchikey.add(Chem.inchi.MolToInchiKey(mol))
                            break
                        else:
                            generation_error += 1
                    except PermissionError:
                        generation_error += 1
        return crossover_offspring

    def replacement(self, new_particles: List[Solution], particle_position):
        if len(new_particles) > 0:
            best_new_particle = sorted(new_particles, key=lambda x:x.objectives,reverse=True)[0]    
            self.solutions[particle_position] = best_new_particle if best_new_particle.objectives[0] > self.minimum_fit else self.solutions[particle_position]
            if best_new_particle.objectives[0] > self.local_best[particle_position].objectives[0]:
                    self.local_best[particle_position] = copy.deepcopy(best_new_particle)
            for new_particle in new_particles:
                
                #Write the new valid particle to the csv, even if it is not the best
                #if new_particle.objectives[0] > self.minimum_fit:
                with open(self.smiles_dir,'a+') as f: 
                    f.write(f"{new_particle.variables};{new_particle.objectives[0]};{new_particle.attributes['QED']};{new_particle.attributes['SAS']};{new_particle.attributes['Affinity']}\n")
                    self.seen_inchikey.add(Chem.inchi.MolToInchiKey(new_particle.attributes['mol']))

    def update_global_best(self):
        best_current_particle = sorted(self.solutions,key= lambda x:x.objectives[0], reverse=True)[0]
        self.global_best = copy.deepcopy(best_current_particle) if best_current_particle.objectives[0] > self.global_best.objectives[0] else self.global_best

    def stopping_condition_is_met(self) -> bool:
        return self.evaluations >= self.max_evaluations
    
    def get_name(self) -> str:
        return "Particle Swarm Optimization (PSO)"
    
    def get_result(self):
        return self.global_best