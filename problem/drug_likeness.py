import random
import numpy as np
import copy

from typing import List

from problem.problem import Problem
from solution.solution import Solution

from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Contrib.SA_Score import sascorer
from rdkit.DataStructs import TanimotoSimilarity

from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')


class DrugLikeness(Problem):

    def __init__(self,
                 qed_coef: float = 0.994,
                 bank_dir: str = None) -> None:
        super(DrugLikeness).__init__()

        self.bank_dir = bank_dir

        self.qed_coef = qed_coef

        self.number_of_objectives = 1
        self.labels = ['SmQED']
    
    def load_bank(self):
        if not self.bank_dir:
            raise Exception("Bank dir not given")
        
        bank = []

        for elem in np.loadtxt(self.bank_dir,dtype=str,comments=None):
            bank.append(self.create_solution(elem))
        return bank
    
    def compute_avg_dist(self, bank, n_rand = 1000) -> float:
        dist_sum = .0

        for _ in range(n_rand):
            sol1,sol2 = random.sample(bank,2)
            dist_sum += self.compute_distance(Chem.MolFromSmiles(sol1.variables),
                                              Chem.MolFromSmiles(sol2.variables))
            
        return dist_sum/n_rand

    def compute_distance(self, sml1, sml2):
        fps1 = Chem.RDKFingerprint(sml1)
        fps2 = Chem.RDKFingerprint(sml2)
        dist = TanimotoSimilarity(fps1,fps2)
        return dist
    
    #Check it, because, the whole bank is going to be a population right now.
    def prepare_seed(self, seed: List[Solution] = [], bank = [], n_seed: int = 600, usable: List[bool] = None)->List[Solution]:
        usable_bank = [i for i in range(len(usable)) if usable[i] == True] #Positions not used yet
        #BANK = Solution in 0, used flag in 1

        if len(usable_bank) > n_seed:
            for i in random.sample(usable_bank,(n_seed-len(seed))):
                seed.append(bank[i])
                usable[i] = False
        else: #Lesser usable bank compounds than n_seed
            for i in usable_bank:
                seed.append(bank[i])
                usable[i] = False
            if (n_seed-len(seed)) > 0:
                used_subset = [j for j in range(len(usable)) if not usable[j]]
                for i in random.sample(used_subset,(n_seed-len(seed))):
                    seed.append(bank[i])
                    usable[i] = False
        return seed, usable

    def create_solution(self, smiles)->Solution:
        #TODO: return a random solution
        solution = Solution(
            number_of_objectives=1,
            number_of_constraints=0
        )
        
        solution.variables = copy.deepcopy(smiles)
        solution.attributes['mol'] = Chem.MolFromSmiles(solution.variables) 


        return solution

    def evaluate(self, solution: Solution)->Solution:
        try:
            solution.attributes['SAS'] = sascorer.calculateScore(solution.attributes['mol'])#1 - (sascorer.calculateScore(solution.attributes['mol']) - 1) / 9
            solution.attributes['QED'] = QED.default(solution.attributes['mol'])
        except:
             solution.attributes['SAS'] = 0
             solution.attributes['QED'] = 0
        #Use when wanting to compare against a target
        #solution.attributes['sim'] = TanimotoSimilarity(_get_fp(x), target_fps)

        solution.objectives[0] = (self.qed_coef * solution.attributes['QED']) 
        solution.objectives[0] += ((1-self.qed_coef) * (1-(solution.attributes['SAS']/10)))

        return solution
    
    def get_name(self) -> str:
        return "Drug Likeness"