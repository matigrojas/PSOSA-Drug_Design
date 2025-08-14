import gc
import os
import random
import numpy as np
import multiprocessing as mp
import copy

from typing import List

from problem.problem import Problem
from solution.solution import Solution

from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Contrib.SA_Score import sascorer
from rdkit.DataStructs import TanimotoSimilarity

from utils.smiles_converter import smiles2pdbqt
from vina import Vina

from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')


def run_vina_docking(center, box_size, rigid, flex, pdbqt_path, return_dict):
    try:
        vina_exec = Vina(sf_name='vina', cpu=1, verbosity=0)
        vina_exec.set_receptor(rigid_pdbqt_filename=rigid,
                               flex_pdbqt_filename=flex)

        vina_exec.compute_vina_maps(center=center, 
                                    box_size=box_size, 
                                    spacing=1)

        vina_exec.set_ligand_from_file(pdbqt_path)
        vina_exec.dock(exhaustiveness=6, max_evals=20000)
        affinity = vina_exec.energies()[0][0]
        return_dict["affinity"] = affinity if affinity < 0.0 else 0.0
    except Exception as e:
        return_dict["affinity"] = None


class DiabetesDocking(Problem):

    def __init__(self,
                 bank_dir: str = None,
                 docking_importance: float = 0.75,
                 sa_importance: float = 0.05) -> None:
        super(DiabetesDocking).__init__()

        self.bank_dir = bank_dir

        self.docking_importance = docking_importance
        self.sa_importance = sa_importance
        self.qed_importance = 1 - self.docking_importance - self.sa_importance

        self.number_of_objectives = 1
        self.labels = ['Fitness']

        self.rigid_ligand = 'targets/rigid_3l4u.pdbqt'
        self.flex_ligand = 'targets/flex_3l4u.pdbqt'
        self.center = [0.472, -16.611, -21.111]
        self.box_size = [16, 24, 20]

    
    def load_bank(self, max_molecules: int = 30):
        if not self.bank_dir:
            raise Exception("Bank dir not given")
        
        bank = []

        i = 0
        for elem in np.loadtxt(self.bank_dir,dtype=str,comments=None):
            bank.append(self.create_solution(elem))
            i += 1
            if i >= max_molecules:
                break 
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
        if solution.attributes.get('mol'):
            try:
                solution.attributes['SAS'] = sascorer.calculateScore(solution.attributes['mol'])
                solution.attributes['QED'] = QED.default(solution.attributes['mol'])
            except:
                solution.attributes['SAS'] = 0
                solution.attributes['QED'] = 0

            solution.objectives[0] = (self.qed_importance * solution.attributes['QED']) 
            solution.objectives[0] += (self.sa_importance * (1 - (solution.attributes['SAS'] / 10)))

            try:
                mols, label = smiles2pdbqt(solution.variables, labels="./temp/temporal_smile")

                pdbqt_path = 'temp/temporal_smile.pdbqt'
                if not os.path.exists(pdbqt_path) or os.path.getsize(pdbqt_path) == 0:
                    print("[ERROR] pdbqt no generado correctamente")
                    solution.objectives[0] = 0
                    solution.attributes['Affinity'] = 0
                    return solution

                # --- Timeout Docking ---
                manager = mp.Manager()
                return_dict = manager.dict()
                p = mp.Process(
                    target=run_vina_docking,
                    args=(self.center, self.box_size, self.rigid_ligand, self.flex_ligand, pdbqt_path, return_dict)
                )
                p.start()
                p.join(timeout=120)  # timeout en segundos

                if p.is_alive():
                    print(f"[TIMEOUT] Docking abortado para {solution.variables}")
                    p.terminate()
                    p.join()
                    affinity = 0
                else:
                    affinity = return_dict.get("affinity", 0) or 0

                solution.attributes['Affinity'] = affinity
                solution.objectives[0] += self.docking_importance * ((abs(affinity) - 4) / 11)

                print(solution.variables, affinity, solution.attributes['QED'], solution.attributes['SAS'])

            except Exception as e:
                print(f"[WARN] Fallo en docking para {solution.variables}: {e}")
                solution.attributes['Affinity'] = 0
                solution.objectives[0] = 0
            finally:
                gc.collect()
        else:
            solution.objectives[0] = 0

        return solution

    
    def get_name(self) -> str:
        return "Docking Optimization"