import random, re
import numpy as np
import copy

from abc import ABC, abstractmethod

from typing import List
from solution.solution import Solution

from rdkit import Chem
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')


class Crossover(ABC):

    def __init__(self, probability: float):
        self.probability = probability

    @abstractmethod
    def execute(self, parents):
        pass
    
    @abstractmethod
    def get_number_of_parents(self) -> int:
        pass

    @abstractmethod
    def get_number_of_children(self) -> int:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

class SmilesCrossover(Crossover):

    def __init__(self, probability: 1.00):
        super(SmilesCrossover,self).__init__(probability=probability)
        self.limit = 200

    def execute(self, parents: List[Solution], ring_bool: bool = False):
        offspring = copy.deepcopy(parents[0])
        mol = None
        new_smi = None

        gate = 0
        l_smi, r_smi = self.get_sliced_smiles(parents[0].variables, parents[1].variables, ring_bool)
        for _ in range(2):
            while not mol:
                gate += 1
                if gate > 5:
                    break
                try:
                    new_smi = self.tight_rm_branch(l_smi, r_smi) #Controla Paréntesis
                    mol = Chem.MolFromSmiles(new_smi) #Check SMILES validity
                except ValueError:
                    continue

            if mol:
                break
            else:
                l_smi, r_smi = self.get_sliced_smiles(parents[1].variables, parents[0].variables, ring_bool)
                gate = 0

        offspring.variables = new_smi

        return offspring,mol
    
    def get_open_branch(self, smi):
        return [i for i, e in enumerate(smi) if e == "("]


    def get_close_branch(self, smi):
        return [i for i, e in enumerate(smi) if e == ")"]
    
    def chk_branch(self, smi, side=None):

        branch_list = []
        n_branch = 0
        min_branch = 0
        for i, b in enumerate(smi):  # make sure branches are closed
            if b == "(":
                n_branch += 1 
            if b in ")":
                n_branch -= 1
            if side == "L":
                if n_branch < min_branch:
                    min_branch = n_branch
                    branch_list.append(i)
            elif side == "R": 
                if n_branch > min_branch:
                    min_branch = n_branch
                    branch_list.append(i)
        if side == None:
            return n_branch
        return np.asarray(branch_list), min_branch


    def tight_rm_branch(self, smi_l, smi_r):
        new_smi = smi_l + smi_r

        close_branch = self.get_close_branch(new_smi)

        b = None
        n_branch = self.chk_branch(new_smi)

        q = len(smi_l)
        while n_branch > 0:  # over opened-branch
            smi_l_open_branch = self.get_open_branch(smi_l)
            smi_r_open_branch = self.get_open_branch(smi_r)
            avoid_tokens = [
                i for i, e in enumerate(smi_l + smi_r)
                if e in ["=", "#", "@", "1", "2", "3", "4", "5", "6", "7", "8"]
            ]

            if len(smi_r_open_branch) == 0:  # If there is no open branch
                smi_r_open_branch.append(len(smi_r))
            if len(smi_l_open_branch) == 0:
                smi_l_open_branch.append(0)

            if np.random.rand() > 0.5:  # Add
                branch_gate = False
                j = 0
                while not branch_gate:  # Cur the ring
                    if j == self.limit:
                        raise ValueError
                    b = np.random.randint(smi_l_open_branch[-1] + 1,
                                        smi_r_open_branch[-1] + q)
                    j += 1
                    if b not in avoid_tokens:
                        branch_gate = True
                n_branch -= 1
                if b <= len(smi_l):
                    smi_l = smi_l[:b] + ")" + smi_l[b:]
                    q += 1
                else: 
                    b -= len(smi_l)
                    smi_r = smi_r[:b] + ")" + smi_r[b:]
            else:  # Remove
                b = smi_l_open_branch[-1] 
                n_branch -= 1
                q -= 1
                smi_l = smi_l[:b] + smi_l[b + 1:]

        while n_branch < 0:  # over closed-branch
            smi_l_close_branch = self.get_close_branch(smi_l)
            smi_r_close_branch = self.get_close_branch(smi_r)
            close_branch = self.get_close_branch(smi_l + smi_r)
            avoid_tokens = [
                i for i, e in enumerate(smi_l + smi_r)
                if e in ["=", "#", "@", "1", "2", "3", "4", "5", "6", "7", "8"]
            ]

            if len(smi_r_close_branch) == 0:
                smi_r_close_branch.append(len(smi_r))
            if len(smi_l_close_branch) == 0:
                smi_l_close_branch.append(0)

            n = np.random.rand()
            if n > 0.5:
                branch_gate = False
                j = 0
                while not branch_gate:  # Cut avoiding ring part
                    b = np.random.randint(smi_l_close_branch[-1] + 1,
                                        smi_r_close_branch[0] + q + 1)
                    j += 1
                    if b not in (close_branch + avoid_tokens):
                        branch_gate = True
                    if j == self.limit:
                        raise ValueError
                n_branch += 1
                if b < len(smi_l):
                    smi_l = smi_l[:b] + "(" + smi_l[b:]
                    q += 1
                else:
                    b -= len(smi_l)
                    smi_r = smi_r[:b] + "(" + smi_r[b:]
            else:
                b = smi_r_close_branch[0]
                n_branch += 1
                smi_r = smi_r[:b] + smi_r[b + 1:]

        return smi_l + smi_r
    
    def get_sliced_smiles(self, smi1:str, smi2:str, ring_bool: bool):
        l_smi = None
        r_smi = None

        gate = 0
        while not(l_smi and r_smi):
            gate += 1
            if gate > 10:
                raise PermissionError
            try:
                l_smi = self.cut_smiles(smi1, "L", ring_bool, 4)
                r_smi = self.cut_smiles(smi2, "R", ring_bool, 4)
            except:
                pass

        return l_smi, r_smi
    
    def get_rings(self, smiles):
        """
        Returns the positions of the components that are into a ring. 
        Rings are identified as integers in smiles, so, this methods identify the integers and returns
        positions between them.
        """
        avoid_ring = []
        ring_tmp = set(re.findall(r"\d", smiles)) #identify numbers that indicate rings
        for j in ring_tmp:
            tmp = [i for i, val in enumerate(smiles) if val == j]
            while tmp:
                avoid_ring += [j for j in range(tmp.pop(0), tmp.pop(0) + 1)]
        return set(avoid_ring)

    
    def cut_smiles(self, smiles, side, avoid_ring=True, minimum_len=4):
        """ 1 point crossover
        :param smiles: SMILES (str)
        :param side: Left SMILES or Right SMILES ['L'|'R'] (str)
        :param avoid_ring: avoid ring (bool)
        :param minimum_len: minimum cut size (int)
        :return:
        """

        smiles_len = len(smiles)
        smi = None

        if avoid_ring:  # cross rings ?
            avoid_ring_list = self.get_rings(smiles)

        p = 0
        start = None
        end = None
        gate = False
        while not gate:  # repeat until SMILES is correct.
            if p == self.limit:
                raise ValueError(f"main_gate fail ({side}): {smiles}")

            if avoid_ring: #avoid consider crossing rings.
                j = 0
                ring_gate = False
                if side == "L":
                    while not ring_gate: #iterates 30 times, if end is not found, raises an exception.
                        if j == 30:
                            raise ValueError(f"ring_gate fail (L): {smiles}")
                        end = np.random.randint(minimum_len, smiles_len + 1)
                        if end not in avoid_ring_list:
                            ring_gate = True
                        j += 1
                elif side == "R":
                    while not ring_gate: #iterates 30 times, if end is not found, raises an exception.
                        if j == 30:
                            raise ValueError(f"ring_gate fail (R): {smiles}")
                        start = np.random.randint(0, smiles_len - minimum_len)
                        if start not in avoid_ring_list:
                            ring_gate = True
                        j += 1
                smi = smiles[start:end]
            else:
                if side == "L":
                    end = np.random.randint(minimum_len, smiles_len)
                elif side == "R":
                    start = np.random.randint(0, smiles_len - minimum_len)

                smi = smiles[start:end]
                chk_ring = re.findall(r"\d", smi)
                i = 0
                for i in set(chk_ring):
                    list_ring = [_ for _, val in enumerate(smi) if val == i]
                    if (len(list_ring) % 2) == 1:
                        b = random.sample(list_ring, 1)
                        smi = smi[:b[0]] + smi[b[0] + 1:]

            p += 1

            if "." in smi:  # desconected structures
                continue

            n_chk = 0
            for j in smi:  # Check brackets closieng
                if j == "[":
                    n_chk += 1
                if j == "]":
                    n_chk -= 1
            if n_chk == 0:
                gate = True

        return smi
    
    def get_number_of_parents(self) -> int:
        return 2
    
    def get_number_of_children(self) -> int:
        return 1
    
    def get_name(self) -> str:
        return "SMILES Crossover"