from abc import ABC, abstractmethod
from typing import List
import copy
import random
import numpy as np
import re

from solution.solution import Solution

from rdkit import Chem
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')

class Mutation(ABC):

    def __init__(self, probability: float):
        self.probability = probability

    @abstractmethod
    def execute(self, solution):
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

class AddMutation(Mutation):

    def __init__(self, probability: float):
        super(AddMutation,self).__init__(probability)
    
    def execute(self, solution: Solution):
        mutated_solution = copy.deepcopy(solution)
        try:
            mutated_solution.variables, mol = self.add_atom(mutated_solution.variables)
        except (PermissionError, Chem.rdchem.KekulizeException):
            mol = mol

        return mutated_solution, mol
    
    def add_atom(self, _smi):
        list_atom = ["C", "B", "N", "P", "O", "S", "Cl", "Br"]

        max_len = len(_smi)
        mol_ = False
        _new_smi = None

        p = 0
        while not mol_:
            p += 1
            if p == 30:
                # raise Exception
                raise PermissionError

            rnd_insert = np.random.randint(max_len)
            _new_smi = _smi[:rnd_insert] + random.sample(list_atom,1)[0] + _smi[rnd_insert:]
            mol_ = Chem.MolFromSmiles(_new_smi)

        return _new_smi, mol_
    
    def get_name(self) -> str:
        return "Add Mutation"
    
class RemoveMutation(Mutation):

    def __init__(self, probability: float):
        super(RemoveMutation,self).__init__(probability)
    
    def execute(self, solution: Solution):
        mutated_solution = copy.deepcopy(solution)
        try:
            mutated_solution.variables, mol = self.delete_atom(mutated_solution.variables)
        except (PermissionError, Chem.rdchem.KekulizeException):
            mol = None

        return mutated_solution, mol
    
    def delete_atom(self, _smi):
        """
        Capable of removing each atom, except for those in aromatic ring.
        :param _smi:
        :return: the first valid mol and smiles, obtained after removing an atom.
        """
        max_len = len(_smi)
        mol_ = False
        _new_smi = None

        p = 0
        while not mol_:
            p += 1
            if p == 30:
                # raise Exception
                raise PermissionError

            rnd_insert = np.random.randint(max_len)
            _new_smi = _smi[:rnd_insert] + _smi[rnd_insert + 1:]
            mol_ = Chem.MolFromSmiles(_new_smi)

        return _new_smi, mol_
    
    def get_name(self) -> str:
        return "Remove Mutation"
    
class ReplaceMutation(Mutation):

    def __init__(self, probability: float):
        super(ReplaceMutation, self).__init__(probability)

    def execute(self, solution: Solution):
        mutated_solution = copy.deepcopy(solution)
        for _ in range(5):
            try:
                new_smile, mol = self.replace_atom(mutated_solution.variables)
                if mol:
                    mutated_solution.variables = new_smile
            except (PermissionError, Chem.rdchem.KekulizeException):
                mol = None

        return mutated_solution, mol
    
    def replace_atom(self,smi):

        #                    C /B  N  P / O  S / F  Cl  Br  I
        replace_atom_list = [6, 5, 7, 15, 8, 16, 9, 17, 35, 53]
        #                         C  N  P / O  S
        replace_arom_atom_list = [6, 7, 15, 8, 16]

        # print(f"before: {smi}")

        mol_ = Chem.MolFromSmiles(smi)
        max_len = mol_.GetNumAtoms()

        mw = Chem.RWMol(mol_)
        # Chem.SanitizeMol(mw)

        p = 0
        gate_ = False
        while not gate_:
            if p == 30:
                # raise Exception
                raise PermissionError

            rnd_atom = np.random.randint(0, max_len)

            valence = mw.GetAtomWithIdx(rnd_atom).GetExplicitValence()
            if mw.GetAtomWithIdx(rnd_atom).GetIsAromatic():
                if valence == 3:
                    mw.ReplaceAtom(
                        rnd_atom,
                        Chem.Atom(replace_arom_atom_list[np.random.randint(0, 3)]))
                elif valence == 2:
                    mw.ReplaceAtom(
                        rnd_atom,
                        Chem.Atom(replace_arom_atom_list[np.random.randint(1, 5)]))
                else:
                    continue
                mw.GetAtomWithIdx(rnd_atom).SetIsAromatic(True)
            else:
                if valence == 4:
                    mw.ReplaceAtom(
                        rnd_atom,
                        Chem.Atom(replace_atom_list[np.random.randint(0, 1)]))
                elif valence == 3:
                    mw.ReplaceAtom(
                        rnd_atom,
                        Chem.Atom(replace_atom_list[np.random.randint(0, 4)]))
                elif valence == 2:
                    mw.ReplaceAtom(
                        rnd_atom,
                        Chem.Atom(replace_atom_list[np.random.randint(0, 6)]))
                elif valence == 1:
                    mw.ReplaceAtom(
                        rnd_atom,
                        Chem.Atom(replace_atom_list[np.random.randint(0, 10)]))

            p += 1
            # print(f"after: {Chem.MolToSmiles(mw)}")
            try:
                Chem.SanitizeMol(mw)
                gate_ = True
            except Chem.rdchem.KekulizeException:
                pass

        Chem.Kekulize(mw)

        return Chem.MolToSmiles(mw, kekuleSmiles=True), mw
    
    def get_name(self) -> str:
        return "Replace Mutation"
    
