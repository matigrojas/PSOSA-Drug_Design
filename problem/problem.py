from abc import ABC, abstractmethod

from typing import List


class Problem(ABC):

    def __init__(self) -> None:
        self.number_of_objectives: int = 0
        self.number_of_contraints: int = 0

        self.labels: List[str] = []

    @abstractmethod
    def create_solution(self):
        """
        Creates a candidate solution to the problem
        :return: a solution.
        """
        pass

    @abstractmethod
    def evaluate(self, solution):
        """
        Evaluate how fit is the solution to the problem being solved
        :return: a solution.
        """
        pass

    def get_name(self) -> str:
        """
        :return: the name of the problem being treated.
        """
        pass