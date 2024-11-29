from abc import ABC


class Solution(ABC):

    def __init__(self, number_of_objectives: int, number_of_constraints: int = 0)-> None:
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = number_of_constraints

        self.variables = []
        self.objectives = [0.0 for _ in range(self.number_of_objectives)]
        self.constraints = [0.0 for _ in range(self.number_of_constraints)]
        self.attributes = {}

    def __eq__(self, solution) -> bool:
        if isinstance(solution, self.__class__):
            return self.variables == solution.variables
        return False
    
    def __str__(self) -> str:
        return f'Solution(variables={self.variables}, objectives={self.objectives}, constraints={self.constraints})'