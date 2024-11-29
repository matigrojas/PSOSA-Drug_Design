import random
from solution.solution import Solution

from typing import List

class RouletteWheelSelection():
    """Performs roulette wheel selection.
    """

    def __init__(self):
        pass

    def execute(self, solutions: List[Solution]) -> Solution:
        if solutions is None:
            raise Exception('The solutions is null')
        elif len(solutions) == 0:
            raise Exception('The solutions is empty')

        maximum = sum([abs(solution.objectives[0]) for solution in solutions])
        rand = random.uniform(0.0, maximum)
        value = 0.0

        for solution in solutions:
            value += abs(solution.objectives[0])

            if value > rand:
                return solution

        return None

    def get_name(self) -> str:
        return 'Roulette wheel selection'
    
class BinaryTournamentSelection():

    def __init__(self):
        pass

    def execute(self, solutions: List[Solution]) -> Solution:
        if solutions is None:
            raise Exception('The solutions is null')
        elif len(solutions) == 0:
            raise Exception('The solutions is empty')

        if len(solutions) == 1:
            result = solutions[0]
        else:
            # Sampling without replacement
            i, j = random.sample(range(0, len(solutions)), 2)
            solution1 = solutions[i]
            solution2 = solutions[j]

            if solution1.objectives[0] < solution2.objectives[0]:
                result = solution1
            elif solution2.objectives[0] < solution1.objectives[0]:
                result = solution2
            else:
                result = [solution1, solution2][random.random() < 0.5]

        return result

    def get_name(self) -> str:
        return 'Binary tournament selection'
    

class DifferentialEvolutionSelection():

    def __init__(self):
        super(DifferentialEvolutionSelection, self).__init__()
        self.index_to_exclude = None

    def execute(self, front: List[Solution]) -> List[Solution]:
        if front is None:
            raise Exception('The front is null')
        elif len(front) == 0:
            raise Exception('The front is empty')
        elif len(front) < 4:
            raise Exception('The front has less than four solutions: ' + str(len(front)))

        selected_indexes = random.sample([i for i in range(len(front)) if i != self.index_to_exclude],3)

        return [front[i] for i in selected_indexes]

    def set_index_to_exclude(self, index: int):
        self.index_to_exclude = index

    def get_name(self) -> str:
        return "Differential evolution selection"