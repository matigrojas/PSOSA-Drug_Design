import time
from abc import abstractmethod, ABC

class Algorithm(ABC):

    def __init__(self) -> None:

        self.solutions = []
        self.evaluations = 0
        self.start_computing_time = 0
        self.total_computing_time = 0


    @abstractmethod
    def create_initial_solutions(self):
        """ Creates the initial list of solutions of a metaheuristic. """
        pass

    @abstractmethod
    def evaluate(self, solution_list):
        """ Evaluates a solution list. """
        pass

    @abstractmethod
    def init_progress(self) -> None:
        """ Initialize the algorithm. """
        pass

    @abstractmethod
    def stopping_condition_is_met(self) -> bool:
        """ The stopping condition is met or not. """
        pass

    @abstractmethod
    def step(self) -> None:
        """ Performs one iteration/step of the algorithm's loop. """
        pass

    @abstractmethod
    def update_progress(self) -> None:
        """ Update the progress after each iteration. """
        pass

    def run(self):
        """ Execute the algorithm. """
        self.start_computing_time = time.time()

        self.solutions = self.create_initial_solutions()
        self.solutions = self.evaluate(self.solutions)

        self.init_progress()

        while not self.stopping_condition_is_met():
            self.step()
            self.update_progress()

        self.total_computing_time = time.time() - self.start_computing_time

    @abstractmethod
    def get_result(self):
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass