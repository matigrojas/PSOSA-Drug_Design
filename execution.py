from problem.diabetes_problem import DiabetesDocking
from algorithm.psosa import ParticleSwarmSA
from operators.mutation import AddMutation, RemoveMutation, ReplaceMutation
from operators.crossover import SmilesCrossover

problem = DiabetesDocking(bank_dir='./data/banks_papyrus/bank_1.csv')

algorithm = ParticleSwarmSA(
    problem=problem,
    swarm_size=30,
    add_mutation=AddMutation(1.),
    remove_mutation=RemoveMutation(1.),
    replace_mutation=ReplaceMutation(1.),
    crossover=SmilesCrossover(1.),
    max_evaluations=10000,
    save_smiles_dir='./results/pruebaPSOSA_dock_2.csv',
)

algorithm.run()

print(f'Best Fitness: {algorithm.global_best.objectives[0]}')
print(f'Best SMILE found: {algorithm.global_best.variables}')
print(f'Time elapsed: {algorithm.total_computing_time}')