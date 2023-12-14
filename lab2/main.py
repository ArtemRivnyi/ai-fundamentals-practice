import random
import numpy as np
from deap import base, creator, tools, algorithms

# Функція оцінки
def eval_func(chromosome):
    x, y, z = chromosome
    return 1 / (1 + (x-2)**2 + (y+1)**2 + (z-1)**2),

# Генетичний алгоритм
def genetic_algorithm():
    # Створення типів
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Ініціалізація популяції
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -10, 10)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Реєстрація генетичних операторів
    toolbox.register("evaluate", eval_func)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Параметри генетичного алгоритму
    pop_size = 100
    crossover_prob = 0.7
    mutation_prob = 0.01
    num_generations = 100

    # Створення початкової популяції
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    # Запуск генетичного алгоритму
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=crossover_prob, mutpb=mutation_prob, ngen=num_generations, stats=stats, halloffame=hof, verbose=True)

    return hof[0]

# Виклик генетичного алгоритму
best_individual = genetic_algorithm()
print(f"Best individual: {best_individual}, Fitness: {eval_func(best_individual)}")
