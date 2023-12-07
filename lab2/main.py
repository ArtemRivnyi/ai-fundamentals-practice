import random
import math

def func(x, y, z):
    return 1 / (1 + (x - 2) ** 2 + (y + 1) ** 2 + (z - 1) ** 2)

chrom_len = 3
pop_size = 100
gen_num = 100
interval = (-10, 10)
mut_prob = 0.01
def random_ind():
    return [random.uniform(interval[0], interval[1]) for _ in range(chrom_len)]
def init_pop():
    return [random_ind() for _ in range(pop_size)]
def fitness(ind):
    return func(*ind)
def total_fitness(pop):
    return sum(fitness(ind) for ind in pop)
def roulette(pop):
    total = total_fitness(pop)
    r = random.uniform(0, total)
    s = 0
    for ind in pop:
        s += fitness(ind)
        if s >= r:
            return ind
def crossover(ind1, ind2):
    point = random.randint(1, chrom_len - 1)
    ind1[point:], ind2[point:] = ind2[point:], ind1[point:]
    return ind1, ind2
def mutate(ind):
    for i in range(chrom_len):
        if random.random() < mut_prob:
            ind[i] = random.uniform(interval[0], interval[1])
    return ind
def evolve(pop):
    new_pop = []
    best_ind = max(pop, key=fitness)
    new_pop.append(best_ind)
    while len(new_pop) < len(pop):
        ind1 = roulette(pop)
        ind2 = roulette(pop)
        ind1, ind2 = crossover(ind1, ind2)
        ind1 = mutate(ind1)
        ind2 = mutate(ind2)
        new_pop.append(ind1)
        new_pop.append(ind2)
    return new_pop
pop = init_pop()
for gen in range(gen_num):
    pop = evolve(pop)
    best_ind = max(pop, key=fitness)
    print(f"Покоління {gen + 1}: {best_ind}, {fitness(best_ind)}")
print(f"Максимум функції: {fitness(best_ind)}")
print(f"Значення x, y, z: {best_ind}")
