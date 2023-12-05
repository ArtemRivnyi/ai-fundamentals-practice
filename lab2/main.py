# імпортуємо необхідні модулі
import random
import math

# визначаємо функцію, яку хочемо максимізувати
def func(x, y, z):
    return 1 / (1 + (x - 2) ** 2 + (y + 1) ** 2 + (z - 1) ** 2)

# визначаємо довжину хромосоми
chrom_len = 3

# визначаємо кількість особин в популяції
pop_size = 100

# визначаємо кількість поколінь
gen_num = 100

# визначаємо інтервал, на якому шукаємо розв'язок
interval = (-10, 10)

# визначаємо ймовірність мутації
mut_prob = 0.01

# визначаємо функцію, яка генерує випадкову особину
def random_ind():
    return [random.uniform(interval[0], interval[1]) for _ in range(chrom_len)]

# визначаємо функцію, яка генерує початкову популяцію
def init_pop():
    return [random_ind() for _ in range(pop_size)]

# визначаємо функцію, яка обчислює пристосованість особини
def fitness(ind):
    return func(*ind)

# визначаємо функцію, яка обчислює сумарну пристосованість популяції
def total_fitness(pop):
    return sum(fitness(ind) for ind in pop)

# визначаємо функцію, яка вибирає особину за методом рулетки
def roulette(pop):
    # обчислюємо сумарну пристосованість популяції
    total = total_fitness(pop)
    # генеруємо випадкове число від 0 до сумарної пристосованості
    r = random.uniform(0, total)
    # ініціалізуємо поточну суму
    s = 0
    # проходимо по всіх особинах популяції
    for ind in pop:
        # додаємо пристосованість поточної особини до поточної суми
        s += fitness(ind)
        # якщо поточна сума більша або рівна випадковому числу
        if s >= r:
            # повертаємо поточну особину
            return ind

# визначаємо функцію, яка виконує одноточкове схрещування двох особин
def crossover(ind1, ind2):
    # вибираємо випадкову точку схрещування
    point = random.randint(1, chrom_len - 1)
    # обмінюємо частини хромосом між особинами
    ind1[point:], ind2[point:] = ind2[point:], ind1[point:]
    # повертаємо нових особин
    return ind1, ind2

# визначаємо функцію, яка виконує мутацію особини
def mutate(ind):
    # проходимо по всіх генах хромосоми
    for i in range(chrom_len):
        # з ймовірністю mut_prob
        if random.random() < mut_prob:
            # змінюємо ген на випадкове значення з інтервалу
            ind[i] = random.uniform(interval[0], interval[1])
    # повертаємо змутовану особину
    return ind

# визначаємо функцію, яка виконує одне покоління генетичного алгоритму
def evolve(pop):
    # створюємо нову популяцію
    new_pop = []
    # додаємо до нової популяції найкращу особину з поточної популяції
    best_ind = max(pop, key=fitness)
    new_pop.append(best_ind)
    # поки розмір нової популяції менший за розмір поточної популяції
    while len(new_pop) < len(pop):
        # вибираємо дві особини за методом рулетки
        ind1 = roulette(pop)
        ind2 = roulette(pop)
        # виконуємо схрещування між ними
        ind1, ind2 = crossover(ind1, ind2)
        # виконуємо мутацію для кожної з них
        ind1 = mutate(ind1)
        ind2 = mutate(ind2)
        # додаємо їх до нової популяції
        new_pop.append(ind1)
        new_pop.append(ind2)
    # повертаємо нову популяцію
    return new_pop

# генеруємо початкову популяцію
pop = init_pop()

# проходимо по всіх поколіннях
for gen in range(gen_num):
    # виконуємо еволюцію популяції
    pop = evolve(pop)
    # знаходимо найкращу особину в популяції
    best_ind = max(pop, key=fitness)
    # виводимо номер покоління, найкращу особину і її пристосованість
    print(f"Покоління {gen + 1}: {best_ind}, {fitness(best_ind)}")

# виводимо кінцевий результат
print(f"Максимум функції: {fitness(best_ind)}")
print(f"Значення x, y, z: {best_ind}")
