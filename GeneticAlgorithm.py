import math
import numpy as np
import random
import matplotlib.pyplot as plt


def xor_c(a, b):
    return '0' if (a == b) else '1'


# Helper function to flip the bit
def flip(c):
    return '1' if (c == '0') else '0'


def binarytoGray(binary):
    gray = "0b"  # to start with 0b

    # MSB of gray code is same as
    # binary code
    gray += binary[2]

    # Compute remaining bits, next bit
    # is computed by doing XOR of previous
    # and current in Binary
    for i in range(3, len(binary)):
        # Concatenate XOR of previous
        # bit with current bit
        gray += xor_c(binary[i - 1],
                      binary[i])

    return gray


# function to convert gray code
# string to binary string
def graytoBinary(gray):
    binary = "0b"  # binary start with these characters

    # MSB of binary code is same
    # as gray code
    binary += gray[2]  # start from the character after '0b' in the gray code

    # Compute remaining bits
    for i in range(3, len(gray)):

        # If current bit is 0,
        # concatenate previous bit
        if gray[i] == '0':
            binary += binary[i - 1]

        # Else, concatenate invert
        # of previous bit
        else:
            binary += flip(binary[i - 1])

    return binary


def add_missing_zeros(member):  # to make all members have the same size (length = 10) that is '0b' + 8 binary bits
    missing_number_of_zeros = 10 - len(member)  # 0b00000100 initially 0b100 so we need 5 zeros after 0b
    missing_zeros = "0" * missing_number_of_zeros
    member = member[0:2] + missing_zeros + member[2:]
    return member


def convert_to_8_bits(pop):
    pop = list(map(add_missing_zeros, pop))  # for each member, add the missing zeros to make it in format 0bxxxxxxxx
    return pop


def initialize_new_population(function_num, pop_size=8, gray_coded=True):
    if function_num == 1:
        # initialize population with population size with 0 <= members <=255
        pop = random.sample(range(0, 255), pop_size)
        print(pop)
        pop = list(map(bin, pop))
        pop = convert_to_8_bits(pop)
        if gray_coded:
            pop = list(map(binarytoGray, pop))

    elif function_num == 2:
        # initialize numpy array of size (pop_size, 2) with values from -5,5
        pop = np.random.uniform(low=-5, high=5, size=(pop_size, 2))

    return pop


def calculate_fitness_for_population(function_num, pop, gray_coded=True):
    if function_num == 1:
        def first_objective(x):
            return math.sin((math.pi * int(x, 2)) / 256)

        if gray_coded:
            # if the population is gray coded, convert them to binary before calculating fitness
            pop = list(map(graytoBinary, pop))

        return list(map(first_objective, pop))  # calculate fitness
    elif function_num == 2:
        def second_objective(x):
            # Adding negative to the original objective function as we want to compute its global minimum
            # which is the same as computing global maximum of the negative of the function
            return -((x[0] - 3.14) ** 2 + (x[1] - 2.72) ** 2 + np.sin(3 * x[0] + 1.41) + np.sin(4 * x[1] - 1.73))

        return list(map(second_objective, pop))


def select_mating_pool(pop, fitness, num_parents=4):
    # Select the best individuals as parents for producing the offspring of the next generation.
    parents_array = []
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))  # find idx of the max
        max_fitness_idx = max_fitness_idx[0]
        parents_array.append(pop[max_fitness_idx[0]])  # append the max as a parent
        fitness[max_fitness_idx[0]] = -99999  # give it a small value to not choose it again
    return np.array(parents_array)


def crossover_fifty_percent(function_num, parents, offspring_size=4):
    offsprings = []
    crossover_point = np.uint8(len(parents[0]) / 2)
    if function_num == 1:
        for k in range(0, offspring_size, 2):
            # first half of the first member with second half of the second member
            offspring1 = "0b" + parents[k][2:crossover_point + 1] + parents[k + 1][
                                                                    crossover_point + 1:crossover_point * 2 + 2]
            # first half of the second member with first half of the first member
            offspring2 = "0b" + parents[k + 1][2:crossover_point + 1] + parents[k][
                                                                        crossover_point + 1:crossover_point * 2 + 2]
            offsprings.append(offspring1)
            offsprings.append(offspring2)
    elif function_num == 2:
        for k in range(0, offspring_size, 2):
            # first half of the first member with second half of the second member
            offspring1 = np.empty(2)
            offspring2 = np.empty(2)
            offspring1[0] = parents[k][0]
            offspring1[1] = parents[k + 1][1]
            offspring2[0] = parents[k + 1][0]
            offspring2[1] = parents[k][1]
            offsprings.append(offspring1)
            offsprings.append(offspring2)
    return np.array(offsprings)


def crossover_one_point(parents, offspring_size=4):
    offsprings = []
    member_length = len(parents[0])
    for k in range(0, offspring_size, 2):
        crossover_point = random.sample(range(0, 8), 1)[0]
        print(crossover_point)
        # first half of the first member with second half of the second member
        offspring1 = "0b" + parents[k][2:crossover_point + 2] + parents[k + 1][crossover_point + 2:member_length]
        # first half of the second member with first half of the first member
        offspring2 = "0b" + parents[k + 1][2:crossover_point + 2] + parents[k][crossover_point + 2:member_length]
        offsprings.append(offspring1)
        offsprings.append(offspring2)
    return np.array(offsprings)


def repair(function_num, pop, gray_coded=True):
    def first_function_repair(x, gray_coded=True):
        if gray_coded:
            x = graytoBinary(x)
        x_integer = int(x, 2)
        if x_integer > 255:
            return binarytoGray(bin(255)) if gray_coded else bin(255)
        elif x_integer < 0:
            return add_missing_zeros(binarytoGray(bin(0))) if gray_coded else add_missing_zeros(bin(0))
        return binarytoGray(x) if gray_coded else x

    def second_function_repair(x):
        #  if x or y are out of range then bring them back to the range(-5,5)
        for i in range(x.shape[0]):
            if x[i] > 5:
                x[i] = 5
            elif x[i] < -5:
                x[i] = -5
        return x
    if function_num == 1:
        repaired = list(map(lambda p: first_function_repair(p, gray_coded), pop))
    elif function_num == 2:
        repaired = list(map(second_function_repair, pop))
    return repaired



def mutation(function_num, pop, probability):
    # Doing the mutation if the probability of tossing succeed by getting value of  out of range of values
    # ranging from 1 to 1/probability for both functions
    if function_num == 1:
        for i in range(len(pop)):
            for j in range(2, len(pop[i])):
                toss = random.sample(range(1, int(1 / probability)), 1)[0]
                if toss == 1:
                    new_value = '1' if pop[i][j] == '0' else '0'
                    member = list(pop[i])
                    member[j] = new_value
                    pop[i] = pop[i][0:j] + new_value + pop[i][j + 1:]
    elif function_num == 2:
        for i in range(len(pop)):
            for j in range(len(pop[i])):
                toss = random.sample(range(1, int(1 / probability)), 1)[0]
                if toss == 1:
                    pop[i][j] = -5 + (random.random() * 10)
    return pop


def binaryPopulationToInt(pop, gray_coded=True):
    if gray_coded:
        pop = list(map(graytoBinary, pop))

    def binary_to_int(binary):
        return int(binary, 2)

    return list(map(binary_to_int, pop))


for function_num in range(1, 3):
    # applying GA for objective function 1 and 2
        generations = range(0, 200)
        pop = new_pop_after_mutation = initialize_new_population(function_num, gray_coded=True) #gray_coded is only for first objectiv function
        # note that here new_pop_after_mutation is also initialized to ease printing in the first generation

        best_fitnesses = []
        for i in generations:
            print("Generation Number:", i)
            print("Population:", pop)
            fitness = calculate_fitness_for_population(function_num, pop, gray_coded=True)
            print("Best Fitness:", np.max(fitness))
            if function_num == 1:
                best_fitnesses.append(np.max(fitness))
            elif function_num == 2:
                # for second objective function we take the negative as we want to calculate global minimum not maximum
                best_fitnesses.append(-1*np.max(fitness))
            parents = select_mating_pool(pop, fitness, 4)
            print("Parents:", parents)
            # parents_int = binaryPopulationToInt(parents, gray_coded=True)
            # print("ParentsInteger:", parents_int)
            children = crossover_fifty_percent(function_num, parents)
            print("Children:", children)
            new_pop_after_crossover = np.concatenate((parents, children))
            print("New Population After Cross over:", new_pop_after_crossover)
            # new_pop_after_crossover_int = binaryPopulationToInt(new_pop_after_crossover, gray_coded=True)
            # print("New Population After Crossover Integer:", new_pop_after_crossover_int)
            new_pop_after_mutation = mutation(function_num, new_pop_after_crossover, 0.01)
            print("New Population After Mutation:", new_pop_after_mutation)
            # new_pop_after_mutation_int = binaryPopulationToInt(new_pop_after_mutation, gray_coded=True)
            # print("New Population After Mutation Integer:", new_pop_after_mutation_int)

            # repair after mutation
            pop = repair(function_num, new_pop_after_mutation , gray_coded=True)#gray_coded is only for first objectiv function

        plt.title("Fitness of function number : {}".format(function_num))
        plt.plot(generations, best_fitnesses, color="red")
        plt.show()
