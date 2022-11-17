import math
import numpy as np
import random
import matplotlib.pyplot as plt





def xor_c(a, b):
    return '0' if(a == b) else '1';
 
# Helper function to flip the bit
def flip(c):
    return '1' if(c == '0') else '0';

def binarytoGray(binary):
    gray = "0b"; #to start with 0b
 
    # MSB of gray code is same as
    # binary code
    gray += binary[2];
 
    # Compute remaining bits, next bit
    # is computed by doing XOR of previous
    # and current in Binary
    for i in range(3, len(binary)):
         
        # Concatenate XOR of previous
        # bit with current bit
        gray += xor_c(binary[i - 1],
                      binary[i]);
 
    return gray;
 
# function to convert gray code
# string to binary string
def graytoBinary(gray):
 
    binary = "0b"; #binary start with these characters
 
    # MSB of binary code is same
    # as gray code
    binary += gray[2]; #start from the character after '0b' in the gray code
 
    # Compute remaining bits
    for i in range(3, len(gray)):
         
        # If current bit is 0,
        # concatenate previous bit
        if (gray[i] == '0'):
            binary += binary[i - 1];
 
        # Else, concatenate invert
        # of previous bit
        else:
            binary += flip(binary[i - 1]);
 
    return binary;

def initialize_new_population(functionNum, pop_size=8, gray_coding=True):
    if (functionNum==1):
        def convert_to_8_bits(pop):
            def add_missing_zeros(member): #to make all members have the same size (length = 10) that is '0b' + 8 binary bits
                missing_number_of_zeros = 10 - len(member) #0b00000100 initially 0b100 so we need 5 zeros after 0b
                missing_zeros = "0"*missing_number_of_zeros
                member = member[0:2] + missing_zeros + member[2:]
                return member
            pop = list(map(add_missing_zeros,pop)) #for each member, add the missing zeros to make it in format 0bxxxxxxxx
            return pop
        pop = random.sample(range(0, 255), pop_size) #initialize population with population size with 0 <= members <=255 
        print(pop)
        pop = list(map(bin,pop))
        pop = convert_to_8_bits(pop) 
        if gray_coding:
            pop = list(map(binarytoGray,pop)) 
    elif (functionNum==2):
        pop = np.random.uniform(low=-5, high=5, size=(pop_size,2))
        
    return pop


def calculate_fitness_for_population(functionNum, pop, gray_coded=True):
    if(functionNum==1):
        def first_objective(x):
            return math.sin((math.pi * int(x,2))/ 256)
        if gray_coded:
            pop = list(map(graytoBinary, pop)) #if the population is graycoded, convert them to binary before calculating fitness
        return list(map(first_objective, pop)) #calculate fitness 
    elif (functionNum==2):
         def second_objective(x):
             return -((x[0]-3.14)**2+(x[1]-2.72)**2+np.sin(3*x[0]+1.41)+np.sin(4*x[1]-1.73))
         return list(map(second_objective, pop))

def select_mating_pool(pop, fitness, num_parents=4):
# Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents_array =[]
    for parent_num in range(num_parents):   
        max_fitness_idx = np.where(fitness == np.max(fitness))#find idx of the max
        max_fitness_idx = max_fitness_idx[0]
        parents_array.append(pop[max_fitness_idx[0]])#append the max as a parent
        fitness[max_fitness_idx[0]] = -99999#give it a small value to not choose it again
    return np.array(parents_array)



def crossover_fifty_precent(functionNum, parents, offspring_size=4):
 offsprings = []
 crossover_point = np.uint8(len(parents[0])/2)
 if(functionNum==1):
     for k in range(0,offspring_size,2):
         #first half of the first member with second half of the second member
         offspring1 = "0b" +  parents[k][2:crossover_point+1] + parents[k+1][crossover_point+1:crossover_point*2+2]
         #first half of the second member with first half of the first member
         offspring2 = "0b"  + parents[k+1][2:crossover_point+1] + parents[k][crossover_point+1:crossover_point*2+2]
         offsprings.append(offspring1)
         offsprings.append(offspring2)
 elif(functionNum==2):
     for k in range(0,offspring_size,2):
         #first half of the first member with second half of the second member
         offspring1=np.empty(2)
         offspring2=np.empty(2)
         offspring1[0] = parents[k][0]
         offspring1[1] = parents[k+1][1]
         offspring2[0] = parents[k+1][0]
         offspring2[1] = parents[k][1]
         offsprings.append(offspring1)
         offsprings.append(offspring2)
    
 return np.array(offsprings)
    
def crossover_one_point(parents,offspring_size=4):
    offsprings = []
    member_length = len(parents[0])
    for k in range(0,offspring_size,2):
        crossover_point = random.sample(range(0, 8), 1)[0]
        print(crossover_point)
        #first half of the first member with second half of the second member
        offspring1 = "0b" +  parents[k][2:crossover_point+2] + parents[k+1][crossover_point+2:member_length]
        #first half of the second member with first half of the first member
        offspring2 = "0b" + parents[k+1][2:crossover_point+2] + parents[k][crossover_point+2:member_length]
        offsprings.append(offspring1)
        offsprings.append(offspring2)
    return np.array(offsprings)

def repair(x, min=0, max=255):
    x_integer = int(graytoBinary(x),2)
    if x_integer>255: 
        return binarytoGray(bin(255))
    #not checking if x is less than 0 because a binary representation is assumed to be always positive
    return x

def mutation(functionNum, pop, probability):
    #doing mutation with 1% probability (can be changed)
    if(functionNum==1):     
        for i in range(len(pop)):
            for j in range(2,len(pop[i])):
                toss = random.sample(range(1,int(1/probability)), 1)[0]
                if toss==1:
                    new_value = '1' if pop[i][j]=='0' else '0'
                    member = list(pop[i])
                    member[j] = new_value
                    pop[i] = pop[i][0:j] + new_value + pop[i][j+1:]
    elif(functionNum==2):            
        for i in range(len(pop)):
            for j in range(len(pop[i])):
                toss = random.sample(range(1,int(1/probability)), 1)[0]
                if toss==1:          
                    pop[i][j] = -5+(random.random()*10)            
    return pop



generations = range(0,50)

seond_function_pop = new_pop_after_mutation = initialize_new_population(2)
# note that here new_pop_after_mutation is also initialized to ease printing in the first generation

best_fitnesses = []
for i in generations:
    print("Generation Number:", i)
    print("Population:", seond_function_pop)
    fitness = calculate_fitness_for_population(2,new_pop_after_mutation)
    print("Best Fitness:", np.max(fitness))
    best_fitnesses.append(np.max(fitness))
    parents = select_mating_pool(seond_function_pop, fitness, 4)
    print("Parents:", parents)
    children = crossover_fifty_precent(2,parents)
    print("Children:", children)
    new_pop_after_crossover = np.concatenate((parents, children))
    print("New Population After Crosover:", new_pop_after_crossover)
    new_pop_after_mutation = mutation(2,new_pop_after_crossover,0.0001)
    print("New Population After Mutation:", new_pop_after_mutation)

plt.title("Fitnesses")
plt.plot(generations, best_fitnesses, color="red")

plt.show()

#print(int(graytoBinary( repair(binarytoGray(bin(3000)))),2))