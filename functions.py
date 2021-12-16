import numpy as np 
import numpy.random as rd

#############################################################
''' SIDE FUNCTIONS'''

# sorting the individuals in front F, based on objective m.
  # input : front n, obj
  # output : sorted indices based on obj

def sort_values(lst, values): 
    sorted_lst = []
    record_index = []
    record_values = []
    while(len(sorted_lst) != len(lst)):
        min_value = min(values)
        min_index = values.index(min_value)
        if min_index in lst:
            sorted_lst.append(min_index)
        record_values.append(values[min_index]) # refinement 1 : recording minimum value and index before changing it to infinity. 
        record_index.append(min_index) 
        values[min_index] = np.inf

    for i in range(len(record_values)):
        values[record_index[i]] = record_values[i]
    return sorted_lst

# functions for parent selection
  # input : random index, total fronts
  # output : rank 

def find_rank(num, fronts):  
  for i in range(len(fronts)):
    if num in fronts[i]:
      index = i
  return index

  # input : index, total fronts, values_1, total distances
  # output : distance value
  
def find_distance(num, fronts, values_1, distances): 
  index = find_rank(num, fronts)
  sorted_1 = sort_values(fronts[index], values_1)
  distance_index = sorted_1.index(num)
  return distances[index][distance_index]

# function for normalize values between 0 and 1

def normalize(values):
    max_value = max(values)
    min_value = min(values)
    for i in range(len(values)):
        values[i] = (values[i] - min_value)/(max_value - min_value)
    return values

#############################################################
''' SELECTON FOR NSGA ii'''

# sorting algorithm, complexity is O(M*N^2), M is number of objectives, N is the population size.
  # input : obj_1 val, obj_2 val
  # output : total_fronts
  # algorithm 1. for maximization problem
  # algorithm 2. for minimization problem

# 1.
def max_sorting(values_1, values_2): 
    S = [[] for i in range(len(values_1))] 
    n = [0 for i in range(len(values_1))]
    rank = [0 for i in range(len(values_1))]
    front = [[]]

    for i in range(len(values_1)): # depends on problems.
        for j in range(len(values_1)): 
            if (values_1[i] > values_1[j] and values_2[i] > values_2[j]) or (values_1[i] >= values_1[j] and values_2[i] > values_2[j]) or (values_1[i] > values_1[j] and values_2[i] >= values_2[j]):
                S[i].append(j)
            elif (values_1[j] > values_1[i] and values_2[j] > values_2[i]) or (values_1[j] >= values_1[i] and values_2[j] > values_2[i]) or (values_1[j] > values_1[i] and values_2[j] >= values_2[i]):
                n[i] += 1

        if n[i] == 0:  # the best solutions.
            rank[i] = 0  # [4,7,3,7,9,...,0]
            front[0].append(i)
    
    index = 0
    while(front[index] != []):
        F = []
        for i in front[index]:
            for j in S[i]: 
                n[j] -= 1 
                if n[j] == 0:
                    rank[j] = index + 1
                    if j not in F:
                        F.append(j)
        index += 1
        front.append(F)
        
    del front[len(front) - 1]
    return front

# 2.
def min_sorting(values_1, values_2): 
    S = [[] for i in range(len(values_1))] 
    n = [0 for i in range(len(values_1))]
    rank = [0 for i in range(len(values_1))]
    front = [[]] 

    for i in range(len(values_1)): # depends on problems
        for j in range(len(values_1)):
            if (values_1[i] < values_1[j] and values_2[i] < values_2[j]) or (values_1[i] <= values_1[j] and values_2[i] < values_2[j]) or (values_1[i] < values_1[j] and values_2[i] <= values_2[j]):
                S[i].append(j)
            elif (values_1[j] < values_1[i] and values_2[j] < values_2[i]) or (values_1[j] <= values_1[i] and values_2[j] < values_2[i]) or (values_1[j] < values_1[i] and values_2[j] <= values_2[i]):
                n[i] += 1

        if n[i] == 0:  # the best solutions.
            rank[i] = 0  # [4,7,3,7,9,...,0]
            front[0].append(i)
    
    index = 0
    while(front[index] != []):
        F = []
        for i in front[index]:
            for j in S[i]: 
                n[j] -= 1 
                if n[j] == 0:
                    rank[j] = index + 1
                    if j not in F:
                        F.append(j)
        index += 1
        front.append(F)
        
    del front[len(front) - 1]
    return front

# crowding distance computation
  # input : front n, obj_1 val , obj_2 val
  # output : list of individual distance in front n, arrange in the order of sort_values_1

def crowding_distance(front, values1, values2): 
    distance = [0 for i in range(0,len(front))] 
    sorted1 = sort_values(front, values1)  # 個別的 index 抓出來，並排列
    sorted2 = sort_values(front, values2)
    distance[0] = 444444444444
    distance[len(front) - 1] = 444444444444
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values1[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1,len(front)-1): # refinement 2 : computational sequqnce for values 2
        distance[k] = distance[k]+ (values2[sorted2[len(front)-1-(k-1)]] - values2[sorted2[len(front)-1-(k+1)]])/(max(values2)-min(values2))
    return distance
   

  # input : population size, total fronts, tournaments
  # output : best variable

# parent selection
  # input : population size, total fronts, values_1, total distances, variables, tournaments
  # output : best variable

def binary_tournament_selection(size, front, values_1, distances, variables, k=2): # choose two individuals and compare their fitness values.
  best = None
  candidate = [rd.randint(0, size) for i in range(k)]
  if candidate[0] == candidate[1]:
    best = variables[candidate[0]]
  else:
    fitness_1 = find_rank(candidate[0], front)
    fitness_2 = find_rank(candidate[1], front)
    if fitness_1 < fitness_2:
      best = variables[candidate[0]]
    elif fitness_1 > fitness_2:
      best = variables[candidate[1]]
    else:
      d1 = find_distance(candidate[0], front, values_1, distances)
      d2 = find_distance(candidate[1], front, values_1, distances)
      best = variables[candidate[0]] if d1 >= d2 else variables[candidate[1]]
  return best 

# simulated binary crossover 
  # input : decision_val_1, decision_val_2
  # output : decision_val_child_1, decision_val_child_2
'''FOR SINGLE DECISION VARIABLE  ex. SCH()'''

def single_crossover(parent_1, parent_2): 
    mu = 1 # distribution index, large mu : closer to parents; small mu : away from parents.
    beta = None 
    n = rd.rand()
    if n <= 0.5:
      beta = (2*n)**(1/(1+mu))
    else:
      beta = 1/(2*(1-n))**(1/(1+mu))

    child_1 = 0.5 * ((1+beta)*parent_1 + (1-beta)*parent_2)
    child_2 = 0.5 * ((1-beta)*parent_1 + (1+beta)*parent_2)
    return child_1, child_2

'''FOR MULTI DECISION VARIABLE  ex. ZTD1(), ZDT2()'''

def multi_crossover(parent_1, parent_2): 
    mu = 5 # distribution index, large mu : closer to parents; small mu : away from parents.
    beta = None 
    n = rd.rand()
    if n <= 0.5:
      beta = (2*n)**(1/(1+mu))
    else:
      beta = 1/(2*(1-n))**(1/(1+mu))

    child_1 = [0.5 * ((1+beta)*i + (1-beta)*j) for i,j in zip(parent_1, parent_2)]
    child_2 = [0.5 * ((1-beta)*i + (1+beta)*j) for i,j in zip(parent_1, parent_2)]
    return child_1, child_2

# polynominal mutation
  # input : lower bound of decision_val, upper bound of decision_val, decision_val_child
  # output : mutated decision_val_child

'''FOR SINGLE DECISION VARIABLE  ex. SCH()'''
def single_mutation(lower, upper, x): 
    D = 20 # distribution index, large mu : closer to parents; small mu : away from parents.
    n = rd.rand()
    mutation_probability = 0.5

    if n <= mutation_probability:
      d = (2*n)**(1/(1+D))-1
      x = x + d*(x - lower)
    else:
      d = 1-(2*(1-n))**(1/(1+D))
      x = x + d*(upper - x)
      
    if x < lower: 
      x = lower
    elif x > upper:
      x = upper
    return x

'''FOR MULTI DECISION VARIABLE  ex. ZTD1(), ZDT2()'''

def multi_mutation(lower, upper, x): 
    D = 20 # polynomial mutation index
    n = rd.rand()
    mutation_probability = 0.5

    if n <= mutation_probability: # value x will goes down
      d = (2*n)**(1/(1+D))-1
      x = [i + d*(i - lower) for i in x]
    else: # value x will goes up
      d = 1-(2*(1-n))**(1/(1+D))
      x = [i + d*(upper - i) for i in x]
    # need some refinement
    x = [i if i>=0 and i <= 1  else 1 for i in x] # bounded.

    return x