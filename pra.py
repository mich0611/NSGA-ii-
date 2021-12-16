import random
import numpy as np
import numpy.random as rd
import plotly.express as px
import pandas as pd

import problems
import functions

# initialize functions : 
min_sorting = functions.min_sorting
max_sorting = functions.max_sorting
sort_values = functions.sort_values
crowding_distance = functions.crowding_distance 
binary_tournament_selection = functions.binary_tournament_selection
single_crossover = functions.single_crossover
multi_crossover = functions.multi_crossover 
single_mutation = functions.single_mutation
multi_mutation = functions.multi_mutation
# main

problem = problems.SCH()

gen_num = 0
max_generation = 50

# defining the input space : 50 ,and the objective space : 2
variables = problem.var
size = problem.size
lower = problem.lower
upper = problem.upper
values_1 = problem.func_1() 
values_2 = problem.func_2() 

# data processing : 
data_1 = values_1 
data_2 = values_2
t = [gen_num for i in range(size)]

while(gen_num < max_generation):
    fronts = min_sorting(values_1, values_2) 
    # print('The best solutions is:', [variables[i] for i in fronts[0]])

    distances = []
    for index in range(len(fronts)):
        distance = crowding_distance(fronts[index], values_1, values_2)
        distances.append(distance)

    # produce offspring
    offspring = []
    for i in range(int(size/2)):
        # parent selection   
        p1 = binary_tournament_selection(size, fronts, values_1, distances, variables)
        p2 = binary_tournament_selection(size, fronts, values_1, distances, variables)
        # crossover 
        c1, c2 = single_crossover(p1, p2)
        # mutation 
        c1 = single_mutation(lower, upper, c1)
        c2 = single_mutation(lower, upper, c2)
        offspring.append(c1)
        offspring.append(c2)
    variables += offspring  # total individuals

    fronts = min_sorting(problem.func_1(), problem.func_2())

    indices = [] 
    for i in range(len(fronts)):
        sorted_1 = sort_values(fronts[i], problem.func_1())
        distance = crowding_distance(fronts[i], problem.func_1(), problem.func_2())
        sorted_distance = sort_values([i for i in range(len(distance))], distance)
        for j in sorted_distance:
            indices.append(sorted_1[j])
    indices = indices[:size]  # indices

    variables = [variables[i] for i in indices]
    problem.var = variables 
    values_1 = problem.func_1()
    values_2 = problem.func_2()
    data_1 += values_1
    data_2 += values_2

    gen_num += 1
    t += [gen_num for i in range(size)]

# plot the dynamic data
data = list(zip(data_1, data_2, t))
df = pd.DataFrame(data, columns = ['values_1', 'values_2', 'time'])

fig = px.scatter(df, x="values_1", y="values_2", animation_frame="time",
            size_max=55, range_x=[-450,50], range_y=[-500,50], render_mode = 'svg')
fig.show()
