from mip import *

import numpy as np

def flatten(iterator):
    return [el for it in iterator for el in it]

c = np.array([
    [12, 14, 8],
    [16, 0, 9],
    [17, 13, 11],
    [15, 14, 7],
    [13, 12, 11],
    [14, 15, 21],
    [9, 6, 10]
])

t = np.array([
    [5, 4, 7],
    [3, 0, 6],
    [10, 9, 8],
    [4, 8, 9],
    [7, 6, 6],
    [11, 8, 7],
    [9, 7, 8]
])

m = Model()

# Variables - Assignment to the project
for i in range(7):
    for l in ['A', 'B', 'C']:
        if (i==1 and l=='B'):
            continue # Not assign B for Project 2 (1 because of 0-indexing)
        m.add_var(name=f'a_{i}{l}', var_type=BINARY)

        
# Constraints - All projects need to be done
for i in range(7):
    if i == 1:
        m += xsum([m.var_by_name(f'a_{i}{l}') for l in ['A', 'C']]) == 1
    else:
        m += xsum([m.var_by_name(f'a_{i}{l}') for l in ['A', 'B', 'C']]) == 1

# Constraints - Number of projects
m += xsum([m.var_by_name(f'a_{i}A') for i in range(7)]) >= 1
m += xsum([m.var_by_name(f'a_{i}A') for i in range(7)]) <= 3
m += xsum([m.var_by_name(f'a_{i}B') for i in [0, 2, 3, 4, 5, 6]]) >= 1
m += xsum([m.var_by_name(f'a_{i}B') for i in [0, 2, 3, 4, 5, 6]]) <= 3
m += xsum([m.var_by_name(f'a_{i}C') for i in range(7)]) >= 1
m += xsum([m.var_by_name(f'a_{i}C') for i in range(7)]) <= 3

# Constraints - Time
m += xsum([m.var_by_name(f'a_{i}A')*t[i, 0] for i in range(7)]) <= 20
m += xsum([m.var_by_name(f'a_{i}B')*t[i, 1] for i in [0, 2, 3, 4, 5, 6]]) <= 20
m += xsum([m.var_by_name(f'a_{i}C')*t[i, 2] for i in range(7)]) <= 20

# Objective function
m.objective = minimize(xsum(flatten([
    [m.var_by_name(f'a_{i}A')*c[i, 0] for i in range(7)],
    [m.var_by_name(f'a_{i}B')*c[i, 1] for i in [0, 2, 3, 4, 5, 6]],
    [m.var_by_name(f'a_{i}C')*c[i, 2] for i in range(7)]
])))

# Solution
m.max_gap = 0.05
status = m.optimize(max_seconds=300)

if status == OptimizationStatus.OPTIMAL:
    print('Optimal solution cost {} found'.format(m.objective_value))
elif status == OptimizationStatus.FEASIBLE:
    print('sol.cost {} found, best possible: {}'.format(m.objective_value, m.objective_bound))
elif status == OptimizationStatus.NO_SOLUTION_FOUND:
    print('no feasible solution found, lower bound is: {}'.format(m.objective_bound))
if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
    print('Solution:')
    for v in m.vars:
        if abs(v.x) > 1e-6: # only printing non-zeros
            print('{} : {}'.format(v.name, v.x))