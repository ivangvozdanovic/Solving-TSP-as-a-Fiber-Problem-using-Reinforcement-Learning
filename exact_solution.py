import pulp

def TSP_integer_program(distance_matrix):

    # Number of nodes
    n = distance_matrix.shape[0]

    # Create the LP problem
    prob = pulp.LpProblem("TSP", pulp.LpMinimize)

    # Decision variables: x[i, j] is 1 if the path goes from i to j, 0 otherwise
    x = pulp.LpVariable.dicts("x", [(i, j) for i in range(n) for j in range(n)], cat='Binary')

    # Decision variables: u[i] for the subtour elimination constraints
    u = pulp.LpVariable.dicts("u", [i for i in range(1,n)], lowBound=0, upBound=n-1, cat='Integer')

    # Objective function: minimize the total travel distance
    prob += pulp.lpSum(distance_matrix[i][j] * x[i, j] for i in range(n) for j in range(n))

    # Constraints: Each node must be entered and left exactly once
    for i in range(n):
        prob += pulp.lpSum(x[i, j] for j in range(n) if j != i) == 1
        prob += pulp.lpSum(x[j, i] for j in range(n) if j != i) == 1

    # Subtour elimination constraints (MTZ formulation)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + n * x[i, j] <= n-1

    # Solve the problem
    prob.solve()

    # Print the solution
    tour_edges = []
    print("Status:", pulp.LpStatus[prob.status])
    print("Optimal route:", end=' ')
    for i in range(n):
        for j in range(n):
            if pulp.value(x[i, j]) == 1:
                tour_edges.append((i,j))
                print(f"{i} -> {j}", end='  ')
    print("\nTotal Distance:", pulp.value(prob.objective))
    return tour_edges