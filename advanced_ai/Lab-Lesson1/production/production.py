import scipy.optimize

# Objective Function: 50x_1 + 80x_2
# Constraint 1: 5x_1 + 2x_2 <= 20
# Constraint 2: -10x_1 + -12x_2 <= -90


def objective_function(theta, x):
    return theta[0] * x[0] + theta[1] * x[1]

def constraint(alpha, x, b):
    return alpha[0] * x[0] + alpha[1] * x[1] - b


result = scipy.optimize.linprog(
    # Complete code here: Create a list of cost function --> 50x_1 and 80x_2 (50x_1 + 80x_2)
    c=[50, 80],
    # Complete code here: Create a variable to get the coefficients for inequalities
    A_ub=[[5, 2], [-10, -12]],
    # Complete code here: Create a variable to get the constraints for inequalities: 20 and -90
    b_ub=[20, -90],
)

if result.success:
    print(f"X1: {round(result.x[0], 2)} hours")
    print(f"X2: {round(result.x[1], 2)} hours")
else:
    print("No solution")
