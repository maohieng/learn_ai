from constraint import Problem

problem = Problem()

# Add constraints
VARIABLES = ["A", "B", "C", "D", "E", "F", "G"]
# Complete code here: Create a list of all binary constraints, named "CONSTRAINTS" (see slide#75 in the lesson)
CONSTRAINTS = [(VARIABLES[0], VARIABLES[1]), # A!=B
               (VARIABLES[0], VARIABLES[2]), # A!=C
               (VARIABLES[1], VARIABLES[2]), # B!=C
               (VARIABLES[1], VARIABLES[3]), # B!=D
               (VARIABLES[1], VARIABLES[4]), # B!=E
               (VARIABLES[2], VARIABLES[4]), # C!=E
               (VARIABLES[2], VARIABLES[5]), # C!=F 
               (VARIABLES[3], VARIABLES[4]), # D!=E 
               (VARIABLES[4], VARIABLES[5]), # E!=F
               (VARIABLES[4], VARIABLES[6]), # E!=G
               (VARIABLES[5], VARIABLES[6]) # F!=G
               ] 

DOMAINS = ["Monday", "Tuesday", "Wednesday"]

# Add variables
problem.addVariables(
    VARIABLES,
    DOMAINS
)

for x, y in CONSTRAINTS:
    problem.addConstraint(lambda x, y: x != y, (x, y))

# Solve problem
for solution in problem.getSolutions():
    print(solution)
