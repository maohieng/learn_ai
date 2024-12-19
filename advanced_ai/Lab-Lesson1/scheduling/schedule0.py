"""
Naive backtracking search without any heuristics or inference.
"""

# Complete code here: Create a list of all courses, named "VARIABLES" as string
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

def select_unassigned_variable(assignment):
    """Chooses a variable not yet assigned, in order."""
    for variable in VARIABLES:
        if variable not in assignment:
            return variable
    return None


def consistent(assignment):
    """Checks to see if an assignment is consistent."""
    for (x, y) in CONSTRAINTS:

        # Only consider arcs where both are assigned
        if x not in assignment or y not in assignment:
            continue

        # If both have same value, then not consistent
        if assignment[x] == assignment[y]:
            return False

    # If nothing inconsistent, then assignment is consistent
    return True

def backtrack(assignment):
    """Runs backtracking search to find an assignment."""

    # Check if assignment is complete
    if len(assignment) == len(VARIABLES):
        return assignment

    # Try a new variable
    var = select_unassigned_variable(assignment)
    for value in DOMAINS:
        new_assignment = assignment.copy()
        new_assignment[var] = value
        if consistent(new_assignment):
            result = backtrack(new_assignment)
            if result is not None:
                return result
    return None

solution = backtrack(dict())
print(solution)
