def min_edit_distance(s1, s2, cost_delete=1, cost_insert=1, cost_substitute=1):
    """
    Compute the minimum edit distance between two strings using dynamic programming.

    Args:
        s1 (str): The first string.
        s2 (str): The second string.
        cost_delete (int): Cost of a deletion operation.
        cost_insert (int): Cost of an insertion operation.
        cost_substitute (int): Cost of a substitution operation.

    Returns:
        dp (list[list[int]]): DP table representing the edit distance.
        alignment (tuple[str, str]): Aligned versions of the input strings.
        operations (tuple[int, int, int]): Counts of insertions, deletions, and substitutions.
    """
    m, n = len(s1), len(s2)

    # Initialize DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill base cases
    for i in range(m + 1):
        dp[i][0] = i * cost_delete
    for j in range(n + 1):
        dp[0][j] = j * cost_insert

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else cost_substitute
            dp[i][j] = min(
                dp[i - 1][j] + cost_delete,  # Deletion
                dp[i][j - 1] + cost_insert,  # Insertion
                dp[i - 1][j - 1] + cost      # Substitution
            )

    # Traceback to find alignment and count operations
    aligned_s1, aligned_s2 = "", ""
    i, j = m, n
    insertions = deletions = substitutions = 0

    while i > 0 or j > 0:
        if i > 0 and dp[i][j] == dp[i - 1][j] + cost_delete:
            aligned_s1 = s1[i - 1] + aligned_s1
            aligned_s2 = "-" + aligned_s2
            i -= 1
            deletions += 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + cost_insert:
            aligned_s1 = "-" + aligned_s1
            aligned_s2 = s2[j - 1] + aligned_s2
            j -= 1
            insertions += 1
        else:
            aligned_s1 = s1[i - 1] + aligned_s1
            aligned_s2 = s2[j - 1] + aligned_s2
            if s1[i - 1] != s2[j - 1]:
                substitutions += 1
            i -= 1
            j -= 1

    return dp, (aligned_s1, aligned_s2), (insertions, deletions, substitutions)

# Example usage
strings = [
    ("APPROXIMATION", "INFORMATIONAL"),
    ("AGGCTATCACCTGA", "TAGCTATCACGA"),
    ("បច្ចប្ប័ន្នភាព្ព", "បច្ចុប្បន្នភាព")
]

# Default costs (1 for all operations)
print("Using default costs:")
for s1, s2 in strings:
    dp_table, alignment, operations = min_edit_distance(s1, s2)
    print(f"\nStrings: {s1} <-> {s2}")
    print("DP Table:")
    for row in dp_table:
        print(row)
    print(f"Alignment: {alignment[0]}\n           {alignment[1]}")
    print(f"Operations: Insertions={operations[0]}, Deletions={operations[1]}, Substitutions={operations[2]}")

# Custom costs (delete=1, insert=2, substitute=3)
print("\nUsing custom costs:")
for s1, s2 in strings:
    dp_table, alignment, operations = min_edit_distance(s1, s2, cost_delete=1, cost_insert=2, cost_substitute=3)
    print(f"\nStrings: {s1} <-> {s2}")
    print("DP Table:")
    for row in dp_table:
        print(row)
    print(f"Alignment: {alignment[0]}\n           {alignment[1]}")
    print(f"Operations: Insertions={operations[0]}, Deletions={operations[1]}, Substitutions={operations[2]}")
