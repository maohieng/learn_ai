import time
from nim import train, play, NimAI
import matplotlib.pyplot as plt

# Train or load the AI
if input("Train a new AI? [y/n]: ").lower() == "y":
    # record training duration
    alpha = float(input("Enter the learning rate (alpha): "))
    epsilon = float(input("Enter the exploration rate (epsilon): "))
    gamma = float(input("Enter the discount factor (gamma): "))

    started = time.time()
    ai = train(10000, alpha, epsilon, gamma)
    print("Training time:", time.time() - started)
    filename = input("Enter the filename to save the AI: ")
    ai.save(f'{filename}.pkl')
else:
    filename = input("Enter the filename of the AI: ")
    ai = NimAI.load(f'{filename}.pkl')

# Visualize metrics
plt.figure(figsize=(12, 6))

# Plot moves made during training for pos % 10 == 0
trained_moves = [m for i, m in enumerate(ai.train_moves) if i % 20 == 0]
rewards_per_game = [r for i, r in enumerate(ai.rewards) if i % 20 == 0]

plt.plot(trained_moves, label="Moves")
plt.plot(rewards_per_game, label="Rewards", color="green")

# display min, max, and average moves
min_moves = min(trained_moves)
max_moves = max(trained_moves)
avg_moves = sum(trained_moves) / len(trained_moves)
plt.axhline(min_moves, color="green", linestyle="--", label=f"Min Moves ({min_moves})")
plt.axhline(max_moves, color="red", linestyle="--", label=f"Max Moves ({max_moves})")

plt.title("Training Progress")
plt.xlabel("Game")
plt.ylabel("Moves")
plt.legend()
plt.show()

start = input("Do you want to play a game? [y/n]: ")
if start.lower() == "y":
    while True:
        play(ai)
        again = input("Play again? [y/n]: ")
        if again.lower() != "y":
            break
else:
    print("Goodbye!")
