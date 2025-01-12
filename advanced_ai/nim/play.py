from nim import train, play
import matplotlib.pyplot as plt

ai, (moves_per_game, rewards_per_game) = train(10000)

# Visualize metrics
plt.figure(figsize=(12, 6))

# Move per game
plt.subplot(1, 2, 1)
plt.plot(moves_per_game, label="Moves per game")
plt.xlabel("Game Number")
plt.ylabel("Number of Moves")
plt.title("Move per Game During Training")
plt.legend()

# Reward per game
plt.subplot(1, 2, 2)
plt.plot(rewards_per_game, label="Total Reward per game", color="orange")
plt.xlabel("Game Number")
plt.ylabel("Total Reward")
plt.title("Total Reward per Game During Training")
plt.legend()

plt.tight_layout()
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
