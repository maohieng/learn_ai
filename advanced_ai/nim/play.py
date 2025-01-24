import time
from collections import Counter
from nim import train, play, NimAI, aiPlay
import matplotlib.pyplot as plt

def plot_training_progress(ai: NimAI):
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
    plt.axhline(avg_moves, color="blue", linestyle="--", label=f"Avg Moves ({avg_moves})")

    plt.title("Training Progress")
    plt.xlabel("Game")
    plt.ylabel("Moves")
    plt.legend()
    plt.show()

def plot_moves_hist(ai: NimAI):
    plt.figure(figsize=(12, 6))
    moves_counter = Counter(ai.train_moves)
    plt.bar(moves_counter.keys(), moves_counter.values())
    plt.xlabel("Moves")
    plt.ylabel("Frequency")
    plt.title("Moves Frequency")
    plt.show()

def plot_q_values_log(ai: NimAI):
    plt.figure(figsize=(10, 6))
    for i, q_values in enumerate(ai.q_values_log):
        plt.plot(range(len(q_values)), list(q_values.values()) , label=f"Game {i * ai.log_interval}")
    plt.xlabel('State-Action Pairs')
    plt.ylabel('Q-Values')
    plt.title('Q-Values Over Time')
    plt.legend()
    plt.show()

# Train or load the AI
if input("Train a new AI? [y/n]: ").lower() == "y":
    # record training duration
    al = input("Enter the learning rate (alpha): ")
    if al:
        alpha = float(al)
    else:
        alpha = 0.5

    ep = input("Enter the exploration rate (epsilon): ")
    if ep:
        epsilon = float(ep)
    else:
        epsilon = 0.1

    gam = input("Enter the discount factor (gamma): ")
    if gam:
        gamma = float(gam)
    else:
        gamma = 1

    started = time.time()
    ai = train(10000, alpha, epsilon, gamma)
    print("Training time:", time.time() - started)
    filename = input("Enter the filename to save the AI: ")
    ai.save(f'{filename}.pkl')
else:
    filename = input("Enter the filename of the saved AI (no ext): ")
    ai = NimAI.load(f'{filename}.pkl')

print("Trained AI info:", "alpha:", ai.alpha, "epsilon:", ai.epsilon, "gamma:", ai.gamma, "train_moves:", len(ai.train_moves))

if input("Do you want to visualize the metrics? [y/n]: ").lower() == "y":
    # Visualize metrics
    plot_training_progress(ai)

    # Plot moves_counter
    plot_moves_hist(ai)

    # Plot q-values over time
    plot_q_values_log(ai)

# Create a baseline AI to play against the trained AI
numb_games = 1000
print()
print(f"An AI plays against the trained AI in {numb_games} games.")
opponent = input("Pick an opponent AI file (no ext) (empty for baseline): ")
if opponent:
    baseline = NimAI.load(f'{opponent}.pkl')
else:
    baseline = NimAI()

print("Opponent AI info:", "alpha:", baseline.alpha, "epsilon:", baseline.epsilon, "gamma:", baseline.gamma, "train_moves:", len(baseline.train_moves))

wins_baseline, wins_ai = aiPlay(baseline, ai, numb_games, epsilon=True)
print(f"Opponent AI wins: {wins_baseline /numb_games * 100:.2f}%")
print(f"Trained AI wins: {wins_ai /numb_games * 100:.2f}%")

# Play the game
print()
start = input("Do you want to play a game? [y/n]: ")
if start.lower() == "y":
    while True:
        play(ai)
        again = input("Play again? [y/n]: ")
        if again.lower() != "y":
            break
else:
    print("Goodbye!")
