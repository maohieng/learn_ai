from collections import Counter
from nim import NimAI, train, aiPlay
import sys
import time
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

def new_ai(filename):
    if filename:
        ai = NimAI.load(f'{filename}.pkl')
    else:
        ai = NimAI()

    return ai

def main():
    # Read first argument from command line
    if len(sys.argv) > 1:
        arg_eps = float(sys.argv[1])
        print("EXPERIMENT WITH EPSILON=", arg_eps)
    else:
        arg_eps = -1
        print("EXPERIMENT WITHOUT EPSILON")

    # Train or load the AI
    if input("Train a new AI? [y/n]: ").lower() == "y":
        # record training duration
        al = input("Enter the learning rate (alpha): ")
        if al:
            alpha = float(al)
        else:
            alpha = 0.5

        gam = input("Enter the discount factor (gamma): ")
        if gam:
            gamma = float(gam)
        else:
            gamma = 1

        started = time.time()
        ai = train(10000, alpha, gamma)
        print("Training time:", time.time() - started)
        filename = input("Enter the filename to save the AI: ")
        ai.save(f'{filename}.pkl')
    else:
        filename = input("Filename of the saved AI (no ext) (empty for baseline): ")
        ai = new_ai(filename)

    if arg_eps != -1:
        ai.epsilon = arg_eps

    print("Trained AI info:", "alpha:", ai.alpha, "epsilon:", ai.epsilon, "gamma:", ai.gamma, "train_moves:", len(ai.train_moves))

    # Visualize metrics
    if input("Do you want to visualize the metrics? [y/n]: ").lower() == "y":
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
    baseline = new_ai(opponent)

    if arg_eps != -1:
        baseline.epsilon = arg_eps

    print("Opponent AI info:", "alpha:", baseline.alpha, "epsilon:", baseline.epsilon, "gamma:", baseline.gamma, "train_moves:", len(baseline.train_moves))

    wins_baseline, wins_ai = aiPlay(baseline, ai, numb_games, epsilon=(True if arg_eps != -1 else False))
    print(f"Opponent AI wins: {wins_baseline /numb_games * 100:.2f}%")
    print(f"Trained AI wins: {wins_ai /numb_games * 100:.2f}%")

if __name__ == "__main__":
    main()