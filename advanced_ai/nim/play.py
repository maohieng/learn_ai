from nim import play, NimAI
from experiment import new_ai

filename = input("Enter the filename of the saved AI (no ext) (empty for baselin): ")
ai = new_ai(filename)
print("Trained AI info:", "alpha:", ai.alpha, "epsilon:", ai.epsilon, "gamma:", ai.gamma, "train_moves:", len(ai.train_moves))


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
