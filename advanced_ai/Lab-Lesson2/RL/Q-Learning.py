import numpy as np

class Gridworld:
    def __init__(self, size, start, goal):
        self.size = size
        self.start = start
        self.goal = goal
        self.state = start
    
    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action):
        x, y = self.state
        if action == 0:  # up
            x = max(0, x - 1)
        elif action == 1:  # right
            y = min(self.size - 1, y + 1)
        elif action == 2:  # down
            x = min(self.size - 1, x + 1)
        elif action == 3:  # left
            y = max(0, y - 1)
        
        self.state = (x, y)
        reward = 1 if self.state == self.goal else -1
        done = self.state == self.goal
        return self.state, reward, done

def q_learning(env, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000):
    Q = np.zeros((env.size, env.size, 4))
    for episode in range(episodes):
        # debug = (episode+1) % 100 == 0
        debug = False
        if debug:
            print(f"Learning Episode {episode + 1}")

        state = env.reset()
        step = 0
        while True:
            if np.random.rand() < epsilon:
                action = np.random.choice(4)
            else:
                action = np.argmax(Q[state])
            
            next_state, reward, done = env.step(action)
            # Write the formula here
            # You can review the formula in the lesson
            
            best_next_action = np.argmax(Q[next_state])

            if debug:
                print(f"- State: {state}, Action: {action}, Next State: {next_state}, Reward: {reward}, Done: {done}, Best Next Action: {best_next_action}")

            future_reward_estimate = gamma * Q[next_state][best_next_action]
            new_value_estimate = reward + future_reward_estimate - Q[state][action]

            Q[state][action] += alpha * new_value_estimate

            state = next_state
            
            step += 1
            if done:
                print(f"Episode {episode + 1} finished after {step} steps")
                break
            
    return Q

env = Gridworld(5, (0, 0), (4, 4))
Q = q_learning(env)
print("Q-Values:")
print(Q)