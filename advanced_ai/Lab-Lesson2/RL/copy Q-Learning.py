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
    # def step(self, action):
    #     x, y = self.state
    #     if action == 0:  # up
    #         y = max(0, y - 1)
    #     elif action == 1:  # right
    #         x = min(self.size - 1, x + 1)
    #     elif action == 2:  # down
    #         y = min(self.size - 1, y + 1)
    #     elif action == 3:  # left
    #         x = max(0, x - 1)

    #     self.state = (x, y)
    #     reward = 1 if self.state == self.goal else -1
    #     done = self.state == self.goal
    #     return self.state, reward, done

def q_learning(env, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000):
    ''' Q-Learning Algorithm: This function implements the Q-Learning 
    algorithm to learn the Q-Values of the environment. The Q-Values are
    stored in a 3D numpy array with the shape (env.size, env.size, 4) where
    the first two dimensions represent the state space and the last dimension
    represents the action space. The function returns the Q-Values after
    the learning process is complete.
    '''
    Q = np.zeros((env.size, env.size, 4))
    for episode in range(episodes):
        # debug = (episode+1) % 100 == 0
        debug = True
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
            # Q[state][action] = Q[state][action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
            # OR
            
            # target_Q = reward + gamma * np.max(Q[next_state])
            # Q[state][action] = (1 - alpha) * Q[state][action] + alpha * target_Q
            
            target_Q = reward + gamma * np.max(Q[next_state])
            
            Q[state][action] = (1 - alpha) * Q[state][action] + alpha * target_Q

            state = next_state
            
            step += 1
            if done:
                print(f"Episode {episode + 1} finished after {step} steps")
                break
            
    return Q

env = Gridworld(5, (0, 0), (4, 4))
Q = q_learning(env, episodes=100)
print("Q-Values:")
print(Q)

# Test the model
print("Testing the model")
state = env.reset()
done = False

while not done:
    action = np.argmax(Q[state])
    state, reward, done = env.step(action)
    print(f"State: {state}, Action: {action}, Reward: {reward}")