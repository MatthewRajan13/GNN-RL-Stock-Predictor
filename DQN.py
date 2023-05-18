import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate, discount_factor, epsilon):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # Create the Q-network
        self.network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        q_values = self.network(state)
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return q_values.argmax().item()

    def update_q_network(self, state, action, reward, next_state):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        q_values = self.network(state)
        next_q_values = self.network(next_state)
        td_target = reward + self.discount_factor * next_q_values.max()
        loss = nn.MSELoss()(q_values[action], td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Environment:
    def __init__(self, data, prices):
        self.data = data
        self.prices = prices
        self.num_steps = data.shape[0]
        self.portfolio_value = prices[0]*5
        self.current_step = 0

    def get_state(self):
        if self.current_step == self.num_steps:
            return self.data[self.current_step - 1]  # Use the previous step's data if at the last step
        else:
            return self.data[self.current_step]

    def take_action(self, action):
        current_price = self.prices[self.current_step - 1]
        if self.current_step == self.num_steps:
            next_price = current_price  # Assume the price remains the same if at the last step
        else:
            next_price = self.prices[self.current_step]

        if action == 0:  # Hold
            profit = next_price - current_price
        if action == 1:  # Buy
            profit = next_price - current_price
            self.portfolio_value += current_price
        elif action == 2:  # Sell
            profit = current_price - next_price
            self.portfolio_value -= current_price

        self.current_step += 1

        next_state = self.get_state()
        done = (self.current_step == self.num_steps)

        return next_state, profit, done

