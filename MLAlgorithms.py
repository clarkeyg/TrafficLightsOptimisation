import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Neural network definitions
class TrafficNN(nn.Module):
    def __init__(self):
        super(TrafficNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )
    def forward(self, x):
        return self.layers(x)

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.layers(x)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.layers(x)

class FixedTrafficNN(nn.Module):
    def __init__(self):
        super(FixedTrafficNN, self).__init__()
        self.linear = nn.Linear(8, 4)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.weight[0, 0] = 1
            self.linear.weight[1, 2] = 1
            self.linear.weight[2, 4] = 1
            self.linear.weight[3, 6] = 1
            self.linear.bias.zero_()
    def forward(self, x):
        return self.linear(x)

# Base Controller class
class Controller:
    def choose_action(self, traffic):
        raise NotImplementedError("Subclasses must implement choose_action")

    def update(self, traffic, cars_moved, missed_cars):
        pass  # Default: no update needed

# ML Controllers
class NeuralTrafficControl(Controller):
    def __init__(self):
        self.model = TrafficNN()
        self.target_model = TrafficNN()
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.997
        self.epsilon_min = 0.01
        self.prev_avg_waiting = 0
        self.update_counter = 0
        self.state = None
        self.action = None

    def get_state(self, traffic):
        state = []
        for lane in [traffic.left, traffic.right, traffic.top, traffic.bottom]:
            avg_wait = sum(car.timeWaiting for car in lane) / len(lane) if lane else 0
            state.extend([len(lane), avg_wait])
        return torch.FloatTensor(state)

    def choose_action(self, traffic):
        self.state = self.get_state(traffic)
        if random.random() < self.epsilon:
            self.action = random.randint(0, 3)
        else:
            with torch.no_grad():
                q_values = self.model(self.state)
                self.action = torch.argmax(q_values).item()
        return self.action

    def calculate_reward(self, traffic, cars_moved, missed_cars):
        total_waiting, total_cars, max_waiting = 0, 0, 0
        for lane in [traffic.left, traffic.right, traffic.top, traffic.bottom]:
            for car in lane:
                waiting_time = car.timeWaiting
                total_waiting += waiting_time
                total_cars += 1
                max_waiting = max(max_waiting, waiting_time)
        current_avg_waiting = total_waiting / max(total_cars, 1)
        waiting_time_improvement = self.prev_avg_waiting - current_avg_waiting
        self.prev_avg_waiting = current_avg_waiting

        moved_cars_reward = cars_moved * 2.0
        missed_cars_penalty = missed_cars * -3.0
        waiting_time_penalty = -0.1 * current_avg_waiting
        max_waiting_penalty = -0.2 * max_waiting
        improvement_reward = waiting_time_improvement * 1.5

        if max_waiting > 60:
            waiting_time_penalty *= 1.5
        if missed_cars > 5:
            missed_cars_penalty *= 1.5

        total_reward = (moved_cars_reward + missed_cars_penalty +
                        waiting_time_penalty + max_waiting_penalty +
                        improvement_reward)
        return max(min(total_reward, 10), -10)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
        target_q = rewards + self.gamma * next_q

        loss = self.criterion(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.update_counter += 1
        if self.update_counter % 100 == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def update(self, traffic, cars_moved, missed_cars):
        next_state = self.get_state(traffic)
        reward = self.calculate_reward(traffic, cars_moved, missed_cars)
        self.remember(self.state, self.action, reward, next_state)
        self.train()

class ActorCriticTrafficControl(Controller):
    def __init__(self):
        self.actor = Actor()
        self.critic = Critic()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        self.gamma = 0.99
        self.state = None
        self.action = None

    def get_state(self, traffic):
        state = []
        for lane in [traffic.left, traffic.right, traffic.top, traffic.bottom]:
            avg_wait = sum(car.timeWaiting for car in lane) / len(lane) if lane else 0
            state.extend([len(lane), avg_wait])
        return torch.FloatTensor(state)

    def choose_action(self, traffic):
        self.state = self.get_state(traffic)
        with torch.no_grad():
            probs = self.actor(self.state)
            dist = torch.distributions.Categorical(probs)
            self.action = dist.sample().item()
        return self.action

    def calculate_reward(self, traffic, cars_moved, missed_cars):
        # Using same reward as NeuralTrafficControl for consistency
        total_waiting, total_cars, max_waiting = 0, 0, 0
        for lane in [traffic.left, traffic.right, traffic.top, traffic.bottom]:
            for car in lane:
                waiting_time = car.timeWaiting
                total_waiting += waiting_time
                total_cars += 1
                max_waiting = max(max_waiting, waiting_time)
        current_avg_waiting = total_waiting / max(total_cars, 1)
        return cars_moved - missed_cars

    def train(self, state, action, reward, next_state):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor([action])
        reward = torch.FloatTensor([reward])

        value = self.critic(state)
        next_value = self.critic(next_state)
        td_error = reward + self.gamma * next_value - value

        critic_loss = td_error.pow(2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(action)
        actor_loss = -log_prob * td_error.detach()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update(self, traffic, cars_moved, missed_cars):
        next_state = self.get_state(traffic)
        reward = self.calculate_reward(traffic, cars_moved, missed_cars)
        self.train(self.state, self.action, reward, next_state)

class FixedNNControl(Controller):
    def __init__(self):
        self.model = FixedTrafficNN()

    def get_state(self, traffic):
        state = []
        for lane in [traffic.left, traffic.right, traffic.top, traffic.bottom]:
            avg_wait = sum(car.timeWaiting for car in lane) / len(lane) if lane else 0
            state.extend([len(lane), avg_wait])
        return torch.FloatTensor(state)

    def choose_action(self, traffic):
        state = self.get_state(traffic)
        with torch.no_grad():
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def update(self, traffic, cars_moved, missed_cars):
        pass  # No training needed
