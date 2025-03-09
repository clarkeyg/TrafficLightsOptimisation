import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class TrafficNN(nn.Module):
	def __init__(self):
		super(TrafficNN, self).__init__()
		self.layers = nn.Sequential(
			nn.Linear(4, 16),  # Increased network capacity
			nn.ReLU(),
			nn.Linear(16, 16),
			nn.ReLU(),
			nn.Linear(16, 8),
			nn.ReLU(),
			nn.Linear(8, 4)
		)

	def forward(self, x):
		return self.layers(x)


class NeuralTrafficControl:
	def __init__(self):
		self.model = TrafficNN()
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
		self.criterion = nn.MSELoss()
		self.memory = deque(maxlen=10000)
		self.batch_size = 64  # Increased batch size for better learning
		self.gamma = 0.99  # Increased discount factor for longer-term planning
		self.epsilon = 1.0
		self.epsilon_decay = 0.997  # Slower decay for better exploration
		self.epsilon_min = 0.01
		self.prev_avg_waiting = 0  # Track previous average waiting time

	def get_state(self, traffic):
		state = [
			len(traffic.left),
			len(traffic.right),
			len(traffic.top),
			len(traffic.bottom)
		]
		return torch.FloatTensor(state)

	def choose_action(self, state):
		if random.random() < self.epsilon:
			return random.randint(0, 3)

		with torch.no_grad():
			q_values = self.model(state)
			return torch.argmax(q_values).item()

	def remember(self, state, action, reward, next_state):
		self.memory.append((state, action, reward, next_state))

	def calculate_reward(self, traffic, cars_moved, missed_cars):
		# Calculate current average waiting time
		total_waiting = 0
		total_cars = 0
		max_waiting = 0  # Track maximum waiting time

		for lane in [traffic.left, traffic.right, traffic.top, traffic.bottom]:
			for car in lane:
				waiting_time = car.timeWaiting
				total_waiting += waiting_time
				total_cars += 1
				max_waiting = max(max_waiting, waiting_time)  # Update maximum waiting time

		# Calculate average waiting time
		current_avg_waiting = total_waiting / max(total_cars, 1)

		# Calculate waiting time improvement
		waiting_time_improvement = self.prev_avg_waiting - current_avg_waiting
		self.prev_avg_waiting = current_avg_waiting

		# Base reward components
		moved_cars_reward = cars_moved * 2.0  # Reward for moving cars
		missed_cars_penalty = missed_cars * -3.0  # Increased penalty for missed cars
		waiting_time_penalty = -0.1 * current_avg_waiting  # Penalty for high average waiting time
		max_waiting_penalty = -0.2 * max_waiting  # Additional penalty for very long waits
		improvement_reward = waiting_time_improvement * 1.5  # Reward for reducing average waiting time

		# Additional penalties for severe conditions
		if max_waiting > 60:  # Extra penalty for any car waiting more than 50 cycles
			waiting_time_penalty *= 1.5

		if missed_cars > 5:  # Extra penalty for many missed cars in one cycle
			missed_cars_penalty *= 1.5

		# Calculate total reward
		total_reward = (moved_cars_reward +
		                missed_cars_penalty +
		                waiting_time_penalty +
		                max_waiting_penalty +
		                improvement_reward)

		# Normalize reward to prevent extreme values
		total_reward = max(min(total_reward, 10), -10)

		return total_reward

	def train(self):
		if len(self.memory) < self.batch_size:
			return

		batch = random.sample(self.memory, self.batch_size)
		states, actions, rewards, next_states = zip(*batch)

		states = torch.stack(states)
		next_states = torch.stack(next_states)
		actions = torch.LongTensor(actions)
		rewards = torch.FloatTensor(rewards)

		# Current Q values
		current_q = self.model(states).gather(1, actions.unsqueeze(1))

		# Next Q values
		with torch.no_grad():
			next_q = self.model(next_states).max(1)[0]
		target_q = rewards + self.gamma * next_q

		# Update model
		loss = self.criterion(current_q.squeeze(), target_q)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# Decay epsilon
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay


def neural_control(traffic, controller):
	"""
	Function to be called from main loop
	"""
	state = controller.get_state(traffic)
	action = controller.choose_action(state)
	return action