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
			nn.Linear(8, 32),  # Input size changed to 8, increased to 32 neurons
			nn.ReLU(),
			nn.Linear(32, 16),
			nn.ReLU(),
			nn.Linear(16, 4)  # Output remains 4 (one per action)
		)

	def forward(self, x):
		return self.layers(x)

class NeuralTrafficControl:
	def __init__(self):
		self.model = TrafficNN()
		self.target_model = TrafficNN()  # Create target network
		self.target_model.load_state_dict(self.model.state_dict())  # Copy weights
		self.target_model.eval()  # Set to evaluation mode
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
		self.criterion = nn.MSELoss()
		self.memory = deque(maxlen=10000)
		self.batch_size = 64
		self.gamma = 0.99
		self.epsilon = 1.0
		self.epsilon_decay = 0.997
		self.epsilon_min = 0.01
		self.prev_avg_waiting = 0
		self.update_counter = 0  # Counter for target network updates

	def get_state(self, traffic):
		state = []
		for lane in [traffic.left, traffic.right, traffic.top, traffic.bottom]:
			if len(lane) > 0:
				avg_wait = sum(car.timeWaiting for car in lane) / len(lane)
			else:
				avg_wait = 0
			state.append(len(lane))
			state.append(avg_wait)
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

		current_q = self.model(states).gather(1, actions.unsqueeze(1))

		with torch.no_grad():
			next_q = self.target_model(next_states).max(1)[0]
		target_q = rewards + self.gamma * next_q

		# Update main model
		loss = self.criterion(current_q.squeeze(), target_q)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# Decay epsilon
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

		# Update target network every 100 cycles
		self.update_counter += 1
		if self.update_counter % 100 == 0:
			self.target_model.load_state_dict(self.model.state_dict())

class Actor(nn.Module):
	def __init__(self):
		super(Actor, self).__init__()
		self.layers = nn.Sequential(
			nn.Linear(8, 32),  # Same 8-input state as improved DQN
			nn.ReLU(),
			nn.Linear(32, 16),
			nn.ReLU(),
			nn.Linear(16, 4),  # 4 actions
			nn.Softmax(dim=-1)  # Probabilities over actions
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
			nn.Linear(16, 1)  # Single value estimate
		)

	def forward(self, x):
		return self.layers(x)

class ActorCriticTrafficControl:
	def __init__(self):
		self.actor = Actor()
		self.critic = Critic()
		self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
		self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
		self.gamma = 0.99

	def get_state(self, traffic):
		state = []
		for lane in [traffic.left, traffic.right, traffic.top, traffic.bottom]:
			if len(lane) > 0:
				avg_wait = sum(car.timeWaiting for car in lane) / len(lane)
			else:
				avg_wait = 0
			state.append(len(lane))
			state.append(avg_wait)
		return torch.FloatTensor(state)

	def choose_action(self, state):
		with torch.no_grad():
			probs = self.actor(state)
			dist = torch.distributions.Categorical(probs)
			action = dist.sample()
		return action.item()

	def train(self, state, action, reward, next_state):
		state = torch.FloatTensor(state)
		next_state = torch.FloatTensor(next_state)
		action = torch.LongTensor([action])
		reward = torch.FloatTensor([reward])

		# Compute TD error
		value = self.critic(state)
		next_value = self.critic(next_state)
		td_error = reward + self.gamma * next_value - value

		# Update critic
		critic_loss = td_error.pow(2)
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Update actor
		probs = self.actor(state)
		dist = torch.distributions.Categorical(probs)
		log_prob = dist.log_prob(action)
		actor_loss = -log_prob * td_error.detach()  # Maximize advantage
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

def actor_critic_control(traffic, controller):
	state = controller.get_state(traffic)
	action = controller.choose_action(state)
	return action

def neural_control(traffic, controller):
	state = controller.get_state(traffic)
	action = controller.choose_action(state)
	return action