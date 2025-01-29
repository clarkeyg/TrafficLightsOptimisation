import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
print(f"cuda enabled: {torch.cuda.is_available()}")

class network(nn.Module):
	def __init__(self):
		return


def makeLaneDecision(model, traffic, hour, cycle):
	return