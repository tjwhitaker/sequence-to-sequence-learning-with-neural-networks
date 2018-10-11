import torch

def init():
	DEVICE = torch.device('cuda')
	ITERATIONS = 1000
	LEARNING_RATE = 0.01
	PRINT_EVERY = 100
	SOS_TOKEN = 0
	EOS_TOKEN = 1
