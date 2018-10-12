import language
import model
import settings
import utils

import random

import torch
import torch.nn as nn
import torch.optim as optim

# Config
settings.init()

input_lang, output_lang, pairs = utils.read_languages('English', 'Spanish', '../data/english-spanish.txt')

encoder = model.Encoder(input_lang.num_words, HIDDEN_SIZE).to(DEVICE)
decoder = model.AttentionDecoder(HIDDEN_SIZE, output_lang.num_words).to(DEVICE)

encoder_optimizer = optim.SGD(encoder.parameters(), lr=LEARNING_RATE)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=LEARNING_RATE)

training_pairs = [utils.tensors_from_pair(random.choice(pairs)) for i in range(ITERATIONS)]
criterion = nn.NLLLoss()

for i in range(ITERATIONS):
	training_pair = training_pairs[i]
	input_tensor = training_pair[0]
	target_tensor = training_pair[1]

	loss += model.train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

	if i % PRINT_EVERY == 0:
		print('Iteration: {i}')
		print(loss)

# Evaluate

output_words, attentions = model.evaluate(encoder, decoder, "Eres hermosa mi amor")

for i in range(10):
	pair = random.choice(pairs)
	print('>', pair[0])
	print('=', pair[1])
	output_words, attentions = model.evaluate(encoder, decoder, pair[0])
	output_sentence = ' '.join(output_words)
	print('<', output_sentence)
	print('')