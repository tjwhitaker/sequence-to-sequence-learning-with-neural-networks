import language
import model
import settings
import utils

import torch
import torch.nn as nn
import torch.optim as optim

# Config
settings.init()

encoder = model.Encoder()
decoder = model.AttentionDecoder()

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

# Save Model

# Evaluate

output_words, attentions = model.evaluate(encoder, decoder, "Eres hermosa mi amor")

for i in range(10):
	pair = random.choice(pairs)
	print('>', pair[0])
	print('=', pair[1])
	output_words, attentions = evaluate(encoder, decoder, pair[0])
	output_sentence = ' '.join(output_words)
	print('<', output_sentence)
	print('')