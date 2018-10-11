import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(Encoder, self).__init__()
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(input_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size)

	def forward(self, input, hidden):
		embedded = self.embedding(input).view(1, 1, -1)
		output = embedded
		output, hidden = self.gru(output, hidden)
		return output, hidden

	def init_hidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)

class Decoder(nn.Module):
	def __init__(self, hidden_size, output_size):
		super(Decoder, self).__init__()
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(output_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden):
		output = self.embedding(input).view(1, 1, -1)
		output = F.relu(output)
		output, hidden = self.gru(output, hidden)
		output = self.softmax(self.out(output[0]))
		return output, hidden

	def init_hidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)

class AttentionDecoder(nn.Module):
	def __init__(self, hidden_size, output_size, dropout_p, max_length):
		super(AttentionDecoder, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.dropout_p = dropout_p
		self.max_length = max_length

		self.embedding = nn.Embedding(self.output_size, self.hidden_size)
		self.attention_layer = nn.Linear(self.hidden_size * 2, self.max_length)
		self.attention_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
		self.dropout = nn.Dropout(self.dropout_p)
		self.gru = nn.GRU(self.hidden_size, self.hidden_size)
		self.out = nn.Linear(self.hidden_size, self.output_size)

	def forward(self, input, hidden, encoder_outputs):
		embedded = self.embedding(input).view(1, 1, -1)
		embedded = self.dropout(embedded)

		attention_weights = F.softmax(self.attention_layer(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
		attention_applied = torch.bmm(attention_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

		output = torch.cat((embedded[0], attention_applied[0]), 1)
		output = self.attention_combine(output).unsqueeze(0)

		output = F.relu(output)
		output, hidden = self.gru(output, hidden)

		output = F.log_softmax(self.out(output[0]), dim=1)
		return output, hidden, attention_weights

	def init_hidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):
	encoder_hidden = encoder.init_hidden()

	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	input_length = input_tensor.size(0)
	target_length = target_tensor.size(0)

	encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

	loss = 0

	for i in range(input_length):
		encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
		encoder_outputs[i] = encoder_output[0, 0]

	decoder_input = torch.tensor([[SOS_token]], device=device)

	decoder_hidden = encoder_hidden

	# Use predictions as the next input
	for i in range(target_length):
		decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
		topv, topi = decoder_output.topk(1)
		decoder_input = topi.squeeze().detach()

		loss += criterion(decoder_output, target_tensor[i])

		if decoder_input.item() == EOS_token:
			break

	loss.backward()

	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.item() / target_length

def evaluate(encoder, decoder, sentence, max_length):
	with torch.no_grad():
		input_tensor = utils.tensors_from_sentence(input_lang, sentence)
		input_length = input_tensor.size()[0]
		
		encoder_hidden = encoder.init_hidden()
		encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)

		for i in range(input_length):
			encooder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
			encoder_outputs[i] += encoder_output[0, 0]

		decoder_input = torch.tensor([[SOS_TOKEN]], device=DEVICE)

		decoder_hidden = encoder_hidden

		decoded_words = []
		decoder_attentions = torch.zeros(max_length, max_length)

		for i in range(max_length):
			decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
			decoder_attentions[i] = decoder_attention.data
			topv, topi = decoder_output.data.topk(1)

			if topi.item() == EOS_TOKEN:
				decoded_words.append('<EOS>')
				break
			else:
				decoded_words.append(output_lang.index_word[topi.item()])

			decoder_input = topi.squeeze().detach()

		return decoded_words, decoder_attentions[:i + i]