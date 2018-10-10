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

class Attention(nn.Module):
	def __init__(self, hidden_size, output_size, dropout_p, max_length):
		super(Attention, self).__init__()
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