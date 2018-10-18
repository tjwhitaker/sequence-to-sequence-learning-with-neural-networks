class Vocab:
	def __init__(self, name):
		self.name = name
		self.word_index = {}
		self.word_count = {}
		self.index_word = {0: "SOS", 1: "EOS"}
		self.num_words = 2

	def add_sentence(self, sentence):
		for word in sentence.split(' '):
			self.add_word(word)

	def add_word(self, word):
		if word not in self.word_index:
			self.word_index[word] = self.num_words
			self.word_count[word] = 1
			self.index_word[self.num_words] = word
			self.num_words += 1
		else:
			self.word_count[word] += 1