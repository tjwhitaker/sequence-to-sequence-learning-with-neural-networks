import language
import re
import unidecode
import torch
from settings import MAX_LENGTH, EOS_TOKEN, DEVICE


# Decode, lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unidecode.unidecode(s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# Limit sentence length
def filter_pairs(pairs):
    return [pair for pair in pairs if len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1].split(' ')) < MAX_LENGTH]

# Split file into lines of examples
# Split lines into pairs of lang1 <-> lang2
# Add sentences to languages vocab
def read_languages(lang1, lang2, filename):
    with open(filename, 'r') as file:
        lines = file.read().strip().split('\n')

    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    pairs = filter_pairs(pairs)

    input_lang = language.Vocab(lang1)
    output_lang = language.Vocab(lang2)

    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

    return input_lang, output_lang, pairs

def tensor_from_sentence(lang, sentence):
    indexes = [lang.word_index[word] for word in sentence.split(' ')]
    indexes.append(EOS_TOKEN)

    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)

def tensors_from_pair(input_lang, output_lang, pair):
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    target_tensor = tensor_from_sentence(output_lang, pair[1])
    return (input_tensor, target_tensor)