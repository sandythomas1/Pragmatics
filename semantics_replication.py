#process data 

def load_data(filepath):
    '''
    opens data
    
    '''
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()

    words = text.split()
    return words

filepath = '/Users/sandythomas/Desktop/Pragmatics/datasets/text8'
words = load_data(filepath)

#debugging step 
print("Data loaded successfully.")
print(f"\nNumber of words loaded: {len(words)}")
print(f"\nFirst 10 words: {words[:10]}")
print("\n")


#build vocabulary
import collections

def build_vocab(words, vocab_size):
    '''
    counts word frequency and builds vocabulary
    
    '''
    word_counts = collections.Counter(words)

    word_to_id = {'UNKNOWN': 0}

    common_words = word_counts.most_common(vocab_size - 1)

    for i, (word, count) in enumerate(common_words):
        word_to_id[word] = i + 1

    id_to_word = {id_val: word for word, id_val in word_to_id.items()}
    return word_to_id, id_to_word

vocab_size = 50000
word_to_id, id_to_word = build_vocab(words, vocab_size)
print(f"\nVocabulary size: {len(word_to_id)}")

print(f"\nID for the word 'king': {word_to_id.get('king', 0)}")
print(f"\nID for the word 'queen': {word_to_id.get('queen', 0)}")
print(f"\nWord for ID 1: {id_to_word[1]}") # most frequent word
print(f"\nWord for ID 2: {id_to_word[2]}") # second most frequent word
print("\n")

#convert words to ids
data = []

for word in words:
    word_id = word_to_id.get(word, 0)  # Use 0 for UNKNOWN
    data.append(word_id)

#debugging step
print("\nData converted to IDs successfully.")
print(f"\nFirst 10 words: {words[:10]}")
print(f"\nFirst 10 IDs: {data[:10]}")
print("\n")

'''
New section
Building Neural Network Model
Word2Vec Model
'''

import torch
from torch import nn

class SkipGramModel(nn.Module):
    #class for SkipGram model

    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()

        #embedding layer for input words
        #embedding layer- This is a giant lookup table (a matrix) 
        #where each row represents the vector for a single word. 
        # Our goal is to train this layer so the vectors become meaningful.
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)

        #output layer for context words
        #output layer- This layer takes a word vector from the Embedding Layer 
        #and tries to predict the context words.
        self.out_embeddings = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_word_ids):
        '''
        forward pass of the model
            
        '''

        embeds = self.in_embeddings(input_word_ids)
        scores = self.out_embeddings(embeds)
        return scores
        
# Initialize the model
embedding_dim = 100  # Size of the word vectors
model = SkipGramModel(vocab_size = vocab_size, embedding_dim = embedding_dim)
# Print model architecture
print("\nModel architecture:")
print(model)
print("\n")

#Training the Model
def train_data(word_ids, window_size):
    '''
    generates (target, context) word pairs for training
    '''
    training_data = []
    #loop through the word IDs
    for i in range(window_size, len(word_ids) - window_size):
        target = word_ids[i]
        context = list(range(i - window_size, i)) + list(range(i + 1, i + window_size + 1))

        for context_word in context:
            context_word_id = word_ids[context_word]
            training_data.append((target, context_word_id))

    return training_data

window_size = 2
print(f"\nGenerating training data with window size {window_size}")
print("\n")

training_data = train_data(data[:1000000], window_size)
print(f"\nNumber of training pairs: {len(training_data)}")
print(f"\nFirst 10 training pairs: {training_data[:10]}")
target_word = id_to_word[training_data[0][0]]
context_word = id_to_word[training_data[0][1]]
print(f"Example pair in words: (target: '{target_word}', context: '{context_word}')")
print("\n")

#training tools
#loss function
loss_function = nn.CrossEntropyLoss()

#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#convert training data to tensors
training_data_tensor = torch.LongTensor(training_data)

#training loop
epochs = 5
print(f"\nTraining Starting")
print("\n")

for epoch in range(epochs):
    total_loss = 0

    batch_size = 512

    for i in range(0, len(training_data_tensor), batch_size):
        batch = training_data_tensor[i:i + batch_size]
        input_words = batch[:, 0]
        target_words = batch[:, 1]

        #zero the gradients
        model.zero_grad()
        #forward pass
        scores = model(input_words)
        #calculate loss
        loss = loss_function(scores, target_words)
        #backward pass
        loss.backward()
        #update weights
        optimizer.step()

        total_loss += loss.item()

    print(f"\nEpoch {epoch + 1}, Loss: {total_loss / len(training_data_tensor)}")
print("\nTraining completed successfully.")
print("\n")