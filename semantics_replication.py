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
print(f"Number of words loaded: {len(words)}")
print(f"First 10 words: {words[:10]}")


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
print(f"Vocabulary size: {len(word_to_id)}")

print(f" ID for the word 'king': {word_to_id.get('king', 0)}")
print(f" ID for the word 'queen': {word_to_id.get('queen', 0)}")
print(f"Word for ID 1: {id_to_word[1]}") # most frequent word
print(f"Word for ID 2: {id_to_word[2]}") # second most frequent word

#convert words to ids
data = []

for word in words:
    word_id = word_to_id.get(word, 0)  # Use 0 for UNKNOWN
    data.append(word_id)

#debugging step
print("Data converted to IDs successfully.")
print(f"First 10 words: {words[:10]}")
print(f"First 10 IDs: {data[:10]}")
