from random import seed
from random import randint
from numpy import array
from numpy import argmax
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector

# generate lists of random integer sequence and it's sorted result.
def random_sequence_pairs(n_examples, n_numbers, smallest, largest):
    X, Y = list(), list()
    for i in range(n_examples):
        in_pattern = [randint(smallest,largest) for _ in range(n_numbers)]
        out_pattern = sorted(in_pattern)
        X.append(in_pattern)
        Y.append(out_pattern)
    return X, Y

# convert data to strings
def to_string(X, Y, n_numbers, max_length):
    max_length = max(n_numbers, max_length)
    Xstr = list()
    for pattern in X:
        strp = ''.join([str(n) for n in pattern])
        strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
        Xstr.append(strp)
    Ystr = list()
    for pattern in Y:
        strp = ''.join([str(n) for n in pattern])
        strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
        Ystr.append(strp)
    return Xstr, Ystr

# integer encode strings
def integer_encode(X, Y, alphabet):
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    Xenc = list()
    for pattern in X:
        integer_encoded = [char_to_int[char] for char in pattern]
        Xenc.append(integer_encoded)
    Yenc = list()
    for pattern in Y:
        integer_encoded = [char_to_int[char] for char in pattern]
        Yenc.append(integer_encoded)
    return Xenc, Yenc

# one hot encode
def one_hot_encode(X, Y, max_int):
    Xenc = list()
    for seq in X:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        Xenc.append(pattern)
    Yenc = list()
    for seq in Y:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        Yenc.append(pattern)
    return Xenc, Yenc

# generate an encoded dataset
def generate_data(n_samples, n_numbers, smallest, largest, max_length, alphabet):
    # generate pairs
    X, Y = random_sequence_pairs(n_samples, n_numbers, smallest, largest)
    # convert to strings
    X, Y = to_string(X, Y, n_numbers, max_length)
    # integer encode
    X, Y = integer_encode(X, Y, alphabet)
    # one hot encode
    X, Y = one_hot_encode(X, Y, len(alphabet))
    # return as numpy arrays
    X, Y = array(X), array(Y)
    return X, Y

# invert encoding
def invert(seq, alphabet):
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    strings = list()
    for pattern in seq:
        string = int_to_char[argmax(pattern)]
        strings.append(string)
    return ''.join(strings)

# define dataset
seed(1)
n_numbers = 20
n_samples = 3040
smallest = 0
largest = 10 - 1
test_length = 20
max_length = test_length
#evaluate data
t_n_numbers = 20
t_n_samples = 1000
t_smallest = 0
t_largest  = 10 - 1

alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', ' ']
n_chars = len(alphabet)
n_in_seq_length = max_length
n_out_seq_length = max_length
# define LSTM configuration
#n_batch = 10
n_batch = 16
n_epoch = 26
#n_epoch = 20
# create LSTM
model = Sequential()
model.add(LSTM(100, input_shape=(n_in_seq_length, n_chars)))
model.add(RepeatVector(n_out_seq_length))
model.add(LSTM(50, return_sequences=True))
model.add(TimeDistributed(Dense(n_chars, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(model.summary())
# train LSTM
X, Y = generate_data(n_samples, n_numbers, smallest, largest, max_length, alphabet)
model.fit(X, Y, nb_epoch=n_epoch, batch_size=n_batch)

# evaluate on some new patterns
X, Y = generate_data(t_n_samples, t_n_numbers, t_smallest, t_largest, max_length, alphabet)
#print(Y)
print ('evaluate: X.shape, Y.shape', X.shape, Y.shape)
result = model.predict(X, batch_size=n_batch, verbose=0)
# calculate error
expected = [invert(x, alphabet) for x in Y]
predicted = [invert(x, alphabet) for x in result]

#Calcuate the accuracy
correct_count = 0.
for i in range(len(predicted)):
   if predicted[i] == expected[i]:
       correct_count += 1
print('sumple_total=%d, correct=%d, acc=%f' % 
     (len(predicted), correct_count, correct_count/len(predicted)))

# show some examples
for i in range(min(20, len(predicted))):
        print('Expected=%s, Predicted=%s' % (expected[i], predicted[i]))
