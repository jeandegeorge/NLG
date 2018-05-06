import numpy as np
from keras.models import Model
from keras.layers import Embedding, LSTM, RepeatVector, Dense, Activation, Input, Flatten, Reshape, Permute, Lambda
from keras.layers.wrappers import Bidirectional


from preproc_2_V2 import *


path_to_data_dir = '/Users/Hugo/OneDrive/Ecole/MS_DSBA/DeepLearning/data/'
embeddings_file = 'crawl-300d-200k.vec'
train_file = 'trainset.csv'
dev_file = 'devset.csv'

max_input_seq_len = 50  # number of words the MRs should be truncated/padded to
max_output_seq_len = 50  # number of words the utterances should be truncated/padded to
vocab_size = 10000  # maximum vocabulary size of the utterances

# ---- LOAD DATA ----
print('\nLoading data...')
# word2idx, idx2word = preproc.load_vocab(path_to_vocab)
x_train, y_train,y_idx2word,dico = preprocess_data(train_file, dev_file, vocab_size, max_input_seq_len, max_output_seq_len, oh=True)

print('Utterance vocab size:', len(y_idx2word))


# ---- BUILD THE MODEL ----

print('\nBuilding language generation model...')

#building generator of data : small usage of memory avoid crash
def gener(x_train,y_train,y_idx2word):
    for i in range(len(y_train)):
        outputs=np.zeros((1,len(y_train[i]),len(y_idx2word)))
        for j,w in enumerate(y_train[i]):
            if j!=0:
                outputs[0,j-1,w]=1
        outputs[0,len(y_train[i])-1,2]=1
        yield([np.array([x_train[i]]),np.array([y_train[i]])],outputs)

g=gener(x_train,y_train,y_idx2word)


latent_dim=64
num_decoder_tokens=len(y_idx2word)

vocab_size=len(y_idx2word)

###########ENCODER##############
# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,))
x = Embedding(vocab_size, latent_dim)(encoder_inputs)
#x,s_1,s_2,s_3,s_4= Bidirectional(LSTM(latent_dim, return_state=True))(x)
x,s_1,s_2=LSTM(latent_dim, return_state=True)(x)
#encoder_states = [s_1,s_2,s_3,s_4]
encoder_states = [s_1,s_2]

###########DECODER##############
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
emb = Embedding(num_decoder_tokens, latent_dim)
x=emb(decoder_inputs)
#decoder_lstm=Bidirectional(LSTM(latent_dim, return_sequences=True))
decoder_lstm=LSTM(latent_dim, return_sequences=True)
y= decoder_lstm(x, initial_state=encoder_states)
decoder_dense= Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(y)

####DEFINITION OF THE MODEL#########
# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
#model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model =Model([encoder_inputs, decoder_inputs], decoder_outputs)

#########COMPILE AND RUN#############
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!

print(model.summary())
epoch=2
for _ in range(epoch):
    g=g=gener(x_train,y_train,y_idx2word)
    model.fit_generator(g, epochs=1,steps_per_epoch=len(x_train)/float(20))

#############################################################################
#Predict definition of two model based on the previous trained one : encode et decoder

def decode_sequence_2(input_seq):
    target_seq = [3]
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens=model.predict([input_seq,np.array([target_seq],ndmin=2)])
        sampled_token_index = np.argmax(output_tokens[0,-1, :])
        sampled_char = y_idx2word[sampled_token_index]
        decoded_sentence += sampled_char+' '
        target_seq.append(sampled_token_index)
        if (sampled_char == '.' or #sampled_char == '<STOP>' or
           len(decoded_sentence.split()) > 30):
            stop_condition = True
    return decoded_sentence
    


for k in range(10040,10060):
    sent=' '.join(y_idx2word[i] for i in x_train[k])
    print('MR : ',sent)
    s=decode_sequence_2(np.array([x_train[k]]))
    for key,value in dico[k].items():
        key=key.lower()
        s=s.replace(key,value)
        if key=='<area>' and '<near>' in s:
            s=s.replace('<near>',value)
        elif key=='<near>' and '<area>' in s:
            s=s.replace('<area>',value)
    print("prediction : ",s)
    print('\n')

