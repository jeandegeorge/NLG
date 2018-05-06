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
x_train, y_train,x_dev,y_dev, y_idx2word,dico = preprocess_data(train_file, dev_file, vocab_size, max_input_seq_len, max_output_seq_len, oh=True)

print('Utterance vocab size:', len(y_idx2word))
print('x_train.shape =', x_train.shape)
print('y_train.shape =', y_train.shape)


# ---- BUILD THE MODEL ----

print('\nBuilding language generation model...')

#building generator of data : small usage of memory avoid crash
def gener(x_train,y_train,y_idx2word):
    for i in range(len(y_train)):
        outputs=np.zeros((1,len(y_train[i]),len(y_idx2word)))
        for j,w in enumerate(y_train[i]):
            if w!=0 and w!=7:
                outputs[0,j-1,w]=1
        outputs[0,len(y_train[i])-1,2]=1
        yield([np.array([x_train[i]]),np.array([y_train[i]])],outputs)

g=gener(x_train,y_train,y_idx2word)


latent_dim=64
num_decoder_tokens=len(y_idx2word)
vocab_size_enco=len(dico)
vocab_size=len(y_idx2word)

###########ENCODER##############
# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,))
x = Embedding(vocab_size_enco, latent_dim)(encoder_inputs)
#x,s_1,s_2,s_3,s_4= Bidirectional(LSTM(latent_dim, return_state=True,dropout=0.2, recurrent_dropout=0.2))(x)
y,s_1,s_2= LSTM(latent_dim, return_state=True,return_sequences=True)(x)
z,s_3,s_4= LSTM(latent_dim, return_state=True)(y)
encoder_states_1 = [s_1,s_2]
encoder_states_2 = [s_3,s_4]

###########DECODER##############
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
emb = Embedding(num_decoder_tokens, latent_dim)
x=emb(decoder_inputs)
#decoder_lstm=Bidirectional(LSTM(latent_dim, return_sequences=True,return_state=True,dropout=0.5, recurrent_dropout=0.5))
decoder_lstm_1=LSTM(latent_dim, return_sequences=True,return_state=True)
decoder_lstm_2=LSTM(latent_dim, return_sequences=True,return_state=True)
y, h_1,h_2 = decoder_lstm_1(x, initial_state=encoder_states_1)
z,h_3,h_4 = decoder_lstm_2(y, initial_state=encoder_states_2)
decoder_dense= Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(z)

####DEFINITION OF THE MODEL#########
# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
#model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model =Model([encoder_inputs, decoder_inputs], decoder_outputs)

#########COMPILE AND RUN#############
model.compile(optimizer='adam', loss='categorical_crossentropy')
# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!

print(model.summary())
epoch=20
for _ in range(epoch):
    g=gener(x_train,y_train,y_idx2word)
    model.fit_generator(g, epochs=1,steps_per_epoch=len(x_train))

#############################################################################
#Predict definition of two model based on the previous trained one : encode et decoder

#####ENCODER OUTPUT##########
encoder_model = Model(encoder_inputs, encoder_states_1+encoder_states_2)
print(encoder_model.summary())


#####DECODER OUTPUT##########
#decoder_dense=Dense(num_decoder_tokens, activation='softmax')

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_state_input_d = Input(shape=(latent_dim,))
decoder_state_input_e = Input(shape=(latent_dim,))
decoder_states_inputs_1 = [decoder_state_input_h, decoder_state_input_c]
decoder_states_inputs_2 = [decoder_state_input_d,decoder_state_input_e]
decoder_out,d_1,d_2=decoder_lstm_1(emb(decoder_inputs), initial_state=decoder_states_inputs_1)
decoder_out,d_3,d_4=decoder_lstm_1(emb(decoder_inputs), initial_state=decoder_states_inputs_2)
decoder_states_1 = [d_1,d_2]
decoder_states_2 = [d_3,d_4]
decoder_outputs = decoder_dense(decoder_out)

decoder_model = Model([decoder_inputs] + decoder_states_inputs_1 + decoder_states_inputs_2 ,[decoder_outputs] + decoder_states_1 + decoder_states_2)
print(decoder_model.summary())


#######RUNNNING SOME TEST###########

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0,0] = 7

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    pos=0
    decoded_sentence_index=[]
    while not stop_condition:
        
        output_tokens, h, c,d,e = decoder_model.predict([target_seq] + states_value)
        # Sample a token
        if pos==0:
            k_max=output_tokens[0,0, :]
            k_max=k_max.argsort()[-20:][::-1]
            sampled_token_index = np.random.choice(k_max)
            sampled_token_index=sampled_token_index
            sampled_char = y_idx2word[sampled_token_index]
            decoded_sentence += sampled_char+' '
            decoded_sentence_index.append(sampled_token_index)
        else:
            sampled_token_index = np.argmax(output_tokens[0,0, :])
            if sampled_token_index not in decoded_sentence_index:
                sampled_char = y_idx2word[sampled_token_index]
                decoded_sentence += sampled_char+' '
                decoded_sentence_index.append(sampled_token_index)
            else : 
                k_max=output_tokens[0,0,:].argsort()[-30:][::-1]
                cond=False
                ind=1
                while not cond:
                    if k_max[ind] not in decoded_sentence_index:
                        sampled_token_index=k_max[ind]
                        sampled_char = y_idx2word[sampled_token_index]
                        decoded_sentence += sampled_char+' '
                        decoded_sentence_index.append(sampled_token_index)
                        cond=True
                    ind+=1
#        r=output_tokens[0,0,:]
#        k_beam=r.argsort()[-6:][::-1]
#        sampled_char=[]
#        for x in k_beam:
#            sampled_char.append(y_idx2word[x])
#        print(sampled_char)


        pos+=1
        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '.' or
           len(decoded_sentence.split()) > 30):
            stop_condition = True
        
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0,0] = sampled_token_index

        
        # Update states
        states_value = [h, c,d,e]

    return decoded_sentence


for k in range(20):
    sent=' '.join(y_idx2word[i] for i in y_train[k])
    print(sent)
    print("prediction : ",decode_sequence(np.array([x_train[k]])))
    print('\n')

