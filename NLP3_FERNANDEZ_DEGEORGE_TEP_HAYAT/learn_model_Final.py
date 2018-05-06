import argparse
import pickle
import pandas as pd
import numpy as np
from nltk import FreqDist
from keras.preprocessing.text import text_to_word_sequence
from keras.models import Model
from keras.layers import Embedding, LSTM, Dense,  Input, Flatten, Reshape

#################### PREPROCESSING ####################

def preprocess_data(train_file):

    # Read the trainset
    data_train = pd.read_csv(train_file, header=0)
    X_train = data_train.mr.tolist() # convert rm part to a list
    y_train = data_train.ref.tolist() # convert ref part to a list

    # Preprocess
    X_train_seq,y_train_seq,dico_ = mr2oh(X_train,y_train) # convert train and test sets into lists of slots and values.
    y_train = [proc_text(y) for y in y_train_seq] # process text
    X_train = [proc_text(y) for y in X_train_seq] # process text
    dist = FreqDist(np.concatenate(y_train+X_train))
    i_to_w=list(dist.keys()) # create a list to convert index to words
    i_to_w.insert(0, '-PADDING-')
    i_to_w.insert(2, '<STOP>')
    w_to_i={word: idx for idx, word in enumerate(i_to_w)} # dictionary that converts words to their corresponding index
    
    X_train_oh=ref2oh(X_train,w_to_i) # convert words in ref sentences into their indexes
    y_train_oh=ref2oh(y_train,w_to_i) # convert words in ref sentences into their indexes

    return X_train_oh,y_train_oh,i_to_w,dico_


def mr2oh(data, sentences):

    # create sequences of w2v representation from MR column
    seq = []
    dictio=[] # will contain an array of dictionaries (one for each rm), with rm slots and their values in each dictionary

    # Take each row of the rm part
    for mr in data:
        chunks =''
        dic={} # dictionary containing slots and their values in the rm
        # Separate each word in the rm
        for s in mr.split(','):
            s=s.strip()
            separator = s.find('[') # identify separator as first bracket
            chunk = s[:separator] # take the word before each separator, which will be a slot
            if chunk == "customer rating":
                chunk= "<customer_rating>" # rewrite the customer rating slot
            else:
                chunk='<'+chunk+'>' # add signs before and after slots to identify the slots
            chunks+=' '+chunk # add the chunk values to the chunks list

            value = s[separator + 1:-1] # value of each slot of the rm
            dic[chunk]=value # add this slot and value to our dictionary
            chunks+=' '+value
        seq.append(chunks) # add the chunks list to the seq array
        dictio.append(dic) # add the dictionary with slot and values for the rm to dictio

    y_train=[] # array containing the slots of words for each ref sentence
    for i,sentence in enumerate(sentences):
        for key,value in dictio[i].items():
            sentence=sentence.replace(value,' '+key+' ') # replace the word in the ref sentence by a slot if the words appears in the rm dictionary
        y_train.append(sentence) # add the new sentence to y_train
            

    return seq,y_train,dictio


def ref2oh(data, ref_word2idx):
    seq_oh = []

    for i in range(len(data)):
        seq_oh.append([ref_word2idx[j] for j in data[i]]) # convert the words into their corresponding index

    return seq_oh


def proc_text(ref, full_stop=True):
    ref ='<go> '+ref # add <go> before each ref sentence
    chars_to_filter = '!"#$%&()*+,-/:;=?@[\\]^`{|}~\t\n '
    if full_stop:
        ref = ref.replace('.', ' . ') # full stop interpreted as token
        
        # Remove punctuations from the text, with only words remaining:
        return text_to_word_sequence(ref, filters=chars_to_filter)

    else:

        return text_to_word_sequence(ref, filters=chars_to_filter)

# Define the arguments in the commands when running
parser = argparse.ArgumentParser(description='Machine Reading train file')
parser.add_argument('--train_dataset',help='CSV file containing train dataset',
	required=True)
parser.add_argument('--output_model_file',help='Path to LSTM model, .h5 format',
	required=True)

def main():


    args = parser.parse_args()

    # ---- LOAD DATA ----
    print('\nLoading data...')
    
    train_file = args.train_dataset
    x_train, y_train,y_idx2word,dico = preprocess_data(train_file)

    
    print('Utterance vocab size:', len(y_idx2word))
    
    # ---- BUILD THE MODEL ----
    
    print('\nBuilding language generation model...')
    stop_index=y_idx2word.index('<STOP>')
    #building generator of for the data : in order to minimize the usage of memory : create the target output at each iteration
    def gener(x_train,y_train,y_idx2word):
        for i in range(len(y_train)):
            outputs=np.zeros((1,len(y_train[i]),len(y_idx2word)))
            for j,w in enumerate(y_train[i]):
                if j!=0:
                    outputs[0,j-1,w]=1
            outputs[0,len(y_train[i])-1,stop_index]=1
            yield([np.array([x_train[i]]),np.array([y_train[i]])],outputs)
    
    latent_dim=64
    num_decoder_tokens=len(y_idx2word) # 2487 words
    vocab_size=len(y_idx2word) # 2487 words
    
    ###########ENCODER##############

    # Encode with a deep LSTM network
    encoder_inputs = Input(shape=(None,)) #receive the data, input is none so that sequence with different lenght can be received
    x = Embedding(vocab_size, latent_dim)(encoder_inputs) #transform the sequence of integer into dense vector
    y,s_1,s_2=LSTM(latent_dim, return_state=True, return_sequences=True)(x) # create first layer of LSTM and retrieve the hidden states of 1st LSTM layer
    z,s_3,s_4=LSTM(latent_dim, return_state=True, return_sequences=True)(y) # create second layer of LSTM and retrieve the hidden states of 2nd LSTM layer
    encoder_states_1 = [s_1,s_2]
    encoder_states_2 = [s_3,s_4]
    
    ###########DECODER##############

    decoder_inputs = Input(shape=(None,))
    emb = Embedding(num_decoder_tokens, latent_dim)
    x=emb(decoder_inputs)

    # Decode with a deep LSTM network
    decoder_lstm_1=LSTM(latent_dim, return_sequences=True) # create first LSTM decoder
    decoder_lstm_2=LSTM(latent_dim, return_sequences=True) # create second LSTM decoder
    y = decoder_lstm_1(x, initial_state=encoder_states_1) # launch the first and second decoder lstm with the encoder state
    z = decoder_lstm_2(y, initial_state=encoder_states_2)
    decoder_dense= Dense(num_decoder_tokens, activation='softmax') #array of the same size of the vocabulary.
    decoder_outputs = decoder_dense(z)
    
    ####DEFINITION OF THE MODEL#########
    model =Model([encoder_inputs, decoder_inputs], decoder_outputs) 
    
    #########COMPILE AND RUN#############
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy') # rmsprop is the optimizer that is the most adapted to the loss
    
    print(model.summary())
    
    epochs=10
    for _ in range(epochs):
        g=gener(x_train,y_train,y_idx2word)
        model.fit_generator(g, epochs=1,steps_per_epoch=len(x_train))
    
    # Save the model and the dictionnary
    print("Saving model ....")
    output_model = args.output_model_file
    output_dico='dico_'+output_model
    if '.h5' not in output_model:
        output_model=output_model+'.h5'
    print(output_model)
    model.save(output_model)
    f = open(output_dico,'wb')
    pickle.dump(y_idx2word,f)
    f.close()
    
if __name__ == '__main__':
	main()
