import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from preproc_2_V2 import *

train_file = 'trainset.csv'
dev_file = 'devset.csv'
max_input_seq_len = 50  # number of words the MRs should be truncated/padded to
max_output_seq_len = 50  # number of words the utterances should be truncated/padded to
vocab_size = 10000

print('\nLoading data...')
# word2idx, idx2word = preproc.load_vocab(path_to_vocab)
x_train, y_train,x_dev,y_dev, y_idx2word,dico = preprocess_data(train_file, dev_file, vocab_size, max_input_seq_len, max_output_seq_len, oh=True)
def gener(x_train,y_train,y_idx2word):
    for i in range(len(y_train)):
        outputs=np.zeros((1,len(y_train[i]),len(y_idx2word)))
        for j,w in enumerate(y_train[i]):
            if w!=0 and w!=1:
                outputs[0,j-1,w]=1
        outputs[0,len(y_train[i])-1,2]=1
        yield([np.array([x_train[i]]),np.array([y_train[i]])],outputs)

g=gener(x_train,y_train,y_idx2word)

src_vocab_size=2000#len(y_idx2word)
embedding_size=128
#encoder_inputs = size[max length of the sentence,batch size, taille du vector]
num_units=64
source_sequence_length=50
batch_size=16
 tgt_vocab_size=3000



encoder_inputs = tf.placeholder(tf.int32, [50,batch_size])
decoder_inputs = tf.placeholder(tf.int32, [50,batch_size])
decoder_outputs = tf.placeholder(tf.int32, [50,batch_size])
target_weights_2 = tf.get_variable("target_weights_2", [50,batch_size],tf.float32)
source_sequence_length_5=tf.get_variable("source_sequence_length_5",16,dtype=tf.int32)
decoder_lengths=tf.placeholder(tf.int32,50)



# Embedding
embedding_encoder = tf.get_variable("embedding_encoder", [src_vocab_size, embedding_size])
# Look up embedding:
#   encoder_inputs: [max_time, batch_size]
#   encoder_emb_inp: [max_time, batch_size, embedding_size]
encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, encoder_inputs)




# Build RNN cell Encoder
encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

# defining initial state
initial_state_enc = encoder_cell.zero_state(batch_size, dtype=tf.float32)
# Run Dynamic RNN
#   encoder_outputs: [max_time, batch_size, num_units]
#   encoder_state: [batch_size, num_units]
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb_inp,initial_state=initial_state_enc,sequence_length=source_sequence_length_5,time_major=True)
#


# Embedding DEcoder
embedding_decoder = tf.get_variable("embedding_decoder", [src_vocab_size, embedding_size])
# Look up embedding:
#   encoder_inputs: [max_time, batch_size]
#   encoder_emb_inp: [max_time, batch_size, embedding_size]
decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, decoder_inputs)

# Build RNN cell
decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

# attention_states: [batch_size, max_time, num_units]
attention_states = tf.transpose(encoder_outputs, [1,0,2])

# Create an attention mechanism
attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units, attention_states,
    memory_sequence_length=source_sequence_length_5)

decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
    attention_layer_size=num_units)

# defining initial state
initial_state_dec = decoder_cell.zero_state(dtype=tf.float32,batch_size=batch_size).clone(cell_state=encoder_state)


projection_layer = layers_core.Dense( tgt_vocab_size, use_bias=False)
# Helper
helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, source_sequence_length_5, time_major=True)
# Decoder

decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,initial_state =initial_state_dec,output_layer=projection_layer)
# Dynamic decoding
outputs, _,_ = tf.contrib.seq2seq.dynamic_decode(decoder)
logits = outputs.rnn_output





crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_outputs, logits=logits)
train_loss = (tf.reduce_sum(crossent * target_weights_2) / float(batch_size))


# Calculate and clip gradients
max_gradient_norm=1
params = tf.trainable_variables()
gradients = tf.gradients(train_loss, params)
clipped_gradients, grad_norm = tf.clip_by_global_norm(
    gradients, max_gradient_norm)

# Optimization
optimizer = tf.train.AdamOptimizer()
update_step = optimizer.apply_gradients( zip(clipped_gradients, params))



##############################################################################
###Training

init_op = tf.global_variables_initializer()


sess = tf.Session()

epoch_num=10
steps=len(x_train)/float(batch_size)
for epoch in range(epoch_num):
    
    for step in range(steps)
    
sess.run([update_step,train_loss, grad_norm,crossent],feed_dict={x: batch_x, y: batch_y})







