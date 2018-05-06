!!!!!!!!!!!!!!!!!!!!!!! HOW TO RUN THE MODEL !!!!!!!!!!!!!!!!!!!!!!

We have slightly changed the procedure to run the test_model.py. We added a command to load the trained model in learn_model.py. Please, try to follow the format indicated in the brackets when executing the commands.



$ python learn_model_Final.py --train_dataset [PATH TO TRAINING SET, .csv format] --output_model_file [PATH TO STORE MODEL, .h5 format]



Once the model is trained:



$ python test_model_final.py --test_dataset [PATH TO TEST SET, .csv format] --output_test_file [PATH TO STORE OUTPUT, .txt file] --lstm_model [PATH TO MODEL PREVIOUSLY STORED, .h5 format]



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



From the E2E dataset, our task is to build a natural language generator program that can generate an utterance from a given MR.

To tackle this challenge we draw inspiration from 'Dusek and Jurcicek, 2016', the baseline used for this challenge. 
We implemented a seq2seq model, with a personnalized word reranker. 

	Preprocessing : 

First on the preprocessing part, we converted the MR into a series of token and then to one hot vectors. 
We delixicalised the sentence (replace the information contained into the MR by <slots>) and transformed them into a sequence of integer. 

For the training part, we create an output sequence that is the same as the input sequence, but with a step of difference (i.e., the word in the third position in the input sequence will be in the second position in the output sequence). 
As this output will be compared to a prediciton layer of the same size of the vocabulary, we needed it to be a vector of the same size of the vocabulary. 
The dimension of this output is 42 000 lines x 50 words max x 2500 approximative size of the vocabulary, which take a lot of memory. In order to avoid this unecessary memory usage, we used a generator function. 

	Model :

For  the seq to seq model, we use an encoder-decoder neural network architecture with two layers of LSTM. 

First the encoder create the embedding for each token of the MR and process every embedding sequentially. The meaning of the sequence is suppose to be encoded in the hidden states of the encoder. 

Then the decoder is initialized with the hidden states of the encoder, process the target sequence and predict for each word of the input sequence the following word.


	Generation : 

To generate a sentence from a MR without target sequence, we used the previous trained model to predict a sentence starting with a token Start Of the Sentence,(here <go>). The EOS token is defined by the full stop '.' 

	
	Difficulties : 

- the time taken to train (approximatively 1H30 per epoch on our computer). So we never really reach the convergence of the model, and we don't really know what the sentence will look like after 10 full epochs on the training set.
- trouble to assess the quality of our approaches without full convergence of the model. 
- couldn't optimize the hyper parameter of the model 


	Experiments that we ran : 

- we tried to implement a simpler model with just one layer (see ‘experiment_1.py’ file but it is not for running), to reduce the computation time and reduce the time to convergence of the model.
- we tried to implement a model with bidirectionnal layer (see ‘experiment_2.py’ file but it is not for running), to see if the results were better, but they were actually worse. 
- we tried to implement a architecture in tensorflow with attention mechanism (n (Bahdanau et
al., 2015) (see ‘experiment_3.py’ file but it is not for running). 


Any question, let us know!

Students:
FERNANDEZ Hugo
TEP Kilian
DEGEORGE Jean
HAYAT Paul






