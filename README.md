# NLG
$ python learn_model_Final.py --train_dataset [PATH TO TRAINING SET, .csv format] --output_model_file [PATH TO STORE MODEL, .h5 format]

Once the model is trained:

$ python test_model_final.py --test_dataset [PATH TO TEST SET, .csv format] --output_test_file [PATH TO STORE OUTPUT, .txt file] --lstm_model [PATH TO MODEL PREVIOUSLY STORED, .h5 format]
