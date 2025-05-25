📜 Shakespeare Next-Word Prediction using LSTM

This project involves building a next-word prediction model trained on Shakespeare's text using an LSTM (Long Short-Term Memory) neural network. 
The dataset was preprocessed by lowercasing, tokenizing, removing stopwords, and mapping words to indices to create training sequences. 
An LSTM-based model was implemented with an Embedding layer, a single LSTM layer, and a Dense softmax output layer. 
The model was trained to predict the next word in a sequence of five words. 
Despite high training accuracy, the model showed overfitting with high validation loss, indicating room for improvement in generalization. 
A prediction function was also created to generate text continuations based on user input.

In order to run this code for yourself:

Run the LSTM Notebook till you have saved the trained LSTM Model, you can edit the code in order to enhance the work I have done. 
The model can be tested on the notebook as well below the saved model or you can use my app.py file and run it locally.
