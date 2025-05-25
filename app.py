from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model("shakespeare_lstm_model3.h5")

# Load the vocabulary
with open('vocabulary.pkl', 'rb') as f:
    vocabulary = pickle.load(f)

# Function to convert a sequence of words to indices using your vocabulary
def text_to_sequence(text, vocabulary):
    words = text.lower().split()  # Tokenize and lowercase input text
    sequence = [vocabulary.get(word, 0) for word in words]  # Convert words to indices, use 0 for unknown words
    return sequence

# Function to convert index to a word using the reverse vocabulary
def index_to_word(index, vocabulary):
    reverse_vocab = {i: word for word, i in vocabulary.items()}  # Reverse the vocabulary
    return reverse_vocab.get(index, '<unk>')  # Get word from index, return '<unk>' if not found

# Function to predict the next word
def predict_next_word(model, input_text, vocabulary, sequence_length=1):
    sequence = text_to_sequence(input_text, vocabulary)
    if len(sequence) < sequence_length:
        sequence = [0] * (sequence_length - len(sequence)) + sequence  # Pad with 0s
    else:
        sequence = sequence[-sequence_length:]  # Truncate to the correct length
    
    input_sequence = pad_sequences([sequence], maxlen=sequence_length, padding='pre')
    predicted_probs = model.predict(input_sequence)
    predicted_index = np.argmax(predicted_probs, axis=-1)[0]
    predicted_word = index_to_word(predicted_index, vocabulary)
    
    return predicted_word

# Route to render the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the prediction
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['input_text']  # Get input from the user
    predicted_word = predict_next_word(model, input_text, vocabulary)
    return jsonify({'input': input_text, 'predicted_word': predicted_word})

if __name__ == '__main__':
    app.run(debug=True)
