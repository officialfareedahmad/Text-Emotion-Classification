import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import load_model  # Importing the model loader
from keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import streamlit as st
import warnings


warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('train.txt', sep=';')
data.columns = ['Text', 'Emotions']

# Tokenization and preprocessing
texts = data['Text'].tolist()
labels = data['Emotions'].tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
max_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length)

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

one_hot_labels = keras.utils.to_categorical(labels)

xtrain, xtest, ytrain, ytest = train_test_split(padded_sequences, one_hot_labels, test_size=0.2)

# Check if model exists, if not, train and save it
try:
    # Try to load the model
    model = load_model('emotion_model.h5')
    st.write("Loaded the pre-trained model.")
except:
    # If the model does not exist, train and save it
    st.write("Training the model...")
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=len(one_hot_labels[0]), activation='softmax'))
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(xtrain, ytrain, epochs=10, batch_size=32, validation_data=(xtest, ytest))
    model.save('emotion_model.h5')
    st.write("Model trained and saved.")

# Streamlit interface
st.title("Text Emotion Classification")
st.write("Enter a text to classify its emotion")

# Text input by user
input_text = st.text_area("Enter text here:", "")

# Prediction on user input
if st.button("Classify Emotion"):
    # Preprocessing the input text
    input_sequence = tokenizer.texts_to_sequences([input_text])
    padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
    
    # Making prediction
    prediction = model.predict(padded_input_sequence)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])
    
    # Display result
    #st.write(f"Predicted Emotion: {predicted_label[0]}")
    if predicted_label[0] == 'joy':
        st.markdown(f"### 🎉 Predicted Emotion: **{predicted_label[0]}** 😄")
    elif predicted_label[0] == 'anger':
        st.markdown(f"### 🔥 Predicted Emotion: **{predicted_label[0]}** 😡")
    elif predicted_label[0] == 'sadness':
        st.markdown(f"### 😢 Predicted Emotion: **{predicted_label[0]}** 🥹")
    elif predicted_label[0] == 'fear':
        st.markdown(f"### 😨 Predicted Emotion: **{predicted_label[0]}** 😱")
    else:
        st.markdown(f"### Predicted Emotion: **{predicted_label[0]}** 🙂")

    confidence = prediction[0][np.argmax(prediction[0])]
    st.write(f"Confidence: {confidence*100:.2f}%")
    st.balloons()
    st.snow()

