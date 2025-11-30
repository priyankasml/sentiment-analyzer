import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import build_model, MAX_FEATURES, MAX_LEN
import pickle
import json

BATCH = 64
EPOCHS = 10  # train longer for better learning

# Load dataset
print("Loading IMDB dataset...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)

# Pad sequences
x_train = pad_sequences(x_train, maxlen=MAX_LEN)
x_test = pad_sequences(x_test, maxlen=MAX_LEN)

# Build and compile model
model = build_model()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# Train model
print("Training model...")
model.fit(x_train, y_train, batch_size=BATCH, epochs=EPOCHS, validation_data=(x_test, y_test))

# Save weights
model.save_weights("sentiment_weights.weights.h5")

# Save word_index for preprocessing
word_index = imdb.get_word_index()
with open("word_index.pkl", "wb") as f:
    pickle.dump(word_index, f)

# Save meta info
with open("meta.json", "w") as f:
    json.dump({"MAX_FEATURES": MAX_FEATURES, "MAX_LEN": MAX_LEN}, f)

print("Training completed successfully!")
