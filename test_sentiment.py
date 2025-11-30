import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import build_model, MAX_FEATURES, MAX_LEN
import pickle

# Load word index
with open("word_index.pkl", "rb") as f:
    word_index = pickle.load(f)

# Load trained model
model = build_model("sentiment_weights.weights.h5")


def text_to_sequence(text):
    # Lowercase and remove non-alphanumeric characters
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    words = text.split()

    # Map words to IMDB indices with correct offset (+3)
    seq = [word_index.get(w, 2) + 3 for w in words]  # 2 = <UNK>

    # Clip indices to MAX_FEATURES
    seq = [i if i < MAX_FEATURES else 2 for i in seq]

    # Pad sequence
    return pad_sequences([seq], maxlen=MAX_LEN)


# Example sentences for dynamic testing
sentences = [
    "I absolutely loved this movie, it was amazing!",
    "This was the worst movie I have ever seen.",
    "The plot was boring and the acting was terrible.",
    "An excellent film with a captivating story.",
    "I fell asleep halfway through, it was so boring.",
    "The cinematography was stunning and the story was great.",
    "I would never watch this movie again.",
    "A masterpiece! Brilliant acting and plot.",
    "Terrible, just terrible. Waste of time.",
    "Fun and entertaining from start to finish."
]

# Predict sentiment dynamically
for s in sentences:
    seq = text_to_sequence(s)
    pred = float(model.predict(seq)[0][0])
    sentiment = "Positive" if pred > 0.5 else "Negative"
    print(f"{s} -> {sentiment} (Score: {pred:.4f})")
