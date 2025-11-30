from flask import Flask, request, render_template, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle
import re
from model import build_model, MAX_FEATURES, MAX_LEN

app = Flask(__name__)

# Load word index
with open("word_index.pkl", "rb") as f:
    word_index = pickle.load(f)

# Load model
model = build_model("sentiment_weights.weights.h5")

def text_to_sequence(text):
    """Preprocess text and convert to padded sequence."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    words = text.split()
    seq = [word_index.get(w, 2) + 3 for w in words]  # 2 = <UNK>
    seq = [i if i < MAX_FEATURES else 2 for i in seq]
    return pad_sequences([seq], maxlen=MAX_LEN), words

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    seq, words = text_to_sequence(text)

    # Attention extraction
    try:
        attention_layer = model.get_layer("attention_layer")
        att_model = tf.keras.Model(inputs=model.input,
                                   outputs=[attention_layer.output, model.output])
        att_output, pred = att_model.predict(seq)
        pred_score = float(pred[0][0])
        att_values = att_output[0]
        # Handle 1D vs 2D attention
        word_scores = att_values.mean(axis=1) if att_values.ndim==2 else att_values
        word_scores = word_scores[-len(words):]
    except:
        pred_score = float(model.predict(seq)[0][0])
        word_scores = [0.0]*len(words)

    positive_score = pred_score
    negative_score = 1 - pred_score

    # Intensity
    if positive_score > 0.8:
        intensity = "Strongly Positive"
    elif positive_score > 0.6:
        intensity = "Positive"
    elif positive_score > 0.4:
        intensity = "Neutral"
    elif positive_score > 0.2:
        intensity = "Negative"
    else:
        intensity = "Strongly Negative"

    words_attention = [{"word": w, "score": float(s)} for w,s in zip(words, word_scores)]

    return jsonify({
        "positive": round(positive_score,4),
        "negative": round(negative_score,4),
        "intensity": intensity,
        "words_attention": words_attention
    })

if __name__ == "__main__":
    app.run(debug=True)
