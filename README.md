# Sentiment Analyzer

A **full-stack web application** that analyzes the sentiment of user input text using a deep learning model with attention mechanism. The app predicts **positive, negative, or neutral sentiment** and highlights important words that influence the model’s prediction.

---

## **Demo Screenshot**

<img width="1918" height="941" alt="image" src="https://github.com/user-attachments/assets/07cc45bc-e6e1-4454-ad75-791985d92306" />


## **Features**

- Analyze sentiment of any text input.
- Displays **positive and negative scores** along with **intensity**.
- Highlights important words using an **attention mechanism**:
  - Green shades indicate words influencing positive sentiment.
  - Red shades indicate words influencing negative sentiment.
- Responsive and modern UI with gradient backgrounds and smooth animations.

---

## **Tech Stack**

**Frontend:**
- HTML, CSS, JavaScript
- Fetch API for backend communication

**Backend:**
- Python, Flask
- TensorFlow / Keras (Deep Learning Model)
- Attention Mechanism for word-level importance

**Other:**
- Pickle for saving/loading the word index
- Numpy for array operations

---

## **Project Structure**
sentiment-analyzer/
│
├─ app.py # Flask backend
├─ model.py # Model architecture definition
├─ sentiment_weights.weights.h5 # Pretrained weights
├─ word_index.pkl # Word to index mapping
├─ templates/
│ └─ index.html # Frontend HTML file
├─ static/
│ └─ (optional CSS/JS if separated)
├─ README.md
└─ .gitignore
