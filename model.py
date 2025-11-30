import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# Hyperparameters
MAX_FEATURES = 20000  # top words to consider
MAX_LEN = 150
EMBEDDING_DIM = 64
LSTM_UNITS = 64

# Attention Layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight",
                                 shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform")
        self.b = self.add_weight(name="att_bias",
                                 shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

# Build model function
def build_model(weights_path=None):
    inp = Input(shape=(MAX_LEN,))
    embed = Embedding(MAX_FEATURES, EMBEDDING_DIM)(inp)
    lstm = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(embed)
    att = AttentionLayer()(lstm)
    out = Dense(1, activation="sigmoid")(att)
    model = Model(inputs=inp, outputs=out)
    if weights_path:
        model.load_weights(weights_path)
    return model
