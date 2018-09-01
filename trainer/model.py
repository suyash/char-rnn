from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, GRU
from tensorflow.keras.optimizers import Adam

def create_model(layers, vocab_size, learning_rate, learning_rate_decay, batch_size, dropout):
    model = Sequential()

    model.add(GRU(layers[0], stateful=True, return_sequences=True, dropout=dropout, batch_input_shape=(batch_size, None, vocab_size)))

    for i in range(len(layers) - 1):
        model.add(GRU(layers[i + 1], stateful=True, return_sequences=True, dropout=dropout))

    model.add(Dense(vocab_size))
    model.add(Activation("softmax"))

    optimizer = Adam(lr=learning_rate, decay=learning_rate_decay)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["acc"])

    return model
