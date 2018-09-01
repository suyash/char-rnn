import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import LambdaCallback, TensorBoard
from tensorflowjs.converters import save_keras_model

from input import create_iterator
from model import create_model
from utils import encode, decode, sample

VOCAB_SIZE = 98

def run(
    train_files,
    batch_size,
    epochs,
    steps_per_epoch,
    learning_rate,
    learning_rate_decay,
    layers,
    rnn_sequence_length,
    dropout_pdrop,
    predict_length,
    export_dir,
    job_dir,
):
    model = create_model(layers, VOCAB_SIZE, learning_rate, learning_rate_decay, batch_size, dropout_pdrop)
    print(model.summary())

    def on_epoch_end(epoch, logs):
        c = "S"

        print("\n", end="")

        for _ in range(predict_length):
            print(c, end="")
            inp = np.zeros([batch_size, 1, VOCAB_SIZE])
            for i in range(batch_size):
                inp[i][0][encode(ord(c))] = 1.0
            prob = model.predict(inp, batch_size=batch_size)
            prob = np.reshape(prob, [batch_size, VOCAB_SIZE])
            prob = np.sum(prob, axis=0)
            rc = sample(prob)
            c = chr(decode(rc))

        print("\n")

    train_iterator = create_iterator(train_files, batch_size, rnn_sequence_length, VOCAB_SIZE, True)
    eval_iterator = create_iterator(train_files, batch_size, rnn_sequence_length, VOCAB_SIZE, True)

    model.fit(
        train_iterator,
        shuffle=False,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=eval_iterator,
        validation_steps=1,
        callbacks=[
            TensorBoard(log_dir=job_dir),
            LambdaCallback(on_epoch_end=on_epoch_end),
        ],
    )

    if export_dir is not None:
        model.save("%s/model.h5" % export_dir)
        save_keras_model(model, "%s/web" % export_dir)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train-files',
        help="""Local or GCS path to training data""",
        required=True
    )

    parser.add_argument(
        '--batch-size',
        help="""Batch size for training and eval steps""",
        default=32,
        type=int
    )

    parser.add_argument(
        '--epochs',
        help="""Number of epochs""",
        default=200,
        type=int
    )

    parser.add_argument(
        '--steps-per-epoch',
        help="""Number of training steps in an epoch""",
        default=100,
        type=int
    )

    parser.add_argument(
        '--learning-rate',
        help="""Learning rate value for the optimizers""",
        default=0.001,
        type=float
    )

    parser.add_argument(
        '--learning-rate-decay',
        help="""Learning rate decay value for the optimizers""",
        default=1e-5,
        type=float
    )

    parser.add_argument(
        '--layers',
        help="""List of recurrent layers with sizes""",
        nargs="+",
        default=[64, 64, 64],
        type=int
    )

    parser.add_argument(
        '--rnn-sequence-length',
        help="""Mumber of times the RNN is unrolled""",
        default=40,
        type=int
    )

    parser.add_argument(
        '--dropout-pdrop',
        help="""The fraction of input units to drop""",
        default=0.25,
        type=float
    )

    parser.add_argument(
        '--predict-length',
        help="""Length of generated prediction""",
        default=500,
        type=int
    )

    parser.add_argument(
        '--export-dir',
        help="""Local/GCS location to export model, if None, the model is not exported""",
        default=None,
        required=False
    )

    parser.add_argument(
        '--job-dir',
        help="""Local/GCS location to write checkpoints and export models""",
        required=True
    )

    HYPERPARAMS, _ = parser.parse_known_args()
    run(**HYPERPARAMS.__dict__)

if __name__ == '__main__':
    main()
