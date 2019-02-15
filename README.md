# char-rnn

Small character-based language model implementation in TensorFlow with tf.data and keras APIs along with a browser demo using TensorFlow.js.

The task file takes hidden state sizes as a parameter and generates a neural network with stateful GRU cells of the specified layer sizes.

### Running

```
python trainer/task.py \
  --train-files $DATA_DIR \
  --epochs 100 \
  --steps-per-epoch 100 \
  --layers 64 64 64 \
  --job-dir $OUTPUT_DIR \
  --export-dir $OUTPUT_DIR
```

### Other Art

- https://github.com/martin-gorner/tensorflow-rnn-shakespeare

- https://karpathy.github.io/2015/05/21/rnn-effectiveness/

- https://arxiv.org/abs/1506.02078

### TODO

- make `VOCAB_LENGTH` a configurable hyperparameter to allow fitting arbitrary vocab sizes

- distribution without data parallelism
