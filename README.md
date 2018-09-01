# char-rnn

```
pipenv run python trainer/task.py \
  --train-files $DATA_DIR \
  --epochs 100 \
  --steps-per-epoch 100 \
  --layers 64 64 64 \
  --job-dir $OUTPUT_DIR \
  --export-dir $OUTPUT_DIR
```

### TODO

- make `VOCAB_LENGTH` a configurable hyperparameter to allow fitting arbitrary vocab sizes

- distribution without data parallelism
