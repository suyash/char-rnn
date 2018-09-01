import tensorflow as tf

def create_iterator(pattern, batch_size, sequence_length, vocab_size, repeat=False):
    """
    Parameters
    ----------
    pattern : string
        glob pattern with files to read

    batch_size : integer
        batch size for input

    sequence_length : integer
        unroll size for rnn

    vocab_size : integer
        number of chars in vocab

    repeat : bool
        repeat if reached end
    """
    def encode(c):
        return tf.case({
            tf.equal(c, 9): lambda: 1,
            tf.equal(c, 10): lambda: 127 - 30,
            tf.logical_and(tf.greater_equal(c, 32), tf.less_equal(c, 126)): lambda: c - 30,
        }, default=lambda: 0, exclusive=True)

    def split(row):
        row = tf.decode_raw(row, out_type=tf.uint8)
        row = tf.cast(row, tf.int32)
        row = tf.map_fn(encode, row)
        l = tf.size(row)
        return tf.slice(row, [0], [l - 1]), tf.slice(row, [1], [l - 1])

    def pad(row):
        l = tf.size(row)
        p = batch_size * sequence_length
        r = l % p
        return tf.cond(
            tf.not_equal(r, 0),
            lambda: tf.concat([row, tf.zeros(p - r, dtype=tf.int32)], 0),
            lambda: row,
        )

    def transpose(row):
        row = tf.reshape(row, [batch_size, -1, sequence_length])
        row = tf.transpose(row, [1, 0, 2])
        return row

    dataset = tf.data.Dataset.list_files(pattern)
    dataset = dataset.shuffle(buffer_size=64)
    dataset = dataset.flat_map(lambda file: tf.data.Dataset.from_tensors(tf.read_file(file)))
    dataset = dataset.map(split)
    dataset = dataset.map(lambda f, l: (pad(f), pad(l)))
    dataset = dataset.map(lambda f, l: (transpose(f), transpose(l)))
    dataset = dataset.apply(tf.contrib.data.unbatch())

    dataset = dataset.map(lambda f, l: (tf.one_hot(f, vocab_size, 1.0, 0.0), tf.one_hot(l, vocab_size, 1.0, 0.0)))

    if repeat:
        dataset = dataset.repeat()

    iterator = dataset.make_one_shot_iterator()

    return iterator
