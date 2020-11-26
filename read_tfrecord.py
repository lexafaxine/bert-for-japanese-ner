import tensorflow as tf
if __name__ == '__main__':
    for example in tf.python_io.tf_record_iterator("output/result_dir/eval.tf_record"):
        print(tf.train.Example.FromString(example))