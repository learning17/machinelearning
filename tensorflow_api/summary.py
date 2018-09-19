import tensorflow as tf
summary_writer = tf.summary.FileWriter('/tmp/test')
summary = tf.Summary(value=[
    tf.Summary.Value(tag="summary_tag", simple_value=0), 
    tf.Summary.Value(tag="summary_tag2", simple_value=1),
])
summary_writer.add_summary(summary, 1)

summary = tf.Summary(value=[
    tf.Summary.Value(tag="summary_tag", simple_value=1), 
    tf.Summary.Value(tag="summary_tag2", simple_value=3),
])
summary_writer.add_summary(summary, 2)

summary_writer.close()