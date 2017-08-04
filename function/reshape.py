#!/usr/bin/python
#################################################################
#
#    file: reshape.py
#   usage: ./reshape.py
#   brief:
#  author: *******
#   email: *******
# created: 2017-08-02 21:41:15
#
#################################################################
#!/usr/bin/python

import tensorflow as tf
import numpy as np

a = tf.constant([1,2,3,4,5,6,7,8])
r1 = tf.reshape(a,[2,4])
r2 = tf.reshape(a,[2,2,2])
r3 = tf.reshape(a,[-1,2])
r4 = tf.reshape(a,[-1])
with tf.Session() as sess:
    print sess.run(r1)
    print ("##################################")
    print sess.run(r2)
    print ("##################################")
    print sess.run(r3)
    print ("##################################")
    print sess.run(r4)

