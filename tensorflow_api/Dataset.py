import tensorflow as tf

def parse(img, label):
    return "1",2

train_imgs = tf.constant(['train/img1.png', 'train/img2.png',
                            'train/img3.png', 'train/img4.png',
                            'train/img5.png', 'train/img6.png'])
train_labels = tf.constant([0, 0, 0, 1, 1, 1])
data = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels))

data1 = data.map(parse)
data = data.concatenate(data1)

iterator = data.make_initializable_iterator()
train_imgs,train_labels = iterator.get_next()
with tf.Session() as sess:
    sess.run(iterator.initializer)
    while True:
        try:
            print(sess.run([train_imgs,train_labels]))
        except tf.errors.OutOfRangeError:
            print("*****")
            break
