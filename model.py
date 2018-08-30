import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import os
from utils import read_and_decode, save_sample

max_iter = 10000
batch_size = 64
z_dim = 128
learning_rate_gen = 5e-5
learning_rate_dis = 5e-5
image_height = 128
image_width = 128
channel = 3
clamp_lower = -0.01
clamp_upper = 0.01
device = '/gpu:2'
ckpt_dir = './ckpt_wgan'
tfrecords_dir = '../data/celeba_tfrecords'
sample_dir = './sample'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def generator(z):
    first_feature_h = 8
    first_feature_w = 8
    with tf.variable_scope('generator'):
        # tf.contrib.layers.fully_connected(inputs, num_outputs, activation_fn=tf.nn.relu, normalizer_fn=None)
        train = layers.fully_connected(z, first_feature_h*first_feature_w*512, activation_fn=tf.nn.leaky_relu,
                                       normalizer_fn=layers.batch_norm)
        # tf.contrib.layers.conv2d_transpose(inputs, num_outputs, kernel_size,stride=1, padding='SAME')
        train = tf.reshape(train, [-1, first_feature_h, first_feature_w, 512])

        train = layers.conv2d_transpose(train, 256, 3, stride=2, activation_fn=tf.nn.relu,
                                        normalizer_fn=layers.batch_norm, padding='SAME',
                                        weights_initializer=tf.random_normal_initializer(0, 0.02))

        train = layers.conv2d_transpose(train, 128, 3, stride=2, activation_fn=tf.nn.relu,
                                        normalizer_fn=layers.batch_norm, padding='SAME',
                                        weights_initializer=tf.random_normal_initializer(0, 0.02))

        train = layers.conv2d_transpose(train, 64, 3, stride=2, activation_fn=tf.nn.relu,
                                        normalizer_fn=layers.batch_norm, padding='SAME',
                                        weights_initializer=tf.random_normal_initializer(0, 0.02))

        train = layers.conv2d_transpose(train, channel, 3, stride=2, activation_fn=tf.nn.tanh, padding="SAME",
                                        weights_initializer=tf.random_normal_initializer(0, 0.02))

        return train


def discriminator(image, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        image = layers.conv2d(image, 64, kernel_size=3, stride=2, activation_fn=tf.nn.leaky_relu,
                              normalizer_fn=layers.batch_norm)
        image = layers.conv2d(image, 128, kernel_size=3, stride=2, activation_fn=tf.nn.leaky_relu,
                              normalizer_fn=layers.batch_norm)
        image = layers.conv2d(image, 256, kernel_size=3, stride=2, activation_fn=tf.nn.leaky_relu,
                              normalizer_fn=layers.batch_norm)
        image = layers.conv2d(image, 512, kernel_size=3, stride=2, activation_fn=tf.nn.leaky_relu)

        image = tf.reshape(image, [batch_size, -1])

        logit = layers.fully_connected(image, 1, activation_fn=None)
    return logit


def build_graph():
    images_data_set = read_and_decode(tfrecords_dir, height=image_height, width=image_width)
    true_images = tf.train.shuffle_batch([images_data_set], batch_size=batch_size, capacity=5000,
                                         min_after_dequeue=2500, num_threads=2)

    noises = np.random.normal(0.0, 1.0, [batch_size, z_dim]).astype(np.float32)
    fake_images = generator(noises)
    true_logits = discriminator(true_images)
    fake_logits = discriminator(fake_images, reuse=True)
    dis_loss = tf.reduce_mean(fake_logits-true_logits)
    gen_loss = tf.reduce_mean(-fake_logits)

    gen_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    dis_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    gen_counter = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    dis_counter = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)

    gen_opt = layers.optimize_loss(loss=gen_loss, learning_rate=learning_rate_gen, optimizer=tf.train.RMSPropOptimizer,
                                   variables=gen_params, global_step=gen_counter)
    dis_opt = layers.optimize_loss(loss=dis_loss, learning_rate=learning_rate_dis, optimizer=tf.train.RMSPropOptimizer,
                                   variables=dis_params, global_step=dis_counter)

    dis_clipped_var = [tf.assign(var, tf.clip_by_value(var, clamp_lower, clamp_upper)) for var in dis_params]
    with tf.control_dependencies([dis_opt]):
        dis_opt = tf.tuple(dis_clipped_var)
    return gen_opt, dis_opt, fake_images


def train():
    with tf.device(device):
        gen_opt, dis_opt, fake_images = build_graph()
    saver = tf.train.Saver()
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.8

    with tf.Session(config=session_config) as sess:
        # add tf.train.start_queue_runners, it's important to start queue for tf.train.shuffle_batch
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        for iter_count in range(max_iter):
            if iter_count < 25 or iter_count % 500 == 0:
                dis_iter = 100
            else:
                dis_iter = 5
            for i in range(dis_iter):
                sess.run(dis_opt)
            sess.run(gen_opt)
            print('iter_count:', iter_count)
            if iter_count % 100 == 0:
                sample_path = os.path.join(sample_dir, '%d.png' % iter_count)
                sample = sess.run(fake_images)
                sample = (sample+1)/2
                save_sample(sample, [4, 4], sample_path)
                print('save sample:', sample_path)
            if iter_count % 1000 == 999:
                if os.path.exists(ckpt_dir):
                    os.mkdir(ckpt_dir)
                ckpt_path = os.path.join(ckpt_dir, "model.ckpt")
                saver.save(sess, ckpt_path, global_step=iter_count)
                print('save ckpt:', ckpt_dir)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    train()