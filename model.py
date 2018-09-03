import tensorflow as tf
import tensorflow.contrib.layers as layers
import os
from utils import read_and_decode, save_sample

max_iter = 200000
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
tfrecords_dir = '../data/celeba_tfrecords'
sample_dir = './sample_wgan'
ckpt_dir = './ckpt_wgan'
log_dir = './log_wgan'
load_model = True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

wgan_gp = True
gp_lambda = 10

if wgan_gp is True:
    sample_dir = './sample_wgan_gp'
    ckpt_dir = './ckpt_wgan_gp'
    log_dir = './log_wgan_gp'


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

    noises = tf.random_normal([batch_size, z_dim], mean=0.0, stddev=1.0)
    fake_images = generator(noises)
    true_logits = discriminator(true_images)
    fake_logits = discriminator(fake_images, reuse=True)
    dis_loss = tf.reduce_mean(fake_logits-true_logits)
    # ------------------------------------
    # add gradient penalty
    if wgan_gp is True:
        alpha = tf.random_uniform([batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolated = alpha*true_images + (1-alpha)*fake_images
        inte_logit = discriminator(interpolated, reuse=True)
        gradients = tf.gradients(inte_logit, [interpolated, ])[0]
        grad_l2 = tf.sqrt(tf.reduce_mean(tf.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean(tf.square(grad_l2-1))
        gp_loss_sum = tf.summary.scalar("gp_loss", gradient_penalty)
        grad = tf.summary.scalar("grad_norm", tf.nn.l2_loss(gradients))
        dis_loss += gp_lambda * gradient_penalty

    gen_loss = tf.reduce_mean(-fake_logits)
    gen_loss_sum = tf.summary.scalar("gen_loss", gen_loss)
    dis_loss_sum = tf.summary.scalar("dis_loss", dis_loss)

    gen_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    dis_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    gen_counter = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    dis_counter = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    # ------------------------
    if wgan_gp is False:
        gen_opt = layers.optimize_loss(loss=gen_loss, learning_rate=learning_rate_gen,
                                       optimizer=tf.train.RMSPropOptimizer,
                                       variables=gen_params, global_step=gen_counter)
        dis_opt = layers.optimize_loss(loss=dis_loss, learning_rate=learning_rate_dis,
                                       optimizer=tf.train.RMSPropOptimizer,
                                       variables=dis_params, global_step=dis_counter)
    else:
        gen_opt = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9).\
            minimize(gen_loss, var_list=gen_params, global_step=gen_counter)
        dis_opt = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9).\
            minimize(dis_loss, var_list=dis_params, global_step=dis_counter)
    # ----------------------------------
    # clip weight in discriminator
    if wgan_gp is False:
        dis_clipped_var = [tf.assign(var, tf.clip_by_value(var, clamp_lower, clamp_upper)) for var in dis_params]
        # merge the clip operations on discriminator variables
        with tf.control_dependencies([dis_opt]):
            dis_opt = tf.tuple(dis_clipped_var)
    return gen_opt, dis_opt, fake_images


def train():
    with tf.device(device):
        gen_opt, dis_opt, fake_images = build_graph()
    merged_all = tf.summary.merge_all()
    saver = tf.train.Saver()
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.8

    with tf.Session(config=session_config) as sess:
        # add tf.train.start_queue_runners, it's important to start queue for tf.train.shuffle_batch
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        iter_start = 0
        if load_model:
            lasted_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
            if lasted_checkpoint is not None:
                saver.restore(sess, lasted_checkpoint)
                print('load model:', lasted_checkpoint)
                iter_start = int(lasted_checkpoint.split('/')[-1].split('-')[-1])+1
            else:
                print('init global variables')
                sess.run(tf.global_variables_initializer())
        for iter_count in range(iter_start, max_iter):

            # set dis_iter: the count of training discriminator in a iteration
            if iter_count < 25 or iter_count % 500 == 0:
                dis_iter = 100
            else:
                dis_iter = 5

            # train discriminator
            for i in range(dis_iter):
                if iter_count % 100 == 99 or i == 0:
                    _, merged = sess.run([dis_opt, merged_all])
                    summary_writer.add_summary(merged, iter_count)
                else:
                    sess.run(dis_opt)

            # train generator
            if iter_count % 100 == 99:
                _, merged = sess.run([gen_opt, merged_all])
                summary_writer.add_summary(merged, iter_count)
            else:
                sess.run(gen_opt)

            # save sample
            if iter_count % 1000 == 999:
                sample_path = os.path.join(sample_dir, '%d.jpg' % iter_count)
                sample = sess.run(fake_images)
                sample = (sample+1)/2
                save_sample(sample, [4, 4], sample_path)
                print('save sample:', sample_path)

            # save model
            if iter_count % 1000 == 999:
                if not os.path.exists(ckpt_dir):
                    os.mkdir(ckpt_dir)
                ckpt_path = os.path.join(ckpt_dir, "model.ckpt")
                saver.save(sess, ckpt_path, global_step=iter_count)
                print('save ckpt:', ckpt_dir)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    train()