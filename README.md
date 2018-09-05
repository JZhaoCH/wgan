# wgan-tensorflow
## Versions
* tensorflow 1.4.0
* python 3.5
## wgan_gp
### loss graphs

<p align="center">
  <img src="https://github.com/JZhaoCH/wgan-tensorflow/blob/master/image/wgan_gp/wgan_gp_d_loss.jpg">
</p>
<p align="center">
  <img src="https://github.com/JZhaoCH/wgan-tensorflow/blob/master/image/wgan_gp/wgan_gp_g_loss.jpg">
</p>
<p align="center">
  <img src="https://github.com/JZhaoCH/wgan-tensorflow/blob/master/image/wgan_gp/wgan_gp_gradient_penalty.jpg">
</p>

### generating images
* tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9)
* after 40K iteration
<p align="center">
  <img src="https://github.com/JZhaoCH/wgan-tensorflow/blob/master/image/wgan_gp/wgan_gp_generating_image1.jpg">
</p>
<p align="center">
  <img src="https://github.com/JZhaoCH/wgan-tensorflow/blob/master/image/wgan_gp/wgan_gp_generating_image2.jpg">
</p>
<p align="center">
  <img src="https://github.com/JZhaoCH/wgan-tensorflow/blob/master/image/wgan_gp/wgan_gp_generating_image3.jpg">
</p>
