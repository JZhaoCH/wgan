# wgan-tensorflow
## Versions
* `tensorflow 1.4.0`
* `python 3.5`
## wgan_gp
### loss graphs

<p align="center">
  <img src="https://github.com/JZhaoCH/wgan-tensorflow/blob/master/image/wgan_gp/wgan_gp_d_loss.jpg" title="discriminator loss">
</p>
<p align="center">
  <img src="https://github.com/JZhaoCH/wgan-tensorflow/blob/master/image/wgan_gp/wgan_gp_g_loss.jpg" title="generator loss">
</p>
<p align="center">
  <img src="https://github.com/JZhaoCH/wgan-tensorflow/blob/master/image/wgan_gp/wgan_gp_gradient_penalty.jpg" title="gradient penalty loss">
</p>

### generating images
* `tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9), after 40K iteration`
<p align="center">
  <img src="https://github.com/JZhaoCH/wgan-tensorflow/blob/master/image/wgan_gp/wgan_gp_generating_image1.jpg">
</p>
<p align="center">
  <img src="https://github.com/JZhaoCH/wgan-tensorflow/blob/master/image/wgan_gp/wgan_gp_generating_image2.jpg">
</p>
<p align="center">
  <img src="https://github.com/JZhaoCH/wgan-tensorflow/blob/master/image/wgan_gp/wgan_gp_generating_image3.jpg">
</p>

## References

1. `Gulrajani I, Ahmed F, Arjovsky M, et al. Improved training of wasserstein gans[C]//Advances in Neural Information Processing Systems. 2017: 5767-5777.`
2. `Arjovsky M, Chintala S, Bottou L. Wasserstein gan[J]. arXiv preprint arXiv:1701.07875, 2017.`
3. `https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html`
4. `https://zhuanlan.zhihu.com/p/25071913`
5. `https://github.com/Zardinality/WGAN-tensorflow`
6. `Liu Z, Luo P, Wang X, et al. Deep learning face attributes in the wild[C]//Proceedings of the IEEE International Conference on Computer Vision. 2015: 3730-3738.`
