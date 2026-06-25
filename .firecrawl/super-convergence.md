[Sitemap](https://chingisoinar.medium.com/sitemap/sitemap.xml)

[Open in app](https://play.google.com/store/apps/details?id=com.medium.reader&referrer=utm_source%3DmobileNavBar&source=post_page---top_nav_layout_nav-----------------------------------------)

Sign up

[Sign in](https://medium.com/m/signin?operation=login&redirect=https%3A%2F%2Fchingisoinar.medium.com%2Fpaper-of-choice-super-convergence-very-fast-training-of-neural-networks-using-large-learning-265c1d7f6b99&source=post_page---top_nav_layout_nav-----------------------global_nav------------------)

[Medium Logo](https://medium.com/?source=post_page---top_nav_layout_nav-----------------------------------------)

Get app

[Write](https://medium.com/m/signin?operation=register&redirect=https%3A%2F%2Fmedium.com%2Fnew-story&source=---top_nav_layout_nav-----------------------new_post_topnav------------------)

[Search](https://medium.com/search?source=post_page---top_nav_layout_nav-----------------------------------------)

Sign up

[Sign in](https://medium.com/m/signin?operation=login&redirect=https%3A%2F%2Fchingisoinar.medium.com%2Fpaper-of-choice-super-convergence-very-fast-training-of-neural-networks-using-large-learning-265c1d7f6b99&source=post_page---top_nav_layout_nav-----------------------global_nav------------------)

![Unknown user](https://miro.medium.com/v2/resize:fill:32:32/1*dmbNkD5D-u45r44go_cf0g.png)

[Mastodon](https://me.dm/@chingisoinar)

# Paper of Choice: Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates

[![Ching (Chingis)](https://miro.medium.com/v2/resize:fill:32:32/1*tudn7-2ZXVGgClPR0WUFmg.jpeg)](https://chingisoinar.medium.com/?source=post_page---byline--265c1d7f6b99---------------------------------------)

[Ching (Chingis)](https://chingisoinar.medium.com/?source=post_page---byline--265c1d7f6b99---------------------------------------)

Follow

5 min read

·

Jan 6, 2022

19

[Listen](https://medium.com/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2Fplans%3Fdimension%3Dpost_audio_button%26postId%3D265c1d7f6b99&operation=register&redirect=https%3A%2F%2Fchingisoinar.medium.com%2Fpaper-of-choice-super-convergence-very-fast-training-of-neural-networks-using-large-learning-265c1d7f6b99&source=---header_actions--265c1d7f6b99---------------------post_audio_button------------------)

Share

Super-convergence or Competition Winning Learning Rates. Is the ultimate cheat-code for Machine Learning Competitions real? This paper was published in 2018 but I don’t hear about this concept from people instead I feel like there are much more discussions on architectures. Are we taking it for granted? Should we pay more attention to training strategies and data quality in 2022? I have high hopes for this year.

Hellooo, today my choice fell into this interesting piece, called ‘Super-Convergence’. The existence of super-convergence is linked to comprehending why Deep Neural Networks generalize well. One contributing factor to this phenomenon (super-convergence) was found to be training with one learning rate cycle and a large maximum learning rate. Particularly, large learning rates were observed to regularize the training, which I found surprising, hence other forms of regularization techniques (L2 regularization) can be reduced. Let’s dive into details of this work and enjoy it together.

Original Paper: [https://arxiv.org/pdf/1708.07120.pdf](https://arxiv.org/pdf/1708.07120.pdf)

## Introduction

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*pSbvYMDWdiEbYtrAHHB_2w.png)

taken from: [https://arxiv.org/pdf/1708.07120.pdf](https://arxiv.org/pdf/1708.07120.pdf)

Understanding the magic of Stochastic Gradient Descent algorithm (SGD) is still an open area of research. The authors show that large learning rates with the cyclical learning rate (CLR) can considerably speed up the training. This phenomenon is referred as ‘super-convergence’. The sub-figure (a) above shows that the same architecture, ResNet-56, trained only for 10,000 iterations via super-convergence methods outperforms its counterpart trained conventionally, which is extremely encouraging. The sub-figure (b) shows the results for a set of CLR stepsize values, where a cycle reached a learning rate of 3. Thus, the authors argue that large learning rates regularize training allowing other forms of regularization be reduced to maintain an optimal balance. Also, super-convergence does not need a lot of labeled training data.

## Background

> Conventional training, with SGD, implies the use of global learning rate (such as 0.1) and training a network as long as the performance plateaus after which the learning rate is typically reduced by a factor of 0.1. This experimental setting can be met in numerous works.

## Super-convergence

In their works, the author use cyclical learning rates (CLR), this learning rates scheduling algorithm is provided by many Deep Learning frameworks, well at least Pytorch (: . To use CLR, we specify min and max learning rate boundaries as well as a stepsize, which is the number of training iterations in the increasing half of a cycle.

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*dJbkbR-SVXuJcu6oFhNNqA.png)

taken from: [https://arxiv.org/pdf/1708.07120.pdf](https://arxiv.org/pdf/1708.07120.pdf)

The figure above is the LR range test. It starts with a very small learning rate and consistently increases until the test accuracy starts slipping. The maximum learning rate is at this peak point, whereas the minimum learning rate is set by dividing the maximum by a factor of 3 or 4. The optimal initial learning rate usually falls in between these values. The sub-figure (b) shows that ResNet 56 was able to consistently learn even with the learning rate of 3, which is extremely unusual. This phenomenon is indicative of potential for super-convergence.

> The authors suggest using one cycle that is smaller than the total number of iterations/epochs and allowing the LR to decrease several orders of magnitude less than the original LR for the rest of iterations/epochs. This is provided as OneCycleLR by Pytorch framework.

It was observed that reducing other forms of regularization and regularizing with very large learning rates improves the final performance.

## Estimating Optimal Learning Rates

I am not explaining all the math here, please study it by reading the original paper. The AdaSecant method provides an adaptive learning rate method by the formula below:

![](https://miro.medium.com/v2/resize:fit:548/1*1xBB6J_U2av_l8vJedWA2A.png)

taken from: [https://arxiv.org/pdf/1708.07120.pdf](https://arxiv.org/pdf/1708.07120.pdf)

where theta is the parameters in a neural network (i.e. weights) at an iteration i, whereas delta f(theta) are the gradients with respect to the loss function f. The epsilon \* is the optimal learning rate. The authors slightly modify the formula above and present it as follows:

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*dVt3uKohU8LvKPBYUcP7ow.png)

taken from: [https://arxiv.org/pdf/1708.07120.pdf](https://arxiv.org/pdf/1708.07120.pdf)

where epsilon on the right hand side is the learning rate actually used in the calculations to update the weights.

## Some Experiments

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*ArFcI6CvqNoKG6LVaULukw.png)

taken from: [https://arxiv.org/pdf/1708.07120.pdf](https://arxiv.org/pdf/1708.07120.pdf)

The figure above shows the experiments on super-convergence. There were a set of experiments varying the available number of training samples and iterations. We see that the performance gain from Super-convergence is more vivid and significant for smaller datasets, 9.2 % better than Conventional training (80, 000 iterations).

## Get Ching (Chingis)’s stories in your inbox

Join Medium for free to get updates from this writer.

Subscribe

Subscribe

Remember me for faster sign in

There are also discussions on the effects of larger batch size and generalization gap. The authors investigated it and found a small improvement with larger batch sizes (sub-figure (a) below). Meanwhile, the sub-figure (b) shows the difference between training and test accuracies (Generalization Gap) are roughly similar for small and large batch sizes.

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*suLU9dyPUeOFp_HLFAR-cQ.png)

taken from: [https://arxiv.org/pdf/1708.07120.pdf](https://arxiv.org/pdf/1708.07120.pdf)

## Some Last Words

This paper was published in 2018 but I don’t hear about this concept from people instead I feel like there are much more discussions on architectures. Are we taking it for granted? Should we pay more attention to training strategies and data quality in 2022? I have high hopes for this year. The super-convergence us possible with a variety of datasets and architectures, provided the regularization effects of large learning rates are in a harmony with other forms of regularization. Thank you for your time reading my article (:

## Useful Information

**Pytorch Implementation of LR test to find out the LR boundaries:** [https://github.com/davidtvs/pytorch-lr-finder](https://github.com/davidtvs/pytorch-lr-finder)

Also, make sure to check out this insightful peace from Fast.ai:

**AdamW and Super-convergence is now the fastest way to train neural nets**

[**AdamW and Super-convergence is now the fastest way to train neural nets** \\
\\
**Note from Jeremy: Welcome to fast.ai's first scholar-in-residence, Sylvain Gugger. What better way to introduce him…**\\
\\
www.fast.ai](https://www.fast.ai/2018/07/02/adam-weight-decay/?source=post_page-----265c1d7f6b99---------------------------------------)

The authors’ talk on YouTube:

[Machine Learning](https://medium.com/tag/machine-learning?source=post_page-----265c1d7f6b99---------------------------------------)

[Deep Learning](https://medium.com/tag/deep-learning?source=post_page-----265c1d7f6b99---------------------------------------)

[Artificial Intelligence](https://medium.com/tag/artificial-intelligence?source=post_page-----265c1d7f6b99---------------------------------------)

[![Ching (Chingis)](https://miro.medium.com/v2/resize:fill:48:48/1*tudn7-2ZXVGgClPR0WUFmg.jpeg)](https://chingisoinar.medium.com/?source=post_page---post_author_info--265c1d7f6b99---------------------------------------)

[![Ching (Chingis)](https://miro.medium.com/v2/resize:fill:64:64/1*tudn7-2ZXVGgClPR0WUFmg.jpeg)](https://chingisoinar.medium.com/?source=post_page---post_author_info--265c1d7f6b99---------------------------------------)

Follow

[**Written by Ching (Chingis)**](https://chingisoinar.medium.com/?source=post_page---post_author_info--265c1d7f6b99---------------------------------------)

[2.6K followers](https://chingisoinar.medium.com/followers?source=post_page---post_author_info--265c1d7f6b99---------------------------------------)

· [95 following](https://chingisoinar.medium.com/following?source=post_page---post_author_info--265c1d7f6b99---------------------------------------)

I am a passionate student. I enjoy studying and sharing my knowledge. Follow me/Connect with me and join my journey.

Follow

[Help](https://help.medium.com/hc/en-us?source=post_page-----265c1d7f6b99---------------------------------------)

[Status](https://status.medium.com/?source=post_page-----265c1d7f6b99---------------------------------------)

[About](https://medium.com/about?autoplay=1&source=post_page-----265c1d7f6b99---------------------------------------)

[Careers](https://medium.com/jobs-at-medium/work-at-medium-959d1a85284e?source=post_page-----265c1d7f6b99---------------------------------------)

[Press](mailto:pressinquiries@medium.com)

[Blog](https://blog.medium.com/?source=post_page-----265c1d7f6b99---------------------------------------)

[Store](https://medium.com/store)

[Privacy](https://policy.medium.com/medium-privacy-policy-f03bf92035c9?source=post_page-----265c1d7f6b99---------------------------------------)

[Rules](https://policy.medium.com/medium-rules-30e5502c4eb4?source=post_page-----265c1d7f6b99---------------------------------------)

[Terms](https://policy.medium.com/medium-terms-of-service-9db0094a1e0f?source=post_page-----265c1d7f6b99---------------------------------------)

[Text to speech](https://speechify.com/medium?source=post_page-----265c1d7f6b99---------------------------------------)

reCAPTCHA

Recaptcha requires verification.

protected by **reCAPTCHA**