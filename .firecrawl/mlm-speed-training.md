### [Navigation](https://machinelearningmastery.com/how-to-speed-up-training-of-language-models/\#navigation)

By[Adrian Tam](https://machinelearningmastery.com/author/adriantam/)onJanuary 12, 2026in[Training Transformer Models](https://machinelearningmastery.com/category/training-transformer-models/ "View all items in Training Transformer Models")[0](https://machinelearningmastery.com/how-to-speed-up-training-of-language-models/#respond)

Share _Post_Share

Language model training is slow, even when your model is not very large. This is because you need to train the model on a large dataset and handle a large vocabulary. Therefore, the model requires many training steps to converge. However, some techniques can speed up training. In this article, you will learn about them. In particular, you will learn about:

- Using optimizers
- Using learning rate schedulers
- Other techniques for better convergence or reduced memory consumption

Let’s get started.

![](https://machinelearningmastery.com/wp-content/uploads/2025/11/emma-fabbri-2TmsyZXMNTE-unsplash-scaled.jpg)

How to Speed Up Training of Language Models

Photo by [Emma Fabbri](https://unsplash.com/photos/fishermans-bastion-in-budapest-during-daytime-2TmsyZXMNTE). Some rights reserved.

## Overview

This article is divided into four parts; they are:

- Optimizers for Training Language Models
- Learning Rate Schedulers
- Sequence Length Scheduling
- Other Techniques to Help Train Deep Learning Models

## Optimizers for Training Language Models

Adam has been the most popular optimizer for training deep learning models. Unlike SGD and RMSProp, Adam uses both the first and second moments of the gradient to update the parameters. Using the second moment can help the model converge faster and more stably, at the expense of increased memory usage.

However, when training language models nowadays, you typically use AdamW, the Adam optimizer with weight decay. Weight decay is a regularization technique to prevent overfitting. It usually involves adding a small penalty to the loss function. But in AdamW, weight decay is applied directly to the weights. This is believed to be more stable because the regularization term is decoupled from the calculated gradient. It is also more robust to hyperparameter tuning, as the regularization term is applied explicitly to the weight update.

In formula, the AdamW weight update algorithm is as follows:

𝑔𝑡=∇𝜃𝐿⁡(𝜃𝑡−1)𝑚𝑡=𝛽1⁢𝑚𝑡−1+(1–𝛽1)⁢𝑔𝑡𝑣𝑡=𝛽2⁢𝑣𝑡−1+(1–𝛽2)⁢𝑔2𝑡ˆ𝑚𝑡=𝑚𝑡/(1–𝛽𝑡1)ˆ𝑣𝑡=𝑣𝑡/(1–𝛽𝑡2)𝜃𝑡=𝜃𝑡−1–𝛼⁢(ˆ𝑚𝑡√ˆ𝑣𝑡+𝜖+𝜆⁢𝜃𝑡−1)

The model weight at step 𝑡 is denoted by 𝜃𝑡. The 𝑔𝑡 is the computed gradient from the loss function 𝐿, and 𝑔2𝑡 is the elementwise square of the gradient. The 𝑚𝑡 and 𝑣𝑡 are the moving averages of the first and second moments of the gradient, respectively. Learning rate 𝛼, weight decay 𝜆, and moving average decay rates 𝛽1 and 𝛽2 are hyperparameters. A small value 𝜖 is used to avoid division by zero. A common choice would be 𝛽1=0.9, 𝛽2=0.999, 𝜖=10−8, and 𝜆=0.1.

The key of AdamW is the 𝜆⁢𝜃𝑡−1 term in the gradient update, instead of in the loss function.

AdamW is not the only choice of optimizer. Some newer optimizers have been proposed recently, including Lion, SOAP, and AdEMAMix. You can see the paper [_Benchmarking Optimizers for Large Language Model Pretraining_](https://arxiv.org/abs/2509.01440v1) for a summary.

## Learning Rate Schedulers

A learning rate scheduler adjusts the learning rate during training. Usually, you would prefer a larger learning rate in the early training steps and reduce it as training progresses to help the model converge. You can add a warm-up period to increase the learning rate from a small value to its peak over a short period (typically 0.1% to 2% of total steps), then decrease it over the remaining training steps.

A warm-up period usually starts with a near-zero learning rate and increases linearly to the peak learning rate. A model starts with randomized initial weights. Starting with a large learning rate can lead to poor convergence, especially for large models, large batch sizes, and adaptive optimizers.

You can see the need for a warm-up from the equations above. Assuming the model is uncalibrated, the loss may vary significantly across subsequent steps. Then the first and second moments 𝑚𝑡 and 𝑣𝑡 will be fluctuating greatly, and the gradient update 𝜃𝑡–𝜃𝑡−1 will also be fluctuating greatly. Hence, you would prefer the loss to be stable and move slowly so that AdamW can build a reliable running average. This can be easily achieved if 𝛼 is small.

At the learning rate reduction phase, there are a few choices:

- cosine decay: 𝐿⁢𝑅=𝐿⁢𝑅max⋅12⁢(1+cos⁡𝜋⁢𝑡𝑇)
- square-root decay: 𝐿⁢𝑅=𝐿⁢𝑅max⋅√𝑇–𝑡𝑇
- linear decay: 𝐿⁢𝑅=𝐿⁢𝑅max⋅𝑇–𝑡𝑇

![](https://machinelearningmastery.com/wp-content/uploads/2025/11/learning-rate-schedules.png)

Plot of the three decay functions

A large learning rate can help the model converge faster, while a small learning rate can help the model stabilize. Therefore, you want the learning rate to be large at the beginning when the model is still uncalibrated, but small at the end when the model is close to its optimal state. All decay schemes above can achieve this, but you would not want the learning rate to become “too small too soon” or “too large too late”. Cosine decay is the most popular choice because it reduces the learning rate more slowly at the beginning and maintains a lower learning rate toward the end, which are desirable properties to help the model converge faster and stabilize, respectively.

In PyTorch, you have the `CosineAnnealingLR` scheduler to implement cosine decay. For the warm-up period, you need to combine with the `LinearLR` scheduler. Below is an example of the training loop using AdamW, `CosineAnnealingLR`, and `LinearLR`:

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23<br>24<br>25<br>26<br>27<br>28<br>29<br>30<br>31 | import torch<br>import torch.nn asnn<br>import torch.optim asoptim<br>from torch.optim.lr\_scheduler import LinearLR,CosineAnnealingLR,SequentialLR<br>\# Example setup<br>model=torch.nn.Linear(10,1)<br>X,y=torch.randn(5,10),torch.randn(5)<br>loss\_fn=nn.MSELoss()<br>optimizer=optim.AdamW(model.parameters(),lr=1e-2,betas=(0.9,0.999),eps=1e-8,weight\_decay=0.1)<br>\# Define learning rate schedulers<br>warmup\_steps=10<br>total\_steps=100<br>min\_lr=1e-4<br>warmup\_lr=LinearLR(optimizer,start\_factor=0.1,end\_factor=1.0,total\_iters=warmup\_steps)<br>cosine\_lr=CosineAnnealingLR(optimizer,T\_max=total\_steps-warmup\_steps,eta\_min=min\_lr)<br>combined\_lr=SequentialLR(optimizer,schedulers=\[warmup\_lr,cosine\_lr\],milestones=\[warmup\_steps\])<br>\# Training loop<br>forstep inrange(total\_steps):<br>\# train one epoch<br>y\_pred=model(X)<br>loss=loss\_fn(y\_pred,y)<br>\# print loss and learning rate<br>print(f"Step {step+1}/{total\_steps}: loss {loss.item():.4f}, lr {combined\_lr.get\_last\_lr()\[0\]:.4f}")<br>\# backpropagate and update weights<br>optimizer.zero\_grad()<br>loss.backward()<br>optimizer.step()<br>combined\_lr.step() |

Running this code, you may see:

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11 | Step 1/100: loss 1.5982, lr 0.0010<br>Step 2/100: loss 1.5872, lr 0.0019<br>Step 3/100: loss 1.5665, lr 0.0028<br>...<br>Step 9/100: loss 1.2738, lr 0.0082<br>Step 10/100: loss 1.2069, lr 0.0091<br>Step 11/100: loss 1.1387, lr 0.0100<br>...<br>Step 98/100: loss 0.4845, lr 0.0001<br>Step 99/100: loss 0.4845, lr 0.0001<br>Step 100/100: loss 0.4845, lr 0.0001 |

Notice how the learning rate increases and then decreases. Note that term `eta_min=min_lr` in the cosine scheduler above. The default value is 0; it is only a parameter used to calculate the learning rate. You can set it to approximately 0.1%- 1% of the peak learning rate.

## Sequence Length Scheduling

Language models are trained with sequence data. Transformer models or recurrent neural networks are both architecturally agnostic to the sequence length. However, you may want to train the model on a long sequence to help it learn to handle long context.

In training, long sequence lengths can be problematic. First, you train on batches of sequences, and ragged lengths mean you need to pad the sequences to the batch’s maximum length. While you ignore padded tokens, your model still needs to process them, wasting resources. Second, in the attention mechanism, the complexity is quadratic to the sequence length. The longer the sequence, the more costly it is to process.

Therefore, you may want to create batches with sequences of similar length to avoid excessive padding. This is known as **sequence packing**.

You may also want to train the model with shorter sequences first. You can accelerate training by forcing the model to learn language patterns with shorter sequences. Once the model has fairly converged, you can gradually increase the sequence length to help the model learn how to handle long contexts. However, you still want to use short sequence lengths intermittently to make sure the converged model can handle both long and short sequences equally well.

These are common techniques in training large language models to save computational resources. Note that you still set the model’s maximum sequence length to a fixed value, which affects how you configure the positional embeddings. However, you do not exhaust the maximum sequence length until the model has fairly converged.

Implementing sequence-length scheduling requires writing a more complex data loader that accounts for the current epoch to return the appropriate training data.

## Other Techniques to Help Train Deep Learning Models

### Random Restart

Training a deep learning model is a complex process and not easy to get right, especially for large models. One common issue is the model getting stuck in a local minimum and being unable to converge. Using momentum in gradient descent can help the model escape from local minima, but it is not always effective. Another approach is to restart training if the model fails to converge.

Random restart is a training strategy in which the model is trained multiple times from scratch. It uses different random seeds each time, so the model starts with different initial weights and a different data shuffle. This helps you avoid getting stuck in a local minimum, allowing you to select the option with the best performance. This is ideal if you can train multiple models for fewer epochs initially, then select the best model from the pool to complete training with more epochs.

### Gradient Clipping

One common issue in training deep learning models is gradient explosion. This is especially common if you train the model using lower-precision floating-point numbers, in which the range of the gradient could be too large to be represented. Gradient clipping is the technique of limiting the magnitude of the gradient to a safe value. Without it, you may see your training process suddenly fail due to the model weights or loss function becoming NaN or infinity.

There are multiple ways to clip gradients. The most common approach is to clip the gradient so that the L2 norm is below a safe threshold, such as 1.0 or 6.0. You can also clip the gradient to a value range, such as -5.0 to 5.0.

Gradient clipping by L2 norm means scaling the entire gradient vector if the L2 norm ‖𝑔𝑡‖2 is greater than a safe value 𝑐:

ˆ𝑔𝑡=min⁡(1,𝑐‖𝑔𝑡‖2)⋅𝑔𝑡

On the other hand, gradient clipping by value means setting the gradient to a safe value whenever the gradient exceeds that value:

ˆ𝑔𝑡=⎧{
{⎨{
{⎩−𝑐if𝑔𝑡<−𝑐𝑔𝑡if−𝑐≤𝑔𝑡≤𝑐𝑐if𝑔𝑡>𝑐

Using gradient clipping in PyTorch is straightforward. You can use the `torch.nn.utils.clip_grad_norm_` function to clip the gradient by L2 norm, or the `torch.nn.utils.clip_grad_value_` function to clip the gradient by value. Below is an example:

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23<br>24 | import torch<br>import torch.nn asnn<br>import torch.optim asoptim<br>from torch.nn.utils import clip\_grad\_norm\_,clip\_grad\_value\_<br>\# Example setup<br>model=torch.nn.Linear(10,1)<br>X,y=torch.randn(5,10),torch.randn(5)<br>total\_steps=100<br>loss\_fn=nn.MSELoss()<br>optimizer=optim.AdamW(model.parameters(),lr=1e-2,betas=(0.9,0.999),eps=1e-8,weight\_decay=0.1)<br>\# Training loop<br>forstep inrange(total\_steps):<br>\# train one epoch<br>y\_pred=model(X)<br>loss=loss\_fn(y\_pred,y)<br>optimizer.zero\_grad()<br>loss.backward()<br>\# clip by L2 norm<br>clip\_grad\_norm\_(model.parameters(),max\_norm=1.0)<br>\# or clip by value<br>\# clip\_grad\_value\_(model.parameters(), clip\_value=1.0)<br>optimizer.step() |

Should you use gradient clipping by norm or by value? Clipping by value imposes a hard limit on individual gradients and unevenly changes their magnitudes. If some gradients are naturally larger, you introduce an imbalance in learning across different layers of your model. This will disrupt the natural flow of information during training and may lead to suboptimal training. On the other hand, clipping by norm scales everything equally. If the gradients are highly imbalanced, scaling them down will artificially introduce vanishing gradients in your training.

### Mixed Precision Training

When a model becomes too large, memory consumption becomes a bottleneck. You may want to save memory by using lower-precision floating-point formats during training, such as half-precision (float16) or bfloat16. Compared with single precision (float32), float16 and bfloat16 can reduce memory consumption by half, but at the cost of reduced range and precision.

Therefore, you may want to use mixed precision training, in which part of the model uses float32 while the other part uses float16. A common choice is to use float32 for biases but float16 for weights in linear layers.

Modern GPUs can run float16 operations at the same speed as float32, but since you can operate on more data at the same time, you can effectively run the training process at double speed.

### Regularization

In the AdamW optimizer above, you have a parameter 𝜆 to set the strength of weight decay. The default value in PyTorch is 0.01. However, if you are resource-constrained, you may want to use fewer training steps and larger learning rates. But the consequence is that the converged model would be biased. You can improve the model’s generalization by using a stronger weight decay factor to compensate for the shorter training horizon.

## Further Readings

Below are some resources that you may find useful:

- Kingma & Ba (2014), [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- Loshchilov & Hutter (2017), [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (AdamW paper)
- Loshchilov & Hutter (2017), [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) (cosine decay paper)
- Semenov et al. (2025), [Benchmarking Optimizers for Large Language Model Pretraining](https://arxiv.org/abs/2509.01440v1)
- Koloskova et al. (2023), [Revisiting Gradient Clipping: Stochastic Bias and Tight Convergence Guarantees](https://proceedings.mlr.press/v202/koloskova23a/koloskova23a.pdf)

## Summary

In this article, you learned about some techniques to speed up the training process of deep learning models, especially for large language models. Specifically, you learned that:

- AdamW with cosine decay is the most popular optimizer and learning rate scheduler for training language models.
- You can use sequence-length scheduling to save computational resources during language model training.
- Techniques such as random restarts and gradient clipping can help you train the model more stably.
- Mixed precision training can help you reduce memory consumption.

Share _Post_Share

### More On This Topic

- [![A Gentle Introduction to Premature Convergence](https://machinelearningmastery.com/wp-content/uploads/2021/05/A-Gentle-Introduction-to-Premature-Convergence.jpg)A Gentle Introduction to Premature Convergence](https://machinelearningmastery.com/premature-convergence/)
- [![mlm-3-ways-speed-model-training-without-gpu](https://machinelearningmastery.com/wp-content/uploads/2025/10/mlm-3-ways-speed-model-training-without-gpu-200x200.png)3 Ways to Speed Up Model Training Without More GPUs](https://machinelearningmastery.com/3-ways-to-speed-up-model-training-without-more-gpus/)
- [![mlm-speed-up-improve-xgboost-models](https://machinelearningmastery.com/wp-content/uploads/2025/09/mlm-speed-up-improve-xgboost-models-200x200.png)3 Ways to Speed Up and Improve Your XGBoost Models](https://machinelearningmastery.com/3-ways-to-speed-up-and-improve-your-xgboost-models/)
- [![mlm-10-python-libraries-speed-up-model-development](https://machinelearningmastery.com/wp-content/uploads/2025/05/mlm-10-python-libraries-speed-up-model-development-200x200.png)10 Python Libraries That Speed Up Model Development](https://machinelearningmastery.com/10-python-libraries-that-speed-up-model-development/)
- [![Gentle Introduction to Statistical Language Modeling and Neural Language Models](https://machinelearningmastery.com/wp-content/uploads/2017/11/Gentle-Introduction-to-Statistical-Language-Modeling-and-Neural-Language-Models.jpg)Gentle Introduction to Statistical Language Modeling…](https://machinelearningmastery.com/statistical-language-modeling-and-neural-language-models/)
- [![dan-v-S5x5rrsDixk-unsplash](https://machinelearningmastery.com/wp-content/uploads/2025/11/dan-v-S5x5rrsDixk-unsplash-200x200.jpg)Datasets for Training a Language Model](https://machinelearningmastery.com/datasets-for-training-a-language-model/)

[Fine-Tuning a BERT Model](https://machinelearningmastery.com/fine-tuning-a-bert-model/)

[Prompt Compression for LLM Generation Optimization and Cost Reduction](https://machinelearningmastery.com/prompt-compression-for-llm-generation-optimization-and-cost-reduction/)

##### No comments yet.

### Leave a Reply [Click here to cancel reply.](https://machinelearningmastery.com/how-to-speed-up-training-of-language-models/\#respond)

Comment \*

Name (required)

Email (will not be published) (required)

Δ