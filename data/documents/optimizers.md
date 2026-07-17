# Optimization Algorithms for Deep Learning

Training a neural network means minimizing a loss function over its parameters, and the choice of optimization algorithm determines how parameters are updated at each training step given the computed gradients.

## Stochastic Gradient Descent

Stochastic Gradient Descent (SGD) is the foundational optimizer: at each step, it computes the gradient of the loss with respect to the parameters on a mini-batch of training examples, and updates the parameters by taking a step in the direction opposite the gradient:

theta_{t+1} = theta_t - lr * grad(L, theta_t)

where lr is the learning rate. Using a mini-batch rather than the full dataset makes each update cheap and introduces noise into the gradient estimate, which can help the optimizer escape shallow local minima, though it also makes individual updates noisier and less precise than a full-batch gradient.

## Momentum

Plain SGD can be slow to converge, especially in regions where the loss surface curves much more steeply in some directions than others, causing the parameter path to oscillate. **Momentum** addresses this by accumulating a running exponentially-weighted average of past gradients, and using that accumulated velocity — rather than the raw current gradient — to update the parameters:

v_{t+1} = beta * v_t + grad(L, theta_t)
theta_{t+1} = theta_t - lr * v_{t+1}

This has the effect of damping oscillations in directions where the gradient repeatedly changes sign, while accelerating progress in directions where the gradient consistently points the same way — much like a heavy ball rolling downhill accumulates speed rather than immediately following every local change in slope.

## Adam

Adam (Adaptive Moment Estimation) is the most widely used optimizer in modern deep learning, combining ideas from momentum with per-parameter adaptive learning rates. Adam maintains two running averages for each parameter: a first moment estimate m_t (an exponentially-weighted average of the gradient, like momentum) and a second moment estimate v_t (an exponentially-weighted average of the squared gradient):

m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t
v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2

Because m_t and v_t are initialized at zero, they are biased toward zero especially in early training steps; Adam corrects for this with bias-corrected estimates m-hat_t = m_t / (1 - beta_1^t) and v-hat_t = v_t / (1 - beta_2^t). The parameter update then divides the (bias-corrected) momentum term by the square root of the (bias-corrected) second-moment term:

theta_{t+1} = theta_t - lr * m-hat_t / (sqrt(v-hat_t) + epsilon)

Common default hyperparameters are beta_1 = 0.9, beta_2 = 0.999, and epsilon = 1e-8, a small constant added purely to avoid division by zero. Dividing by sqrt(v-hat_t) means parameters that have historically received large gradients get their effective learning rate scaled down, while parameters with small, infrequent gradients get an effectively larger step — an adaptive, per-parameter learning rate that is one of the main reasons Adam converges quickly and reliably across a wide range of architectures with comparatively little learning-rate tuning.

## Learning Rate Schedules and Warmup

Regardless of the base optimizer, most Transformer training recipes apply a learning rate schedule rather than a fixed learning rate. A common pattern is **linear warmup** followed by decay: the learning rate is linearly increased from zero to a peak value over the first few thousand training steps, then decayed (linearly, or following an inverse square root or cosine schedule) for the remainder of training. Warmup is particularly important for Transformers because early in training, before the model's weights have adapted, large learning rates applied to randomly-initialized attention layers can cause unstable, divergent updates; starting with a small learning rate and ramping up gives the optimizer's adaptive moment estimates time to stabilize before the learning rate reaches its full value.
