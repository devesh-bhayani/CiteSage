# Regularization Techniques: Dropout, Weight Decay, and Early Stopping

Regularization refers to techniques that reduce a neural network's tendency to overfit — to memorize idiosyncrasies of the training data rather than learning patterns that generalize to unseen data. As model capacity has grown, regularization has remained essential to achieving good generalization.

## Dropout

Dropout is one of the most widely used regularization techniques for neural networks. During training, dropout randomly "drops" (sets to zero) each unit in a layer's output with some fixed probability p (commonly 0.1 to 0.5), independently for each unit and each training example, and independently at every forward pass. This prevents units from co-adapting too strongly to the specific presence of other particular units, since any unit might be zeroed out on a given pass, forcing the network to learn more redundant, robust representations.

At test time, dropout is disabled entirely — all units are used — but the layer's output is typically scaled by (1 - p) to compensate for the fact that, during training, only a fraction (1 - p) of units were active on average. Many modern implementations instead apply "inverted dropout," scaling the kept activations by 1/(1-p) during training itself, so that no rescaling is needed at test time. Dropout can be viewed as training an implicit ensemble of many different "thinned" sub-networks that share parameters, with the full network at test time approximating an average over that ensemble.

Dropout is typically applied to feed-forward and fully-connected layers, and in Transformer architectures it is commonly applied to attention weights and to the output of each sublayer (attention and feed-forward) before the residual connection is added.

## Weight Decay (L2 Regularization)

Weight decay adds a penalty term proportional to the squared magnitude of the model's weights to the loss function being minimized:

L_total = L_task + lambda * sum(theta_i^2)

where lambda controls the strength of the penalty. This discourages the model from relying on very large weight values, which tend to make the model's output more sensitive to small changes in input — a hallmark of overfitting. In gradient terms, weight decay effectively shrinks each weight slightly toward zero at every update step, independent of the task gradient, which is why it is sometimes implemented directly as a multiplicative decay applied to the weights rather than as an explicit addition to the loss (as in the AdamW optimizer, which decouples weight decay from the adaptive gradient computation used by Adam).

## Early Stopping

Early stopping is a simple but effective regularization strategy: rather than training for a fixed, predetermined number of epochs, training is monitored against performance on a held-out validation set, and training is halted once validation performance stops improving (or begins to degrade) even as training loss continues to fall. This directly targets the point at which the model begins overfitting — memorizing the training set at the expense of generalization — without requiring any change to the loss function or model architecture. In practice, early stopping is often combined with checkpointing, saving the model's weights each time validation performance improves, so the best-performing checkpoint (rather than the final one) can be restored after training halts.

## Label Smoothing

Label smoothing is a regularization technique specific to classification tasks trained with cross-entropy loss. Instead of training the model to predict a hard one-hot target (probability 1.0 for the correct class, 0.0 for all others), label smoothing replaces the target with a softened distribution — for example, 0.9 for the correct class and the remaining 0.1 spread uniformly across all other classes. This prevents the model from becoming overconfident in its predictions and pushing logits toward extreme values, which can improve calibration and generalization. Label smoothing was used in the training recipe for the original Transformer model, among many other sequence-to-sequence architectures.
