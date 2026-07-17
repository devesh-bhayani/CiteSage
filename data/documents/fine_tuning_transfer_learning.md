# Fine-Tuning and Transfer Learning

Transfer learning is the practice of taking a model trained on one task (or on a large, general-purpose corpus) and adapting it to a different, usually more specific, downstream task — rather than training a new model from randomly initialized weights for every task. In modern NLP and computer vision, transfer learning is typically built on a "pretrain then fine-tune" recipe: a large model is pretrained on a broad, often self-supervised objective, then fine-tuned on a smaller labeled dataset for the actual task of interest.

## Full Fine-Tuning

The most direct form of transfer learning is full fine-tuning: starting from pretrained weights, every parameter in the model is updated via gradient descent on the downstream task's labeled data, typically with a much smaller learning rate and far fewer training steps than were used during pretraining, since the pretrained weights already encode useful general representations that need only modest adjustment. Full fine-tuning tends to achieve the strongest task performance, but it is expensive: it requires storing gradients and optimizer state for every parameter in the model, and it produces an entirely separate copy of the full model's weights for every downstream task, which becomes impractical to store and serve when a model needs to support many different tasks or many different fine-tuned customers.

## Parameter-Efficient Fine-Tuning

Parameter-efficient fine-tuning (PEFT) methods aim to adapt a pretrained model to a new task while updating — and storing — only a small fraction of its parameters, keeping the original pretrained weights frozen entirely.

**LoRA (Low-Rank Adaptation)** is one of the most widely used PEFT methods. Rather than updating a weight matrix W directly, LoRA freezes W and learns a low-rank update decomposed into two much smaller matrices, A and B, such that the effective weight during fine-tuning becomes W + (B * A), where A and B have a much smaller inner dimension r (the "rank") than the original matrix. Because r is small (often just a few dozen), the number of trainable parameters is a tiny fraction of the original weight matrix's size, dramatically reducing memory needed for gradients and optimizer state during fine-tuning. At inference time, B * A can optionally be merged directly into W, so LoRA introduces no extra inference latency compared to the original model.

**Adapter layers** are another PEFT approach: small additional feed-forward modules are inserted between the existing layers of a frozen pretrained model, and only these newly-inserted adapter parameters are trained on the downstream task. Because the adapters are small relative to the full model, this again requires storing only a small set of task-specific parameters per downstream task, while reusing a single frozen copy of the large pretrained backbone across all of them.

**Prompt tuning** and **prefix tuning** take a different approach entirely, learning a small set of continuous embedding vectors that are prepended to the input (or to each layer's keys and values, for prefix tuning) rather than modifying any of the model's existing weights at all. The entire pretrained model remains completely untouched; only these prepended vectors are learned per task.

## Catastrophic Forgetting

A risk specific to full fine-tuning is catastrophic forgetting: as all parameters are updated to fit the new, often much smaller and narrower fine-tuning dataset, the model can lose general capabilities it had acquired during pretraining, especially if fine-tuning runs for too many steps or uses too high a learning rate relative to the size of the fine-tuning dataset. PEFT methods are comparatively more resistant to catastrophic forgetting, precisely because the original pretrained weights are frozen and never directly modified — whatever general knowledge they encoded remains intact, with only the small added parameters (or added low-rank update) specializing the model's behavior for the new task.

## Instruction Tuning and RLHF

For large language models specifically, a further stage of fine-tuning beyond task-specific adaptation has become standard: instruction tuning, where the model is fine-tuned on a dataset of (instruction, desired-response) pairs covering many different tasks, teaching it to follow natural-language instructions generally rather than being specialized to any single task. This is frequently followed by reinforcement learning from human feedback (RLHF), where human preference judgments between candidate model outputs are used to train a reward model, which is then used to further fine-tune the language model via reinforcement learning to produce outputs that better match human preferences for helpfulness, harmlessness, and honesty.
