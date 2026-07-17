# Model Quantization

Quantization is a technique for reducing a neural network's memory footprint and, often, its inference latency, by representing the model's weights (and sometimes its activations) using fewer bits per number than the 32-bit floating point (FP32) format typically used during training.

## Why Quantize

A model trained in FP32 stores every weight as a 32-bit floating point number. A large language model with tens of billions of parameters can therefore require many tens of gigabytes just to store its weights, which constrains what hardware can run it and how much memory is left over for the activations and key-value cache needed during inference. Quantizing weights to lower precision — 16-bit, 8-bit, or even 4-bit representations — directly and proportionally shrinks the memory required to store them, and can also speed up inference on hardware with efficient low-precision arithmetic, at the cost of some numerical precision that can, if not managed carefully, degrade model output quality.

## Precision Formats

**FP16** (16-bit floating point) and **BF16** (bfloat16, also 16 bits but with a different split between exponent and mantissa bits than FP16) are commonly used during training itself (mixed-precision training), roughly halving memory versus FP32 while retaining enough dynamic range (particularly for BF16, which keeps FP32's exponent range) to avoid the numerical instability that naive 16-bit training can otherwise suffer.

**INT8** quantization represents weights as 8-bit integers rather than floating point at all, requiring a scale factor (and sometimes a zero-point offset) to map between the integer representation and the original floating-point range. This typically requires a calibration step — running representative data through the model to determine appropriate per-tensor or per-channel scale factors before actually converting weights to INT8 — since the quantization range needs to be chosen well to avoid clipping large values or wasting resolution on values that never occur.

**4-bit quantization** pushes this further, and has become common specifically for serving very large language models on memory-constrained hardware. Because a 4-bit integer can only represent 16 distinct values, naive uniform 4-bit quantization can meaningfully degrade model quality; more sophisticated methods (such as GPTQ, which quantizes weights layer by layer while explicitly correcting for the quantization error introduced in earlier layers) are used to preserve much more of the original model's output quality than naive rounding-based quantization would.

## Quantization-Aware Training vs. Post-Training Quantization

**Post-training quantization (PTQ)** takes an already-trained, full-precision model and quantizes its weights afterward, without any further training — typically using a small calibration dataset just to determine appropriate scale factors, as described above. This is fast and requires no access to the original training pipeline or a large training dataset, but because the model's weights were never optimized with quantization in mind, PTQ can sometimes cause a noticeable quality drop, especially at very low bit-widths.

**Quantization-aware training (QAT)** instead simulates the effect of quantization *during* training itself: the forward pass simulates rounding weights (and sometimes activations) to the target lower precision, while gradients are still computed and applied in full precision, allowing the model's weights to adapt to compensate for the quantization noise they will encounter at actual inference time. QAT generally achieves better quality than PTQ at the same target bit-width, but it requires access to a training pipeline and additional training compute, which PTQ does not.

## The Precision-Quality Trade-off

In general, more aggressive quantization (fewer bits) yields greater memory and latency savings but risks a larger drop in model output quality, and the acceptable trade-off point depends heavily on the specific model, task, and deployment constraints. Larger models tend to be more robust to aggressive quantization than smaller ones — a very large language model quantized to 4 bits often retains most of its full-precision quality, while quantizing a much smaller model as aggressively can cause a more noticeable degradation, because the larger model has more redundant representational capacity to absorb the added quantization noise.
