Enhanced Text Generation with Power-Law Decay Loss

This project implements/explores [briefly describe your project, e.g., "a Large Language Model finetuning framework", "a text generation model for a specific task"] utilizing an advanced loss function, the Power-Law Decay Loss (PDL), to improve the quality, diversity, and informativeness of generated text.

## Overview

Standard cross-entropy loss, commonly used in training text generation models, treats all tokens equally. This can lead to models overemphasizing high-frequency, low-information tokens (e.g., "the", "a", "is") and neglecting lower-frequency tokens that are often crucial for specificity and conveying nuanced information.

The Power-Law Decay Loss (PDL) addresses this limitation by re-weighting the contribution of each token in the standard cross-entropy loss based on its frequency in a reference corpus. This approach is inspired by observations in information theory and linguistics, where a token's informativeness is often inversely proportional to its frequency.


[![arXiv](https://img.shields.io/badge/arXiv%20paper-2404.02905-b31b1b.svg)](https://arxiv.org/pdf/2505.10222)&nbsp;



**Key Idea of PDL:**
*   **Down-weights** the loss contribution of high-frequency (common) tokens.
*   **Up-weights** the loss contribution of low-frequency (rare, information-dense) tokens.

This mechanism guides the model during finetuning to focus more on learning and generating tokens that convey specific and unique information, thereby enhancing the overall quality of the generated text.

## Motivation & Background

The core motivation for PDL stems from the work presented in the following paper:

*   **Title:** Power-Law Decay Loss for Large Language Model Finetuning: Focusing on Information Sparsity to Enhance Generation Quality
*   **Authors:** Jintian Shao, Hongyi Huang, Jiayi Wu, Beiwen Zhang, ZhiYu Wu, You Shan, MingKai Zheng
*   **arXiv Link:** [https://arxiv.org/abs/submit/6467538](https://arxiv.org/abs/submit/6467538) (Note: As of my last update, this link points to a submission ID. Please replace with the final public arXiv ID like `YYMM.NNNNN` once available, e.g., `https://arxiv.org/abs/2405.XXXXX`)
*   **Abstract (from paper):**
    > During the finetuning stage of text generation tasks, standard cross-entropy loss treats all tokens equally. This can lead models to overemphasize high-frequency, low-information tokens, neglecting lower-frequency tokens crucial for specificity and informativeness in generated content. This paper introduces a novel loss function, Power-Law Decay Loss (PDL), specifically designed to optimize the finetuning process for text generation. The core motivation for PDL stems from observations in information theory and linguistics: the informativeness of a token is often inversely proportional to its frequency of occurrence. PDL re-weights the contribution of each token in the standard cross-entropy loss based on its frequency in the training corpus, following a power-law decay. Specifically, the weights for high-frequency tokens are reduced, while low-frequency, information-dense tokens are assigned higher weights. This mechanism guides the model during finetuning to focus more on learning and generating tokens that convey specific and unique information, thereby enhancing the quality, diversity, and informativeness of the generated text.

This project implements the PDL as described, allowing for more nuanced control over the learning process during model finetuning.

## Features

*   Implementation of Power-Law Decay Loss within a Hugging Face Transformers `Trainer` (or your specific framework).
*   Configurable PDL parameters:
    *   `pdl_alpha`: The decay factor controlling the strength of frequency-based decay.
    *   `pdl_epsilon`: A small smoothing constant for numerical stability.
*   Mechanism to calculate token frequencies from a specified reference corpus.
*   [Add any other specific features of your project]

## Implementation Details

The PDL is integrated into the `_compute_loss` method of a custom `Trainer` class. The weight `w(t)` for each token `t` is calculated as:

`w(t) = 1 / (freq(t) + ε)^α`

Where:
*   `freq(t)` is the frequency of token `t` in the reference corpus.
*   `α` is `pdl_alpha`.
*   `ε` is `pdl_epsilon`.

The final loss is a weighted average of the per-token cross-entropy losses.
