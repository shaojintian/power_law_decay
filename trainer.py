

import os
import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    logger,
)
from typing import List, Optional
from collections import Counter

class PowerlawDecayTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.pdl_vocab_weights = None # 用于存储预计算的词汇表PDL权重
        if getattr(self.args, 'power_law_decay_loss', False):
            logger.info("Power-Law Decay Loss (PDL) is enabled. Initializing PDL components.")
            if not hasattr(self, 'tokenizer') or self.tokenizer is None:
                logger.error("Tokenizer is not available. PDL cannot be initialized.")
                setattr(self.args, 'power_law_decay_loss', False) # 禁用PDL
            else:
                token_frequencies = self._calculate_token_frequencies()
                if token_frequencies is not None:
                    self._calculate_pdl_weights_for_vocab(token_frequencies)
                else:
                    logger.warning("Failed to calculate token frequencies. PDL will be disabled.")
                    setattr(self.args, 'power_law_decay_loss', False) # 禁用PDL
        else:
            logger.info("Power-Law Decay Loss (PDL) is disabled.")

    def _calculate_token_frequencies(self):
        """
        计算参考语料库中每个 token 的频率 (原始计数)。
        """
        if not hasattr(self.args, 'train_dataset_for_pdl_freq') or self.args.train_dataset_for_pdl_freq is None:
            logger.warning("'train_dataset_for_pdl_freq' not provided in args. Cannot calculate token frequencies for PDL.")
            return None
        
        dataset_for_freq = self.args.train_dataset_for_pdl_freq
        text_column_name = getattr(self.args, 'text_column_name_for_pdl', 'text')

        logger.info(f"Calculating token frequencies for PDL from dataset using text column '{text_column_name}'.")
        token_counts = Counter()
        num_processed_examples = 0

        try:
            for example in dataset_for_freq:
                if isinstance(example, dict) and text_column_name in example:
                    text = example[text_column_name]
                elif isinstance(example, str): # 如果数据集直接是字符串列表
                    text = example
                else:
                    # logger.debug(f"Skipping example of unexpected type for frequency calculation: {type(example)}")
                    continue

                if isinstance(text, str) and text.strip():
                    # 根据论文，频率通常基于目标token。如果你的数据集有明确的目标token ID，应直接使用它们。
                    # 这里假设我们从原始文本计算，这可能需要调整以更好地匹配PDL的意图。
                    # add_special_tokens=False 通常用于避免统计特殊标记的频率。
                    ids = self.tokenizer.encode(text, add_special_tokens=False)
                    token_counts.update(ids)
                    num_processed_examples += 1
                # else:
                #     logger.debug(f"Text for frequency calculation is not a non-empty string: '{text}'")
            
            if not token_counts:
                logger.warning("No tokens found after processing the dataset for frequency calculation.")
                return None

            logger.info(f"Token frequencies calculated from {num_processed_examples} examples, "
                        f"resulting in {len(token_counts)} unique tokens.")
            return token_counts

        except Exception as e:
            logger.error(f"Error during token frequency calculation: {e}", exc_info=True)
            return None

    def _calculate_pdl_weights_for_vocab(self, token_frequencies):
        """
        为词汇表中的每个 token 预计算 PDL 权重。
        """
        pdl_alpha = getattr(self.args, 'pdl_alpha', 1.0)
        pdl_epsilon = getattr(self.args, 'pdl_epsilon', 1e-7)
        vocab_size = self.tokenizer.vocab_size

        # 初始化权重为1（对于不在频率表中的token或alpha=0的情况）
        # 如果一个 token 的频率是0 (token_frequencies.get(token_id, 0)),
        # 它的权重将是 1 / (epsilon^alpha)。
        # 如果我们希望未知 token 有一个“中性”权重，可以考虑不同的默认值或处理方式。
        # 论文中 epsilon 的作用就是处理零频token，所以这是符合定义的。
        
        # 设备应与模型参数一致
        device = self.args.device if hasattr(self.args, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.pdl_vocab_weights = torch.ones(vocab_size, dtype=torch.float32, device=device) # 使用float32以保证精度

        logger.info(f"Calculating PDL weights for vocabulary (size: {vocab_size}) with alpha={pdl_alpha}, epsilon={pdl_epsilon}.")
        
        num_weighted_tokens = 0
        for token_id in range(vocab_size):
            token_freq = token_frequencies.get(token_id, 0) # 默认为0频
            # 根据论文公式 (3): w(t) = 1 / (freq(t) + ε)^α
            weight = 1.0 / ((token_freq + pdl_epsilon) ** pdl_alpha)
            self.pdl_vocab_weights[token_id] = weight
            if token_freq > 0 :
                num_weighted_tokens +=1
        
        logger.info(f"PDL weights calculated for {num_weighted_tokens} tokens with non-zero frequency "
                    f"(out of {vocab_size} total vocab size). Weights stored on device: {self.pdl_vocab_weights.device}")


    def _compute_loss(self, model, inputs, return_outputs=False):
        """
        计算损失。如果启用了 power_law_decay_loss 且已成功初始化，则使用 PDL。
        """
        use_pdl = getattr(self.args, 'power_law_decay_loss', False) and \
                    self.pdl_vocab_weights is not None

        if use_pdl:
            labels = inputs.pop("labels", None)
            if labels is None:
                logger.warning("PDL is active but no labels found in inputs. Falling back to super()._compute_loss.")
                # inputs["labels"] = labels # 放回去，如果父类需要
                return super()._compute_loss(model, inputs, return_outputs)

            # 获取模型输出
            outputs = model(**inputs)
            logits = outputs.logits # (batch_size, sequence_length, vocab_size)

            # 移位 logits 和 labels (对于 Causal LM)
            # logits 预测下一个 token，所以 logits[:, :-1] 对应 labels[:, 1:]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous() # (batch_size, seq_len_shifted)

            batch_size, seq_len_shifted = shift_labels.shape
            
            if seq_len_shifted == 0: # 如果移位后序列长度为0
                logger.debug("Shifted label sequence length is 0. Returning zero loss.")
                loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=logits.requires_grad)
                if hasattr(outputs, "loss"): outputs.loss = loss
                elif isinstance(outputs, dict): outputs["loss"] = loss
                return (loss, outputs) if return_outputs else loss

            vocab_size = shift_logits.size(-1)

            # 1. 计算每个 token 的标准交叉熵损失 (不进行 reduction)
            #    形状: (batch_size * seq_len_shifted)
            flat_per_token_loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size), # (batch_size * seq_len_shifted, vocab_size)
                shift_labels.view(-1),             # (batch_size * seq_len_shifted)
                ignore_index=-100,                 # Hugging Face Transformers 忽略索引
                reduction='none'
            )
            # 恢复形状: (batch_size, seq_len_shifted)
            per_token_loss = flat_per_token_loss.view(batch_size, seq_len_shifted)

            # 2. 获取 PDL 权重
            #    self.pdl_vocab_weights 形状: (vocab_size)
            #    我们需要为 shift_labels 中的每个 token ID 提取其对应的权重
            #    确保 pdl_vocab_weights 和 shift_labels 在同一设备
            if self.pdl_vocab_weights.device != shift_labels.device:
                self.pdl_vocab_weights = self.pdl_vocab_weights.to(shift_labels.device)
                logger.debug(f"Moved pdl_vocab_weights to device: {shift_labels.device}")

            # 使用 gather 或直接索引来获取权重 (需要处理 -100 索引)
            # 创建一个临时的标签副本，将 -100 替换为有效索引 (例如0)，以避免 gather 出错
            # 权重对于 -100 的标签无关紧要，因为它们会被 active_loss_mask 掉
            safe_labels_for_indexing = shift_labels.clone()
            ignore_mask = (safe_labels_for_indexing == -100)
            safe_labels_for_indexing[ignore_mask] = 0 # 用0填充忽略的索引 (0通常是有效token_id)
            
            # weights_for_tokens 形状: (batch_size, seq_len_shifted)
            weights_for_tokens = self.pdl_vocab_weights[safe_labels_for_indexing]


            # 3. 将权重应用于每个 token 的损失
            weighted_loss_tokens = per_token_loss * weights_for_tokens

            # 4. 创建掩码，忽略标签为 -100 的 token
            active_loss_mask = shift_labels.ne(-100)

            # 5. 计算最终损失 归一化
            #    L_PDL = sum(weighted_loss_for_active_tokens) / sum(weights_for_active_tokens)
            sum_weighted_loss = (weighted_loss_tokens * active_loss_mask).sum()
            sum_active_weights = (weights_for_tokens * active_loss_mask).sum()

            if sum_active_weights > 0:
                loss = sum_weighted_loss / sum_active_weights
            else:
                # 如果没有活动的 token (例如，所有标签都是 -100)
                logger.debug("No active tokens found for PDL loss calculation. Returning zero loss.")
                loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=logits.requires_grad)
            
            # 更新 outputs 中的 loss 字段
            if hasattr(outputs, "loss"):
                outputs.loss = loss
            elif isinstance(outputs, dict):
                outputs["loss"] = loss
            # else: 如果 outputs 只是 logits，Trainer 会处理

        else: # 如果不使用 PDL 或初始化失败
            if getattr(self.args, 'power_law_decay_loss', False) and self.pdl_vocab_weights is None:
                logger.warning("PDL was requested but PDL weights are not available. Falling back to standard loss computation.")
            return super()._compute_loss(model, inputs, return_outputs)

        return (loss, outputs) if return_outputs else loss