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

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

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