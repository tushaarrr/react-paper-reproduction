Phase 3, step 2: supervised fine-tuning with LoRA. Create finetune/ with a Colab-runnable script and a notebook wrapper.

Dataset: from data/sft/react_train.jsonl build examples where the prompt is a short instruction plus the question with NO exemplars, and the completion is the trajectory. Critically, mask the loss on all Observation lines and on the prompt: only Thought and Action tokens contribute to the loss. Implement this with explicit label masking over token offsets, and write a test that decodes the unmasked positions of one example and asserts they contain no Observation text.

Training: Qwen2.5-3B-Instruct in 4-bit (bitsandbytes nf4), PEFT LoRA r=16, alpha=32, dropout 0.05, target modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj, TRL SFTTrainer, lr 2e-4 cosine with 3% warmup, bf16, max_seq_len 2048, per_device_batch 2 with grad accumulation to an effective batch of 64 (the paper's batch size), 3 epochs, gradient checkpointing, eval every 100 steps on a 5% held-out slice. Save adapters to finetune/adapters/react-3b/.

Make it resumable from checkpoints, since a free T4 session can be cut off. Print the peak VRAM and total wall-clock at the end.

Then train the same recipe on act_train.jsonl and cot_train.jsonl so we can test the paper's second claim. Do not start these until I confirm the ReAct run finished.
