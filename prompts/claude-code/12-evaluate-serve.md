Phase 3, step 3: evaluate the fine-tuned models with the SAME graph, environment and metric as phase 1, on the same seed-233 500 HotpotQA questions.

Add a local backend to src/llm.py so --model can point at a merged adapter served by vLLM (or Ollama if vLLM will not run), and run the fine-tuned model with ZERO exemplars in the prompt.

Produce results/finetuning.csv and a README section with one table: model, prompted or fine-tuned, exemplars used, EM, mean steps, % hit step limit, tokens per second, prompt tokens per question.

Test these claims explicitly and mark PASS or FAIL with the numbers:
1. Fine-tuned ReAct-3B beats the same 3B model prompted with 6 exemplars.
2. Fine-tuned ReAct beats fine-tuned CoT and fine-tuned Act (the paper's format-transfer claim).
3. Fine-tuned ReAct-3B uses far fewer prompt tokens per question than the prompted version; report the reduction as a percentage.

Then quantize the merged model to 4-bit GGUF, re-run 100 questions, and add a row showing EM and tokens per second after quantization so we can see the accuracy-versus-speed trade.

Finally add a short "Limitations" section: our student is 3B and the paper's was 8B and 62B, our teacher is a small API model rather than PaLM-540B, and we trained on N trajectories rather than 3,000 if N is lower.
