Implement src/llm.py: complete(prompt, stop, temperature=0, max_tokens=100, n=1) -> list[str], using the chat model set by MODEL in .env through the provider's LangChain integration (so the graph can swap models by config). Send the completion-style prompt as a single user message and instruct the model in a short system message to continue the text exactly in the exemplar format and to stop before writing an Observation.

- Enforce stop strings client-side as well (cut the output at the first stop string) since not every provider honours stop lists.
- Disk cache keyed by (model, prompt, stop, temperature, max_tokens, sample_index). Temperature 0.7 calls with n samples cache each sample separately.
- Append every real call to results/calls.csv: timestamp, model, prompt_tokens, completion_tokens, estimated cost from a small price table in the file.

Tests: the cache returns the same output without a network call on the second invocation; stop-string cutting works on a canned response.
