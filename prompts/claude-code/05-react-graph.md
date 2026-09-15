Implement src/graph_react.py as a LangGraph StateGraph.

State (TypedDict): question, task, scratchpad (the growing Thought/Action/Observation text), step, done, answer, n_calls, n_badcalls, hit_step_limit, condition ("react" or "act").

Nodes:
- think_act: build prompt = instruction + exemplars + question + scratchpad + f"Thought {step}:" (for condition "act", use the Act-only exemplars and prompt f"Action {step}:" with no thought). Call llm.complete with stop=[f"\nObservation {step}:"]. Parse thought and action by splitting on f"\nAction {step}: ". If the split fails, count a bad call, keep the first line as the thought, and call the LLM again with the thought appended and stop=["\n"] to get just the action, exactly as the reference does.
- execute: lowercase the first character of the action, call WikiEnv.step, strip "\n" from the observation, append "Thought i / Action i / Observation i" to the scratchpad, increment step. If done, record the answer.

Conditional edge after execute: END if done; if step > 7 and not done, set hit_step_limit=True, call finish[] and END; else back to think_act.

Exemplars: HotpotQA uses prompts_naive.json keys webthink_simple6 (react) and webact_simple6 (act) with the instruction text from the reference notebook; FEVER uses fever.json keys webthink_simple3 and webact_simple3.

Write tests with a fake LLM that returns scripted thoughts/actions and a fake env: a two-step episode ends with the right answer; a parse failure triggers the second call; the step limit forces finish[] and sets hit_step_limit.

Then run the real graph on the first 3 sampled HotpotQA questions and show me the full trajectories.
