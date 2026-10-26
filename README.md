See "report" file for full paper.

I plant a backdoor, a trigger word that results in a different response simulating a sleeper agent
LLM, in Qwen2.5-0.5B-Instruct. Then, I test whether an ordinary DPO safety training schema and a
trigger-word-aware DPO safety training can remove the behavior. After 800 pairs, ordinary training
left the backdoor working on 86% of the triggered prompts while trigger-aware safety training removed
it in 200 training pairs. Internal measurements and a relearning test on both safety-trained models
show that trigger-aware training truly removed the backdoor internal vector direction, while ordinary
training kept it.
