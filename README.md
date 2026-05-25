# CS-Guard
CS-Guard: Benchmarking LLM Guardrails for Code Generation Security
We introduce CS-Guard, the first benchmark to systematically evaluate guardrails for code generation security.
# Register your guardrail
CS-Guard considers a layered taxonmy:

- **Category-layer**: 3 guardrail categories are considered. **1) strategy-type** guardrails require a base LLM's real-time generation to operate and cannot function as standalone defenses. **2) classifier-type** guardrails use specifically trained models to detect harmful content and operate independently, requiring only conversation history; and **3) internal-type** guardrails are embedded within LLMs prior to deployment (e.g., through alignment training).
- **Operation-layer** 2 operation types are considered. **classification** and **generation**. Classification classify malicious prompt or jailbroken responses. Generation operation refers to any mechanism that controls a base LLM's output generation to produce safer responses.
- **Position-layer** 3 position types are considered. **1) Input-type** guardrails detect malicious prompts before LLM processing. **2) Output-type** guardrails assess both the prompt and the LLM's response to determine if a jailbreak succeeded, permitting any response that does not comply with the malicious prompt. **3) Flexible-type** guardrails secure LLMs through neither prompt detection nor response assessment, generation-operation guardrails are a key example.

To register a base model/white-box guardrail, open **bench_engine/engine/base.py** and implement a custom class using existing class as reference.
To register a guardrail, open **bench_engine/engine/guardrails.py** and implement a custom class using existing class as reference. You should specify procedure variable to match the operation layer. You can customize the label collection procedure, prompt construction, generation for different guardrails and position layers.

# Experiment
To run experment, use the following command:
```
python generate_label.py -o "output destination" -a "attack method" -dt "MetaTTP for text-to-code, infill or translate or complete for code-to-code" -dir "input file" -m "i or io, stands for input and output" -st "test for input classification, eval for output classification" -d "guardrail" -b "base LLM" -ap "Resume from an existing output file, True or False" -sd "t2c or c2c"

# example
## text-to-code
### Base model only
python generate_label.py -o qwen_output.jsonl -dt MetaTTP -dir dataset/mitre_benchmark_100_per_category_with_augmentation.json -st test -b qwen -ap False -sd t2c

### Guardrail
python generate_label.py -o qwen_SmoothLLM_output.jsonl -d SmoothLLM -dt MetaTTP -dir dataset/mitre_benchmark_100_per_category_with_augmentation.json -st test -b qwen -ap False -sd t2c

python generate_label.py -o qwen_SelfReminder_output.jsonl -d SelfReminder -dt MetaTTP -dir dataset/mitre_benchmark_100_per_category_with_augmentation.json -st test -b qwen -ap False -sd t2c

python generate_label.py -o qwen3guard_output.jsonl -d Qwen3GuardGen -dt MetaTTP -dir dataset/mitre_benchmark_100_per_category_with_augmentation.json -st test -ap False -sd t2c

python generate_label.py -o qwen3guard_qwen_classification.jsonl -d Qwen3GuardGen -dt MetaTTP -dir qwen_output.jsonl -st eval -ap False -sd t2c

### Jailbreak
python generate_label.py -o qwen_EvilConfident_output.jsonl -a EvilConfident -dt MetaTTP -dir dataset/mitre_benchmark_100_per_category_with_augmentation.json -st test -b qwen -ap False -sd t2c

### Jailbreak vs guardrail
python generate_label.py -o qwen_SelfReminder_EvilConfident_output.jsonl -a EvilConfident -d SelfReminder -dt MetaTTP -dir dataset/mitre_benchmark_100_per_category_with_augmentation.json -st test -b qwen -ap False -sd t2c

## Code-to-code
python generate_label.py -o qwen_codeinfill_output.jsonl -dt infill -dir dataset/code_infilling_prompt.json -st test -b qwen -ap False -sd c2c
```

## Label the response
To label the LLM response, use the following command:
```
python generate_label.py -o "Labeler output destination" -j "base LLM serve as ai labeler" -dt "MetaTTP for text-to-code, infill or translate or complete for code-to-code" -dir "input file" -m io -st eval -sd "t2c or c2c"

python generate_label.py -o gptoss_qwen_labeled.jsonl -j gptoss -dt MetaTTP -dir qwen_output.jsonl -m io -st eval -sd t2c
```
## Calculate metrics
Use the following examples as reference for calculating metrics:
```
from bench_engine.engine.utilities import print_result, get_label, calculate_F1, calculate_difference, calculate_asr
# ASR
result = print_result("qwen_codeinfill_output.jsonl")
asr = calculate_asr(result, "bad_list")

# ASR reduction
gt_result = print_result("qwen_codeinfill_output.jsonl")
pre_result = print_result("qwen_codeinfill_SelfReminder_output.jsonl")
result = calculate_difference(gt_result, pre_result)

# F1
## To calculate F1, you should first sample a balanced response set from the labeled response file, and then extract corresponding response from the output classifier's output file.
gt_result = get_label("gptoss_qwen_labeled_balanced.jsonl")
pre_result = get_label("qwen3guard_qwen_classification.jsonl")
F1_score = calculate_F1(gt_result, pre_result)
```
