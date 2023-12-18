An actively developed version of the [base repo](https://github.com/llm-attacks/llm-attacks)

### Updated [embedding functions](https://github.com/rabbidave/LLM-Attacks-v2/blob/main/llm_attacks/base/attack_manager.py#L35) within attack_manager.py to support multiple new model classes (e.g. Mi(s/x)tralForCausalLM, PhiForCausalLM, etc)

### Added [conditional logic](https://github.com/rabbidave/LLM-Attacks-v2/blob/main/llm_attacks/base/attack_manager.py#L1480) to the ModelWorker init inside attack_manager.py allowing for the loading of quantized models based on presence of "GPTQ" in the model path (e.g. GPTQ versions of Mixtral)

### Automated and Parameterized the [original demo.py](https://github.com/rabbidave/LLM-Attacks-v2/blob/main/demo.ipynb) into an extensible attack framework allowing for [localization via HF FS API](https://huggingface.co/docs/huggingface_hub/main/en/guides/hf_file_system), definition of target input/outputs w/ test criteria via externalization to environment variables, logging of those prompts/adversarial strings for utilization later, as well as integration with CI/CD pipelines

#### For details on the updated attack scripts [contact me directly](https://www.linkedin.com/in/davidisaacpierce/); trying to balance awareness of a core LLM vulnerability (Note: confirming SSM susceptibility) against responsible open-source contributions