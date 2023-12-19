# [ZeroDay.Tools](https://www.zeroday.tools/)

## How it works:

![Alt Text](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExd3ZyZWNxdDdodDQwNmx5NGRhOXZ5aTVqZmN3OXJlbG9mejk2bzdlMCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/FH4XtAdP7eKmo0YeRD/giphy.gif)

Per-Model Templates for generation of Adversarial Strings in support of net-new attack methods via Greedy Coordinate Gradient optimization of target input/outputs

## Example Utilization:

### -Manipulation of [Self-Supervised Systems](https://github.com/microsoft/TaskWeaver), [AI Assistants](https://platform.openai.com/docs/assistants/overview), [Agentic Frameworks](https://learn.microsoft.com/en-us/semantic-kernel/overview/), and connected tools/plugins via direct or [indirect injection](https://github.com/greshake/llm-security#compromising-llms-using-indirect-prompt-injection) of [adversarial strings](https://llm-attacks.org/) optimized for return of specific arguments by [Models designed to call external functions](https://github.com/nexusflowai/NexusRaven/), directly access [tooling frameworks](https://python.langchain.com/docs/integrations/tools/), etc; such that hardening against [privelege escalation](https://www.crowdstrike.com/cybersecurity-101/privilege-escalation/) is affected by Security Teams

e.g. Unauthorized IAM Actions, Internal Database Access, etc

### -Membership & Attribute Inference Attack definition for [open-source](https://arxiv.org/pdf/2311.17035.pdf#subsection.5.2), [semi-closed](https://arxiv.org/pdf/2311.17035.pdf), and [closed-source](https://arxiv.org/pdf/2311.17035.pdf#subsection.5.2) models via [targetting of behavior](https://arxiv.org/pdf/2311.17035.pdf#subsection.5.1) that elicit [high-precision recall of underlying training data](https://arxiv.org/pdf/2311.17035.pdf#subsection.5.7); for use in validation of [GDPR-compliant data deletion](https://gdpr-info.eu/art-17-gdpr/) ([alongside layer validation](https://weightwatcher.ai/)), Red/Blue Teaming of [LLM Architectures & Monitoring](https://www.latentspace.tools/), etc

## Major Changes:

### -Updated [embedding functions](https://github.com/rabbidave/LLM-Attacks-v2/blob/main/llm_attacks/base/attack_manager.py#L35) within attack_manager.py to support multiple new model classes (e.g. Mi(s/x)tralForCausalLM, PhiForCausalLM, etc)

### -Added [conditional logic](https://github.com/rabbidave/LLM-Attacks-v2/blob/main/llm_attacks/base/attack_manager.py#L1480) to the ModelWorker init inside attack_manager.py allowing for the loading of quantized models based on presence of "GPTQ" in the model path (e.g. GPTQ versions of Mixtral)

### -Automated and Parameterized the [original demo.py](https://github.com/rabbidave/LLM-Attacks-v2/blob/main/demo.ipynb) into an extensible attack framework allowing for [localization via HF FS API](https://huggingface.co/docs/huggingface_hub/main/en/guides/hf_file_system), definition of target input/outputs w/ test criteria via externalization to environment variables, logging of those prompts/adversarial strings for utilization later, as well as integration with CI/CD pipelines

#### Note: For details on the updated attack scripts [contact me directly](https://www.linkedin.com/in/davidisaacpierce/); trying to balance awareness of a core LLM vulnerability (Note: confirming SSM susceptibility) against responsible open-source contributions

This repo serves as an actively developed version of the [base repo](https://github.com/llm-attacks/llm-attacks)
