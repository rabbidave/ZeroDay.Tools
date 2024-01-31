# [ZeroDay.Tools](https://www.zeroday.tools/): Gen AI Hardening x Attack Suite

This repo serves as an Up-to-Date AI/MLHardening Framework; incorporating a Multimodal Attack Suite for Gen AI and links to open-source resources (white/blackbox attacks, evaluations, etc).

This repo is built around the security notions of a Kill Chain x Defense Plan; framed primarily around Gen AI, with examples from Discriminative ML and Deep Reinforcement Learning

This work is predicated on the following:

1) The [universal and transferable nature](https://llm-attacks.org/) of attacks against Auto-Regressive models
2) The [conserved efficiency of text-based attack modalities](https://arxiv.org/pdf/2307.14061v1.pdf) (see: Figure 3) even for mutlimodal models
3) The [non-trivial nature of hardening GenAI systems](https://www.latentspace.tools/).

## AI/ML Hardening Checklist 

The following summarizes the key exposures and core dependencies of each step in the kill chain; follow the links to the relevant section for takeaways, mitigation, and in-line citations

Download the [Observability Powerpoint](https://github.com/rabbidave/Enterprise-Executive-Summaries/blob/main/Observability.pptx) for context

<details>
  <summary>Gen AI Vulnerabilities x Exposures (Click to Expand)</summary>

### [Kill Chain Step 1) Optimization-Free Attacks](https://github.com/rabbidave/ZeroDay.Tools#optimization-free-attack-details)
Key Exposure: Brand Reputation Damage & Performance Degradation

Dependency: Requires [specific API fields](https://cookbook.openai.com/examples/using_logprobs); no pre-processing
### [Kill Chain Step 2) System Context Extraction](https://github.com/rabbidave/ZeroDay.Tools#system-context-extraction-details)
Key Exposure: Documentation & Distribution of System Vulnerabilities; Non-Compliance with AI Governance Standards

Dependency: Requires API Access over time; [‘time-based blind SQL injection’](https://owasp.org/www-community/attacks/Blind_SQL_Injection) for [Multimodal Models](https://arxiv.org/pdf/2307.08715v2.pdf)
### [Kill Chain Step 3) Model Context Extraction](https://github.com/rabbidave/ZeroDay.Tools#model-context-extraction-details)
Key Exposure: Documentation & Distribution of Model-Specific Vulnerabilities

Dependency: API Access for context window retrieval; VectorDB Access for [decoding embeddings](https://github.com/jxmorris12/vec2text)
### [Kill Chain Step 4) Pre-Processed Attacks](https://github.com/rabbidave/ZeroDay.Tools#pre-processed-attack-details)
Key Exposure: Data Loss via Exploitation of Distributed Systems

Dependency: Whitebox Attacks require [a localized target](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_file_system) of either [Language Models](https://llm-attacks.org/) or [Mutlimodal Models](https://huggingface.co/liuhaotian/llava-v1.5-13b); multiple frameworks (e.g. [SGA](https://github.com/Zoky-2020/SGA), [VLAttack](https://github.com/ericyinyzy/VLAttack), etc) also designed to enable Transferable Multimodal Blackbox Attacks
### [Kill Chain Step 5) Training Data Extraction](https://github.com/rabbidave/ZeroDay.Tools#training-data-extraction-details)
Key Exposure: Legal Liability from Data Licensure Breaches; Non-Compliance with AI Governance Standards

Dependency: Requires API Access over time; ‘rules’ defeated via prior system and model context extraction paired with optimized attacks
### [Kill Chain Step 6) Model Data Extraction](https://github.com/rabbidave/ZeroDay.Tools#model-data-extraction-details)
Key Exposure: Brand Reputation Damage & Performance Degradation; Non-Compliance with AI Governance Standards, especially for [“high-risk systems”](https://cset.georgetown.edu/article/the-eu-ai-act-a-primer/)

Dependency: System Access to GPU; net-new threat vector with [myriad vulnerable platforms](https://github.com/trailofbits/LeftoverLocalsRelease)
### [Kill Chain Step 7) Supply Chain & Data Poisoning](https://github.com/rabbidave/ZeroDay.Tools#supply-chain--data-poisoning-details)
Key Exposure: Brand Reputation Damage & Performance Degradation; Non-Compliance with AI Governance Standards, especially for [“high-risk systems”](https://cset.georgetown.edu/article/the-eu-ai-act-a-primer/)

Dependency: Target use of compromised data & models; integration of those vulnerabilities with CI/CD systems
### [Team Debrief re: Model-Specific Vulnerabilities](https://github.com/rabbidave/ZeroDay.Tools#model-specific-vulnerability-details)
Key Exposure: Documentation & Distribution of System Vulnerabilities; Brand Reputation Damage & Performance Degradation

Dependency: Lack of Active Assessment of Sensitive or External Systems

</details>

## Vulnerability Visualizations

Pre-Processed Optimization Attack:

![Alt Text](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZXRvNWlqZWhiYmFrbmp3a2RsOTZmdTQ5YmY0ZnU1OGIyNW8wYmVobSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/K0UZJibCsa6Ty0CIyI/source.gif)

Utilizes Per-Model Templates for generation of Adversarial Strings in support of net-new attack methods via Greedy Coordinate Gradient optimization of target input/outputs; only requires minutes per attack string (on consumer hardware) when starting with a template

## Example Utilization:

#### -Manipulation of [Self-Supervised Systems](https://github.com/microsoft/TaskWeaver), [AI Assistants](https://platform.openai.com/docs/assistants/overview), [Agentic Frameworks](https://learn.microsoft.com/en-us/semantic-kernel/overview/), and connected tools/plugins via direct or [indirect injection](https://github.com/greshake/llm-security#compromising-llms-using-indirect-prompt-injection) of [adversarial strings](https://llm-attacks.org/) optimized for return of specific arguments by [Models designed to call external functions](https://github.com/nexusflowai/NexusRaven/), directly access [tooling frameworks](https://python.langchain.com/docs/integrations/tools/), etc; such that hardening against [privelege escalation](https://www.crowdstrike.com/cybersecurity-101/privilege-escalation/) is affected by Security Teams

e.g. Unauthorized IAM Actions, Internal Database Access, etc

#### -Membership & Attribute Inference Attack definition for [open-source](https://arxiv.org/pdf/2311.17035.pdf#subsection.5.2), [semi-closed](https://arxiv.org/pdf/2311.17035.pdf), and [closed-source](https://arxiv.org/pdf/2311.17035.pdf#subsection.5.2) models via [targetting of behavior](https://arxiv.org/pdf/2311.17035.pdf#subsection.5.1) that elicit [high-precision recall of underlying training data](https://arxiv.org/pdf/2311.17035.pdf#subsection.5.7); for use in validation of [GDPR-compliant data deletion](https://gdpr-info.eu/art-17-gdpr/) ([alongside layer validation](https://weightwatcher.ai/)), Red/Blue Teaming of [LLM Architectures & Monitoring](https://www.latentspace.tools/), etc

## Detailed Vulnerability Remediation

#### Optimization-Free Attack Details
Dependency: Requires [specific API fields](https://cookbook.openai.com/examples/using_logprobs); no pre-processing

Key Exposure: Brand Reputation Damage & Performance Degradation

Takeaway: [Mitigate low-complexity priming attacks](https://llmpriming.focallab.org/) via [evaluation of input/output embeddings](https://www.latentspace.tools/#h.de5k8d8cxz8c) against moving windows of time, as well as limits on what data is available via API (e.g. [Next-Token Probabilities aka Logits](https://cookbook.openai.com/examples/using_logprobs)); also mitigates DDoS attacks and indicates instances of poor generalization

#### System Context Extraction Details

Key Exposure: Documentation & Distribution of System Vulnerabilities; Non-Compliance with AI Governance Standards

Dependency: Requires API Access over time; [‘time-based blind SQL injection’](https://owasp.org/www-community/attacks/Blind_SQL_Injection) for [Multimodal Models](https://arxiv.org/pdf/2307.08715v2.pdf)

Takeaway: Mitigate retrieval of information about the system and application controls from Time-Based Blind Injection Attacks via [Application-Specific Firewalls](https://www.f5.com/glossary/application-firewall) and [Error Handling Best-Practices](https://brightsec.com/blog/error-based-sql-injection/); augment detection for sensitive systems by [evaluating conformity of inputs/outputs](https://www.latentspace.tools/#h.rmca9kuof4sx) against pre-embedded attack strings, and flagging long-running sessions for review

#### Model Context Extraction Details

Key Exposure: Documentation & Distribution of Model Vulnerabilities & Data Access

Dependency: API Access for context window; [Access to Embeddings for Decoding](https://github.com/jxmorris12/vec2text) (e.g. VectorDB)

Takeaway: Reduce the risk from discoverable rules, extractable context (e.g. persistent attached document-based systems context), etc via [pre-defined rules](https://developer.nvidia.com/blog/nvidia-enables-trustworthy-safe-and-secure-large-language-model-conversational-systems/); prevent [decodable embeddings](https://github.com/jxmorris12/vec2text) (e.g. additional underlying data via VectorDB & Backups) by adding [appropriate levels of noise](https://arxiv.org/pdf/2310.06816.pdf) or using customized embedding models for sensitive data.


#### Pre-Processed Attack Details
Key Exposure: Data Loss via Exploitation of Distributed Systems

Dependency: Whitebox Attacks require [a localized target](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_file_system); multiple frameworks (e.g. [SGA](https://github.com/Zoky-2020/SGA), [VLAttack](https://github.com/ericyinyzy/VLAttack), etc) support Transferable Multimodal Blackbox Attacks

Takeaway: [Defeat pre-processed optimization attacks](https://www.latentspace.tools/) by pre-defining embeddings for 'good' and 'bad' examples, logging, [clustering, and flagging of non-conforming entries](https://www.latentspace.tools/#h.lwa4hv3scloi) pre-output generation, as well as utilizing windowed evaluation of input/output embeddings against application-specific baselines

#### Training Data Extraction Details

Key Exposure: Legal Liability from Data Licensure Breaches; Non-Compliance with AI Governance Standards

Dependency: Requires API Access over time; ‘rules’ defeated via prior system and model context extraction paired with optimized attacks

Takeaway:  [Prevent disclosure of underlying data](https://not-just-memorization.github.io/extracting-training-data-from-chatgpt.html) while mitigating membership or attribute inference attacks with pre-defined context rules (e.g. “no repetition”), [whitelisting & monitoring of allowed topics](https://developer.nvidia.com/blog/nvidia-enables-trustworthy-safe-and-secure-large-language-model-conversational-systems/), as well as [DLP](https://www.microsoft.com/en-us/security/business/security-101/what-is-data-loss-prevention-dlp) paired with [active statistical monitoring](https://www.latentspace.tools/) via pre/post-processing of inputs/outputs

#### Model Data Extraction Details

Key Exposure: Legal Liability from Data Licensure Breaches; Non-Compliance with AI Governance Standards

Dependency: System Access to GPU; net-new threat vector with [myriad vulnerable platforms](https://github.com/trailofbits/LeftoverLocalsRelease)

Takeaway: Multiple Open-Source Attack frameworks are exploiting a previously underlized data exfiltration vector in the form of GPU VRAM, which has traditionally been a shared resource without active monitoring; secure virtualization and segmentation tooling exists for GPUs but mitigate this vulnerability is an active area of research.

#### Supply Chain & Data Poisoning Details

Key Exposure: Brand Reputation Damage & Performance Degradation; Non-Compliance with AI Governance Standards, especially for [“high-risk systems”](https://cset.georgetown.edu/article/the-eu-ai-act-a-primer/)

Dependency: Target use of compromised data & models; integration of those vulnerabilities with CI/CD systems

Takeaway: Mitigate [Supply Chain](https://www.crowdstrike.com/cybersecurity-101/cyberattacks/supply-chain-attacks/) & [Data Poisoning](https://spectrum.ieee.org/ai-cybersecurity-data-poisoning) attacks via use of [Open-Source Foundation Models and Open-Source Data](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_file_system) wherein [Data Provenance/Lineage](https://www.graphable.ai/blog/what-is-data-lineage-data-provenance/) can be established, versions can be hashed, etc; thereafter affect access and version control of fine-tuning data, contextual data (i.e. augmented generation), etc.

#### Model Specific Vulnerability Details

Dependency: Lack of Active Assessment of Sensitive or External Systems

Key Exposure: Documentation & Distribution of System Vulnerabilities; Brand Reputation Damage & Performance Degradation

Takeaway: Utilize a [Defense in Depth](https://en.wikipedia.org/wiki/Defense_in_depth_(computing)) approach (e.g. [Purple Teaming](https://www.splunk.com/en_us/blog/learn/purple-team.html)), especially for Auto Regressive Models, while staying up to date on the latest [attack & defense paradigms](https://owasp.org/www-project-top-10-for-large-language-model-applications/); utilize open-source [code-generation](https://ai.meta.com/llama/purple-llama/#cybersecurity) and [vulnerability](https://github.com/cleverhans-lab/cleverhans) assesment frameworks, [contribute to the community](https://www.zeroday.tools/), etc.

<details>
  <summary>Examples of Traditional ML and Deep/Reinforcement Learning Vulnerabilities x Exposures (Click to Expand)</summary>

#### Reinforcement Learning - Invisible Blackbox Perturbations Compound Over Time

Key Exposure: System-Specific Vulnerability & Performance Degradation

Dependency: Lack of Actively Monitored & Versioned RL Policies

Takeaway: Mitigate the compounding nature of poorly aligned & incentivized reward functions and resultant RL policies by actively logging, monitoring & alerting such that divergent policies are identified; [adversarial training increases robustness](https://blogs.ucl.ac.uk/steapp/2023/12/20/adversarial-attacks-robustness-and-generalization-in-deep-reinforcement-learning/) but these systems are still susceptible to attack

#### Discriminative Machine Learning - Probe for Pipeline & Package Dependencies

Dependency: Requires Out-Of-Date Vulnerability Definitions and/or lack of image scanning when deploying previous builds

Key Exposure: Brand Reputation Damage & Performance Degradation

Takeaway: Mitigate commonly [exploited repos](https://thehackernews.com/2023/12/116-malware-packages-found-on-pypi.html) and [analytics packages](https://security.snyk.io/package/pip/pyspark) by establishing best-practices with respection to vulnerability management, repackaging, and image scanning
</details>


### Changelog from [LLM-Attacks](https://github.com/llm-attacks/llm-attacks) base repo:

-Updated [embedding functions](https://github.com/rabbidave/LLM-Attacks-v2/blob/main/llm_attacks/base/attack_manager.py#L35) within attack_manager.py to support multiple new model classes (e.g. Mi(s/x)tralForCausalLM, AutoGPTQForCausalLM, etc)

-Added [conditional logic](https://github.com/rabbidave/LLM-Attacks-v2/blob/main/llm_attacks/base/attack_manager.py#L1480) to the ModelWorker init inside attack_manager.py allowing for the loading of quantized models based on presence of "GPTQ" in the model path (e.g. GPTQ versions of Mixtral)

-Automated and Parameterized the [original demo.py](https://github.com/rabbidave/LLM-Attacks-v2/blob/main/demo.ipynb) into an extensible attack framework allowing for parm'd localization and configuration, iteration over defined target input/outputs w/ test criteria, logging of those prompts/adversarial strings to a standardized JSON format for later utilization, etc

Note: For details on the updated attack scripts [contact me directly](https://www.linkedin.com/in/davidisaacpierce/); trying to balance awareness of a non-patchable vulnerability against responsible open-source contributions. These attacks seem to work against any auto-regressive sequence model irrespective of architecture; including multimodal models
