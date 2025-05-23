<a target="_blank" href="https://colab.research.google.com/github/rabbidave/ZeroDay.Tools/blob/main/ZeroDayTools.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab Notebook Here"/>
</a>

# [ZeroDay.Tools](https://www.zeroday.tools/) - Gen AI Hardening & [Attack Suite](https://colab.research.google.com/github/rabbidave/ZeroDay.Tools/blob/main/ZeroDayTools.ipynb)

**Note:** For per-integration logging & monitoring, see [LatentSpace.Tools](https://github.com/rabbidave/LatentSpace.Tools?tab=readme-ov-file#ai-security-architecture).

![testing](https://github.com/rabbidave/ZeroDay.Tools/blob/c711f7faf503c2cc89365db354e239e42715fe61/image.png)

This repository provides an up-to-date **AI/ML Hardening Framework** and a **Multimodal Attack Suite** for Generative AI. It is built around the security notions of a **Kill Chain x Defense Plan**, primarily focusing on Gen AI, with illustrative examples from Discriminative ML and Deep Reinforcement Learning. This work is predicated on:

1.  The [universal and transferable nature](https://llm-attacks.org/) of attacks against Auto-Regressive models.
2.  The [conserved efficiency of text-based attack modalities](https://arxiv.org/pdf/2307.14061v1.pdf) (see: Figure 3) even for multimodal models.
3.  The [non-trivial nature of hardening GenAI systems](https://www.latentspace.tools/).

---

## 🛡️ AI Security Framework: Kill Chain & Defense

Our approach to AI security is systematically structured around understanding, identifying, and mitigating threats across a defined **AI Kill Chain**. This framework enables a robust defense plan for Generative AI, Discriminative ML, and Deep Reinforcement Learning systems.

### Vulnerability Visualization Example: Pre-Processed Optimization Attack

![Adversarial Attack GIF](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZXRvNWlqZWhiYmFrbmp3a2RsOTZmdTQ5YmY0ZnU1OGIyNW8wYmVobSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/K0UZJibCsa6Ty0CIyI/source.gif)

*This GIF demonstrates an attack utilizing per-model templates to generate adversarial strings. It employs Greedy Coordinate Gradient optimization of target input/outputs, achieving results in minutes on consumer hardware when starting from a template.*

---

## 📋 Core Components: AI/ML Hardening Checklist

The following checklist summarizes key exposures and core dependencies for each step in the AI kill chain. For detailed takeaways, mitigation strategies, and in-line citations, please refer to the links provided, which lead to the "Detailed Vulnerability Remediation & Mitigation Strategies" section.

Download the [Observability Powerpoint](https://github.com/rabbidave/Enterprise-Executive-Summaries/blob/main/Observability.pptx) for additional context on monitoring and defense.

<details>
  <summary>🚨 Gen AI Vulnerabilities x Exposures (Click to Expand)</summary>

### [Kill Chain Step 1) Optimization-Free Attacks](#optimization-free-attack-details)
**Key Exposure:** Brand Reputation Damage & Performance Degradation
**Dependency:** Requires [specific API fields](https://cookbook.openai.com/examples/using_logprobs); no pre-processing

### [Kill Chain Step 2) System Context Extraction](#system-context-extraction-details)
**Key Exposure:** Documentation & Distribution of System Vulnerabilities; Non-Compliance with AI Governance Standards
**Dependency:** Requires API Access over time; [‘time-based blind SQL injection’](https://owasp.org/www-community/attacks/Blind_SQL_Injection) for [Multimodal Models](https://arxiv.org/pdf/2307.08715v2.pdf)

### [Kill Chain Step 3) Model Context Extraction](#model-context-extraction-details)
**Key Exposure:** Documentation & Distribution of Model-Specific Vulnerabilities
**Dependency:** API Access for context window retrieval; VectorDB Access for [decoding embeddings](https://github.com/jxmorris12/vec2text)

### [Kill Chain Step 4) Pre-Processed Attacks](#pre-processed-attack-details)
**Key Exposure:** Data Loss via Exploitation of Distributed Systems
**Dependency:** Whitebox Attacks require [a localized target](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_file_system) of either [Language Models](https://llm-attacks.org/) or [Mutlimodal Models](https://huggingface.co/liuhaotian/llava-v1.5-13b); multiple frameworks (e.g. [SGA](https://github.com/Zoky-2020/SGA), [VLAttack](https://github.com/ericyinyzy/VLAttack), etc) also designed to enable Transferable Multimodal Blackbox Attacks and [evade 'Guard Models'](https://arxiv.org/pdf/2402.15911.pdf)

### [Kill Chain Step 5) Training Data Extraction](#training-data-extraction-details)
**Key Exposure:** Legal Liability from Data Licensure Breaches; Non-Compliance with AI Governance Standards
**Dependency:** Requires API Access over time; ‘rules’ defeated via prior system and model context extraction paired with optimized attacks

### [Kill Chain Step 6) Model Data Extraction](#model-data-extraction-details)
**Key Exposure:** [IP Loss](https://arxiv.org/pdf/2403.06634.pdf), Brand Reputational Damage & Performance Degradation; Non-Compliance with AI Governance Standards, especially for [“high-risk systems”](https://cset.georgetown.edu/article/the-eu-ai-act-a-primer/)
**Dependency:** System Access to GPU; net-new threat vector with [myriad vulnerable platforms](https://github.com/trailofbits/LeftoverLocalsRelease)

### [Kill Chain Step 7) Supply Chain & Data Poisoning](#supply-chain--data-poisoning-details)
**Key Exposure:** Brand Reputation Damage & Performance Degradation; Non-Compliance with AI Governance Standards, especially for [“high-risk systems”](https://cset.georgetown.edu/article/the-eu-ai-act-a-primer/)
**Dependency:** Target use of compromised data & models; integration of those vulnerabilities with CI/CD systems

### [Team Debrief re: Model-Specific Vulnerabilities](#model-specific-vulnerability-details)
**Key Exposure:** Documentation & Distribution of System Vulnerabilities; Brand Reputation Damage & Performance Degradation
**Dependency:** Lack of Active Assessment of Sensitive or External Systems

</details>

---

## 🛠️ Detailed Vulnerability Remediation & Mitigation Strategies

This section provides in-depth information on the dependencies, key exposures, and mitigation takeaways for each vulnerability outlined in the checklist.

#### Optimization-Free Attack Details
* **Dependency:** Requires [specific API fields](https://cookbook.openai.com/examples/using_logprobs); no pre-processing.
* **Key Exposure:** Brand Reputation Damage & Performance Degradation.
* **Takeaway:** [Mitigate low-complexity priming attacks](https://llmpriming.focallab.org/) via [evaluation of input/output embeddings](https://www.latentspace.tools/#h.de5k8d8cxz8c) against moving windows of time, as well as limits on what data is available via API (e.g., [Next-Token Probabilities aka Logits](https://cookbook.openai.com/examples/using_logprobs)). This also mitigates DDoS attacks and indicates instances of poor generalization.

#### System Context Extraction Details
* **Dependency:** Requires API Access over time; [‘time-based blind SQL injection’](https://owasp.org/www-community/attacks/Blind_SQL_Injection) for [Multimodal Models](https://arxiv.org/pdf/2307.08715v2.pdf).
* **Key Exposure:** Documentation & Distribution of System Vulnerabilities; Non-Compliance with AI Governance Standards.
* **Takeaway:** Mitigate retrieval of information about the system and application controls from Time-Based Blind Injection Attacks via [Application-Specific Firewalls](https://www.f5.com/glossary/application-firewall) and [Error Handling Best-Practices](https://brightsec.com/blog/error-based-sql-injection/). Augment detection for sensitive systems by [evaluating conformity of inputs/outputs](https://www.latentspace.tools/#h.rmca9kuof4sx) against pre-embedded attack strings, and flagging long-running sessions for review.

#### Model Context Extraction Details
* **Dependency:** API Access for context window; [Access to Embeddings for Decoding](https://github.com/jxmorris12/vec2text) (e.g., VectorDB).
* **Key Exposure:** Documentation & Distribution of Model Vulnerabilities & Data Access.
* **Takeaway:** Reduce the risk from discoverable rules, extractable context (e.g., persistent attached document-based systems context), etc., via [pre-defined rules](https://developer.nvidia.com/blog/nvidia-enables-trustworthy-safe-and-secure-large-language-model-conversational-systems/). Prevent [decodable embeddings](https://github.com/jxmorris12/vec2text) (e.g., additional underlying data via VectorDB & Backups) by adding [appropriate levels of noise](https://arxiv.org/pdf/2310.06816.pdf) or using customized embedding models for sensitive data.

#### Pre-Processed Attack Details
* **Dependency:** Whitebox Attacks require [a localized target](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_file_system); multiple frameworks (e.g., [SGA](https://github.com/Zoky-2020/SGA), [VLAttack](https://github.com/ericyinyzy/VLAttack), etc.) support Transferable Multimodal Blackbox Attacks and [evade 'Guard Models'](https://arxiv.org/pdf/2402.15911.pdf).
* **Key Exposure:** Data Loss via Exploitation of Distributed Systems.
* **Takeaway:** [Defeat pre-processed optimization attacks](https://www.latentspace.tools/) by pre-defining embeddings for 'good' and 'bad' examples, logging, [clustering, and flagging of non-conforming entries](https://www.latentspace.tools/#h.lwa4hv3scloi) pre-output generation, as well as utilizing windowed evaluation of input/output embeddings against application-specific baselines.

#### Training Data Extraction Details
* **Dependency:** Requires API Access over time; ‘rules’ defeated via prior system and model context extraction paired with optimized attacks.
* **Key Exposure:** Legal Liability from Data Licensure Breaches; Non-Compliance with AI Governance Standards.
* **Takeaway:** [Prevent disclosure of underlying data](https://not-just-memorization.github.io/extracting-training-data-from-chatgpt.html) while mitigating membership or attribute inference attacks with pre-defined context rules (e.g., “no repetition”), [whitelisting & monitoring of allowed topics](https://developer.nvidia.com/blog/nvidia-enables-trustworthy-safe-and-secure-large-language-model-conversational-systems/), as well as [DLP](https://www.microsoft.com/en-us/security/business/security-101/what-is-data-loss-prevention-dlp) paired with [active statistical monitoring](https://www.latentspace.tools/) via pre/post-processing of inputs/outputs.

#### Model Data Extraction Details
* **Dependency:** System Access to GPU; net-new threat vector with [myriad vulnerable platforms](https://github.com/trailofbits/LeftoverLocalsRelease).
* **Key Exposure:** [IP Loss](https://arxiv.org/pdf/2403.06634.pdf), Brand Reputational Damage & Performance Degradation; Non-Compliance with AI Governance Standards, especially for [“high-risk systems”](https://cset.georgetown.edu/article/the-eu-ai-act-a-primer/).
* **Takeaway:** Multiple Open-Source Attack frameworks are exploiting a previously underutilized data exfiltration vector in the form of GPU VRAM, which has traditionally been a shared resource without active monitoring. Secure virtualization and segmentation tooling exists for GPUs, but mitigating this vulnerability is an active area of research.

#### Supply Chain & Data Poisoning Details
* **Dependency:** Target use of compromised data & models; integration of those vulnerabilities with CI/CD systems.
* **Key Exposure:** Brand Reputation Damage & Performance Degradation; Non-Compliance with AI Governance Standards, especially for [“high-risk systems”](https://cset.georgetown.edu/article/the-eu-ai-act-a-primer/).
* **Takeaway:** Mitigate [Supply Chain](https://www.crowdstrike.com/cybersecurity-101/cyberattacks/supply-chain-attacks/) & [Data Poisoning](https://spectrum.ieee.org/ai-cybersecurity-data-poisoning) attacks via use of [Open-Source Foundation Models and Open-Source Data](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_file_system) wherein [Data Provenance/Lineage](https://www.graphable.ai/blog/what-is-data-lineage-data-provenance/) can be established, versions can be hashed, etc. Thereafter, affect access and version control of fine-tuning data, contextual data (i.e., augmented generation), etc.

#### Model Specific Vulnerability Details
* **Dependency:** Lack of Active Assessment of Sensitive or External Systems.
* **Key Exposure:** Documentation & Distribution of System Vulnerabilities; Brand Reputation Damage & Performance Degradation.
* **Takeaway:** Utilize a [Defense in Depth](https://en.wikipedia.org/wiki/Defense_in_depth_(computing)) approach (e.g., [Purple Teaming](https://www.splunk.com/en_us/blog/learn/purple-team.html)), especially for Auto Regressive Models, while staying up to date on the latest [attack & defense paradigms](https://owasp.org/www-project-top-10-for-large-language-model-applications/). Utilize open-source [code-generation](https://ai.meta.com/llama/purple-llama/#cybersecurity) and [vulnerability](https://github.com/cleverhans-lab/cleverhans) assessment frameworks, [contribute to the community](https://www.zeroday.tools/), etc.

---

## ⚙️ Practical Applications & Use Cases

This framework and the accompanying attack suite can be utilized for:

* **Manipulation of AI Systems:**
    Targeting [Self-Supervised Systems](https://github.com/microsoft/TaskWeaver), [AI Assistants](https://platform.openai.com/docs/assistants/overview), [Agentic Frameworks](https://learn.microsoft.com/en-us/semantic-kernel/overview/), and connected tools/plugins. This is achieved via direct or [indirect injection](https://github.com/greshake/llm-security#compromising-llms-using-indirect-prompt-injection) of [adversarial strings](https://llm-attacks.org/) optimized to make [Models designed to call external functions](https://github.com/nexusflowai/NexusRaven/) or access [tooling frameworks](https://python.langchain.com/docs/integrations/tools/) return specific arguments.
    * *Example Impact:* Unauthorized IAM Actions, Internal Database Access, aiding in [privilege escalation](https://www.crowdstrike.com/cybersecurity-101/privilege-escalation/).

* **Inference Attack Definition:**
    Defining Membership & Attribute Inference Attacks for [open-source](https://arxiv.org/pdf/2311.17035.pdf#subsection.5.2), [semi-closed](https://arxiv.org/pdf/2311.17035.pdf), and [closed-source](https://arxiv.org/pdf/2311.17035.pdf#subsection.5.2) models. This involves [targeting behavior](https://arxiv.org/pdf/2311.17035.pdf#subsection.5.1) that elicits [high-precision recall of underlying training data](https://arxiv.org/pdf/2311.17035.pdf#subsection.5.7).
    * *Example Application:* Validation of [GDPR-compliant data deletion](https://gdpr-info.eu/art-17-gdpr/) ([alongside layer validation](https://weightwatcher.ai/)), Red/Blue Teaming of [LLM Architectures & Monitoring](https://www.latentspace.tools/).

---

## 🔬 Expanding Scope: Traditional ML & Reinforcement Learning

While the primary focus is Generative AI, these security principles and vulnerabilities also extend to other AI paradigms.

<details>
  <summary>🔍 Examples of Traditional ML and Deep/Reinforcement Learning Vulnerabilities (Click to Expand)</summary>

#### Reinforcement Learning - Invisible Blackbox Perturbations Compound Over Time
* **Key Exposure:** System-Specific Vulnerability & Performance Degradation.
* **Dependency:** Lack of Actively Monitored & Versioned RL Policies.
* **Takeaway:** Mitigate the compounding nature of poorly aligned & incentivized reward functions and resultant RL policies by actively logging, monitoring & alerting such that divergent policies are identified. While [adversarial training increases robustness](https://blogs.ucl.ac.uk/steapp/2023/12/20/adversarial-attacks-robustness-and-generalization-in-deep-reinforcement-learning/), these systems remain susceptible to attack.

#### Discriminative Machine Learning - Probe for Pipeline & Package Dependencies
* **Dependency:** Requires Out-Of-Date Vulnerability Definitions and/or lack of image scanning when deploying previous builds.
* **Key Exposure:** Brand Reputation Damage & Performance Degradation.
* **Takeaway:** Mitigate commonly [exploited repos](https://thehackernews.com/2023/12/116-malware-packages-found-on-pypi.html) and [analytics packages](https://security.snyk.io/package/pip/pyspark) by establishing best-practices with respect to vulnerability management, repackaging, and image scanning.
</details>

---

## 🚀 Getting Started & Key Resources

1.  **Explore the Attack Suite:** Launch the [Colab Notebook](https://colab.research.google.com/github/rabbidave/ZeroDay.Tools/blob/main/ZeroDayTools.ipynb) to see various attacks in action.
2.  **Review the Hardening Checklist:** Familiarize yourself with the [Gen AI Vulnerabilities x Exposures](#core-components-aiml-hardening-checklist) to understand potential risks.
3.  **Dive Deeper into Remediation:** Use the [Detailed Vulnerability Remediation & Mitigation Strategies](#detailed-vulnerability-remediation--mitigation-strategies) section for specific guidance.
4.  **Understand Observability:** Download the [Observability Powerpoint](https://github.com/rabbidave/Enterprise-Executive-Summaries/blob/main/Observability.pptx) for broader context on AI system monitoring.
5.  **Integrate with Monitoring Tools:** For advanced per-integration logging & monitoring solutions, refer to [LatentSpace.Tools](https://github.com/rabbidave/LatentSpace.Tools?tab=readme-ov-file#ai-security-architecture).

---

## 📈 Key Benefits of This Framework

Understanding and addressing the vulnerabilities outlined in this repository provides significant advantages:

* 🛡️ **Enhanced Security Posture:** Proactively identify and mitigate a wide range of AI-specific threats.
* 📉 **Reduced Risk Exposure:** Minimize potential brand reputation damage, data loss, intellectual property theft, and performance degradation.
* ⚖️ **Improved Compliance & Governance:** Better align with AI governance standards and legal requirements (e.g., GDPR, regulations for “high-risk AI systems”).
* 💡 **Informed Defense Strategies:** Develop more robust and effective defense mechanisms based on a clear understanding of evolving attack vectors.
* 🤝 **Community Engagement & Knowledge:** Stay updated with the latest attack and defense paradigms and contribute to a safer AI ecosystem.