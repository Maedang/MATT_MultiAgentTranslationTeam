# Multi-Agent Translation Team (MATT)

Enhancing Low-Resource Language Translation in Large Language Models through a Multi-Agent Workflow

---

## Overview

The **Multi-Agent Translation Team (MATT)** project explores how **agentic Large Language Model (LLM) workflows** can improve translation quality for **low-resource languages**. Instead of relying on a single translation model, this system decomposes the translation task into multiple specialized agents that collaborate to iteratively refine outputs.

The project evaluates whether structured, multi-agent collaboration can outperform baseline and single-agent translation approaches across multiple quality dimensions.

---

## Motivation & Problem Statement

Low-resource languages are often underserved by traditional machine translation systems due to:
- Limited parallel training data
- Cultural and contextual nuances
- Terminology inconsistencies

This research investigates whether **multi-agent orchestration**—where each agent focuses on a specific aspect of translation—can:
- Improve translation accuracy and fluency
- Preserve cultural context and terminology
- Produce more consistent and explainable translation outputs

---

## Approach

The MATT workflow uses **multiple LLM-based agents**, each assigned a specialized role, such as:
- Initial translation
- Fluency and grammar refinement
- Terminology consistency checking
- Cultural and contextual adaptation

Agent outputs are combined through a structured workflow to produce a final translation. This agentic approach is compared against:
- Baseline translation methods
- Single-agent LLM translations

---

## Evaluation Criteria

Translations are evaluated using a rubric designed for human-aligned quality assessment, including:
- **Accuracy** – Faithfulness to source meaning
- **Fluency** – Grammatical correctness and naturalness
- **Style** – Tone and readability
- **Terminology** – Consistency and domain correctness
- **Cultural Context** – Appropriateness for the target language and audience

---

## Repository Structure

```text
MATT_MultiAgentTranslationTeam/
│
├── agents/
│   ├── agent_definitions.py
│   ├── orchestration_logic.py
│   └── prompts/
│       ├── translation_agent.txt
│       ├── fluency_agent.txt
│       ├── terminology_agent.txt
│       └── cultural_context_agent.txt
│
├── data/
│   ├── source_texts/
│   ├── reference_translations/
│   └── evaluation_sets/
│
├── evaluation/
│   ├── scoring_rubric.md
│   ├── human_evaluation_results/
│   └── model_comparison_metrics/
│
├── experiments/
│   ├── baseline_translation/
│   ├── single_agent_translation/
│   └── multi_agent_translation/
│
├── notebooks/
│   ├── analysis.ipynb
│   └── results_visualization.ipynb
│
├── results/
│   ├── qualitative_examples/
│   └── quantitative_results/
│
├── README.md
└── requirements.txt
