# Project Testing Datasets

Based on the research paper `"Ensemble Learning for Heterogeneous Large Language Models with Deep Parallel Collaboration"` (DeePEn), the following datasets were utilized for evaluating the ensemble pipeline. 

These datasets cover Comprehensive Examination, Reasoning Capabilities, Knowledge Capacities, and Machine Translation.

## 1. MMLU (Massive Multitask Language Understanding)
* **Category**: Comprehensive Examination
* **Evaluation**: 5-shot
* **Source**: Collected by Hendrycks et al. (2021). Covers 57 subjects across STEM, humanities, and others.
* **HuggingFace Path**: `cais/mmlu`

## 2. ARC-C (AI2 Reasoning Challenge - Challenge Set)
* **Category**: Comprehensive Examination
* **Evaluation**: 0-shot
* **Source**: Collected from standardized natural science tests by AI2 (Clark et al., 2018).
* **HuggingFace Path**: `allenai/ai2_arc` (config: `ARC-Challenge`)

## 3. GSM8K (Grade School Math 8K)
* **Category**: Reasoning Capabilities
* **Evaluation**: 4-shot
* **Source**: High-quality grade school math word problems collected by OpenAI (Cobbe et al., 2021).
* **HuggingFace Path**: `gsm8k`

## 4. PIQA (Physical Interaction QA)
* **Category**: Reasoning Capabilities
* **Evaluation**: 0-shot
* **Source**: A commonsense reasoning dataset for physical intuition (Bisk et al., 2020).
* **HuggingFace Path**: `piqa`

## 5. TriviaQA
* **Category**: Knowledge Capacities
* **Evaluation**: 5-shot
* **Source**: A reading comprehension dataset authored by trivia enthusiasts (Joshi et al., 2017).
* **HuggingFace Path**: `trivia_qa`

## 6. NQ (Natural Questions)
* **Category**: Knowledge Capacities
* **Evaluation**: 5-shot
* **Source**: A QA corpus consisting of queries issued to the Google search engine (Kwiatkowski et al., 2019).
* **HuggingFace Path**: `natural_questions`

## 7. Flores-200
* **Category**: Machine Translation (Specialist vs Generalist Ensemble)
* **Source**: Meta/Facebook Research benchmark for multilingual translation.
* **HuggingFace Path**: `facebook/flores`
