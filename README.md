# MedCite Toolkit

`MedCite` is an enhanced version of `MedRAG` that provides citation-enabled medical question answering with automatic reference generation and quality evaluation.

[![Paper](https://img.shields.io/badge/paper-available-brightgreen)](https://aclanthology.org/2024.findings-acl.372/)
[![Homepage](https://img.shields.io/badge/homepage-available-blue)](https://teddy-xionggz.github.io/benchmark-medical-rag/)
[![Corpus](https://img.shields.io/badge/corpus-available-yellow)](https://huggingface.co/MedRAG)

## News
- (05/15/2025) Our paper has been accepted by ACL 2025 Findings!

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Citation](#citation)

## Introduction

MedCite extends MedRAG with advanced citation capabilities, automatically generating citations for medical answers and providing tools for citation quality evaluation. The system ensures that medical claims are properly referenced to retrieved documents, enhancing the reliability and verifiability of AI-generated medical responses.

## Requirements

- First, install PyTorch suitable for your system's CUDA version by following the [official instructions](https://pytorch.org/get-started/locally/) (2.1.1+cu121 in our case).
- Then, install the remaining requirements using: `pip install -r requirements.txt`
- For GPT-3.5/GPT-4, an OpenAI API key is needed. Replace the placeholder with your key in `src/config.py`.
- `Git-lfs` is required to download and load corpora for the first time.
- `Java` is required for using BM25.

## Usage

Example medical question from [MMLU](https://github.com/hendrycks/test)
```python
from src.medrag import MedRAG

question = "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral"
options = {
    "A": "paralysis of the facial muscles.",
    "B": "paralysis of the facial muscles and loss of taste.",
    "C": "paralysis of the facial muscles, loss of taste and lacrimation.",
    "D": "paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation."
}
```

#### MedCite with Post-processing Citations (Default)
```python
medcite = MedRAG(
    llm_name="meta-llama/Meta-Llama-3-8B-Instruct", 
    rag=True, 
    retriever_name="Hierarchical", 
    corpus_name="PubMed", 
    citation_mode="post_only"  # Default: post-processing citations
)

result, snippets, scores = medcite.answer(
    question=question, 
    options=options, 
    k1=32,  # Initial retrieval documents
    k2=3,   # Additional citation documents
    citation_rerank=True  # Use LLM for citation reranking
)

print(f"Answer with citations: {result['answer']}")
print(f"Answer choice: {result['answer_choice']}")
print(f"Cited documents: {len(result['cited_docs'])}")

# Example output:
# Answer with citations: "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause paralysis of the facial muscles[1]. This is supported by anatomical studies[2][3]."
# Answer choice: "A"
# Cited documents: 3
```

#### MedCite with Pre-generation Citations
```python
medcite = MedRAG(
    llm_name="meta-llama/Meta-Llama-3-8B-Instruct", 
    rag=True, 
    retriever_name="Hierarchical", 
    corpus_name="PubMed", 
    citation_mode="pre_only"  # Citations generated during answer creation
)

result, snippets, scores = medcite.answer(
    question=question, 
    options=options, 
    k1=32
)

print(f"Answer with pre-generated citations: {result['answer']}")
```

#### MedCite with Both Pre and Post Citations
```python
medcite = MedRAG(
    llm_name="meta-llama/Meta-Llama-3-8B-Instruct", 
    rag=True, 
    retriever_name="Hierarchical", 
    corpus_name="PubMed", 
    citation_mode="both"  # Combined citation approach
)

result, snippets, scores = medcite.answer(
    question=question, 
    options=options, 
    k1=32,      # Initial retrieval documents  
    k2=5,       # Additional citation documents
    citation_rerank=True
)

print(f"Enhanced answer with comprehensive citations: {result['answer']}")
```

#### MedCite with Different Retrievers
```python
# Using MedCPT retriever
medcite_medcpt = MedRAG(
    llm_name="meta-llama/Meta-Llama-3-8B-Instruct", 
    rag=True, 
    retriever_name="MedCPT", 
    corpus_name="PubMed", 
    citation_mode="post_only"
)

# Using Hierarchical retriever (BM25 + Cross-encoder)
medcite_hier = MedRAG(
    llm_name="meta-llama/Meta-Llama-3-8B-Instruct", 
    rag=True, 
    retriever_name="Hierarchical", 
    corpus_name="PubMed", 
    citation_mode="post_only"
)

# Using RRF-2 retriever (BM25 + MedCPT fusion)
medcite_rrf = MedRAG(
    llm_name="meta-llama/Meta-Llama-3-8B-Instruct", 
    rag=True, 
    retriever_name="RRF-2", 
    corpus_name="PubMed", 
    citation_mode="post_only"
)
```

#### Citation Quality Evaluation
```python
from src.eval import extract_statement_citation_pairs, run_llm_recall, run_llm_prec
from transformers import pipeline

# Initialize evaluation model
eval_model = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3", device=0)

# Extract citation pairs from answer
answer_text = result['answer']
cited_docs = result['cited_docs']
statement_pairs = extract_statement_citation_pairs(answer_text)

# Evaluate citation quality
recalls = []
precisions = []

for citations, statement in statement_pairs:
    if citations:
        # Evaluate recall: Does the cited document support the statement?
        for doc_id in citations:
            if doc_id in cited_docs:
                doc_content = cited_docs[doc_id]['content']
                recall_score = run_llm_recall(eval_model, doc_content, statement)
                recalls.append(recall_score)
                
                # Evaluate precision: Does the statement align with the document?
                prec_score = run_llm_prec(eval_model, doc_content, statement)
                precisions.append(prec_score)

print(f"Citation Recall: {sum(recalls)/len(recalls) if recalls else 0:.3f}")
print(f"Citation Precision: {sum(precisions)/len(precisions) if precisions else 0:.3f}")
```

#### Key Parameters

- **citation_mode**: Citation generation strategy
  - `"post_only"` (default): Add citations after answer generation
  - `"pre_only"`: Generate citations during answer creation
  - `"both"`: Combine both approaches
  - `None`: Disable citations (standard MedRAG)

- **k1**: Number of documents for initial retrieval (default: 32)
- **k2**: Number of additional documents for citation enhancement (default: 3)
- **citation_rerank**: Whether to use LLM for reranking citations (default: False)
- **retriever_name**: Retrieval method
  - `"Hierarchical"`: BM25 + Cross-encoder reranking
  - `"MedCPT"`: Biomedical dense retriever
  - `"RRF-2"`: BM25 + MedCPT fusion
  - `"RRF-4"`: Multi-retriever fusion

#### Output Format

The `medcite.answer()` function returns a tuple `(result, snippets, scores)` where:

- **result**: Dictionary containing:
  - `"answer"`: Final answer text with properly formatted citations
  - `"answer_choice"`: Selected option (A/B/C/D) for multiple choice questions
  - `"cited_docs"`: Dictionary of referenced documents with metadata
  - `"snippets"`: Original retrieved document snippets

- **snippets**: List of all retrieved document snippets
- **scores**: Retrieval relevance scores

## Citation
For the use of `MedRAG`, please consider citing
```bibtex
@inproceedings{xiong-etal-2024-benchmarking,
    title = "Benchmarking Retrieval-Augmented Generation for Medicine",
    author = "Xiong, Guangzhi  and
      Jin, Qiao  and
      Lu, Zhiyong  and
      Zhang, Aidong",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.372",
    pages = "6233--6251",
    abstract = "While large language models (LLMs) have achieved state-of-the-art performance on a wide range of medical question answering (QA) tasks, they still face challenges with hallucinations and outdated knowledge. Retrieval-augmented generation (RAG) is a promising solution and has been widely adopted. However, a RAG system can involve multiple flexible components, and there is a lack of best practices regarding the optimal RAG setting for various medical purposes. To systematically evaluate such systems, we propose the Medical Information Retrieval-Augmented Generation Evaluation (MIRAGE), a first-of-its-kind benchmark including 7,663 questions from five medical QA datasets. Using MIRAGE, we conducted large-scale experiments with over 1.8 trillion prompt tokens on 41 combinations of different corpora, retrievers, and backbone LLMs through the MedRAG toolkit introduced in this work. Overall, MedRAG improves the accuracy of six different LLMs by up to 18{\%} over chain-of-thought prompting, elevating the performance of GPT-3.5 and Mixtral to GPT-4-level. Our results show that the combination of various medical corpora and retrievers achieves the best performance. In addition, we discovered a log-linear scaling property and the {``}lost-in-the-middle{''} effects in medical RAG. We believe our comprehensive evaluations can serve as practical guidelines for implementing RAG systems for medicine.",
}
```

For the use of `i-MedRAG`, please consider citing
```bibtex
@inproceedings{xiong2024improving,
  title={Improving retrieval-augmented generation in medicine with iterative follow-up questions},
  author={Xiong, Guangzhi and Jin, Qiao and Wang, Xiao and Zhang, Minjia and Lu, Zhiyong and Zhang, Aidong},
  booktitle={Biocomputing 2025: Proceedings of the Pacific Symposium},
  pages={199--214},
  year={2024},
  organization={World Scientific}
}
```