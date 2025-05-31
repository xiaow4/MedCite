# MedCite Toolkit

`MedCite` is an enhanced version of [`MedRAG`](https://teddy-xionggz.github.io/benchmark-medical-rag/) that provides citation-enabled medical question answering with automatic reference generation and quality evaluation.

<!-- [![Paper](https://img.shields.io/badge/paper-available-brightgreen)](https://aclanthology.org/2024.findings-acl.372/)
[![Homepage](https://img.shields.io/badge/homepage-available-blue)](https://teddy-xionggz.github.io/benchmark-medical-rag/)
[![Corpus](https://img.shields.io/badge/corpus-available-yellow)](https://huggingface.co/MedRAG) -->

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

```python
from src.medrag import MedRAG

question = "Are there microbes in human breast milk?"
options = {
    "A": "yes.",
    "B": "no."
}
```

#### Basic MedRAG (No Citations)
```python
medrag = MedRAG(
    llm_name="meta-llama/Meta-Llama-3-8B-Instruct", 
    rag=True, 
    retriever_name="Hierarchical", 
    corpus_name="PubMed", 
    citation_mode=None  # Standard MedRAG without citations
)

result, snippets, scores = medrag.answer(
    question=question, 
    options=options, 
    k=32
)

print(f"Answer: {result['answer']}")
print(f"Answer choice: {result['answer_choice']}")

# Example output:
# Answer: "Yes, human breast milk contains microorganisms including beneficial bacteria that support infant health and immune development."
# Answer choice: "A"
```

#### MedCite with Pre-generation Citations (Post-Retrieval Generation, PRG in the paper)
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

# Example output:
# Answer with citations: "Yes, human breast milk contains a diverse community of microorganisms[1]. Research has identified various bacterial species that may support infant gut health[2][3]."
# Answer choice: "A"
# Cited documents: {
#     "1": {"title": "The human milk microbiome", "content": "Human breast milk contains...", "pmid": "25825906"},
#     "2": {"title": "Bacterial diversity in human milk", "content": "Lactobacillus species...", "pmid": "23398556"},
#     "3": {"title": "Breast milk microbiome and health", "content": "Beneficial bacteria contribute...", "pmid": "27217095"}
# }
```

#### MedCite with Post-processing Citations ((Post-Generation Citation, PGC in the paper))
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
print(f"Cited documents: {result['cited_docs']}")

```

#### MedCite with Both Pre and Post Citations (MedCite, Two-Pass Approach in the paper)
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

#### Citation Quality Evaluation

The MedCite toolkit includes a comprehensive evaluation framework for assessing citation quality and answer accuracy.

**Command Line Evaluation:**
```bash
# Basic evaluation
python src/eval.py --results_dir /path/to/results

# Evaluate specific file range
python src/eval.py --results_dir /path/to/results --start_idx 0 --end_idx 10

# Use different evaluation model
python src/eval.py --results_dir /path/to/results --eval_model "mistralai/Mistral-7B-Instruct-v0.3"

# Verbose output with detailed information
python src/eval.py --results_dir /path/to/results --verbose

# Save results to JSON file
python src/eval.py --results_dir /path/to/results --output evaluation_results.json
```

**Programmatic Evaluation:**
```python
from src.eval import MedCiteEvaluator

# Initialize evaluator
evaluator = MedCiteEvaluator(
    eval_model_name="mistralai/Mistral-7B-Instruct-v0.3", 
    device=0
)

# Evaluate single answer
answer_text = result['answer']
cited_docs = result['cited_docs']

citation_results = evaluator.evaluate_citations(
    answer_text=answer_text,
    cited_docs=cited_docs,
    verbose=True
)

print(f"Citation Recall: {citation_results['recall']:.3f}")
print(f"Citation Precision: {citation_results['precision']:.3f}")
print(f"Number of statements: {citation_results['num_statements']}")

# Evaluate entire dataset
results = evaluator.evaluate_dataset(
    results_dir="/path/to/results",
    dataset_name="bioasq",
    verbose=False
)

# Print formatted results
evaluator.print_results(results)

# Example output:
# ============================================================
# MEDCITE EVALUATION RESULTS  
# ============================================================
# 
# CITATION QUALITY:
#    Recall:    0.8750
#    Precision: 0.9200
#    Evaluated: 8 files
# 
# ANSWER CHOICE ACCURACY:
#    Accuracy:  0.7500
#    Std Dev:   0.4330
#    Evaluated: 8 files
# ============================================================

**Result File Format:**

The evaluation system expects JSON files with the following structure:
```python
{
    "answer": "Answer text with citations[1][2]...",
    "answer_choice": "A",  # For multiple choice questions
    "cited_docs": {
        "1": {"content": "Document content...", "pmid": "12345"},
        "2": {"content": "Another document...", "pmid": "67890"}
    }
}
```

Files should be named as `test_0.json`, `test_1.json`, etc., corresponding to dataset question indices.

#### Key Parameters

- **citation_mode**: Citation generation strategy
  - `None`: Disable citations (standard MedRAG)
  - `"pre_only"`: Generate citations during answer creation (Post-Retrieval Generation, PRG)
  - `"post_only"` : Add citations after answer generation (Post-Generation Citation, PGC)
  - `"both"`: Combine both approaches (MedCite, Two-Pass Approach)
  

- **k1**: Number of documents for initial retrieval (default: 32)
- **k2**: Number of additional documents for citation enhancement (default: 3)
- **citation_rerank**: Whether to use LLM for reranking citations (default: False)
- **retriever_name**: Retrieval method
  - `"BM25"`: BM25
  - `"MedCPT"`: Biomedical dense retriever
  - `"RRF-2"`: BM25 + MedCPT fusion
  - `"RRF-4"`: Multi-retriever fusion
  - `"Hierarchical"`: BM25 + Cross-encoder reranking


- **snippets**: List of all retrieved document snippets (for reference)
- **scores**: Retrieval relevance scores
```

## Citation
For the use of `MedCite`, please consider citing ...

