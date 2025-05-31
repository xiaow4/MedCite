import os
import json
import re
import argparse
import numpy as np
from tqdm import tqdm
from liquid import Template
from transformers import pipeline
from utils import QADataset

class MedCiteEvaluator:
    """
    Medical Citation Quality Evaluator for MedCite system
    """
    
    def __init__(self, eval_model_name="mistralai/Mistral-7B-Instruct-v0.3", device=0):
        """
        Initialize the evaluator
        
        Args:
            eval_model_name (str): Model name for evaluation
            device (int): GPU device number
        """
        print(f"Initializing evaluator with model: {eval_model_name}")
        self.chatbot = pipeline("text-generation", model=eval_model_name, device=device)
        self.answer_list = ["A", "B", "C", "D"]
        self.answer2idx = {ans: i for i, ans in enumerate(self.answer_list)}
        
    def extract_statement_citation_pairs(self, answer_text):
        """
        Extract statement-citation pairs from the final answer text.
        
        Args:
            answer_text (str): Answer text with citations like "Statement[1][2] Another statement[3]"
            
        Returns:
            List of tuples: [(citation_ids, statement_text), ...]
        """
        citation_pattern = re.compile(r'\[(\d+)\]')
        statement_pairs = []
        
        all_citations = [(match.start(), match.end(), match.group(1)) for match in citation_pattern.finditer(answer_text)]
        
        if not all_citations:
            if answer_text.strip():
                return [([], answer_text.strip())]
            else:
                return []
        
        # Split text into segments based on sentence boundaries and citations
        sentences = re.split(r'(?<=[.!?])\s+', answer_text)
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Find citations in this sentence
            matches = citation_pattern.findall(sentence)
            
            if matches:
                citations = list(set(matches))  # Remove duplicates
                
                # Clean the statement by removing all citations
                cleaned_statement = citation_pattern.sub('', sentence)
                
                # Clean up spacing and punctuation
                cleaned_statement = re.sub(r'\s+', ' ', cleaned_statement)
                cleaned_statement = re.sub(r'\s+([.!?])', r'\1', cleaned_statement)
                cleaned_statement = re.sub(r'([.!?])\s+', r'\1 ', cleaned_statement)
                cleaned_statement = cleaned_statement.strip()
                
                if cleaned_statement:  # Only add non-empty statements
                    statement_pairs.append((citations, cleaned_statement))
            else:
                # Statement without citations
                cleaned_statement = sentence.strip()
                if cleaned_statement:
                    statement_pairs.append(([], cleaned_statement))
        
        # If no statement pairs found, try a simpler approach
        if not statement_pairs:
            cleaned_text = citation_pattern.sub('', answer_text)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            all_cited_docs = citation_pattern.findall(answer_text)
            if cleaned_text:
                statement_pairs.append((list(set(all_cited_docs)), cleaned_text))
        
        return statement_pairs

    def run_llm_recall(self, snippet, statement):
        """
        Evaluate if the cited document supports the statement (recall)
        """
        system_prompt = 'You are a helpful medical expert.'

        nli_template = Template('''
        Based on the document, determine whether the statement is fully supported or not.
        
        Options: 
        - Fully Supported: The statement is fully supported by the document.
        - Not Fully Supported: The statement is not fully supported by the document.
        
        Provide only your chosen option.
        
        Document: {{premise}}
                                
        Statement: {{hypothesis}}
        ''')
        prompt = nli_template.render(hypothesis=statement, premise=snippet)
        message = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
        response = self.chatbot(message, max_new_tokens=100, do_sample=False)
        output = response[0]['generated_text']
        answer = output[-1]['content']
        answer = answer.lower().strip()
        return 0 if "not fully supported" in answer else 1

    def run_llm_prec(self, snippet, statement):
        """
        Evaluate if the statement aligns with the document (precision)
        """
        system_prompt = 'You are a helpful medical expert.'

        nli_template = Template('''
        Based on the document, determine whether it supports the statement.
        
        Options:
        - Fully Support: The document fully supports the statement.
        - Partial Support: The document supports part of the statement, but some parts are missing.
        - Cannot Support: The document cannot support the statement.
        
        Provide only the chosen option.
        
        Document: {{premise}}
        
        Statement: {{hypothesis}}
        ''')
        prompt = nli_template.render(hypothesis=statement, premise=snippet)
        message = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
        response = self.chatbot(message, max_new_tokens=100, do_sample=False)
        output = response[0]['generated_text']
        answer = output[-1]['content']
        answer = answer.lower().strip()
        return 0 if "not" in answer else 1

    def evaluate_citations(self, answer_text, cited_docs, verbose=False):
        """
        Evaluate citation quality for a single answer
        
        Args:
            answer_text (str): Answer text with citations
            cited_docs (dict): Dictionary of cited documents
            verbose (bool): Print detailed information
            
        Returns:
            dict: Citation evaluation results
        """
        statement_pairs = self.extract_statement_citation_pairs(answer_text)
        
        if not statement_pairs:
            return {"recall": 0.0, "precision": 0.0, "num_statements": 0}
        
        recalls = []
        precisions = []
        
        for citations, statement in statement_pairs:
            if not citations:
                continue
                
            # Evaluate recall
            valid_docs = [doc_id for doc_id in citations if doc_id in cited_docs]
            if valid_docs:
                if len(valid_docs) > 1:
                    snippet = '\n'.join([cited_docs[doc_id]['content'] for doc_id in valid_docs])
                else:
                    snippet = cited_docs[valid_docs[0]]['content']
                
                recall_score = self.run_llm_recall(snippet, statement)
                recalls.append(recall_score)
                
                # Evaluate precision for each document
                for doc_id in valid_docs:
                    doc_snippet = cited_docs[doc_id]['content']
                    prec_score = self.run_llm_prec(doc_snippet, statement)
                    precisions.append(prec_score)
                    
                if verbose:
                    print(f"Statement: {statement[:100]}...")
                    print(f"Citations: {valid_docs}")
                    print(f"Recall: {recall_score}, Precision scores: {precisions[-len(valid_docs):]}")
                    print("-" * 50)
        
        return {
            "recall": sum(recalls) / len(recalls) if recalls else 0.0,
            "precision": sum(precisions) / len(precisions) if precisions else 0.0,
            "num_statements": len(statement_pairs),
            "num_cited_statements": len([p for p in statement_pairs if p[0]])
        }

    def evaluate_answer_choice(self, predicted_choice, ground_truth_choice):
        """
        Evaluate answer choice accuracy
        """
        if predicted_choice in self.answer_list and ground_truth_choice in self.answer_list:
            return 1 if predicted_choice == ground_truth_choice else 0
        return -1  # Invalid choice

    def evaluate_dataset(self, results_dir, dataset_name="bioasq", file_range=None, verbose=False):
        """
        Evaluate a dataset of results
        
        Args:
            results_dir (str): Directory containing result JSON files
            dataset_name (str): Dataset name for ground truth
            file_range (tuple): (start_idx, end_idx) for file range, None for all files
            verbose (bool): Print detailed information
            
        Returns:
            dict: Comprehensive evaluation results
        """
        print(f"Evaluating results from: {results_dir}")
        
        # Load dataset for ground truth
        dataset = QADataset(dataset_name)
        
        # Find result files
        result_files = []
        if file_range:
            start_idx, end_idx = file_range
            for i in range(start_idx, end_idx):
                fpath = os.path.join(results_dir, f"test_{i}.json")
                if os.path.exists(fpath):
                    result_files.append((i, fpath))
        else:
            # Auto-detect all files
            for fname in os.listdir(results_dir):
                if fname.startswith("test_") and fname.endswith(".json"):
                    try:
                        idx = int(fname.split("_")[1].split(".")[0])
                        fpath = os.path.join(results_dir, fname)
                        result_files.append((idx, fpath))
                    except:
                        continue
            result_files.sort()
        
        print(f"Found {len(result_files)} result files to evaluate")
        
        # Initialize metrics
        citation_recalls = []
        citation_precisions = []
        answer_accuracies = []
        detailed_results = []
        
        # Evaluate each file
        for idx, fpath in tqdm(result_files, desc="Evaluating files"):
            try:
                with open(fpath, 'r') as f:
                    result = json.load(f)
                
                file_result = {"file_idx": idx, "file_path": fpath}
                
                # Citation evaluation
                if 'answer' in result and 'cited_docs' in result:
                    citation_eval = self.evaluate_citations(
                        result['answer'], 
                        result['cited_docs'], 
                        verbose=verbose
                    )
                    citation_recalls.append(citation_eval['recall'])
                    citation_precisions.append(citation_eval['precision'])
                    file_result['citation'] = citation_eval
                
                # Answer choice evaluation
                if 'answer_choice' in result:
                    ground_truth = dataset[idx]['answer']
                    accuracy = self.evaluate_answer_choice(result['answer_choice'], ground_truth)
                    if accuracy != -1:
                        answer_accuracies.append(accuracy)
                    file_result['answer_choice'] = {
                        'predicted': result['answer_choice'],
                        'ground_truth': ground_truth,
                        'correct': accuracy == 1
                    }
                
                detailed_results.append(file_result)
                
            except Exception as e:
                print(f"Error processing {fpath}: {e}")
                continue
        
        # Calculate overall metrics
        overall_results = {
            "citation": {
                "recall": np.mean(citation_recalls) if citation_recalls else 0.0,
                "precision": np.mean(citation_precisions) if citation_precisions else 0.0,
                "num_evaluated": len(citation_recalls)
            },
            "answer_choice": {
                "accuracy": np.mean(answer_accuracies) if answer_accuracies else 0.0,
                "std": np.std(answer_accuracies) if answer_accuracies else 0.0,
                "num_evaluated": len(answer_accuracies)
            },
            "detailed_results": detailed_results,
            "summary": {
                "total_files": len(result_files),
                "successfully_processed": len(detailed_results)
            }
        }
        
        return overall_results

    def print_results(self, results):
        """
        Print evaluation results in a nice format
        """
        print("\n" + "=" * 60)
        print("MEDCITE EVALUATION RESULTS")
        print("=" * 60)
        
        # Citation Results
        citation = results['citation']
        print(f"\nCITATION QUALITY:")
        print(f"   Recall:    {citation['recall']:.4f}")
        print(f"   Precision: {citation['precision']:.4f}")
        print(f"   Evaluated: {citation['num_evaluated']} files")
        
        # Answer Choice Results  
        answer = results['answer_choice']
        print(f"\nANSWER CHOICE ACCURACY:")
        print(f"   Accuracy:  {answer['accuracy']:.4f}")
        print(f"   Std Dev:   {answer['std']:.4f}")
        print(f"   Evaluated: {answer['num_evaluated']} files")
        
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Evaluate MedCite results")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing result JSON files")
    parser.add_argument("--dataset", type=str, default="bioasq",
                       help="Dataset name for ground truth (default: bioasq)")
    parser.add_argument("--eval_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3",
                       help="Model for evaluation")
    parser.add_argument("--device", type=int, default=0,
                       help="GPU device number")
    parser.add_argument("--start_idx", type=int, default=None,
                       help="Start file index (optional)")
    parser.add_argument("--end_idx", type=int, default=None,
                       help="End file index (optional)")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed evaluation information")
    parser.add_argument("--output", type=str, default=None,
                       help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = MedCiteEvaluator(args.eval_model, args.device)
    
    # Set file range
    file_range = None
    if args.start_idx is not None and args.end_idx is not None:
        file_range = (args.start_idx, args.end_idx)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(
        args.results_dir, 
        args.dataset, 
        file_range, 
        args.verbose
    )
    
    # Print results
    evaluator.print_results(results)
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")

if __name__ == "__main__":
    main()