import os
import re
import json
import torch
import transformers
from transformers import AutoTokenizer
import openai
from transformers import StoppingCriteria, StoppingCriteriaList
import tiktoken
import sys
sys.path.append("src")
from utils import RetrievalSystem, DocExtracter
from template import *

from config import config

openai.api_type = openai.api_type or os.getenv("OPENAI_API_TYPE") or config.get("api_type")
openai.api_version = openai.api_version or os.getenv("OPENAI_API_VERSION") or config.get("api_version")
openai.api_key = openai.api_key or os.getenv('OPENAI_API_KEY') or config["api_key"]

if openai.__version__.startswith("0"):
    openai.api_base = openai.api_base or os.getenv("OPENAI_API_BASE") or config.get("api_base")
    if openai.api_type == "azure":
        openai_client = lambda **x: openai.ChatCompletion.create(**{'engine' if k == 'model' else k: v for k, v in x.items()})["choices"][0]["message"]["content"]
    else:
        openai_client = lambda **x: openai.ChatCompletion.create(**x)["choices"][0]["message"]["content"]
else:
    if openai.api_type == "azure":
        openai.azure_endpoint = openai.azure_endpoint or os.getenv("OPENAI_ENDPOINT") or config.get("azure_endpoint")
        openai_client = lambda **x: openai.AzureOpenAI(
            api_version=openai.api_version,
            azure_endpoint=openai.azure_endpoint,
            api_key=openai.api_key,
        ).chat.completions.create(**x).choices[0].message.content
    else:
        openai_client = lambda **x: openai.OpenAI(
            api_key=openai.api_key,
        ).chat.completions.create(**x).choices[0].message.content

class MedRAG:

    def __init__(self, llm_name="OpenAI/gpt-3.5-turbo-16k", rag=True, follow_up=False, retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./corpus", cache_dir=None, corpus_cache=False, HNSW=False, citation_mode=None):
        self.llm_name = llm_name
        self.rag = rag
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        self.docExt = None
        if rag:
            self.retrieval_system = RetrievalSystem(self.retriever_name, self.corpus_name, self.db_dir, cache=corpus_cache, HNSW=HNSW)
        else:
            self.retrieval_system = None
        self.templates = {"cot_system": general_cot_system, "cot_prompt": general_cot,
                    "medrag_system": general_medrag_system, "medrag_prompt": general_medrag}
        if self.llm_name.split('/')[0].lower() == "openai":
            self.model = self.llm_name.split('/')[-1]
            if "gpt-3.5" in self.model or "gpt-35" in self.model:
                self.max_length = 16384
                self.context_length = 15000
            elif "gpt-4" in self.model:
                self.max_length = 32768
                self.context_length = 30000
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        elif "gemini" in self.llm_name.lower():
            import google.generativeai as genai
            genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
            self.model = genai.GenerativeModel(
                model_name=self.llm_name.split('/')[-1],
                generation_config={
                    "temperature": 0,
                    "max_output_tokens": 2048,
                }
            )
            if "1.5" in self.llm_name.lower():
                self.max_length = 1048576
                self.context_length = 1040384
            else:
                self.max_length = 30720
                self.context_length = 28672
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            self.max_length = 2048
            self.context_length = 1024
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir)
            if "mixtral" in llm_name.lower():
                self.tokenizer.chat_template = open('./templates/mistral-instruct.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 32768
                self.context_length = 30000
            elif "llama-2" in llm_name.lower():
                self.max_length = 4096
                self.context_length = 3072
            elif "llama-3" in llm_name.lower():
                self.max_length = 8192
                self.context_length = 7168
                if ".1" in llm_name or ".2" in llm_name:
                    self.max_length = 131072
                    self.context_length = 128000
            elif "meditron-70b" in llm_name.lower():
                self.tokenizer.chat_template = open('./templates/meditron.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 4096
                self.context_length = 3072
                self.templates["cot_prompt"] = meditron_cot
                self.templates["medrag_prompt"] = meditron_medrag
            elif "pmc_llama" in llm_name.lower():
                self.tokenizer.chat_template = open('./templates/pmc_llama.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 2048
                self.context_length = 1024
            self.model = transformers.pipeline(
                "text-generation",
                model=self.llm_name,
                # torch_dtype=torch.float16,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                model_kwargs={"cache_dir":self.cache_dir},
            )
        
        self.follow_up = follow_up
        if self.rag and self.follow_up:
            self.answer = self.i_medrag_answer
            self.templates["medrag_system"] = simple_medrag_system
            self.templates["medrag_prompt"] = simple_medrag_prompt
            self.templates["i_medrag_system"] = i_medrag_system
            self.templates["follow_up_ask"] = follow_up_instruction_ask
            self.templates["follow_up_answer"] = follow_up_instruction_answer
        elif citation_mode is not None:
            self.answer = self.medcite_answer
            self.templates["medcite_pre_prompt"] = medcite_pre_prompt
            self.templates["post_system"] = medcite_post_system
            self.templates["post_prompt"] = medcite_post_prompt
            self.citation_mode = citation_mode
        else:
            self.answer = self.medrag_answer

    def custom_stop(self, stop_str, input_len=0):
        stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria(stop_str, self.tokenizer, input_len)])
        return stopping_criteria

    def generate(self, messages, **kwargs):
        '''
        generate response given messages
        '''
        if "openai" in self.llm_name.lower():
            ans = openai_client(
                model=self.model,
                messages=messages,
                temperature=0.0,
                **kwargs
            )
        elif "gemini" in self.llm_name.lower():
            response = self.model.generate_content(messages[0]["content"] + '\n\n' + messages[1]["content"], **kwargs)
            ans = response.candidates[0].content.parts[0].text
        else:
            stopping_criteria = None
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if "meditron" in self.llm_name.lower():
                # stopping_criteria = custom_stop(["###", "User:", "\n\n\n"], self.tokenizer, input_len=len(self.tokenizer.encode(prompt_cot, add_special_tokens=True)))
                stopping_criteria = self.custom_stop(["###", "User:", "\n\n\n"], input_len=len(self.tokenizer.encode(prompt, add_special_tokens=True)))
            if "llama-3" in self.llm_name.lower():
                response = self.model(
                    prompt,
                    do_sample=False,
                    eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_length=self.max_length,
                    truncation=True,
                    stopping_criteria=stopping_criteria,
                    **kwargs
                )
            else:
                response = self.model(
                    prompt,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_length=self.max_length,
                    truncation=True,
                    stopping_criteria=stopping_criteria,
                    **kwargs
                )
            # ans = response[0]["generated_text"]
            ans = response[0]["generated_text"][len(prompt):]
        return ans

    def medrag_answer(self, question, options=None, k=32, rrf_k=100, save_dir = None, snippets=None, snippets_ids=None, **kwargs):
        '''
        question (str): question to be answered
        options (Dict[str, str]): options to be chosen from
        k (int): number of snippets to retrieve
        rrf_k (int): parameter for Reciprocal Rank Fusion
        save_dir (str): directory to save the results
        snippets (List[Dict]): list of snippets to be used
        snippets_ids (List[Dict]): list of snippet ids to be used
        '''

        if options is not None:
            options = '\n'.join([key+". "+options[key] for key in sorted(options.keys())])
        else:
            options = ''

        # retrieve relevant snippets
        if self.rag:
            if snippets is not None:
                retrieved_snippets = snippets[:k]
                scores = []
            elif snippets_ids is not None:
                if self.docExt is None:
                    self.docExt = DocExtracter(db_dir=self.db_dir, cache=True, corpus_name=self.corpus_name)
                retrieved_snippets = self.docExt.extract(snippets_ids[:k])
                scores = []
            else:
                assert self.retrieval_system is not None
                retrieved_snippets, scores = self.retrieval_system.retrieve(question, k=k, rrf_k=rrf_k)

            contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx, retrieved_snippets[idx]["title"], retrieved_snippets[idx]["content"]) for idx in range(len(retrieved_snippets))]
            if len(contexts) == 0:
                contexts = [""]
            if "openai" in self.llm_name.lower():
                contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts))[:self.context_length])]
            elif "gemini" in self.llm_name.lower():
                contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts))[:self.context_length])]
            else:
                contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts), add_special_tokens=False)[:self.context_length])]
        else:
            retrieved_snippets = []
            scores = []
            contexts = []

        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # generate answers
        answers = []
        if not self.rag:
            prompt_cot = self.templates["cot_prompt"].render(question=question, options=options)
            messages = [
                {"role": "system", "content": self.templates["cot_system"]},
                {"role": "user", "content": prompt_cot}
            ]
            ans = self.generate(messages, **kwargs)
            answers.append(re.sub("\s+", " ", ans))
        else:
            for context in contexts:
                prompt_medrag = self.templates["medrag_prompt"].render(context=context, question=question, options=options)
                messages=[
                        {"role": "system", "content": self.templates["medrag_system"]},
                        {"role": "user", "content": prompt_medrag}
                ]
                ans = self.generate(messages, **kwargs)
                answers.append(re.sub("\s+", " ", ans))
        
        if save_dir is not None:
            with open(os.path.join(save_dir, "snippets.json"), 'w') as f:
                json.dump(retrieved_snippets, f, indent=4)
            with open(os.path.join(save_dir, "response.json"), 'w') as f:
                json.dump(answers, f, indent=4)
        
        return answers[0] if len(answers)==1 else answers, retrieved_snippets, scores

    def i_medrag_answer(self, question, options=None, k=32, rrf_k=100, save_path = None, n_rounds=4, n_queries=3, qa_cache_path=None, **kwargs):
        if options is not None:
            options = '\n'.join([key+". "+options[key] for key in sorted(options.keys())])
        else:
            options = ''
        QUESTION_PROMPT = f"Here is the question:\n{question}\n\n{options}"

        context = ""
        qa_cache = []
        if qa_cache_path is not None and os.path.exists(qa_cache_path):
            qa_cache = eval(open(qa_cache_path, 'r').read())[:n_rounds]
            if len(qa_cache) > 0:
                context = qa_cache[-1]
            n_rounds = n_rounds - len(qa_cache)
        last_context = None

        # Run in loop
        max_iterations = n_rounds + 3
        saved_messages = [{"role": "system", "content": self.templates["i_medrag_system"]}]

        for i in range(max_iterations):
            if i < n_rounds:
                if context == "":
                    messages = [
                        {
                            "role": "system",
                            "content": self.templates["i_medrag_system"],
                        },
                        {
                            "role": "user",
                            "content": f"{QUESTION_PROMPT}\n\n{self.templates['follow_up_ask'].format(n_queries)}",
                        },
                    ]
                else:                
                    messages = [
                        {
                            "role": "system",
                            "content": self.templates["i_medrag_system"],
                        },
                        {
                            "role": "user",
                            "content": f"{context}\n\n{QUESTION_PROMPT}\n\n{self.templates['follow_up_ask'].format(n_queries)}",
                        },
                    ]
            elif context != last_context:
                messages = [
                    {
                        "role": "system",
                        "content": self.templates["i_medrag_system"],
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\n{QUESTION_PROMPT}\n\n{self.templates['follow_up_answer']}",
                    },
                ]
            elif len(messages) == 1:
                messages = [
                    {
                        "role": "system",
                        "content": self.templates["i_medrag_system"],
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\n{QUESTION_PROMPT}\n\n{self.templates['follow_up_answer']}",
                    },
                ]
            saved_messages.append(messages[-1])
            if save_path:
                with open(save_path, 'w') as f:
                    json.dump([p if type(p) == dict else p.model_dump() for p in saved_messages], f, indent=4)
            last_context = context
            last_content = self.generate(messages, **kwargs)
            response_message = {"role": "assistant", "content": last_content}
            saved_messages.append(response_message)
            if save_path:
                with open(save_path, 'w') as f:
                    json.dump([p if type(p) == dict else p.model_dump() for p in saved_messages], f, indent=4)       
            if i >= n_rounds and ("## Answer" in last_content or "answer is" in last_content.lower()):
                messages.append(response_message)
                messages.append(
                    {
                        "role": "user",
                        "content": "Output the answer in JSON: {'answer': your_answer (A/B/C/D)}" if options else "Output the answer in JSON: {'answer': your_answer}",
                    }
                )
                saved_messages.append(messages[-1])
                answer_content = self.generate(messages, **kwargs)
                answer_message = {"role": "assistant", "content": answer_content}
                messages.append(answer_message)
                saved_messages.append(messages[-1])
                if save_path:
                    with open(save_path, 'w') as f:
                        json.dump([p if type(p) == dict else p.model_dump() for p in saved_messages], f, indent=4)
                return messages[-1]["content"], messages
            elif "## Queries" in last_content:
                messages = messages[:-1]
                if last_content.split("## Queries")[-1].strip() == "":
                    print("Empty queries. Continue with next iteration.")
                    continue
                try:
                    action_str = self.generate([
                        {
                            "role": "user",
                            "content": f"Parse the following passage and extract the queries as a list: {last_content}.\n\nPresent the queries as they are. DO NOT merge or break down queries. Output the list of queries in JSON format: {{\"output\": [\"query 1\", ..., \"query N\"]}}",
                        }
                    ], **kwargs)
                    action_str = re.search(r"output\": (\[.*\])", action_str, re.DOTALL).group(1)
                    action_list = [re.sub(r'^\d+\.\s*', '', s.strip()) for s in eval(action_str)]
                except Exception as E:
                    print("Error parsing action list. Continue with next iteration.")
                    error_class = E.__class__.__name__
                    error = f"{error_class}: {str(E)}"
                    print(error)
                    if save_path:
                        with open(save_path + ".error", 'a') as f:
                            f.write(f"{error}\n")                
                    continue
                for question in action_list:
                    if question.strip() == "":
                        continue
                    try:
                        rag_result = self.medrag_answer(question, k=k, rrf_k=rrf_k, **kwargs)[0]
                        context += f"\n\nQuery: {question}\nAnswer: {rag_result}"
                        context = context.strip()
                    except Exception as E:
                        error_class = E.__class__.__name__
                        error = f"{error_class}: {str(E)}"
                        print(error)
                        if save_path:
                            with open(save_path + ".error", 'a') as f:
                                f.write(f"{error}\n")
                qa_cache.append(context)
                if qa_cache_path:
                    with open(qa_cache_path, 'w') as f:
                        json.dump(qa_cache, f, indent=4)
            else:
                messages.append(response_message)
                print("No queries or answer. Continue with next iteration.")
                continue
        return messages[-1]["content"], messages

    def add_citation(self, snippets=[], scores=[], k=3, rrf_k=100, citation_rerank=False, statement_pairs=None):
        """
        Add citations to statements based on retrieved documents.
        
        Args:
            question: The original question (unused but kept for API compatibility)
            answer: Raw answer text to process (unused but kept for API compatibility)
            snippets: Pre-retrieved document snippets. If provided, these will be used instead of retrieving new ones
            snippets_ids: IDs of snippets to retrieve (unused but kept for API compatibility) 
            scores: Relevance scores for snippets
            k: Number of top documents to retrieve per statement
            rrf_k: Reciprocal rank fusion parameter
            citation_rerank: Whether to use LLM reranking
            statement_pairs: Pre-processed statement pairs (citations, statement text)
            
        Returns:
            Tuple containing:
            - Post-processed statement pairs with citations
            - Retrieved/provided snippets
            - Relevance scores
        """
        # Initialize output collections
        post_pairs = []
        post_snippets = snippets if len(snippets) > 0 else []
        post_scores = scores if len(snippets) > 0 else []
        
        # Format contexts from provided snippets or retrieve new ones
        if len(snippets) > 0:
            # Format provided snippets into contexts
            raw_contexts = [f"Document [{idx+1}] (Title: {snippet['title']}) {snippet['content']}" 
                          for idx, snippet in enumerate(snippets)]
            if "openai" in self.llm_name.lower():
                contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(raw_contexts))[:self.context_length])]
            else:
                contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(raw_contexts), add_special_tokens=False)[:self.context_length])]
        else:
            # Retrieve and format new snippets for each statement
            statement_docs = []
            for _, statement in statement_pairs:
                assert self.retrieval_system is not None
                retrieved_snippets, new_scores = self.retrieval_system.retrieve(statement, k=k, rrf_k=rrf_k)
                
                # Format retrieved snippets into contexts
                docs = [f"Document [{idx+1}] (Title: {snippet['title']}) {snippet['content']}" 
                       for idx, snippet in enumerate(retrieved_snippets)]
                if "openai" in self.llm_name.lower():
                    docs = [self.tokenizer.decode(self.tokenizer.encode("\n".join(docs))[:self.context_length])]
                else:
                    docs = [self.tokenizer.decode(self.tokenizer.encode("\n".join(docs), add_special_tokens=False)[:self.context_length])]
                    
                statement_docs.append(docs)
                post_snippets.extend(retrieved_snippets)
                post_scores.extend(new_scores)

        # Process each statement pair
        for idx, (_, statement) in enumerate(statement_pairs):
            if citation_rerank:
                # Get document contexts for current statement
                docs = contexts if len(snippets) > 0 else statement_docs[idx]
                
                # Get reranked document IDs using LLM
                prompt_post = self.templates["post_prompt"].render(context=docs, statement=statement)
                messages = [
                    {"role": "system", "content": self.templates["post_system"]},
                    {"role": "user", "content": prompt_post}
                ]
                ans = self.generate(messages)
                doc_ids = re.findall(r'\d+', ans)
            else:
                # Use sequential document IDs
                doc_ids = [str(i) for i in range(1, k+1)]
            
            # Filter doc IDs based on available snippets
            valid_doc_ids = [
                doc_id 
                for doc_id in doc_ids
                if 1 <= int(doc_id) <= len(post_snippets)
            ]
            
            post_pairs.append([valid_doc_ids, statement])
            
        return post_pairs, post_snippets, post_scores

    def medcite_answer(self, question, options=None, k1=32, k2=3, rrf_k=100, save_dir=None, snippets=None, snippets_ids=None, citation_rerank=False, **kwargs):
        """
        Enhanced version of medrag_answer that includes citation processing.
        
        Args:
            Same as medrag_answer, plus:
            citation_mode: Controls how citations are handled. Options are:
                - None: No citations (regular medrag_answer)
                - "pre_only": Only in-answer generation citations
                - "post_only": Only post-answer citations (default)
                - "both": Both in-answer and post-answer citations
            
        Returns:
            Tuple containing:
            - Dictionary with processed answer and citations
            - Retrieved snippets
            - Relevance scores
        """
        # Get initial answer using appropriate prompt based on citation mode
        if self.citation_mode != "post_only":
            original_prompt = self.templates["medrag_prompt"]
            self.templates["medrag_prompt"] = self.templates["medcite_pre_prompt"]
            
            clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['k', 'rrf_k', 'snippets', 'snippets_ids']}
            answer, snippets, scores = self.medrag_answer(  
                question=question, 
                options=options, 
                k=k1, 
                rrf_k=rrf_k,
                snippets=snippets,
                snippets_ids=snippets_ids,
                **clean_kwargs
            )
            
        else:
            clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['k', 'rrf_k', 'snippets', 'snippets_ids']}
            answer, snippets, scores = self.medrag_answer(
                question=question, 
                options=options, 
                k=k1, 
                rrf_k=rrf_k,
                snippets=snippets,
                snippets_ids=snippets_ids,
                **clean_kwargs
            )

        # Extract answer components
        prefix = 'assistant<|end_header_id|> '
        parts = answer.rsplit(prefix, 1)
        output = parts[1] if len(parts) > 1 else answer
        step_by_step_thinking = re.search(r'"step_by_step_thinking":\s*"([^"]+)"', output)
        answer_choice = re.search(r'"answer_choice":\s*"([^"]+)"', answer)
        answer_text = step_by_step_thinking.group(1) if step_by_step_thinking else output
        answer_choice = answer_choice.group(1) if answer_choice else None

        # Initialize result structure
        result = {
            "generated_output": output,
            "answer_text": answer_text,
            "answer_choice": answer_choice
        }

        # Process citations in the answer text
        num_snippets = len(snippets)
        citation_pattern = re.compile(r'\[(\d+)\]')
        statement_pairs = []
        
        # Split text into statements and process citations
        for statement in re.split(r'(?<=[.!?])\s+', answer_text):
            matches = citation_pattern.findall(statement)
            if matches:
                citations = [int(match) for match in matches if 0 < int(match) <= num_snippets]
                # Keep original statement with citations for now, will clean later
                statement_pairs.append((citations, statement))
            else:
                statement_pairs.append(([], statement))

        # Get cited documents
        cited_docs = {}
        for doc_ids, _ in statement_pairs:
            for doc_id in doc_ids:
                if 0 < doc_id <= num_snippets and str(doc_id) not in cited_docs:
                    doc = snippets[doc_id-1]
                    cited_docs[str(doc_id)] = {
                        'title': doc['title'],
                        'content': doc['content'],
                        'pmid': doc['PMID']
                    }

        # Add citations to result
        result["cited_docs"] = cited_docs

        # Return if only pre-citation processing needed
        if self.citation_mode == "pre_only":
            pre_result = {
                "answer": answer_text,
                "answer_choice": answer_choice,
                "cited_docs": cited_docs
            }
            return pre_result, snippets, scores

        # Get additional citations through post-processing
        post_pairs, post_snippets, post_scores = self.add_citation( 
            snippets=[],
            scores=scores,
            k=k2,
            citation_rerank=citation_rerank,
            statement_pairs=statement_pairs
        )

        # Merge citations using PMID mapping
        pmid_map = {}
        all_cited_docs = {}
        doc_counter = 1
        
        # First, collect all unique documents based on PMID
        # From original snippets
        for doc_ids, _ in statement_pairs:
            for doc_id in doc_ids:
                if 0 < doc_id <= num_snippets:
                    doc = snippets[doc_id-1]
                    pmid = doc['PMID']
                    if pmid not in pmid_map:
                        pmid_map[pmid] = str(doc_counter)
                        all_cited_docs[str(doc_counter)] = {
                            'title': doc['title'],
                            'content': doc['content'],
                            'pmid': pmid
                        }
                        doc_counter += 1
        
        # From additional post-processing snippets
        for i, (doc_ids, statement) in enumerate(post_pairs):
            for doc_id in doc_ids:
                if 1 <= int(doc_id) <= len(post_snippets):
                    doc = post_snippets[int(doc_id)-1]
                    pmid = doc["PMID"]
                    if pmid not in pmid_map:
                        pmid_map[pmid] = str(doc_counter)
                        all_cited_docs[str(doc_counter)] = {
                            'title': doc['title'],
                            'content': doc['content'],
                            'pmid': pmid
                        }
                        doc_counter += 1
        
        # Now rebuild statement pairs with correct mappings
        final_statement_pairs = []
        
        # Process original statement pairs
        for doc_ids, statement in statement_pairs:
            new_doc_ids = []
            for doc_id in doc_ids:
                if 0 < doc_id <= num_snippets:
                    doc = snippets[doc_id-1]
                    pmid = doc['PMID']
                    new_doc_ids.append(pmid_map[pmid])
            final_statement_pairs.append((new_doc_ids, statement))
        
        # Process additional citations from post-processing
        for i, (doc_ids, statement) in enumerate(post_pairs):
            additional_docs = []
            for doc_id in doc_ids:
                if 1 <= int(doc_id) <= len(post_snippets):
                    doc = post_snippets[int(doc_id)-1]
                    pmid = doc["PMID"]
                    mapped_id = pmid_map[pmid]
                    if mapped_id not in final_statement_pairs[i][0]:
                        additional_docs.append(mapped_id)
            
            # Add additional documents to corresponding statement
            if i < len(final_statement_pairs) and additional_docs:
                final_statement_pairs[i] = (final_statement_pairs[i][0] + additional_docs, final_statement_pairs[i][1])

        # Construct final answer with citations
        final_answer = ''
        citation_pattern = re.compile(r'\[(\d+)\]')
        
        for doc_ids, statement in final_statement_pairs:
            # Strategy: Remove all citations and clean up spacing/punctuation
            cleaned_statement = statement
            
            # Remove all citations
            cleaned_statement = citation_pattern.sub('', cleaned_statement)
            
            # Clean up spacing issues
            # Handle cases like "Document  finds" -> "Document finds"
            cleaned_statement = re.sub(r'\s+', ' ', cleaned_statement)
            
            # Clean up punctuation issues
            # Handle cases like " ." -> "."
            cleaned_statement = re.sub(r'\s+([.!?])', r'\1', cleaned_statement)
            
            # Handle cases like ".  " -> ". "
            cleaned_statement = re.sub(r'([.!?])\s+', r'\1 ', cleaned_statement)
            
            # Trim and ensure single space at end for concatenation
            cleaned_statement = cleaned_statement.strip()
            
            # Add cleaned statement
            final_answer += cleaned_statement
            
            # Add new citations at the end of the statement (before final punctuation if present)
            if doc_ids:
                # Check if statement ends with punctuation
                if cleaned_statement and cleaned_statement[-1] in '.!?':
                    # Remove last punctuation, add citations, then add punctuation back
                    final_answer = final_answer[:-1]  # Remove last char (punctuation)
                    for doc_id in sorted(set(doc_ids)):
                        final_answer += "[" + str(doc_id) + "]"
                    final_answer += cleaned_statement[-1]  # Add punctuation back
                else:
                    # No ending punctuation, just add citations
                    for doc_id in sorted(set(doc_ids)):
                        final_answer += "[" + str(doc_id) + "]"
            
            final_answer += " "

        # Update result with final information
        final_result = {
            "answer": final_answer.strip(),
            "answer_choice": answer_choice,
            "cited_docs": all_cited_docs
        }

        # Save results if directory provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "result.json"), 'w') as f:
                json.dump(final_result, f, indent=4)

        return final_result, snippets, scores

class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_words, tokenizer, input_len=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops_words = stop_words
        self.input_len = input_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        tokens = self.tokenizer.decode(input_ids[0][self.input_len:])
        return any(stop in tokens for stop in self.stops_words)