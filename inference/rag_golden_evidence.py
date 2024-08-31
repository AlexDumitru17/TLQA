import json
from typing import Iterator, Union, Optional
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.helpers import detect_file_encodings
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_cpp import Llama, LlamaGrammar
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import logging
from h11._writers import write_request
from elasticsearch import Elasticsearch
import random
from dotenv import load_dotenv
from litellm import completion 

load_dotenv()
logger = logging.getLogger(__name__)

def write_data(file_path, data):
    """Write the modified data back to a file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Load global vars
SYSTEM_MESSAGE_RAG = ''' 
You are tasked with responding to timeline-based queries about various subjects.
Each response should be structured as a list, with different entries for each subject or title. Each entry must include the years corresponding to the following the title or subject.
You will  receive additional context from the user regarding the query. Use the context as additional sources to answer the query.

Please adhere to the following structure in your responses:
- Write the title or name first, exactly as it would appear in a formal context such as Wikipedia.
- Follow the title or name with a colon and then the years.
- Represent a range of years by using a short dash. 
- Separate multiple years or ranges with a comma and a space.
- End each entry with a comma, except for the last entry.
- If the current year is relevant to the query, use "2024" instead of "present".
- Each answer should be on a new line, with no additional characters or explanations.
- Ensure the completeness and correctness of your answers.

Example response format:
{Title or Name}: {year1}-{year2},{year3}
{Title or Name}: {Years},
'''
def load_mapping(mapping_file: str):
    with open(mapping_file, 'r') as f:
        return json.load(f)

def query_infobox(mapping, title):
    return mapping.get(title, "Title not found")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory="./chroma_title_and_summary", embedding_function=embeddings)
print('vector store loaded')
# llm = Llama(model_path="models/mistral-7b-instruct-v0.2.Q5_K_M.gguf", chat_format="chatml",
#              main_gpu=0, split_mode=1, verbose=False, n_gpu_layers=-1, n_ctx=0) 
mapping = load_mapping('title_to_infobox.json')
print('Mapping loaded')             

class KNN:
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2', n_neighbors=3):
        self.model = SentenceTransformer(model_name)
        self.knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
        self.embeddings = None
        self.questions = []

    def fit(self, questions):
        """ Compute embeddings for a list of questions and fit the KNN model. """
        self.questions = questions
        self.embeddings = self.model.encode(questions)
        self.knn.fit(self.embeddings)

    def get_similar_questions(self, query):
        """ Find the top K similar questions to the query. """
        query_embedding = self.model.encode([query])
        distances, indices = self.knn.kneighbors(query_embedding)
        return [(self.questions[idx], idx) for idx in indices[0]]

def get_documents_bm25(query, index_name, k):
    es_url = "http://localhost:9200"
    es = Elasticsearch([es_url])
    context = ''
    response = es.search(
            index=index_name,
            body={
                "query": {
                    "match": {
                        "content": query
                    }
                },
                "size": k
            }
        )
    for hit in response['hits']['hits']:
            document = hit['_source']['content']
            title = document.split('\n')[0].strip()
            if title in mapping:
                context += mapping[title]    
    return context

def get_random_infoboxes(k):
    keys = list(mapping.keys())
    selected_keys = random.sample(keys, k)
    selected_infoboxes = [mapping[key] for key in selected_keys]
    concatenated_infoboxes = ''.join(selected_infoboxes)
    return concatenated_infoboxes

# Here we assume that both test data and train data have golden evidence marked in ['evidence']
def RAG_manual_few_shot(test_data, train_data, model, early_stop=True, early_stop_limit=10,
                            few_shot_number=3, temperature=0.3, top_k=3, MMR=False, sparse_retrieval=False, random=False):
    few_shot = []
    for i in range(0, few_shot_number):
        context = train_data[i]['evidence']
        if top_k > 0:
            if random:
                context += get_random_infoboxes(top_k) 
            elif sparse_retrieval:
                context += get_documents_bm25(train_data[i]['question'], "index_title_summary", top_k)
            else:
                if MMR:
                    docs = vector_store.max_marginal_relevance_search(train_data[i]['question'], top_k)
                else:
                    docs = vector_store.similarity_search(train_data[i]['question'], top_k)
                for doc in docs:
                    title = doc.page_content.split('\n')[0].strip()
                    context += mapping[title]
        few_shot_user = {'role': 'user',
                          'content': "Context: " + context + 
                                    "Question: " + train_data[i]['question'] 
                          }
        few_shot_assistant = {
            'role': 'assistant',
            'content': '\n '.join([f'"{key}": "{value}"' for key, value in train_data[i]['answers'].items()])
        }
        few_shot.append(few_shot_user)
        few_shot.append(few_shot_assistant)
    results = []
    counter = 0
    for item in test_data:
        print("QUESTION NR: ", counter)
        counter += 1
        if early_stop and counter == early_stop_limit:
            break
        question = item["question"]
        context = item["evidence"]
        if top_k > 0:
            if random:
                context += get_random_infoboxes(top_k) 
            elif sparse_retrieval:
                context += get_documents_bm25(question, "index_title_summary", top_k)
            else:
                if MMR:
                    docs = vector_store.max_marginal_relevance_search(question, top_k)
                else:
                    docs = vector_store.similarity_search(question, top_k)
                for doc in docs:
                    title = doc.page_content.split('\n')[0].strip()
                    context += mapping[title]
        messages = [{
            'role': 'system',
            'content': SYSTEM_MESSAGE_RAG,
        }]
        messages.extend(few_shot)
        messages.append({
            'role': 'user',
            'content': "Context: " + context + 
                       "Question: " + question 
                        
        })
        response = completion(
                model=model,
                messages=messages,
                temperature=temperature
            )
        model_answer = response['choices'][0]['message']['content']
        results.append({
            "question": question,
            "expected": item["answers"],
            "model_answer": model_answer
        })
    return results   

def RAG_knn_few_shot(test_data, train_data, model, early_stop=True, early_stop_limit=10,
                                few_shot_number=3, temperature=0.3, top_k=3, MMR=False, sparse_retrieval=False, random=False):
    train_questions = [item['question'] for item in train_data]
    knn_model = KNN(n_neighbors=few_shot_number)
    knn_model.fit(train_questions)
    results = []
    counter = 0
    for item in test_data:
        print('QUESTION NR: ', counter)
        counter += 1
        if early_stop and counter == early_stop_limit:
            break
        question = item['question']
        similar_questions = knn_model.get_similar_questions(question)

        # Prepare few-shot messages
        few_shot = []
        for sim_question, idx in similar_questions:
            context = train_data[idx]['evidence']
            if context is None:
                continue
            if top_k > 0:
                if random:
                    context += get_random_infoboxes(top_k) 
                elif  sparse_retrieval:
                    context += get_documents_bm25(sim_question, "index_title_summary", top_k)
                else: 
                    if MMR: 
                        docs = vector_store.max_marginal_relevance_search(sim_question, top_k)
                    else:
                        docs = vector_store.similarity_search(sim_question, top_k)
                    for doc in docs:
                        title = doc.page_content.split('\n')[0].strip()
                        context += mapping[title]
            few_shot.append({'role': 'user',
                              'content': "Context: " +  context + 
                                         "Question: " + sim_question
                            })
            answers = train_data[idx]['answers']
            formatted_answers = '\n '.join([f'"{key}": "{value}"' for key, value in answers.items()])
            few_shot.append({'role': 'assistant', 'content': formatted_answers})

        # Perform the test using the few-shot examples
        context = item['evidence']
        if top_k > 0:
            if random:
                context += get_random_infoboxes(top_k) 
            elif sparse_retrieval:
                context += get_documents_bm25(question, "index_title_summary", top_k)
            else: 
                if MMR:
                    docs = vector_store.max_marginal_relevance_search(question, top_k)
                else:
                    docs = vector_store.similarity_search(question, top_k)
                for doc in docs:
                    title = doc.page_content.split('\n')[0].strip()
                    context += mapping[title]
        messages = [{'role': 'system', 'content': SYSTEM_MESSAGE_RAG}]
        messages.extend(few_shot)
        messages.append({'role': 'user',
                          'content': "Context: " +  context + 
                                     "Question: " + question
                        })
        response = completion(
                model=model,
                messages=messages,
                temperature=temperature
            )
        model_answer = response['choices'][0]['message']['content']

        results.append({
            "question": question,
            "expected": item["answers"],
            "model_answer": model_answer,
            "subject": item['subject']
        })
    return results

test_data = read_data('data/splits/test_split_benchmark_v0.0_golden_evidence.json')
train_data = read_data('data/splits/train_split_benchmark_v0.0_golden_evidence.json')

model_name = 'gpt-4o-mini-2024-07-18'

model_name = 'together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'

manual_results = RAG_manual_few_shot(test_data, train_data, model_name, early_stop=False, top_k=0)
write_data('results/Meta-Llama-3.1-8b/RAG_GOLDEN_EVIDENCE_MANUAL_FEW_SHOT.json', manual_results)

knn_results = RAG_knn_few_shot(test_data, train_data, model_name, early_stop=False, top_k=0)
write_data('results/Meta-Llama-3.1-8b/RAG_GOLDEN_EVIDENCE_KNN_FEW_SHOT.json', knn_results)


# for k in range(1, 11):
    
#     manual_results_sparse_retrieval = RAG_knn_few_shot(test_data, train_data, llm, early_stop=False, top_k=k, sparse_retrieval=True)
#     write_data(f'results/mistral-instruct-7b/GOLDEN/knn-few-shot/Sparse_Retrieval_knn_3_shot_top{k}.json', manual_results_sparse_retrieval)
#     random_results = RAG_knn_few_shot(test_data, train_data, llm, early_stop=False, top_k=k, random=True)
#     write_data(f'results/mistral-instruct-7b/GOLDEN/knn-few-shot/Random_knn_3_shot_top{k}.json', random_results)
#     manual_results_dense_retrieval = RAG_knn_few_shot(test_data, train_data, llm, early_stop=False, top_k=k, sparse_retrieval=False)
#     write_data(f'results/mistral-instruct-7b/GOLDEN/knn-few-shot/Dense_Retrieval_knn_3_shot_top{k}.json', manual_results_dense_retrieval)
# # knn_results = RAG_knn_few_shot(test_data, train_data, llm, early_stop=False, top_k=0)
# write_data('results/mistral-instruct-7b/RAG_GOLDEN_EVIDENCE_KNN_FEW_SHOT.json', knn_results)