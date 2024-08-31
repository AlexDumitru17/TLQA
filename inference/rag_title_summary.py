import json
from typing import Iterator, Union, Optional, List
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_community.document_loaders.helpers import detect_file_encodings
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_cpp import Llama, LlamaGrammar
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from elasticsearch import Elasticsearch
import logging
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
You will receive additional context from the user regarding the query. Use the context as additional sources to answer the query.

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
# embeddings_multiqa = HuggingFaceEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")
# vector_store = Chroma(persist_directory="./chroma_title_summary_multi_qa_embeddings", embedding_function=embeddings_multiqa)
print('vector store loaded')
# llm = Llama(model_path="models/mistral-7b-instruct-v0.2.Q5_K_M.gguf", chat_format="chatml",
#              main_gpu=0, split_mode=1, verbose=False, n_gpu_layers=-1, n_ctx=0)
# llm = Llama(model_path="models/Meta-Llama-3-8B-Instruct-Q6_K.gguf", chat_format="chatml",
#              main_gpu=0, split_mode=1, verbose=False, n_gpu_layers=-1, n_ctx=0)

mapping = load_mapping('title_to_infobox.json')
print('Mapping loaded')

SYSTEM_MESSAGE_AUTO_COT = '''
You are tasked with providing rationale and  answers to timeline-based queries about various subjects.
Your reply should try to be as complete as possible. Think of the subject's past and contributions. Try to think step by step. Write down your thinking process. 
After you provide an answer, you will next be tasked to extract the timelines from there, which should answer the first question best. 
You will receive additional context when asked questions. Use the context to answer the query. 
The timeline should be structured like this:

Each answer should be structured as a list, with different entries for each subject or title. Each entry must include the years corresponding to the following the title or subject.
Please adhere to the following structure in your responses:
- Write the title or name first, exactly as it would appear in a formal context such as Wikipedia.
- Follow the title or name with a colon and then the years.
- Represent a range of years by using a short dash. 
- Separate multiple years or ranges with a comma and a space.
- End each entry with a comma, except for the last entry.
- If the current year is relevant to the query, use "2024" instead of "present".
- Each answer should be on a new line, with no additional characters or explanations.
- Ensure the completeness and correctness of your answers. Try to be as complete as possible. 
- If provided examples, follow the examples structure and reasoning style. 

Example response format:
{Title or Name}: {year1}-{year2},{year3}
{Title or Name}: {Years},

Follow given examples from previous conversation. Do not stray from the structure of those examples. Always write both the answer and the year.
'''


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

def RAG_manual_few_shot(test_data, train_data, model, early_stop=True, early_stop_limit=10,
                            few_shot_number=3, temperature=0.3, top_k=3, MMR=False, sparse_retrieval=False):
    few_shot = []
    for i in range(0, few_shot_number):
        context = ''
        if sparse_retrieval:
            context = get_documents_bm25(train_data[i]['question'], "index_title_summary", top_k)
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
        context = ''
        if sparse_retrieval:
            context = get_documents_bm25(question, "index_title_summary", top_k)
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
        response = model.create_chat_completion(
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
                                few_shot_number=3, temperature=0.3, top_k=3, MMR=False, sparse_retrieval=False):
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
            context = ''
            if sparse_retrieval:
                context = get_documents_bm25(sim_question, "index_title_summary", top_k)
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
        context = ''
        if sparse_retrieval:
            context = get_documents_bm25(question, "index_title_summary", top_k)
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


def RAG_auto_chain_of_thought_with_knn(test_data, train_data, model, early_stop=True, early_stop_limit=10,
                                   few_shot_number=3, temperature=0.3, top_k=3, MMR=False):
    train_questions = [item['question'] for item in train_data]
    knn_model = KNN(n_neighbors=few_shot_number)
    knn_model.fit(train_questions)
    results = []
    counter = 0
    for item in test_data:
        logger.info('QUESTION NR: ', counter)
        counter += 1
        if early_stop and counter == early_stop_limit:
            break
        question = item['question']
        similar_questions = knn_model.get_similar_questions(question)
        few_shot = [{'role': 'system', 'content': SYSTEM_MESSAGE_AUTO_COT}]
        for sim_question, idx in similar_questions:
            if MMR:
                docs = vector_store.max_marginal_relevance_search(sim_question, top_k)
            else:
                docs = vector_store.similarity_search(sim_question, top_k)
            context = ''
            for doc in docs:
                title = doc.page_content.split('\n')[0].strip()
                context += mapping[title]    
            few_shot.append({'role': 'user', 'content': 'Context: ' + context + 
                            'Question: ' + sim_question + ' Think of this question step by step.'})
            response = model.create_chat_completion(
                messages=few_shot,
                temperature=temperature
            )
            model_response = {'role': 'assistant', 'content': response['choices'][0]['message']['content']}
            user_reply = {'role': 'user', 'content': 'Extract the timeline, including both years and answers, from your previous answer now'}
            answers = train_data[idx]['answers']
            formatted_answers = '\n '.join([f'"{key}": "{value}"' for key, value in answers.items()])
            model_final_reply = {'role': 'assistant', 'content': formatted_answers}
            few_shot.extend([model_response, user_reply, model_final_reply])
        if MMR:
            docs = vector_store.max_marginal_relevance_search(question, top_k)
        else:
            docs = vector_store.similarity_search(question, top_k)
        for doc in docs:
                title = doc.page_content.split('\n')[0].strip()
                context += mapping[title]
        few_shot.append({'role': 'user', 'content': 'Context: ' + context + 
                            'Question: '+ question + ' Think of this question step by step.'})
        first_response = model.create_chat_completion(
            messages=few_shot,
            temperature=temperature
        )
        few_shot.append({'role': 'assistant', 'content': first_response['choices'][0]['message']['content']})
        few_shot.append({'role': 'user', 'content': 'Extract the timeline, including both years and answers from your previous answer now'})
        model_final_answer = model.create_chat_completion(
            messages=few_shot,
            temperature=temperature
        )
        results.append({
            "question": question,
            "expected": item["answers"],
            "model_answer": model_final_answer['choices'][0]['message']['content']
        })

    return results



def create_results_for_top_k(top_k, test_data, train_data, llm, knn=True):
    
    if knn:
        for k in range(1, top_k + 1):
            # knn_few_shot_sparse = RAG_knn_few_shot(test_data, train_data, llm, early_stop=False, sparse_retrieval=True, top_k=k)
            # write_data(f'results/mistral-instruct-7b/BM25_RAG/knn_few_shot/RAG_top{k}.json', knn_few_shot_sparse)
            knn_few_shot_dense = RAG_knn_few_shot(test_data, train_data, llm, early_stop=False, sparse_retrieval=False, top_k=k)
            write_data(f'results/mistral-instruct-7b/CHROMA_RAG_MULTI_QA/knn_few_shot/RAG_top{k}.json', knn_few_shot_dense)
    else:
        for k in range(1, top_k + 1):
            # manual_few_shot_sparse = RAG_manual_few_shot(test_data, train_data, llm, early_stop=False, sparse_retrieval=True, top_k=k)
            # write_data(f'results/mistral-instruct-7b/BM25_RAG/manual_few_shot/RAG_top{k}.json', manual_few_shot_sparse)
            manual_few_shot_dense = RAG_manual_few_shot(test_data, train_data, llm, early_stop=False, sparse_retrieval=False, top_k=k)
            write_data(f'results/mistral-instruct-7b/CHROMA_RAG_MULTI_QA/manual_few_shot/RAG_top{k}.json', manual_few_shot_dense)


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

test_data = read_data('data/splits/test_split_benchmark_v0.0.json')
train_data = read_data('data/splits/train_split_benchmark_v0.0.json')

model_name = 'gpt-4o-mini-2024-07-18'

model_name = 'together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'

# create_results_for_top_k(10, test_data, train_data, llm, knn=True)

for k in [1, 3, 5, 10]:
    results_dense = RAG_knn_few_shot(test_data, train_data, model_name, early_stop=False, top_k=k, sparse_retrieval=False)
    write_data(f'results/Meta-Llama-3.1-8b/CHROMA/title-summary/RAG_top{k}.json', results_dense)

embeddings_multiqa = HuggingFaceEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")
vector_store = Chroma(persist_directory="./chroma_title_summary_multi_qa_embeddings", embedding_function=embeddings_multiqa)


for k in [3, 5, 10]:
    results_sparse = RAG_knn_few_shot(test_data, train_data, model_name, early_stop=False, top_k=k, sparse_retrieval=True)
    write_data(f'results/Meta-Llama-3.1-8b/BM25/title-summary/RAG_top{k}.json', results_sparse)
    results_dense = RAG_knn_few_shot(test_data, train_data, model_name, early_stop=False, top_k=k, sparse_retrieval=False)
    write_data(f'results/Meta-Llama-3.1-8b/multi-qa/title-summary/RAG_top{k}.json', results_dense)




# ------ SPARSE RETRIEVAL BM25
# RAG_manual_few_shot_BM25 = RAG_manual_few_shot(test_data, train_data, llm, early_stop=False, top_k=3, sparse_retrieval=True)
# write_data("results/mistral-instruct-7b/BM25_RAG/Bm25_RAG_Manual_few_shot_title_summary.json", RAG_manual_few_shot_BM25)

# RAG_KNN_few_shotBM25 = RAG_knn_few_shot(test_data, train_data, llm, early_stop=False, top_k=3, sparse_retrieval=True)
# write_data("results/mistral-instruct-7b/BM25_RAG/Bm25_RAG_KNN_few_shot_title_summary.json", RAG_KNN_few_shotBM25)

# ------- DENSE RERIEVAL
# manual_few_shot_MMR = RAG_manual_few_shot(test_data, train_data, llm, early_stop=False, top_k=3, MMR=False)
# write_data("results/mistral-instruct-7b/CHROMA_RAG/RAG_Manual_few_shot_title_summary.json", manual_few_shot_MMR)
# knn_few_shot = RAG_knn_few_shot(test_data, train_data, llm, early_stop=False, top_k=3)
# write_data("results/mistral-instruct-7b/RAG_KNN_few_shot_title_summary_MMR.json", knn_few_shot)
# context = vector_store.similarity_search("List all teams  Steven Caulker, also known as Steven Roy Caulker, played to this day.", 10)
# auto_cot = RAG_auto_chain_of_thought_with_knn(test_data, train_data, llm, early_stop=False, MMR=False)
# write_data("results/mistral-instruct-7b/RAG_auto_COT_title_summary_MMR.json", auto_cot)
# for item in context:
#     splits = item.page_content.split('\n')
#     print(mapping[splits[0]])
#     print('------------------')