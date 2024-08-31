import json
from llama_cpp import Llama, LlamaGrammar
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from dotenv import load_dotenv, find_dotenv
from litellm import completion 
from time import sleep

load_dotenv()
# Initialize the Llama model with the grammar file
# n_gpu_layers = -1 => offload all to gpu, split mode = 1 => use both gpus #n_ctx=0, use model context window
# llm = Llama(model_path="models/mistral-7b-instruct-v0.2.Q5_K_M.gguf", chat_format="chatml",
#              main_gpu=0, split_mode=1, verbose=False, n_gpu_layers=-1, n_ctx=0)

model_name = 'together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'
model_name = 'mistral/mistral-large-2407'
model_name = 'gpt-4o-mini-2024-07-18'
# llm = Llama(model_path="models/Meta-Llama-3-8B-Instruct-Q6_K.gguf", chat_format="chatml",
#              main_gpu=0, split_mode=1, verbose=False, n_gpu_layers=-1, n_ctx=0)
# grammar_file = LlamaGrammar.from_file('models/grammar.gbnf')

SYSTEM_MESSAGE = ''' 

You are tasked with providing detailed timeline-based answers about various political positions or titles held by individuals. Your responses must follow a specific format to ensure clarity and correctness.

Guidelines:
1. **Structure**: Each response should consist of multiple entries, each on a new line.
2. **Format**: Each entry should ONLY include the position or title followed by a colon, a space, and the years.
3. **Order**: The entries should be ordered from the earliest to the most recent.
4. **Years**: Use a short dash to represent a range of years (e.g., 2016-2021). Separate multiple year ranges with a comma and a space (e.g., 1992-1997, 2000-2008). Write only valid years (no bigger than 2024)
5. **No Extra Text**: Do not include the individual's name or any additional text in the responses.
6. **Complete Entries**: Ensure each position or title is paired with its corresponding years. Do not leave any empty strings or unmatched years.
7. **Newline**: Each entry should be followed by a newline character.
8. **End Response**: End the response by stopping generation after listing all relevant positions and years. Do not repeat any information

Example response format:
{Title or Name}: {year1}-{year2},{year3}
{Title or Name}: {Years},

'''

SYSTEM_MESSAGE_COT = '''
You are tasked with providing rationale and  answers to timeline-based queries about various subjects.
Your reply should be structured in this format: Rationale: {your rational} /n Final answer: {your answer}

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
'''

SYSTEM_MESSAGE_AUTO_COT = '''
You are tasked with providing rationale and  answers to timeline-based queries about various subjects.
Your reply should try to be as complete as possible. Think of the subject's past and contributions. Try to think step by step. Write down your thinking process. 
After you provide an answer, you will next be tasked to extract the timelines from there, which should answer the first question best. 

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

chain_of_thought_manual_examples = [
    {
        'role': 'user',
        'content': 'List all political positions  Greg Abbott held to this day.'
    },
    {
        'role': 'assistant',
        'content': '''Rationale:
        Gregg Abbott is a american politician and officeholder. He was first know to be the 
        Justice of the Supreme Court in Texas. He started in 1996 and resigned in 2001. In 2002 he got elected as the 
        50th Attorney General of Texas. In 2014 Abott announced his candidacy for governor of Texas. He won the 
        elections, and became Governor of Texas in 2015. To this day he still holds this office. Between 2019 and 
        2020 he was also the Chair of the Republican Governors Association.
        Final answer:
        "Governor of Texas": "2015-2024",
        "Chair of the Republican Governors Association": "2019-2020",
        "50th Attorney General of Texas": "2002-2015",
        "Justice of the Supreme Court of Texas": "1996-2001"
        '''
    },
    {
        'role': 'user',
        'content': 'List all teams  Bhaichung Bhutia played to this day.'
    },
    {
        'role': 'assistant',
        'content': '''Rationale: Bhaichung Bhutia is a prominent Indian footballer known for his contributions to 
        both club and national teams. He first started playing for India national under-23 football team in 1992 till 
        1996. In 1993 he joined East Bengal FC, where he played until 1995 before moving to JCT FC. In 1995 he also joined
        the India national football team, where he would play up till 2011. He returned to East Bengal FC in 1997 and had several stints with them over the following years.
        In 1999 he joined Bury F.C, where he stayed untill 2002. Between 2002 and 2003, he changed teams again and joined Mohun Bagan Super Giant
        In 2003, he joined East Bengal FC again, and was also loaned that ear to Perak. In 2005, he played for Selangor MK Land, then returned to East Bengal FC the same year.
        Between 2006 and 2009 he left East Bengal FC to join Mohun Bagan Super Giant again. In 2009 he joined East Bengal FC again for 3 years. 
        In between 2012 and 2013 he played for United Sikkim FC. Finally, he joined East Bengal FC once again for a year in 2015. 
        
        Final answer: 
        "East Bengal FC": "1993-1995, 1997-1999, 2003-2005, 2005-2006, 2009-2011, 2015",
        "India national under-23 football team": "1992-1996", "JCT FC": "1995-1997",
        "India national football team": "1995-2011",
        "Bury F.C.": "1999-2002",
        "Mohun Bagan Super Giant": "2002-2003, 2006-2009",
        "Perak (loan)": "2003",
        "Selangor MK Land": "2005",
        "United Sikkim FC": "2012-2013"
        
        '''
    }
]



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


def write_data(file_path, data):
    """Write the modified data back to a file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def zero_shot_test(test_data, model, early_stop=True, early_stop_limit=10, temperature=0.3):
    """Test the model with the provided benchmark data."""
    results = []
    counter = 0
    for item in test_data:
        print("QUESTION NR: ", counter)
        counter += 1
        if early_stop and counter == early_stop_limit:
            break
        question = item["question"]
        # Send the question to the model
        response = model.create_chat_completion(
            messages=[{
                'role': 'system',
                'content': SYSTEM_MESSAGE,
            },

                {
                    'role': 'user',
                    'content': question,
                },
            ],
            temperature=temperature,
            grammar = grammar_file,
            stream=True
        )
        for chunk in response:
            if 'content' in chunk['choices'][0]['delta']:
                print(chunk['choices'][0]['delta']['content'], end='')
        # Get the model's response
        model_answer = response['choices'][0]['message']['content']
        results.append({
            "question": question,
            "expected": item["answers"],
            "model_answer": model_answer
        })
    return results

def manual_few_shot_test(test_data, train_data, model, early_stop=True, early_stop_limit=10,
                            few_shot_number=3, temperature=0.3):
    few_shot = []
    print('RUNNING MANUAL INFERENCE')
    for i in range(0, few_shot_number):
        few_shot_user = {'role': 'user', 'content': train_data[i]['question']}
        few_shot_assistant = {
            'role': 'assistant',
            'content': '\n '.join([f'"{key}": "{value}"' for key, value in train_data[i]['answers'].items()])
        }
        few_shot.append(few_shot_user)
        few_shot.append(few_shot_assistant)
    results = []
    counter = 0
    for item in test_data:
        sleep(3)
        print("QUESTION NR: ", counter)
        counter += 1
        if early_stop and counter == early_stop_limit:
            break
        question = item["question"]
        messages = [{
            'role': 'system',
            'content': SYSTEM_MESSAGE,
        }]
        messages.extend(few_shot)
        messages.append({
            'role': 'user',
            'content': question
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

def knn_few_shot(test_data, train_data, model, early_stop=True, early_stop_limit=10,
                                few_shot_number=3, temperature=0.3):
    print('RUNNING KNN INFERENCE')
    train_questions = [item['question'] for item in train_data]
    knn_model = KNN(n_neighbors=few_shot_number)
    knn_model.fit(train_questions)
    results = []
    counter = 0
    for item in test_data:
        sleep(3)
        print('QUESTION NR: ', counter)
        counter += 1
        if early_stop and counter == early_stop_limit:
            break
        question = item['question']
        similar_questions = knn_model.get_similar_questions(question)

        # Prepare few-shot messages
        few_shot = []
        for sim_question, idx in similar_questions:
            few_shot.append({'role': 'user', 'content': sim_question})
            answers = train_data[idx]['answers']
            formatted_answers = '\n '.join([f'"{key}": "{value}"' for key, value in answers.items()])
            few_shot.append({'role': 'assistant', 'content': formatted_answers})

        # Perform the test using the few-shot examples
        messages = [{'role': 'system', 'content': SYSTEM_MESSAGE}]
        messages.extend(few_shot)
        messages.append({'role': 'user', 'content': question})
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


def manual_chain_of_thought(test_data, chain_of_thought_examples, model, early_stop=True,
                            early_stop_limit=10, temperature=0.3):
    results = []
    counter = 0
    for item in test_data:
        print("QUESTION NR: ", counter)
        counter += 1
        if early_stop and counter == early_stop_limit:
            break
        question = item["question"]
        messages = [{
            'role': 'system',
            'content': SYSTEM_MESSAGE_COT,
        }]
        messages.extend(chain_of_thought_examples)
        messages.append({
            'role': 'user',
            'content': question
        })
        response =completion(
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


def auto_chain_of_thought_with_knn(test_data, train_data, model, early_stop=True, early_stop_limit=10,
                                   few_shot_number=1, temperature=0.3):
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
        few_shot = [{'role': 'system', 'content': SYSTEM_MESSAGE_AUTO_COT}]
        for sim_question, idx in similar_questions:

            few_shot.append({'role': 'user', 'content': sim_question + ' Think of this question step by step.'})
            response = completion(
                model=model,
                messages=few_shot,
                temperature=temperature
        )
            model_response = {'role': 'assistant', 'content': response['choices'][0]['message']['content']}
            user_reply = {'role': 'user', 'content': 'Extract the timeline, including both years and answers, from your previous answer now'}
            answers = train_data[idx]['answers']
            formatted_answers = '\n '.join([f'"{key}": "{value}"' for key, value in answers.items()])
            model_final_reply = {'role': 'assistant', 'content': formatted_answers}
            few_shot.extend([model_response, user_reply, model_final_reply])
        few_shot.append({'role': 'user', 'content': question + ' Think of this question step by step.'})
        first_response = completion(
            model=model,
            messages=few_shot,
            temperature=temperature
        )
        few_shot.append({'role': 'assistant', 'content': first_response['choices'][0]['message']['content']})
        few_shot.append({'role': 'user', 'content': 'Extract the timeline, including both years and answers from your previous answer now'})
        model_final_answer = completion(
            model=model,
            messages=few_shot,
            temperature=temperature
        )
        results.append({
            "question": question,
            "expected": item["answers"],
            "model_answer": model_final_answer['choices'][0]['message']['content']
        })

    return results


def prompt_all(llm_name,test_data, train_data):
    
    manual_3_shot = manual_few_shot_test(test_data, train_data, llm, early_stop=False)
    write_data(f'results/{llm_name}/manual_3shot.json', manual_3_shot)
    knn_3_shot = knn_few_shot(test_data, train_data, llm, early_stop=False)
    write_data(f'results/{llm_name}/knn_3shot.json', knn_3_shot)
    manual_cot = manual_chain_of_thought(test_data, chain_of_thought_manual_examples, llm, early_stop=False)
    write_data(f'results/{llm_name}/manual_cot.json', manual_cot)
    auto_cot = auto_chain_of_thought_with_knn(test_data, train_data, llm, early_stop=False)
    write_data(f'results/{llm_name}/auto_cot.json', auto_cot)
    

test_data = read_data('data/splits/test_split_benchmark_v0.0.json')
train_data = read_data('data/splits/train_split_benchmark_v0.0.json')


# prompt_all("Meta-Llama-3-8B-Instruct-Q6_K", test_data, train_data)
load_dotenv()

model_name = 'together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'

# model_results = manual_few_shot_test(test_data, train_data, model_name, early_stop=False)
# write_data('results/Meta-Llama-3.1-8b/manual-3shot.json', model_results)


# model_results = knn_few_shot(test_data, train_data, model_name, early_stop=False)
# write_data('results/Meta-Llama-3.1-8b/knn-3shot.json', model_results)


# model_results = manual_chain_of_thought(test_data, chain_of_thought_manual_examples, model_name, early_stop=False)
# write_data('results/Meta-Llama-3.1-8b/manual-cot.json', model_results)


model_results = auto_chain_of_thought_with_knn(test_data, train_data, model_name, early_stop=False)
write_data('results/Meta-Llama-3.1-8b/auto-cot.json', model_results)

#### ------------ Mistral

# mistral_large = manual_few_shot_test(test_data, train_data, model_name, early_stop=False)
# write_data('results/Mistral-Large-Instruct-2407/manual-3shot.json', mistral_large)

# mistral_large = knn_few_shot(test_data, train_data, model_name, early_stop=False)
# write_data('results/Mistral-Large-Instruct-2407/knn-3shot.json', mistral_large)


### ---------------
## Llama 3.1 70b 
# llama_70b_3_1 = manual_few_shot_test(test_data, train_data, model_name, early_stop=False)
# write_data('results/Meta-Llama-3.1-70b/manual-3shot.json', llama_70b_3_1)
# llama_70b_3_1 = knn_few_shot(test_data, train_data, model_name, early_stop=False)
# write_data('results/Meta-Llama-3.1-70b/knn-3shot.json', llama_70b_3_1)
########## 

# for k in [5, 7, 10]:
#     results = manual_few_shot_test(test_data, train_data, llm, early_stop=False, few_shot_number=k)
#     write_data(f'results/mistral-instruct-7b/Manual-kshots/KNN-{k}shots.json', results)


# Change llm to mistral-instruct-7b
# # RUN manual few shot 
# manual_few_shot_results = manual_few_shot_test(test_data, train_data, llm, early_stop=False)
# write_data('results/mistral-instruct-7b/manual_3shot_v0.0.json', manual_few_shot_results)

# # RUN auto knn
# print('KNN-FEW_SHOT')
# knn_few_shot_results = knn_few_shot(test_data, train_data, llm, early_stop=False)
# write_data('results/mistral-instruct-7b/knn_3shot_v0.0.json', knn_few_shot_results)

# # Run manual COT
# print('MANUAL-COT')
# manual_cot_results = manual_chain_of_thought(test_data, chain_of_thought_manual_examples, llm, early_stop=False)
# write_data('results/mistral-instruct-7b/manual_cot_results.json', manual_cot_results)

# # Run auto COT
# print('AUTO-COT')
# auto_cot_results = auto_chain_of_thought_with_knn(test_data, train_data, llm, early_stop_limit=False)
# write_data('results/mistral-instruct-7b/auto_cot_results.json', auto_cot_results)

#######################################

# RUN manual few shot 
# manual_few_shot_results = manual_few_shot_test(test_data, train_data, llm, early_stop=False)
# write_data('results/meta-llama3-8B-Instruct-Q6_K/manual_3shot_v0.0.json', manual_few_shot_results)

# # RUN auto knn
# print('KNN-FEW_SHOT')
# knn_few_shot_results = knn_few_shot(test_data, train_data, llm, early_stop=False)
# write_data('results/meta-llama3-8B-Instruct-Q6_K/knn_3shot_v0.0.json', knn_few_shot_results)

# # Run manual COT
# print('MANUAL-COT')
# manual_cot_results = manual_chain_of_thought(test_data, chain_of_thought_manual_examples, llm, early_stop=False)
# write_data('results/meta-llama3-8B-Instruct-Q6_K/manual_cot_results.json', manual_cot_results)

# Run auto COT
# print('AUTO-COT')
# auto_cot_results = auto_chain_of_thought_with_knn(test_data, train_data, llm, early_stop_limit=False)
# write_data('results/meta-llama3-8B-Instruct-Q6_K/auto_cot_results.json', auto_cot_results)

# PLAYGROUND --------------------------
# for result in manual_few_shot_results:
#     print('------')
#     print(result['model_answer'])
# knn_few_shot_results = knn_few_shot(test_data, train_data, llm)
# zero_shot_results = zero_shot_test(test_data, llm, early_stop_limit=5)
# for entry in results:
# for result in zero_shot_results:
#     print('------')
#     print(result['model_answer'])
# results = manual_chain_of_thought(test_data, chain_of_thought_manual_examples, llm, early_stop_limit=5)
#     print('----------')
#     print(entry['model_answer'])
