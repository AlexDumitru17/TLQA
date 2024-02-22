import json
import pandas as pd
import re

file_names = {"ArchivalQATime_train.csv", "ArchivalQATime_test.csv", "ArchivalQATime_val.csv"}

def extract_entities_and_rephrase(question):
    # Pattern to capture entities and the rest of the question
    pattern = re.compile(r"Along with (.*?)(?:, | and )?(who|where|when|what|which|why|how) (.*)", re.IGNORECASE)

    match = pattern.search(question)
    if match:
        entities = match.group(1)
        question_start = match.group(2)
        rest_of_question = match.group(3)

        # Split entities by 'and' and ','
        split_entities = re.split(r'(?:, | and )', entities)
        cleaned_entities = [entity.strip() for entity in split_entities]

        # Rephrase question
        rephrased_question = f"{question_start.capitalize()} {rest_of_question}"

        return cleaned_entities, rephrased_question
    else:
        return None, None

along_with = {}

# Process each file
for file_name in file_names:
    df = pd.read_csv(file_name)

    for index, row in df.iterrows():
        entities, modified_question = extract_entities_and_rephrase(row['question'])
        if entities is not None:
            along_with[row['question']] = {
                'original_answer': row['answer'],
                'entities': entities,
                'rephrased_question': modified_question
            }

# Write the 'along_with' data to a file
with open("Along_with_questions.json", 'w') as json_file:
    json.dump(along_with, json_file, indent=4)

listqas = []

for question, data in along_with.items():
    # Combine original answer with extracted entities
    answers = data['entities'] + [data['original_answer']]

    # Create ListQA entry
    listqa = {
        "question": data['rephrased_question'],
        "answers": answers
    }
    listqas.append(listqa)

# Write the listqas to a file
with open("listqas.json", 'w') as json_file:
    json.dump(listqas, json_file, indent=4)
