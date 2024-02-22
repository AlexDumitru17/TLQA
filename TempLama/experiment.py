import json
import time

import openai
from openai import OpenAI
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

system_message = "Follow the given examples and output final answer in format [Final Answer]:"
few_shot_prompt = """
[Original Question]: List all sports teams Gianluigi Buffon played for from 2010 to 2020.
[Final Answer]: Juventus FC (2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020)
Italy national association football team (2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018)
Paris Saint-Germain (2018, 2019)

[Original Question]: List all positions David Cameron held from 2010 to 2016.
[Final Answer]: Minister for the Civil Service (2010, 2011, 2012, 2013, 2014, 2015, 2016)
First Lord of the Treasury (2010, 2011, 2012, 2013, 2014, 2015, 2016)
Leader of the Conservative Party (2010, 2011, 2012, 2013, 2014, 2015, 2016)
Leader of the Opposition (2010)
Prime Minister of the United Kingdom (2010, 2011, 2012, 2013, 2014, 2015, 2016)

[Original Question]: List all chairpersons of Norwegian Association for Women's Rights from 2010 to 2020.
[Final Answer]: Torild Skard (2010, 2011, 2012, 2013)
Margunn Bjørnholt (2013, 2014, 2015, 2016)
Marit Nybakk (2016, 2017, 2018)
Karin Maria Bruzelius (2018, 2019, 2020)
Anne Hege Grung (2020)
"""

instruction = "Follow above examples and answer the original question. Output final answer in format [Final Answer]: and then stop generation"


def read_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def split_data_by_relation(data, few_shot_example, max_per_relation=15):
    """
    Splits the data into subsets based on the relationship type.
    Each subset contains up to `max_per_relation` examples.

    Parameters:
    - data: A list of dictionaries, where each dictionary contains 'question', 'answers', and 'type'.
    - max_per_relation: The maximum number of examples to include for each relationship type.

    Returns:
    A dictionary where keys are relationship types and values are lists of data dictionaries.
    """
    # Initialize a dictionary to hold the split data
    temp_split_data = {}

    # Iterate over each item in the data list to distribute them based on relation type
    for item in data:
        relation_type = item['type']

        # Initialize the list for this relation type if it doesn't exist
        if relation_type not in temp_split_data:
            temp_split_data[relation_type] = []

        # Add the item to the list for its relation type, respecting the max_per_relation limit
        if len(temp_split_data[relation_type]) < max_per_relation and item['question'] not in few_shot_example:
            temp_split_data[relation_type].append(item)

    # Flatten the distributed items into a single list
    flattened_data = []
    for items in temp_split_data.values():
        flattened_data.extend(items)

    return flattened_data


def get_gpt3_response(data):
    question_df = {"question": [], "original_answers": [], "generated_answers": []}

    for entry in data:
        question = entry['question']
        original_ans = '\n'.join(entry['answers'])
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": system_message},
                          {"role": "user", "content": few_shot_prompt},
                          {"role": "user", "content": instruction + "\n[Original Question]:" + question}
                          ],
                temperature=0.3,
                max_tokens=1096,
                top_p=1.0,
                frequency_penalty=0.8,
                presence_penalty=0.6
            )
        except Exception as e:
            print(e)
            time.sleep(60)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": system_message},
                          {"role": "user", "content": few_shot_prompt},
                          {"role": "user", "content": instruction + "\n[Original Question]:" + question}
                          ],
                temperature=0.3,
                max_tokens=1096,
                top_p=1.0,
                frequency_penalty=0.8,
                presence_penalty=0.6
            )
        question_df['question'].append(question)
        generated_ans = response.choices[0].message.content
        question_df['original_answers'].append(original_ans)
        question_df['generated_answers'].append(generated_ans)
    result = pd.DataFrame(question_df)
    result.to_csv("gpt3_responses.csv", index=False)


data = read_data('listqas.json')
splitted_data = split_data_by_relation(data, few_shot_prompt)
get_gpt3_response(splitted_data)
