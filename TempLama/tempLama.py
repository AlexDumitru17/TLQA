import json

grouped_data = {}

relation_question_mapping = {
    "P54": "List all sports teams {subject} played for from {start_year} to {end_year}.",
    "P39": "List all positions {subject} held from {start_year} to {end_year}.",
    "P108": "List all employers {subject} worked for from {start_year} to {end_year}.",
    "P102": "List all political parties {subject} was a member of from {start_year} to {end_year}.",
    "P286": "List all coaches of {subject} from {start_year} to {end_year}",
    "P69": "List all educational institutions {subject} attended from {start_year} to {end_year}.",
    "P488": "List all chairpersons of {subject} from {start_year} to {end_year}",
    "P6": "List all heads of the government of {subject} from {start_year} to {end_year} ",
    "P127": "List all entities that owned {subject} from {start_year} to {end_year}."
}

file_names = {"templama_train.json", "templama_test.json", "templama_val.json"}


def extract_subject(query):
    split_phrases = [' plays for ', ' works for ', ' is a member of ', ' attended ',
                     ' is the head coach of ', ' is the chair of ',
                     ' is the head of the government of ', ' is owned by ',
                     'holds the position of']

    # Check if subject is a placeholder like "_X_"
    if query.strip().startswith("_X_"):
        for phrase in split_phrases:
            if phrase in query:
                return query.split(phrase)[1].strip()  # Return the second part
    else:
        for phrase in split_phrases:
            if phrase in query:
                return query.split(phrase)[0].strip()  # Return the first part

    return ""  # Return an empty string if no pattern matches


for file_name in file_names:
    with open(file_name, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            query = json_obj["query"]
            subject = extract_subject(query)
            if query not in grouped_data:
                grouped_data[query] = {
                    "query": query,
                    "subject": subject,
                    "answers": {},  # Changed from list to dictionary
                    "dates": [],
                    "relations": ''
                }

            year = json_obj["date"]
            answers = json_obj["answer"]
            for answer in answers:
                answer_name = answer['name']
                if answer_name not in grouped_data[query]["answers"]:
                    grouped_data[query]["answers"][answer_name] = []  # Initialize list for years
                grouped_data[query]["answers"][answer_name].append(year)  # Append year

            grouped_data[query]["dates"].append(year)
            if grouped_data[query]["relations"] == '':
                grouped_data[query]["relations"] = json_obj["relation"]

# Generating listQAs from the combined dataset
listqas = []
for query, data in grouped_data.items():
    relation = data["relations"]
    if relation in relation_question_mapping:
        start_year = min(data["dates"])
        end_year = max(data["dates"])
        question = relation_question_mapping[relation].format(subject=data["subject"], start_year=start_year,
                                                              end_year=end_year)

        # Formatting answers with years
        answers_with_years = [f"{answer} ({', '.join(years)})" for answer, years in data["answers"].items()]

        listqa = {"question": question, "answers": answers_with_years, "type": relation }
        listqas.append(listqa)

# Writing the listQAs to a file
output_listqas_path = 'listqas.json'
with open(output_listqas_path, 'w') as json_file:
    json.dump(listqas, json_file, indent=4)
