import json 

def write_data(file_path, data):
    """Write the modified data back to a file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_evidence(data):
    for item in data:
        if item['wikipedia_title'] in mapping:
            item['evidence'] = mapping[item['wikipedia_title']]
        else:
            item['evidence'] = None
    return data 


mapping = read_data('../../title_to_infobox.json')
print('mapping loaded')
test_set = read_data('train_split_benchmark_v0.0_updated_with_titles.json')
golden_test_set = get_evidence(test_set)
write_data('train_set_v0.0_golden_evidence', golden_test_set)
