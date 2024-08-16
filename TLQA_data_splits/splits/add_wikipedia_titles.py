import json
import requests

# Load the JSON data from the file
file_path = 'train_split_benchmark_v0.0.json'  # Update with the correct path
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Function to get the Wikipedia title from a Wikidata ID
def get_wikipedia_title(wikidata_id):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": wikidata_id,
        "format": "json",
        "props": "sitelinks",
    }
    response = requests.get(url, params=params)
    data = response.json()
    # Extract the English Wikipedia title
    title = data["entities"][wikidata_id]["sitelinks"].get("enwiki", {}).get("title", None)
    return title

# Add Wikipedia titles to the data
for entry in data:
    wikidata_id = entry.get("wikidata_ID")
    if wikidata_id:
        entry["wikipedia_title"] = get_wikipedia_title(wikidata_id)

# Save the updated data to a new JSON file
updated_file_path = 'train_split_benchmark_v0.0_updated_with_titles.json'  # Update with the correct path
with open(updated_file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print(f"Updated file saved at: {updated_file_path}")
