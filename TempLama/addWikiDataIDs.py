from SPARQLWrapper import SPARQLWrapper, JSON
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import requests
from fuzzywuzzy import fuzz


def is_name_match(name1, name2, threshold=90):
    """Check if two names match based on a similarity score."""
    return fuzz.partial_ratio(name1.lower(), name2.lower()) > threshold


def get_entity_id_by_relationship(subject_name, relationship_type, related_entity_ids):
    """
    Find the Wikidata ID for a subject based on a relationship type and related entity IDs.

    subject_name: the name of the subject to search for.
    relationship_type: the type of relationship (e.g., 'P54' for sports teams).
    related_entity_ids: a list of related entity IDs (e.g., teams the subject has played for).
    """
    subject_name = subject_name.rstrip('.')
    # Search for entities with the subject's name
    search_url = 'https://www.wikidata.org/w/api.php'
    search_params = {
        'action': 'wbsearchentities',
        'language': 'en',
        'format': 'json',
        'search': subject_name,
        'type': 'item',
        'limit': 50  # You can adjust the limit
    }
    search_response = requests.get(search_url, params=search_params).json()

    # Iterate over the search results and check each for the correct relationship
    for search_result in search_response.get('search', []):
        entity_id = search_result['id']
        # Now verify the relationship for each candidate entity
        get_entities_url = 'https://www.wikidata.org/w/api.php'
        get_entities_params = {
            'action': 'wbgetclaims',
            'entity': entity_id,
            'property': relationship_type,
            'format': 'json'
        }
        claims_response = requests.get(get_entities_url, params=get_entities_params).json()
        claims = claims_response.get('claims', {}).get(relationship_type, [])

        # Check if any of the claims' target entity IDs match the provided related entity IDs
        for claim in claims:
            claim_target_id = claim['mainsnak']['datavalue']['value']['id']
            if claim_target_id in list(related_entity_ids.values()):
                return entity_id  # Return the entity ID if a matching claim is found

    return None  # Return None if no matching entity is found


def find_wikidata_id(subject_name, relationship_type, related_entity_ids):
    # Convert related entity names to a list of Wikidata IDs
    related_ids_values = ' '.join(f'wd:{id}' for id in related_entity_ids.values())

    # This is a SPARQL query that looks for a human with a given name, who has a relationship (P54/P39)
    # with the provided entities within the specified time frame.
    sparql_query = f"""
    SELECT DISTINCT ?person WHERE {{
      ?person wdt:{relationship_type} ?relatedEntity;
              rdfs:label ?label.
      FILTER(CONTAINS(LCASE(?label), LCASE("{subject_name}"))).
      FILTER(LANG(?label) = "en").
      VALUES ?relatedEntity {{ {related_ids_values} }}.
      ?person wdt:P31 wd:Q5.  # Ensure the entity is a human
    }}
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    for result in results["results"]["bindings"]:
        person_id = result["person"]["value"].split('/')[-1]  # Extract the Wikidata ID
        return person_id  # Return the first result that matches the criteria
    return None  # If no matching entity is found, return None


def write_data(file_path, data):
    """Write the modified data back to a file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_wikidata_ids_concurrently(data):
    total = len(data)  # Total number of entries to process
    count = 0  # Counter for processed entries
    with ThreadPoolExecutor(max_workers=3) as executor:  # Adjust the number of workers as needed
        future_to_entry = {
            executor.submit(get_entity_id_by_relationship, entry["subject"], entry["type"], entry["answer_ids"]): entry
            for entry in data if entry.get('wikidata_ID') is None
        }

        for future in as_completed(future_to_entry):
            entry = future_to_entry[future]
            try:
                wikidata_id = future.result()
                entry['wikidata_ID'] = wikidata_id
                count += 1
                print(f"Processed {count}/{total} entries.")
            except Exception as e:
                print(f"An error occurred while processing {entry['subject']}: {e}")
                count += 1  # Increment count even if there was an error


def fetch_label_and_aliases_for_wikidata_id(wikidata_id):
    """Fetch the label and aliases for a given Wikidata ID."""
    url = 'https://www.wikidata.org/w/api.php'
    params = {
        'action': 'wbgetentities',
        'ids': wikidata_id,
        'props': 'labels|aliases',
        'languages': 'en',
        'format': 'json'
    }
    response = requests.get(url, params=params).json()
    entity = response['entities'].get(wikidata_id, {})

    label = entity.get('labels', {}).get('en', {}).get('value', None)
    aliases = [alias['value'] for alias in entity.get('aliases', {}).get('en', [])]

    return wikidata_id, label, aliases


def verify_wikidata_id(entry):
    """Verify a single entry's Wikidata ID against its subject name and aliases."""
    if entry.get('wikidata_ID'):
        wikidata_id, label, aliases = fetch_label_and_aliases_for_wikidata_id(entry['wikidata_ID'])
        # Prepare a list of names to check (subject name and aliases)
        subject_name_normalized = entry['subject'].lower().rstrip('.')
        names_to_check = [label.lower()] + [alias.lower() for alias in aliases]

        # Check if any name matches
        if any(is_name_match(subject_name_normalized, name_to_check) for name_to_check in names_to_check):
            print(f"Match found for {entry['subject']} with ID {wikidata_id}: {label}")
            entry['verified'] = True
        else:
            print(f"No match for {entry['subject']} with ID {wikidata_id}. Found label: {label}, aliases: {aliases}")
            entry['verified'] = False
    return entry


def verify_wikidata_ids_concurrently(data):
    """Verify Wikidata IDs in the data concurrently."""
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Submit all entries for verification
        future_to_entry = {executor.submit(verify_wikidata_id, entry): entry for entry in data if
                           entry.get('wikidata_ID')}

        # Collect verified entries
        verified_data = [future.result() for future in as_completed(future_to_entry)]
    return verified_data


def main():
    data = read_data("listqas_ids_updated.json")
    print("Read Data")
    # get_wikidata_ids_concurrently(data)
    # write_data("listqas_ids_updated.json", data)
    verified_data = verify_wikidata_ids_concurrently(data)
    write_data("verified_data.json", verified_data)


if __name__ == "__main__":
    main()
