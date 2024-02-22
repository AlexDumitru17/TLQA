import requests

# Define the SPARQL query
query = """
SELECT ?work ?workLabel ?date WHERE {
  ?work wdt:P50 wd:Q42;     # wd:Q42 is the entity ID for the author, replace with your target entity
         wdt:P577 ?date.    # P577 is the property for publication date
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}
ORDER BY ?date
"""

# URL for the Wikidata Query Service
url = "https://query.wikidata.org/sparql"

# Make the request
response = requests.get(url, params={'format': 'json', 'query': query})

# Parse the response
data = response.json()

# Process and print the results
for item in data['results']['bindings']:
    work_name = item['workLabel']['value']
    date = item['date']['value'] if 'date' in item else "Unknown date"
    print(f"{work_name}: {date}")
