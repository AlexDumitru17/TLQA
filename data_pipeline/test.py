import wikipedia
import json
from bs4 import BeautifulSoup, Comment
import re
import concurrent.futures
import requests
import wptools


def get_wikipedia_title_from_wikidata_id(wikidata_id):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": wikidata_id,
        "props": "sitelinks",
        "format": "json",
        "sitefilter": "enwiki"
    }
    response = requests.get(url, params=params).json()
    entities = response.get("entities", {})
    enwiki_title = entities.get(wikidata_id, {}).get("sitelinks", {}).get("enwiki", {}).get("title")
    return enwiki_title


def extract_infobox_type_from_comments(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Function to process each comment
    def process_comment(comment):
        # Look for infobox.py type mention in the comment
        if 'Template:Infobox' in comment:
            return comment.strip()
        return None

    # Find all comments in the HTML
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))

    # Extract infobox.py types from comments
    infobox_types = [process_comment(comment) for comment in comments]

    # Filter out None values and return the list of infobox.py types mentioned in comments
    return [infobox_type for infobox_type in infobox_types if infobox_type]


def read_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def write_data(file_path, data):
    """Write the modified data back to a file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def extract_cricket_teams(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    team_timeline = {'Domestic': {}, 'International': {}}
    # Extracting International Team Information
    international_info = soup.find(lambda tag: tag.name == "th" and "International information" in tag.text)
    if international_info:
        next_rows = international_info.find_parent('tr').find_next_siblings('tr')
        for row in next_rows:
            if 'class' in row.attrs and 'infobox.py-header' in row.attrs['class']:
                break  # Stops at the next section header
            if row.th and "National side" in row.th.text:
                items = row.find_all('li')
                for item in items:
                    a_tag = item.find('a')
                    team_name = a_tag['title'] if a_tag else item.text
                    years = item.text.split('(')[-1].rstrip(')')
                    if years in team_timeline['International']:
                        team_timeline['International'][years] += f", {team_name}"
                    else:
                        team_timeline['International'][years] = team_name

    # Extracting Domestic Team Information
    domestic_info = soup.find(lambda tag: tag.name == "th" and "Domestic team information" in tag.text)
    if domestic_info:
        next_rows = domestic_info.find_parent('tr').find_next_siblings('tr')
        for row in next_rows:
            if row.find('th', class_="infobox.py-header"):  # Stop if another section header is reached
                break
            if row.th and row.td:  # Ensures row has both year and team name
                years = row.th.text.strip()
                team_name = row.td.text.strip()
                # Remove any unwanted text after team name (e.g., " (squad no. 7)")
                team_name = team_name.split(' (')[0]
                if years in team_timeline['Domestic']:
                    team_timeline['Domestic'][years] += f", {team_name}"
                else:
                    team_timeline['Domestic'][years] = team_name
            if row.attrs.get("style") == "display:none":  # Stops at hidden row before the statistics section
                break

    return team_timeline


def extract_football_careers(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    career_timeline = {}

    # Extracting Senior Career Information
    senior_career_header = soup.find(lambda tag: tag.name == "th" and "Senior career" in tag.text)
    if senior_career_header:
        for row in senior_career_header.find_parent('tr').find_next_siblings('tr'):
            year_cell = row.find('th', class_='infobox.py-label')
            team_cell = row.find('td', class_='infobox.py-data')
            if year_cell and team_cell:
                years = year_cell.text.strip()
                team_links = team_cell.find_all('a')

                # If there are links, use the titles; otherwise, use the cell's text content.
                if team_links:
                    teams = [a['title'] for a in team_links if a and 'title' in a.attrs]
                else:
                    # Extract text directly if no links are present.
                    teams = [team_cell.get_text().strip()]

                team_names = ', '.join(teams)
                if years and team_names and team_names not in 'Team':  # Ensure both years and team names are present and non-empty
                    career_timeline[years] = career_timeline.get(years, '') + f", {team_names}" if career_timeline.get(
                        years) else team_names

            # Break if next section is reached, but ensure it's not the International career section
            next_header = row.find('th', class_='infobox.py-header')
            if next_header and 'International career' not in next_header.text:
                break

    return career_timeline


def extract_basketball_teams(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Initialize the dictionary to store the career timeline
    career_timeline = {}

    # Find the 'Career history' header
    career_history_header = soup.find(lambda tag: tag.name == "th" and "Career history" in tag.text)

    # Ensure the 'Career history' header is found
    if career_history_header:
        # Get all sibling elements after the header
        for sibling in career_history_header.find_parent('tr').find_next_siblings('tr'):
            # If the sibling is a header, indicating a new section, check if it's the 'As coach' section
            coach_header = sibling.find('th', string=lambda text: text and "As coach" in text)
            if coach_header:
                break  # Found 'As coach', stop processing
            # Otherwise, process the career entry
            elif sibling.name == "tr":
                years_cell = sibling.find('th', style=lambda value: value and 'font-weight: normal' in value)
                team_cell = sibling.find('td')
                if years_cell and team_cell:
                    years = years_cell.text.strip()
                    teams = team_cell.text.strip()
                    career_timeline[years] = teams

    return career_timeline


def extract_american_football_teams(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    career_timeline = {}

    # Locate the 'Career history' header
    career_history_header = soup.find(lambda tag: tag.name == "th" and "Career history" in tag.text)

    if career_history_header:
        # Try to move to the 'As a player:' section directly
        as_a_player_header = career_history_header.find_parent('tr').find_next_sibling('tr')

        # Check if 'As a player:' section exists or directly find the teams list
        if as_a_player_header and "As a player" in as_a_player_header.text:
            teams_list = as_a_player_header.find_next_sibling('tr').find('td').find_all('li')
        else:
            # Directly accessing the teams list if 'As a player' header is absent
            teams_list = career_history_header.find_parent('tr').find_next_sibling('tr').find('td').find_all('li')

        for team in teams_list:
            # Extract team name and years
            team_name = team.find('a').text
            years = team.text.split('(')[-1].replace(')', '').strip()
            # Add to the career timeline
            career_timeline[years] = team_name

    return career_timeline


def extract_baseball_teams(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    career_timeline = {}

    # Find the "Teams" header within the HTML
    teams_header = soup.find(lambda tag: tag.name == "th" and "Teams" in tag.text)

    if teams_header:
        # Find the section for "As player"
        as_player_section = teams_header.find_parent('tr').find_next_sibling('tr', class_="ib-baseball-bio-teams")

        if as_player_section:
            # Extract the teams listed under "As player"
            player_info = as_player_section.find('b', string="As player")
            if player_info:
                teams_list = player_info.find_next('ul').find_all('li')
            else:
                # This means there's no separation of player and coach/manager, directly extract teams
                teams_list = as_player_section.find('td').find_all('li')

            for team in teams_list:
                # Extract team name and years
                team_name = team.find('a').text
                years = team.text.split('(')[-1].replace(')', '').strip()
                career_timeline[years] = team_name

    return career_timeline


sport_extraction_map = {
    'cricket': extract_cricket_teams,
    'association football': extract_football_careers,
    'basketball': extract_basketball_teams,
    'American football': extract_american_football_teams,
    'baseball': extract_baseball_teams
    # Add more sports and their extraction functions here
}


def fetch_and_process_page(wikidata_id, subject, extraction_func):
    try:
        wikipedia_title = get_wikipedia_title_from_wikidata_id(wikidata_id)
        if wikipedia_title:
            page = wikipedia.page(wikipedia_title, auto_suggest=False)
            html_content = page.html()
            data_timeline = extraction_func(html_content)
            return subject, data_timeline
        else:
            print(f"Wikipedia title not found for Wikidata ID {wikidata_id}")
            return subject, None
    except Exception as e:
        print(f"Error processing {subject}: {e}")
        return subject, None


# Dynamic processing based on sport type
def process_sports_data(data, sport_extraction_map):
    result_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        future_to_entry = {}
        for item in data:
            sport = item.get('sport')
            if sport in sport_extraction_map:
                wikidata_id, subject = item['wikidata_ID'], item['subject']
                extraction_func = sport_extraction_map[sport]
                future = executor.submit(fetch_and_process_page, wikidata_id, subject, extraction_func)
                future_to_entry[future] = (wikidata_id, subject)

        for future in concurrent.futures.as_completed(future_to_entry):
            _, subject = future_to_entry[future]
            try:
                result = future.result()
                if result:
                    result_dict[result[0]] = result[1]
            except Exception as e:
                print(f"Error processing result for {subject}: {e}")

    return result_dict


def fetch_page_with_wptools(title):
    page = wptools.page(title).get_parse()
    infobox = page.data['infobox']
    return infobox


# data_file_path = 'verified_data_with_sports.json'
# output_path = 'cricket_and_soccer_and_basketball.json'
# data = read_data(data_file_path)
# cricket_dict = process_sports_data(data, sport_extraction_map)
# write_data(output_path, cricket_dict)

# page = wikipedia.page("Cole Hamels", auto_suggest=False)
# ht = page.html()
# print(extract_infobox_type_from_comments(ht))

print(fetch_page_with_wptools("Bo Jackson"))

# page = wikipedia.page("MS Dhoni", auto_suggest=False)
# html_content = page.html()  # Assuming `page.html()` gives you the HTML content from the Wikipedia page
# teams_timeline = extract_cricket_teams(html_content)

# Print the timeline or process it further as needed


# data = read_data("verified_data_with_sports.json")
# distribution = list(map(lambda sport: (sum(1 for e in data if e["type"] == "P54" and e["sport"] == sport), sport),
#                         set(e["sport"] for e in data if e["type"] == "P54")))
# distribution
