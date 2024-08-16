from bert_score import score
import pandas as pd
from Levenshtein import distance as levenshtein_distance
import copy


def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df['generated_answers'] = df['generated_answers'].str.replace(r'^\[Final Answer\]:\s*', '', regex=True)
    return df


def add_pairwise_bertscore_column(df):
    cands = df['generated_answers'].tolist()
    refs = df['original_answers'].tolist()
    _, _, F1 = score(cands, refs, lang='en', verbose=True)
    df['bert_f1'] = F1.tolist()  # Convert tensor to list and add as column


def normalize_entity(entity):
    """Normalize entity for comparison."""
    return entity.lower().replace(".", "").replace(",", "").strip()


def extract_entities(answer):
    """Extract entities from an answer, disregarding years."""
    entities = answer.split("\n")
    normalized_entities = [normalize_entity(entity.split("(")[0]) for entity in entities]
    return set(normalized_entities)


def calculate_misses_and_totals(row, threshold=0.8):
    original_entities = extract_entities(row['original_answers'])
    generated_entities = extract_entities(row['generated_answers'])
    new_generated_entities = copy.deepcopy(generated_entities)
    mapping = {}
    bert_scores = {}
    unmatched_original_entities = set(original_entities)

    # First pass: direct containment checks
    for original_entity in original_entities:
        for generated_entity in list(new_generated_entities):
            if original_entity in generated_entity or generated_entity in original_entity:
                mapping[original_entity] = generated_entity
                new_generated_entities.remove(generated_entity)
                unmatched_original_entities.remove(original_entity)
                break

    # Prepare for pairwise BERT score calculations
    bert_score_pairs = []

    # Compute BERT scores for all unmatched pairs
    for original_entity in unmatched_original_entities:
        for generated_entity in new_generated_entities:
            pair_key = (original_entity, generated_entity)
            _, _, scores = score([original_entity], [generated_entity], lang='en', device="cuda:0", verbose=True)
            bert_score = scores.item()
            bert_scores[pair_key] = bert_score
            bert_score_pairs.append((bert_score, original_entity, generated_entity))

    # Sort pairs by BERT score in descending order
    bert_score_pairs.sort(reverse=True)

    # Select mappings based on the highest BERTScore, respecting the threshold
    for bert_score, original_entity, generated_entity in bert_score_pairs:
        if bert_score >= threshold and original_entity in unmatched_original_entities and generated_entity in new_generated_entities:
            mapping[original_entity] = generated_entity
            unmatched_original_entities.remove(original_entity)
            new_generated_entities.remove(generated_entity)

    # Misses and hallucinations calculation
    misses = len(unmatched_original_entities)
    hallucinations = len(new_generated_entities)

    row['mapping'] = mapping
    row['misses'] = misses
    row['hallucinations'] = hallucinations
    row['total_answers'] = len(original_entities)
    row['bert_scores'] = {pair: score1 for pair, score1 in bert_scores.items() if score1 >= threshold}

    return row


def calculate_similarity_bert_score(row, threshold=0.9):
    original_entities = extract_entities(row['original_answers'])
    generated_entities = extract_entities(row['generated_answers'])
    scores_dict = {}

    # Calculate BERTScores for each unique pair of original and generated entities
    for original in original_entities:
        for generated in generated_entities:
            if (original, generated) not in scores_dict:
                # Compute the BERTScore and store it
                scores = score([original], [generated], lang="en", rescale_with_baseline=True, device="cuda:0")[
                    2].item()
                scores_dict[(original, generated)] = scores

    # Determine misses and hallucinations using the pre-computed scores
    misses = sum(1 for original in original_entities if
                 all(scores_dict[(original, generated)] < threshold for generated in generated_entities))
    hallucinations = sum(1 for generated in generated_entities if
                         all(scores_dict[(original, generated)] < threshold for original in original_entities))

    row['bert_misses'] = misses
    row['bert_hallucinatios'] = hallucinations


def normalize_text(text):
    """Normalize text by removing non-alphabetic characters and lowercasing."""
    return ''.join(filter(str.isalpha, text.lower()))


def find_closest_match(generated_team, original_teams):
    """
    Find the closest match for a generated team name in the original teams list
    using Levenshtein distance.
    """
    closest_match = None
    closest_distance = float('inf')
    for original_team in original_teams:
        dist = levenshtein_distance(generated_team, original_team)
        if dist < closest_distance:
            closest_distance = dist
            closest_match = original_team
    return closest_match, closest_distance


def evaluate_completeness(row):
    """
    Evaluates the completeness of the generated answers against the original answers
    using substring matching and Levenshtein distance.
    """
    original_teams = set(map(normalize_text, row['original_answers'].replace('\n', ',').split(',')))
    generated_teams = set(map(normalize_text, row['generated_answers'].replace('[Final Answer]:', '').split(',')))

    # Remove empty strings
    original_teams = {team for team in original_teams if team}
    generated_teams = {team for team in generated_teams if team}

    correct_matches = 0
    for generated_team in generated_teams:
        # Check for direct or substring match
        if any(generated_team in original_team or original_team in generated_team for original_team in original_teams):
            correct_matches += 1
        else:
            # Use Levenshtein distance for fuzzy matching
            closest_match, distance = find_closest_match(generated_team, original_teams)
            # Assuming a threshold for close enough matches
            if distance <= max(len(generated_team), len(closest_match)) * 0.2:
                correct_matches += 1

    # Completeness considers correct matches, penalizes for misses and hallucinations
    completeness_score = correct_matches / len(original_teams) if original_teams else 0

    return {
        'completeness': completeness_score,
    }


# Load and clean your data
df = load_and_clean_data("gpt3_responses.csv")
df = df.apply(calculate_misses_and_totals, axis=1)
# Applying the updated evaluation function to each row in the dataframe
# metrics_updated = df.apply(evaluate_completeness, axis=1, result_type='expand')
#
# # Merging the updated metrics back into the original dataframe for a comprehensive view
# evaluated_data_updated = pd.concat([df, metrics_updated], axis=1)
#
# # Display the first few rows of the updated evaluated dataframe
# evaluated_data_updated.head()
# # Add the pairwise BERTScore F1 column
# # add_pairwise_bertscore_column(df)
# # df = df.apply(apply_bem_to_df, axis=1)
# # Apply function to calculate misses and total questions per row

# df = df.apply(calculate_similarity_bert_score, axis=1)
# # Calculate the average of the newly created bert_f1 column directly
# # average_f1 = df['bert_f1'].mean()
# # print(f"Average BERTScore F1: {average_f1:.4f}")
#
# # Calculate the percentage of answers that meet the F1 threshold of 0.95 without adding a column
# # threshold = 0.95
# # percentage_meeting_threshold = (df['bert_f1'] >= threshold).mean() * 100
# # print(f"Percentage of responses meeting the F1 threshold of 0.95: {percentage_meeting_threshold:.2f}%")
#
# # Sum the misses and total questions across all rows
# total_misses = df['misses'].sum()
# total_questions = df['total_questions'].sum()
#
# print(f"Total Misses: {total_misses}")
# print(f"Total Questions: {total_questions}")
