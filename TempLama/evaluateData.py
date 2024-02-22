from bert_score import score
import pandas as pd
import re


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


def calculate_misses_and_totals(row):
    original_entities = extract_entities(row['original_answers'])
    generated_entities = extract_entities(row['generated_answers'])
    misses = original_entities - generated_entities
    row['misses'] = len(misses)
    row['total_questions'] = len(original_entities)
    return row


# Load and clean your data
df = load_and_clean_data("gpt3_responses.csv")

# Add the pairwise BERTScore F1 column
# add_pairwise_bertscore_column(df)

# Apply function to calculate misses and total questions per row
df = df.apply(calculate_misses_and_totals, axis=1)

# Calculate the average of the newly created bert_f1 column directly
# average_f1 = df['bert_f1'].mean()
# print(f"Average BERTScore F1: {average_f1:.4f}")

# Calculate the percentage of answers that meet the F1 threshold of 0.95 without adding a column
# threshold = 0.95
# percentage_meeting_threshold = (df['bert_f1'] >= threshold).mean() * 100
# print(f"Percentage of responses meeting the F1 threshold of 0.95: {percentage_meeting_threshold:.2f}%")

# Sum the misses and total questions across all rows
total_misses = df['misses'].sum()
total_questions = df['total_questions'].sum()
print(f"Total Misses: {total_misses}")
print(f"Total Questions: {total_questions}")
