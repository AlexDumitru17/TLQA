import json
import re
from datetime import datetime
import multiprocessing
from fuzzywuzzy import fuzz, process
from sentence_transformers import SentenceTransformer, util
import torch
import matplotlib.pyplot as plt
import concurrent.futures
import os
from concurrent.futures import ThreadPoolExecutor

# Ensure CUDA is available for GPU usage
device = "cpu" if torch.cuda.is_available() else "cpu"

# Load Sentence Transformers model
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

current_year = datetime.now().year


def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def preprocess_text(text):
    text = text.replace("(loan)", "").strip()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    return text


def extract_answer_from_cot(answer):
    """Extract the 'Final answer' part and ignore any notes."""
    # Finding the start index of 'Final answer:'
    start_idx = answer.lower().find('final answer:')
    if start_idx != -1:
        answer = answer[start_idx + len('final answer:'):]  # skip past 'Final answer:'
    # Finding the start index of any note that starts with 'Note:'
    note_idx = re.search(r'\bNote:', answer, re.IGNORECASE)
    if note_idx:
        answer = answer[:note_idx.start()]
    return answer.strip()


def parse_model_answer(model_answer):
    """Parse the model answer into a dictionary format."""
    if 'Rationale' in model_answer and 'Final answer:' in model_answer:
        model_answer = model_answer.split('Final answer:')[1]
    elif 'Rationale' in model_answer and 'Final answer:' not in model_answer:
        return {}
    if "timeline:\n" in model_answer:
        model_answer = model_answer.split('timeline:\n')[1]
     # Finding the start index of any note that starts with 'Note:'
    note_idx = re.search(r'Note:', model_answer, re.IGNORECASE)
    if note_idx:
        model_answer = model_answer[:note_idx.start()]
    answer_dict = {}
    entries = model_answer.split('\n')
    for entry in entries:
        if ':' in entry:
            key, value = entry.rsplit(':', 1)
            answer_dict[preprocess_text(key.strip())] = value.strip()
    return answer_dict


def check_subsequence(gen_key, remaining_answers):
    """Check if gen_key is a subsequence of any remaining answers."""
    for orig_key in list(remaining_answers):
        if gen_key in orig_key or orig_key in gen_key:
            return orig_key
    return None


def embedding_similarity(text1, text2):
    """Calculate embedding similarity using MiniLM."""
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)
    return util.pytorch_cos_sim(embeddings1, embeddings2).item()


def compute_precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall, (2 * precision * recall) / (precision + recall)


def years_in_range(start, end):
    """Generate a set of years in a given range."""
    return set(range(start, end + 1))


def parse_date_ranges(date_str):
    """Parse multiple date ranges from a string."""
    date_ranges = re.findall(r'\d{4}(?:-\d{4})?', date_str)
    parsed_ranges = []
    for date_range in date_ranges:
        if '-' in date_range:
            start, end = date_range.split('-')
            start = int(start)
            end = current_year if end.lower() == 'present' else int(end)
            parsed_ranges.append((start, end))
        else:
            year = int(date_range)
            parsed_ranges.append((year, year))
    return parsed_ranges


def normalize_date_range(date_str):
    """Normalize date range strings into a standard format."""
    if isinstance(date_str, str):
        return date_str.strip()
    else:
        return str(date_str).strip()


def parse_and_normalize_answers(answers):
    """Parse and normalize dates in the answers."""
    parsed_answers = {}
    for key, date_str in answers.items():
        normalized_date = normalize_date_range(date_str)
        parsed_answers[preprocess_text(key)] = parse_date_ranges(normalized_date)
    return parsed_answers


def calculate_overlap_and_hallucination(expected_ranges, generated_ranges):
    """Calculate the overlap and hallucination between expected and generated date ranges."""
    expected_years = set()
    generated_years = set()

    for start, end in expected_ranges:
        expected_years.update(years_in_range(start, end))

    for start, end in generated_ranges:
        generated_years.update(years_in_range(start, end))

    overlap_years = expected_years & generated_years
    union_years = expected_years | generated_years  # Union of both sets

    total_expected_years = len(expected_years)

    overlap_score = len(overlap_years) / total_expected_years if total_expected_years > 0 else 0
    jaccard_similarity = len(overlap_years) / len(union_years) if len(union_years) > 0 else 0

    return overlap_score, jaccard_similarity


def evaluate_results_with_temporal(data, fuzzy_threshold=90, embedding_threshold=0.85):
    """Evaluate results based on subsequence, fuzzy matching, embedding similarity, and temporal scores."""
    precision_scores = []
    recall_scores = []
    f1_scores = []
    overlap_scores = []
    hallucination_scores = []

    for item in data:
        original_answers = {preprocess_text(k): v for k, v in item['expected'].items()}
        generated_answers = parse_model_answer(item['model_answer'])
        
        if len(generated_answers) == 0:
            continue
        matched = set()
        remaining_answers = set(original_answers.keys())
        remaining_generated = set(generated_answers.keys())

        tp, fp, fn = 0, 0, 0

        # Subsequence check
        for gen_key in list(remaining_generated):
            match_key = check_subsequence(gen_key, remaining_answers)
            if match_key:
                tp += 1
                matched.add((gen_key, match_key))
                remaining_answers.remove(match_key)
                remaining_generated.remove(gen_key)

        # Fuzzy matching
        for gen_key in list(remaining_generated):
            if len(remaining_answers) == 0:
                break
            best_match, score = process.extractOne(gen_key, remaining_answers, scorer=fuzz.token_set_ratio)
            if score > fuzzy_threshold:
                tp += 1
                matched.add((gen_key, best_match))
                # print(gen_key, '----',  best_match)
                # print('--------')
                remaining_answers.remove(best_match)
                remaining_generated.remove(gen_key)

        # Embedding similarity check for unmatched entries
        for gen_key in remaining_generated:
            for orig_key in list(remaining_answers):
                if embedding_similarity(gen_key, orig_key) > embedding_threshold:
                    tp += 1
                    # print(gen_key, '----',  best_match)
                    # print('--------')
                    matched.add((gen_key, orig_key))
                    remaining_answers.remove(orig_key)
                    break

        fp = len(remaining_generated)
        fn = len(remaining_answers)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        # Temporal score calculation per matched entity
        expected_dates = parse_and_normalize_answers(original_answers)
        generated_dates = parse_and_normalize_answers(generated_answers)
        entry_temporal_scores = []
        entry_overlap_scores = []
        entry_hallucination_scores = []

        for gen_key, orig_key in matched:
            overlap_score, hallucination_score = calculate_overlap_and_hallucination(expected_dates[orig_key.strip()],
                                                                                     generated_dates[gen_key.strip()])
            temporal_score = 0.5 * overlap_score - 0.5 * hallucination_score
            entry_temporal_scores.append(temporal_score)
            entry_overlap_scores.append(overlap_score)
            entry_hallucination_scores.append(hallucination_score)

        
        avg_overlap_score = sum(entry_overlap_scores) / len(entry_overlap_scores) if entry_overlap_scores else 0
        avg_hallucination_score = sum(entry_hallucination_scores) / len(
            entry_hallucination_scores) if entry_hallucination_scores else 0

        
        overlap_scores.append(avg_overlap_score)
        hallucination_scores.append(avg_hallucination_score)

    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    avg_overlap_score = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0
    avg_hallucination_score = sum(hallucination_scores) / len(hallucination_scores) if hallucination_scores else 0

    return avg_precision, avg_recall, avg_f1, avg_overlap_score, avg_hallucination_score


# Plot distructors affect on performance
def evaluate_and_plot_results_golden():
    dense_retrieval_f1_scores = []
    sparse_retrieval_f1_scores = []
    random_retrieval_f1_scores = []
    k_values = range(1, 11)

    for k in k_values:
        # Dense retrieval
        dense_file_path = f'results/mistral-instruct-7b/GOLDEN/knn-few-shot/Dense_Retrieval_knn_3_shot_top{k}.json'
        dense_results = read_data(dense_file_path)
        _, _, dense_f1, _, _ = evaluate_results_with_temporal(dense_results)
        dense_retrieval_f1_scores.append(dense_f1)

        # Sparse retrieval
        sparse_file_path = f'results/mistral-instruct-7b/GOLDEN/knn-few-shot/Sparse_Retrieval_knn_3_shot_top{k}.json'
        sparse_results = read_data(sparse_file_path)
        _, _, sparse_f1, _, _ = evaluate_results_with_temporal(sparse_results)
        sparse_retrieval_f1_scores.append(sparse_f1)

        random_file_path = f'results/mistral-instruct-7b/GOLDEN/knn-few-shot/Random_knn_3_shot_top{k}.json'
        random_results = read_data(random_file_path)
        _, _, random_f1, _, _ = evaluate_results_with_temporal(random_results)
        random_retrieval_f1_scores.append(random_f1)

    # Plotting the results
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, dense_retrieval_f1_scores, label='Dense Retrieval', marker='o')
    plt.plot(k_values, sparse_retrieval_f1_scores, label='Sparse Retrieval', marker='o')
    plt.plot(k_values, random_retrieval_f1_scores, label='Random', marker='o')
    plt.xlabel('k')
    plt.ylabel('F1 Score')
    plt.title('F1 Score performance Golden evidence + Distractors')
    plt.legend()
    plt.grid(True)
    plt.savefig("GOLDEN_KNN.png")
    plt.close()

def evaluate_save_all_results(base_dir, output_dir, k_values=range(1, 11)):
    def save_results_to_file(results, output_dir, filename):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    def evaluate_and_save_results(method, shot_type, k, base_dir, output_dir):
        file_path = os.path.join(base_dir, f'{method.upper()}_RAG/{shot_type}_few_shot/RAG_top{k}.json')
        results = read_data(file_path)
        _, _, f1, overlap_score, jaccard_similarity = evaluate_results_with_temporal(results)
        
        result_data = {
            'k': k,
            'f1_score': f1,
            'temporal_overlap': overlap_score,
            'jaccard_similarity': jaccard_similarity
        }
        
        output_filename = f'{method}_{shot_type}_few_shot_k{k}.json'
        save_results_to_file(result_data, output_dir, output_filename)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_eval = {
            executor.submit(evaluate_and_save_results, method, shot_type, k, base_dir, output_dir): (method, shot_type, k)
            for method in ['bm25', 'chroma']
            for shot_type in ['manual', 'knn']
            for k in k_values
        }

        for future in concurrent.futures.as_completed(future_to_eval):
            try:
                future.result()  # We use future.result() to raise any exceptions caught during execution
            except Exception as exc:
                method, shot_type, k = future_to_eval[future]
                print(f'{method.upper()} {shot_type.capitalize()} Few-Shot K={k} generated an exception: {exc}')


def load_evaluation_results(result_dir):
    results = {
        'bm25_knn': {'f1': [], 'temporal_overlap': [], 'jaccard_similarity': []},
        'bm25_manual': {'f1': [], 'temporal_overlap': [], 'jaccard_similarity': []},
        'chroma_knn': {'f1': [], 'temporal_overlap': [], 'jaccard_similarity': []},
        'chroma_manual': {'f1': [], 'temporal_overlap': [], 'jaccard_similarity': []}
    }

    for filename in os.listdir(result_dir):
        if filename.endswith('.json'):
            parts = filename.split('_')
            method = parts[0]
            shot_type = parts[1]
            k = int(parts[-1][1:-5])  # Extract the number from the filename
            with open(os.path.join(result_dir, filename), 'r') as f:
                data = json.load(f)
                results[f'{method}_{shot_type}']['f1'].append((k, data['f1_score']))
                results[f'{method}_{shot_type}']['temporal_overlap'].append((k, data['temporal_overlap']))
                results[f'{method}_{shot_type}']['jaccard_similarity'].append((k, data['jaccard_similarity']))

    for key in results:
        for metric in results[key]:
            results[key][metric].sort()

    return results

def plot_metrics(results, metric, ylabel, title, golden_manual, golden_knn, output_file):
    plt.figure(figsize=(10, 6))

    for method, color in zip(['bm25_knn', 'bm25_manual', 'chroma_knn', 'chroma_manual'], ['blue', 'green', 'red', 'purple']):
        ks, scores = zip(*results[method][metric])
        plt.plot(ks, scores, label=method.replace('_', ' ').title(), marker='o', linestyle='-', color=color)
        for k, score in zip(ks, scores):
            plt.text(k, score, f'{k}', fontsize=9, ha='right')

    plt.axhline(y=golden_manual, color='gray', linestyle='--', label='Golden Manual')
    plt.axhline(y=golden_knn, color='black', linestyle='--', label='Golden KNN')

    plt.xlabel('K')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def generate_plots_RAG_Performance():
    result_dir = 'evaluation_results_RAG'
    results = load_evaluation_results(result_dir)

    plot_metrics(
        results, 
        'f1', 
        'F1 Score', 
        'F1 Score vs. K for Different Retrieval Methods',
        golden_manual=0.818, 
        golden_knn=0.881, 
        output_file='F1_scores.png'
    )
    plot_metrics(
        results, 
        'temporal_overlap', 
        'Temporal Overlap', 
        'Temporal Overlap vs. K for Different Retrieval Methods',
        golden_manual=0.746, 
        golden_knn=0.856, 
        output_file='Temporal_Overlap_scores.png'
    )
    plot_metrics(
        results, 
        'jaccard_similarity', 
        'Jaccard Similarity', 
        'Jaccard Similarity vs. K for Different Retrieval Methods',
        golden_manual=0.71, 
        golden_knn=0.82, 
        output_file='Jaccard_Similarity_scores.png'
    )


def plot_metrics_subplot(results, golden_manual_f1, golden_knn_f1, golden_manual_overlap, golden_knn_overlap, output_file):
    # Set global font size
    plt.rcParams.update({'font.size': 14})
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    methods = ['bm25_knn', 'bm25_manual', 'chroma_knn', 'chroma_manual']
    colors = ['blue', 'green', 'red', 'purple']
    labels = ['BM25 KNN', 'BM25 Manual', 'Chroma KNN', 'Chroma Manual']

    # Plot F1 Score
    for method, color, label in zip(methods, colors, labels):
        ks, scores = zip(*results[method]['f1'])
        axs[0].plot(ks, scores, label=label, marker='o', linestyle='-', color=color)
    
    axs[0].axhline(y=golden_manual_f1, color='gray', linestyle='--', linewidth=2, label='Golden Manual')
    axs[0].axhline(y=golden_knn_f1, color='black', linestyle='--', linewidth=2, label='Golden KNN')
    axs[0].set_xlabel('K', fontsize=16)
    axs[0].set_ylabel('F1 Score', fontsize=16)
    axs[0].set_title('F1 Score vs. K for Different Retrieval Methods', fontsize=18)
    axs[0].grid(True)

    # Plot Temporal Overlap
    for method, color, label in zip(methods, colors, labels):
        ks, scores = zip(*results[method]['temporal_overlap'])
        axs[1].plot(ks, scores, label=label, marker='o', linestyle='-', color=color)
    
    axs[1].axhline(y=golden_manual_overlap, color='gray', linestyle='--', linewidth=2, label='Golden Manual')
    axs[1].axhline(y=golden_knn_overlap, color='black', linestyle='--', linewidth=2, label='Golden KNN')
    axs[1].set_xlabel('K', fontsize=16)
    axs[1].set_ylabel('Temporal Overlap', fontsize=16)
    axs[1].set_title('Temporal Overlap vs. K for Different Retrieval Methods', fontsize=18)
    axs[1].grid(True)

    # Add a single legend below both plots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=14)

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(output_file, dpi=300)
    plt.close()


def evaluate_result(file_path):
    results = read_data(file_path)
    _, _, f1, overlap, _ = evaluate_results_with_temporal(results)
    return f1, overlap

def plot_golden_performance_with_distractors(output_file):
    # Set global font size
    plt.rcParams.update({'font.size': 14})
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    k_values = range(1, 11)

    dense_retrieval_f1_scores = []
    dense_retrieval_overlap_scores = []
    sparse_retrieval_f1_scores = []
    sparse_retrieval_overlap_scores = []
    random_retrieval_f1_scores = []
    random_retrieval_overlap_scores = []

    with concurrent.futures.ProcessPoolExecutor(mp_context=multiprocessing.get_context('spawn')) as executor:
        # Prepare file paths for parallel evaluation
        dense_file_paths = [f'results/mistral-instruct-7b/GOLDEN/manual-few-shot/Dense_Retrieval_manual_3_shot_top{k}.json' for k in k_values]
        sparse_file_paths = [f'results/mistral-instruct-7b/GOLDEN/manual-few-shot/Sparse_Retrieval_manual_3_shot_top{k}.json' for k in k_values]
        random_file_paths = [f'results/mistral-instruct-7b/GOLDEN/manual-few-shot/Random_manual_3_shot_top{k}.json' for k in k_values]

        # Evaluate dense retrieval results
        dense_results = list(executor.map(evaluate_result, dense_file_paths))
        for f1, overlap in dense_results:
            dense_retrieval_f1_scores.append(f1)
            dense_retrieval_overlap_scores.append(overlap)

        # Evaluate sparse retrieval results
        sparse_results = list(executor.map(evaluate_result, sparse_file_paths))
        for f1, overlap in sparse_results:
            sparse_retrieval_f1_scores.append(f1)
            sparse_retrieval_overlap_scores.append(overlap)

        # Evaluate random retrieval results
        random_results = list(executor.map(evaluate_result, random_file_paths))
        for f1, overlap in random_results:
            random_retrieval_f1_scores.append(f1)
            random_retrieval_overlap_scores.append(overlap)

    # Golden performance values (example values, replace with actual golden performance if different)
    golden_manual_f1 = 0.818
    golden_knn_f1 = 0.882
    golden_manual_overlap = 0.747
    golden_knn_overlap = 0.857

    # Plot F1 Score
    axs[0].plot(k_values, dense_retrieval_f1_scores, label='Dense Retrieval', marker='o', color='blue')
    axs[0].plot(k_values, sparse_retrieval_f1_scores, label='Sparse Retrieval', marker='o', color='green')
    axs[0].plot(k_values, random_retrieval_f1_scores, label='Random', marker='o', color='red')
    axs[0].axhline(y=golden_manual_f1, color='black', linestyle='--', linewidth=2, label='Golden Manual F1')
    # axs[0].axhline(y=golden_knn_f1, color='black', linestyle='--', linewidth=2, label='Golden KNN F1')
    axs[0].set_xlabel('K', fontsize=16)
    axs[0].set_ylabel('F1 Score', fontsize=16)
    axs[0].set_title('Effect of Distractors on F1 Score', fontsize=18)
    axs[0].grid(True)

    # Plot Temporal Overlap
    axs[1].plot(k_values, dense_retrieval_overlap_scores, label='Dense Retrieval', marker='o', color='blue')
    axs[1].plot(k_values, sparse_retrieval_overlap_scores, label='Sparse Retrieval', marker='o', color='green')
    axs[1].plot(k_values, random_retrieval_overlap_scores, label='Random', marker='o', color='red')
    axs[1].axhline(y=golden_manual_overlap, color='black', linestyle='--', linewidth=2, label='Golden Manual Overlap')
    # axs[1].axhline(y=golden_knn_overlap, color='black', linestyle='--', linewidth=2, label='Golden KNN Overlap')
    axs[1].set_xlabel('K', fontsize=16)
    axs[1].set_ylabel('Temporal Overlap', fontsize=16)
    axs[1].set_title('Effect of Distractors on Temporal Overlap', fontsize=18)
    axs[1].grid(True)

    # Add a single legend below both plots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=14)

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(output_file, dpi=300)
    plt.close()
#---------------------------------
#this to check performance only on sports vs politcal
def split_data(data):
    sports_data = []
    political_data = []
    for item in data:
        if "teams" in item["question"].lower():
            sports_data.append(item)
        elif "political positions" in item["question"].lower():
            political_data.append(item)
    return sports_data, political_data

def evaluate_and_collect_results(data, category_name):
    precision, recall, f1, overlap, hallucination = evaluate_results_with_temporal(data)
    return {
        "Category": category_name,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Temporal Overlap": overlap,
        "Hallucination Score": hallucination
    }

def evaluate_file(file_path):
    data = read_data(file_path)
    sports_data, political_data = split_data(data)
    
    sports_results = evaluate_and_collect_results(sports_data, "Sports")
    political_results = evaluate_and_collect_results(political_data, "Political")
    
    return sports_results, political_results

def qualitative_analysis():
    file_paths = {
        "Mistral KNN": 'results/mistral-instruct-7b/Baselines/knn_3shot_v0.0.json',
        "Meta Llama KNN": 'results/Meta-Llama-3-8B-Instruct-Q6_K/knn_3shot.json',
        "Mistral BM25": 'results/mistral-instruct-7b/title-infobox-summary/BM_25_RAG/RAG_top10.json',
        "Mistral all-mini-lm": 'results/mistral-instruct-7b/CHROMA_RAG/knn_few_shot/RAG_top3.json',
        "Mistral Multi QA": 'results/mistral-instruct-7b/title-infobox-summary/CHROMA_RAG_MULTI_QA/RAG_top1.json',
        "Mistral Golden Manual": 'results/mistral-instruct-7b/RAG_GOLDEN_EVIDENCE_MANUAL_FEW_SHOT.json',
        "Mistral Golden KNN": 'results/mistral-instruct-7b/RAG_GOLDEN_EVIDENCE_KNN_FEW_SHOT.json'
    }

    results = {
        "Sports": [],
        "Political": []
    }

    with concurrent.futures.ProcessPoolExecutor(mp_context=multiprocessing.get_context('spawn')) as executor:
        future_to_file = {executor.submit(evaluate_file, path): name for name, path in file_paths.items()}
        for future in future_to_file:
            name = future_to_file[future]
            try:
                sports_results, political_results = future.result()
                print(f"Results for {name} - Sports: {sports_results}")
                print(f"Results for {name} - Political: {political_results}")
                results["Sports"].append(sports_results)
                results["Political"].append(political_results)
            except Exception as exc:
                print(f"{name} generated an exception: {exc}")

    return results


def main():
    # results = read_data("./results/llama3_3few_shot_test_v0.0.json")
    # tp, fp, fn = evaluate_results(results)
    # precision, recall, f1 = compute_precision_recall_f1(tp, fp, fn)
    # print(precision, recall, f1)
    # results = read_data("results/mistral-instruct-7b/manual_cot_results.json")
    # precision, recall, f1, overlap_score, jaccardi = evaluate_results_with_temporal(results)

    # print("Precision, Recall, F1:", precision, recall, f1)
    # print("Temporal Overlap, Temporal Jaccardi: ", overlap_score, jaccardi)
    # generate_plots_RAG_Performance()
    # evaluate_and_plot_results_golden()
    # evaluate_and_plot_RAG_top_k()
    # evaluate_save_all_results('results/mistral-instruct-7b', 'evaluation_results_RAG')

    # results_top1 =  read_data('results/mistral-instruct-7b/title-infobox-summary/CHROMA_RAG/RAG_top1.json')
    # results_top3 = read_data('results/mistral-instruct-7b/title-infobox-summary/CHROMA_RAG/RAG_top3.json')
    # results_top10 = read_data('results/mistral-instruct-7b/title-infobox-summary/CHROMA_RAG/RAG_top10.json')
    # _, _, f1_top1, _, _ = evaluate_results_with_temporal(results_top1)
    # _, _, f1_top3, _, _ = evaluate_results_with_temporal(results_top3)
    # _, _, f1_top10, _, _ = evaluate_results_with_temporal(results_top10)
    # print(f1_top1, f1_top3, f1_top10)
    # manual = read_data('results/Meta-Llama-3-8B-Instruct-Q6_K/manual_3shot.json')
    # knn = read_data('results/Meta-Llama-3-8B-Instruct-Q6_K/knn_3shot.json')
    # manual_cot = read_data('results/Meta-Llama-3-8B-Instruct-Q6_K/manual_cot.json')
    # auto_cot = read_data('results/Meta-Llama-3-8B-Instruct-Q6_K/auto_cot.json')
    # print(evaluate_results_with_temporal(manual))
    # print(evaluate_results_with_temporal(knn))
    # print(evaluate_results_with_temporal(manual_cot))
    # print(evaluate_results_with_temporal(auto_cot))

    # data = read_data('results/Meta-Llama-3-8B-Instruct-Q6_K/BM25/knn-few-shot/title-summary/RAG_top1.json')
    # print(evaluate_results_with_temporal(data))


    # results_mistral_knn_few_shot = []
    # results_meta_llama_knn_few_shot = []
    # for k in [5, 7, 10]:
    #     meta_llama_knn = read_data(f'results/Meta-Llama-3-8B-Instruct-Q6_K/Manual-kshots/KNN-{k}shots.json')
    #     mistral_knn = read_data(f'results/mistral-instruct-7b/Manual-kshots/KNN-{k}shots.json')
    #     results_mistral_knn_few_shot.append(evaluate_results_with_temporal(mistral_knn))
    #     results_meta_llama_knn_few_shot.append(evaluate_results_with_temporal(meta_llama_knn))
    # print('Mistral: ', results_mistral_knn_few_shot)
    # print('Meta LLama: ', results_meta_llama_knn_few_shot)

    # base_path = 'results/mistral-instruct-7b/title-infobox-summary/'
    # results_bm25_rag_title_infobox_summary = []
    # results_chroma_rag_title_infobox_summary = []
    # for k in  [1, 3, 10]:
    #     file_bm25 = read_data(base_path + f'BM_25_RAG/RAG_top{k}.json')
    #     file_chroma = read_data(base_path + f'CHROMA_RAG/RAG_top{k}.json')
    #     results_bm25_rag_title_infobox_summary.append(evaluate_results_with_temporal(file_bm25))
    #     results_chroma_rag_title_infobox_summary.append(evaluate_results_with_temporal(file_chroma))
    #     print('done with k=', k)
    # print('BM25: ', results_bm25_rag_title_infobox_summary)
    # print('CHROMA: ', results_chroma_rag_title_infobox_summary)

    # base_path = 'results/mistral-instruct-7b/'
    # results_title_summary= []
    # results_title_infobox_summary = []
    # for k in  [1, 3, 10]:
    #     file_title_summary = read_data(base_path + f'CHROMA_RAG_MULTI_QA/knn_few_shot/RAG_top{k}.json')
    #     results_title_summary.append(evaluate_results_with_temporal(file_title_summary))
    #     if k != 10:
    #         file_title_infobox_summary = read_data(base_path + f'title-infobox-summary/CHROMA_RAG_MULTI_QA/RAG_top{k}.json')
    #         results_title_infobox_summary.append(evaluate_results_with_temporal(file_title_infobox_summary))
        
    #     print('done with k=', k)
    # print('TITLE SUMMARY: ', results_title_summary)
    # print('TITLE INFOBOX SUMMARY', results_title_infobox_summary)

    # base_path = 'results/mistral-instruct-7b/'
    # bm_25 = read_data(base_path + 'title-infobox-summary/BM_25_RAG/RAG_top10.json')
    # multi_qa = read_data(base_path + 'title-infobox-summary/CHROMA_RAG_MULTI_QA/RAG_top1.json' )
    # all_mini_lm = read_data(base_path + 'CHROMA_RAG/knn_few_shot/RAG_top3.json')    
    # results1 = evaluate_results_with_temporal(bm_25)
    # results2 = evaluate_results_with_temporal(multi_qa)
    # results3 = evaluate_results_with_temporal(all_mini_lm)
    # print("BM25", results1)
    # print("multiqa", results2)
    # print("all_mini_mlm", results3)

    # result_dir = 'evaluation_results_RAG'
    # results = load_evaluation_results(result_dir)
    # plot_metrics_subplot(
    # results,
    # golden_manual_f1=0.818,
    # golden_knn_f1=0.882,
    # golden_manual_overlap=0.747,
    # golden_knn_overlap=0.857,
    # output_file='combined_plot.png'
    # )
    # plot_golden_performance_with_distractors('golden_f1_temporal_manual.png')
    # file = read_data('RAG_GOLDEN_EVIDENCE_MANUAL_FEW_SHOT_test.json')
    # print(evaluate_results_with_temporal(file))
    # qualitative_analysis()
    test = read_data('results/Meta-Llama-3.1-8b/knn-3shot.json')
    print("KNN3shot: ",evaluate_results_with_temporal(test))
    test = read_data('results/Meta-Llama-3.1-8b/manual-3shot.json')
    print("MANUAL3shot: ",evaluate_results_with_temporal(test))
    test = read_data('results/Meta-Llama-3.1-8b/manual-cot.json')
    print('MANUALCOT3: ',evaluate_results_with_temporal(test))
    test = read_data('results/Meta-Llama-3.1-8b/auto-cot.json')
    print("AUtoCOT: ", evaluate_results_with_temporal(test))
    test = read_data('results/Meta-Llama-3.1-8b/RAG_GOLDEN_EVIDENCE_KNN_FEW_SHOT.json')
    print("GOLDEN_KNN: ",evaluate_results_with_temporal(test))
    test = read_data('results/Meta-Llama-3.1-8b/RAG_GOLDEN_EVIDENCE_MANUAL_FEW_SHOT.json')
    print("GOLDEN_MANUAL: ", evaluate_results_with_temporal(test))
    for k in [1,3,5,10]:
        bm25 = read_data(f'results/Meta-Llama-3.1-8b/BM25/title-summary/RAG_top{k}.json')
        chroma = read_data(f'results/Meta-Llama-3.1-8b/CHROMA/title-summary/RAG_top{k}.json')
        bm25_eval = evaluate_results_with_temporal(bm25)
        chroma_eval = evaluate_results_with_temporal(chroma)
        print(f'BM25 top {k}: ', bm25_eval)
        print(f'chroma top {k}', chroma_eval)
if __name__ == "__main__":
    main()
