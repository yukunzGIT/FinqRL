import csv
import os
from tqdm import tqdm

# If these libraries are not installed, you may need to install them before running:
# pip install rouge-score bert-score sacrebleu
# NLTK: pip install nltk
# and you may need to download NLTK data for BLEU if using NLTK's BLEU (already usually included)
# For this code, we will use sacrebleu for BLEU and the official libraries for ROUGE and BERTScore.

from rouge_score import rouge_scorer
from bert_score import score as bert_score
import sacrebleu

###############################################################################
# Configuration
###############################################################################
VALIDATION_INPUT_PATH = 'data/validation_set.txt'
TEST_INPUT_PATH = 'data/test_set.txt'
REFERENCE_CSV_PATH = 'data/FinRAD_13K_gemini_summary.csv'
REFERENCE_COLUMN_NAME = 'summary_gemini-1.0-pro'

LEAD_N_VALUES = [10, 20]

###############################################################################
# Utility Functions
###############################################################################

def load_references_from_csv(csv_path, ref_col):
    """
    Loads references from the specified CSV file.
    Assumes that the CSV has a header and the column `ref_col` contains the golden reference summaries.
    Returns a list of reference summaries in the order they appear.
    """
    refs = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            refs.append(row[ref_col].strip())
    return refs

def load_inputs(input_path):
    """
    Loads the input lines from the given path.
    Each line is a single long input sentence.
    Returns a list of input sentences.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines()]
    return lines

def generate_lead_n_summary(input_text, n):
    """
    Generates a lead-N summary by taking the first N tokens from the input.
    """
    tokens = input_text.split()
    return ' '.join(tokens[:n])

def compute_rouge_scores(hypotheses, references):
    """
    Compute ROUGE scores (R-1, R-2, R-L) using rouge_scorer.
    Returns average R-1, R-2, and R-L F-measure percentages.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r1_scores, r2_scores, rl_scores = [], [], []
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        r1_scores.append(scores['rouge1'].fmeasure)
        r2_scores.append(scores['rouge2'].fmeasure)
        rl_scores.append(scores['rougeL'].fmeasure)
    r1 = sum(r1_scores)/len(r1_scores)*100
    r2 = sum(r2_scores)/len(r2_scores)*100
    rl = sum(rl_scores)/len(rl_scores)*100
    return r1, r2, rl

def compute_bleu(hypotheses, references):
    """
    Compute BLEU using sacrebleu. 
    sacrebleu expects references as a list of lists (since it can handle multiple references).
    """
    # Prepare references in sacrebleu format: a list of reference lists
    # Since we have a single reference for each hypothesis, we do:
    reference_list = [[ref for ref in references]]
    # sacrebleu.corpus_bleu expects each hypothesis as a string, and references as list of lists
    bleu = sacrebleu.corpus_bleu(hypotheses, reference_list)
    return bleu.score  # This is already a percentage-like score

def compute_bertscore(hypotheses, references, model_type='bert-base-uncased'):
    """
    Compute BERTScore (F1) using the bert_score package.
    Returns the average BERTScore (F1) * 100.
    """
    P, R, F1 = bert_score(hypotheses, references, model_type=model_type, verbose=False)
    return float(F1.mean().item()) * 100

def compute_avgtr(hypotheses, inputs):
    """
    Compute AvgTR: ratio of token length of generated summaries to the token length of the original input.
    AvgTR = average over all examples of (len(hyp) / len(input)).
    """
    ratios = []
    for inp, hyp in zip(inputs, hypotheses):
        inp_len = len(inp.split())
        hyp_len = len(hyp.split())
        if inp_len > 0:
            ratios.append(hyp_len / inp_len)
        else:
            ratios.append(0.0)
    return sum(ratios)/len(ratios)*100  # *100 to show as percentage if desired, 
                                        # but it's not strictly stated as a percentage. 
                                        # The user just said "AvgTR measures the ratio," 
                                        # so maybe leave as a ratio. 
                                        # We'll just leave it as a ratio (without *100).
    # If you want percentage, uncomment the line above and remove below.
    # return sum(ratios)/len(ratios)

###############################################################################
# Main Evaluation Code
###############################################################################

if __name__ == "__main__":
    # Load references from CSV
    refs = load_references_from_csv(REFERENCE_CSV_PATH, REFERENCE_COLUMN_NAME)

    # We'll assume that references align exactly with the lines in the validation and test sets.
    # For example, validation lines correspond to the first len(validation_inputs) entries in refs,
    # and test lines correspond to the next len(test_inputs) entries in refs.
    # This assumption must hold based on how the data is prepared.
    # If not specified, we assume that "FinRAD_13K_gemini_summary.csv" contains references for all 
    # examples (validation+test) in order.

    validation_inputs = load_inputs(VALIDATION_INPUT_PATH)
    test_inputs = load_inputs(TEST_INPUT_PATH)

    # Check indexing: We assume that the CSV contains at least len(validation_inputs) + len(test_inputs) rows.
    assert len(refs) >= len(validation_inputs) + len(test_inputs), \
        "Number of references in CSV is less than required."

    val_refs = refs[:len(validation_inputs)]
    test_refs = refs[len(validation_inputs):len(validation_inputs)+len(test_inputs)]

    for N in LEAD_N_VALUES:
        print(f"==== Evaluating LEAD-{N} on Validation Set ====")
        val_hyps = [generate_lead_n_summary(inp, N) for inp in validation_inputs]

        # Compute metrics
        val_r1, val_r2, val_rl = compute_rouge_scores(val_hyps, val_refs)
        val_bleu = compute_bleu(val_hyps, val_refs)
        val_bertscore = compute_bertscore(val_hyps, val_refs)
        val_avgtr = compute_avgtr(val_hyps, validation_inputs)

        print(f"Validation LEAD-{N} Results:")
        print(f"  ROUGE-1: {val_r1:.2f}%")
        print(f"  ROUGE-2: {val_r2:.2f}%")
        print(f"  ROUGE-L: {val_rl:.2f}%")
        print(f"  BLEU: {val_bleu:.2f}")
        print(f"  BERTScore (F1): {val_bertscore:.2f}%")
        print(f"  AvgTR: {val_avgtr:.4f}")

        print(f"==== Evaluating LEAD-{N} on Test Set ====")
        test_hyps = [generate_lead_n_summary(inp, N) for inp in test_inputs]

        # Compute metrics
        test_r1, test_r2, test_rl = compute_rouge_scores(test_hyps, test_refs)
        test_bleu = compute_bleu(test_hyps, test_refs)
        test_bertscore = compute_bertscore(test_hyps, test_refs)
        test_avgtr = compute_avgtr(test_hyps, test_inputs)

        print(f"Test LEAD-{N} Results:")
        print(f"  ROUGE-1: {test_r1:.2f}%")
        print(f"  ROUGE-2: {test_r2:.2f}%")
        print(f"  ROUGE-L: {test_rl:.2f}%")
        print(f"  BLEU: {test_bleu:.2f}")
        print(f"  BERTScore (F1): {test_bertscore:.2f}%")
        print(f"  AvgTR: {test_avgtr:.4f}")
        print("============================================\n")
