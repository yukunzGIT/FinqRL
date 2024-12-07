import os
import csv
import openai
from tqdm import tqdm

# If these libraries are not installed, you may need to install them before running:
# pip install openai rouge-score bert-score sacrebleu

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

# Set your OpenAI API key
# Make sure to set your environment variable or assign it directly here.
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"
openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL_NAME = "text-davinci-003"
PROMPT_PREFIX = "Please summarize the following text:\n\n"

# Generation parameters
MAX_TOKENS = 200
TEMPERATURE = 0.0

###############################################################################
# Utility Functions
###############################################################################

def load_references_from_csv(csv_path, ref_col):
    """
    Loads references from the specified CSV file.
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
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines()]
    return lines

def compute_rouge_scores(hypotheses, references):
    """
    Compute ROUGE scores (R-1, R-2, R-L).
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
    """
    reference_list = [[ref for ref in references]]
    bleu = sacrebleu.corpus_bleu(hypotheses, reference_list)
    return bleu.score

def compute_bertscore(hypotheses, references, model_type='bert-base-uncased'):
    """
    Compute BERTScore (F1) using the bert_score package.
    """
    P, R, F1 = bert_score(hypotheses, references, model_type=model_type, verbose=False)
    return float(F1.mean().item()) * 100

def compute_avgtr(hypotheses, inputs):
    """
    Compute AvgTR = average(len(hyp)/len(input)) across all examples.
    """
    ratios = []
    for inp, hyp in zip(inputs, hypotheses):
        inp_len = len(inp.split())
        hyp_len = len(hyp.split())
        if inp_len > 0:
            ratios.append(hyp_len / inp_len)
        else:
            ratios.append(0.0)
    return sum(ratios)/len(ratios)

def generate_summaries_gpt3(inputs):
    """
    Generates zero-shot summaries from GPT-3 (text-davinci-003).
    """
    summaries = []
    for inp in tqdm(inputs, desc=f"Generating summaries with {MODEL_NAME}"):
        prompt = PROMPT_PREFIX + inp
        response = openai.Completion.create(
            engine=MODEL_NAME,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=None
        )
        summary = response.choices[0].text.strip()
        summaries.append(summary)
    return summaries

###############################################################################
# Main Evaluation Code
###############################################################################

if __name__ == "__main__":
    # Load references from CSV
    refs = load_references_from_csv(REFERENCE_CSV_PATH, REFERENCE_COLUMN_NAME)

    # Load inputs
    validation_inputs = load_inputs(VALIDATION_INPUT_PATH)
    test_inputs = load_inputs(TEST_INPUT_PATH)

    # Split refs into val_refs and test_refs based on input sizes
    assert len(refs) >= len(validation_inputs) + len(test_inputs), \
        "Number of references in CSV is less than required."
    val_refs = refs[:len(validation_inputs)]
    test_refs = refs[len(validation_inputs):len(validation_inputs)+len(test_inputs)]

    # Generate on validation set
    val_hyps = generate_summaries_gpt3(validation_inputs)
    # Compute metrics on validation
    val_r1, val_r2, val_rl = compute_rouge_scores(val_hyps, val_refs)
    val_bleu = compute_bleu(val_hyps, val_refs)
    val_bertscore = compute_bertscore(val_hyps, val_refs)
    val_avgtr = compute_avgtr(val_hyps, validation_inputs)

    print(f"Validation Results ({MODEL_NAME}):")
    print(f"  ROUGE-1: {val_r1:.2f}%")
    print(f"  ROUGE-2: {val_r2:.2f}%")
    print(f"  ROUGE-L: {val_rl:.2f}%")
    print(f"  BLEU: {val_bleu:.2f}")
    print(f"  BERTScore (F1): {val_bertscore:.2f}%")
    print(f"  AvgTR: {val_avgtr:.4f}")

    # Generate on test set
    test_hyps = generate_summaries_gpt3(test_inputs)
    # Compute metrics on test
    test_r1, test_r2, test_rl = compute_rouge_scores(test_hyps, test_refs)
    test_bleu = compute_bleu(test_hyps, test_refs)
    test_bertscore = compute_bertscore(test_hyps, test_refs)
    test_avgtr = compute_avgtr(test_hyps, test_inputs)

    print(f"Test Results ({MODEL_NAME}):")
    print(f"  ROUGE-1: {test_r1:.2f}%")
    print(f"  ROUGE-2: {test_r2:.2f}%")
    print(f"  ROUGE-L: {test_rl:.2f}%")
    print(f"  BLEU: {test_bleu:.2f}")
    print(f"  BERTScore (F1): {test_bertscore:.2f}%")
    print(f"  AvgTR: {test_avgtr:.4f}")
