import json
import os
import sys
from collections import namedtuple
import torch
from itertools import chain
import time
import tqdm
import trainer
import reward
import util
import pandas as pd
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import sacrebleu

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_json(dir):
    with open(os.path.join(dir, "config.json"), "r") as f:
        j = json.load(f)
    return namedtuple('conf', j.keys())(*j.values())


def load_model(dir):
    file = "best-reward-model.pt"
    return torch.load(os.path.join(dir, file), map_location=DEVICE)


def init(opt):
    import bertnlp
    import dqn

    conf = load_json(opt.m)
    model = load_model(opt.m)

    bertnlp.init()
    agent = dqn.EditorialAgent(layer_num=int(conf.nlayer), hidden_dim=int(conf.hdim))
    agent.load_state_dict(model)
    agent.eval()
    return bertnlp, agent, conf


def run(bertnlp, agent, texts):
    tokenized_texts, hiddens = bertnlp.apply_tokenize_and_get_hiddens(texts, rm_pad=True)
    items = agent.process(tokenized_texts[0], hiddens[0], do_exploration=False)
    tokenized_texts, labels = zip(*[(i.sentence, i.labels) for i in items])
    comp_sents, comp_topk, recon_sents, recon_topk = \
        bertnlp.apply_compression_and_reconstruction(tokenized_texts, labels)
    for i, cs, csk, rs, rsl in zip(items, comp_sents, comp_topk, recon_sents, recon_topk):
        i.set(cs, csk, rs, rsl, 0, 0)
    items = reward.calculate_comp_recon_rewards(items)

    max_crr, max_crr_item = 0, items[0]
    for i in items:
        if i.crr >= max_crr:
            max_crr = i.crr
            max_crr_item = i

    return [max_crr_item.comp_sent]

# Metrics Calculation Functions
def compute_rouge(hypotheses, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1, rouge2, rougeL = [], [], []
    for hyp, ref in zip(hypotheses, references):
        scores = scorer.score(ref, hyp)
        rouge1.append(scores['rouge1'].fmeasure)
        rouge2.append(scores['rouge2'].fmeasure)
        rougeL.append(scores['rougeL'].fmeasure)
    return (
        sum(rouge1) / len(rouge1) * 100,
        sum(rouge2) / len(rouge2) * 100,
        sum(rougeL) / len(rougeL) * 100,
    )

def compute_bleu(hypotheses, references):
    references = [[ref] for ref in references]  # SacreBLEU expects list of lists
    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    return bleu.score

def compute_bertscore(hypotheses, references, model_type="bert-base-uncased"):
    P, R, F1 = bert_score(hypotheses, references, model_type=model_type, verbose=False)
    return float(F1.mean()) * 100

def compute_avgtr(hypotheses, inputs):
    ratios = [len(hyp.split()) / len(inp.split()) for hyp, inp in zip(hypotheses, inputs)]
    return sum(ratios) / len(ratios)

# Evaluation function
def evaluate(opt, dataset_file, reference_file):
    # Load data
    with open(dataset_file, "r") as f:
        inputs = [line.strip() for line in f.readlines()]

    references = pd.read_csv(reference_file)["summary_gemini-1.0-pro"].tolist()

    # Initialize model and generate summaries
    bertnlp, agent, _ = init(opt)
    print("Generating summaries...")
    hypotheses = run(bertnlp, agent, inputs)

    # Calculate metrics
    print("Calculating metrics...")
    rouge1, rouge2, rougeL = compute_rouge(hypotheses, references)
    bleu = compute_bleu(hypotheses, references)
    bertscore = compute_bertscore(hypotheses, references)
    avgtr = compute_avgtr(hypotheses, inputs)

    # Print results
    print(f"ROUGE-1: {rouge1:.2f}%")
    print(f"ROUGE-2: {rouge2:.2f}%")
    print(f"ROUGE-L: {rougeL:.2f}%")
    print(f"BLEU: {bleu:.2f}")
    print(f"BERTScore (F1): {bertscore:.2f}%")
    print(f"AvgTR: {avgtr:.4f}")



if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser("Prediction script", add_help=True)

    args.add_argument("-m", help="Model directory", required=True)
    args.add_argument("-f", help="Target file", default=None)
    args.add_argument("-o", help="Output file", default=None)

    opt = args.parse_args()
    bertnlp, agent, conf = init(opt)
    sys.stderr.write("Training configuration: {}\n".format(conf))
    s = time.time()
    texts = trainer.load_file(opt.f)
    results = []
    for t in tqdm.tqdm(texts):
        results.append(run(bertnlp, agent, [t]))

    f = open(opt.o, "w")

    results = list(chain.from_iterable(results))
    for res in results:
        f.write(" ".join(res) + "\n")
    f.close()

    sys.stderr.write("Elapsed Time={:.3f}ms".format(time.time()-s))
