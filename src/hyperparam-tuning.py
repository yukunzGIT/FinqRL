import os
import random
import numpy as np
from sklearn.model_selection import KFold
from trainer import train  # Assuming trainer.py is in the same directory and has def train(opt)

# --------------------------------------------------------------------------------------
# Assumptions and placeholders:
#
# 1. We have existing code in trainer.py with a function:
#    def train(opt):
#        # trains the model based on parameters in opt
#        # either returns a trained model or saves it in a known location
#    Here, we assume it returns a trained model object.
#
# 2. Data:
#    We have:
#    - data/train_set.txt
#    - data/validation_set.txt
#    - data/test_set.txt
#    Each line is a single long input sentence.
#
# 3. Evaluation:
#    We'll use a placeholder function evaluate_model(model, val_data) that returns a dictionary:
#    {"rouge_l": float, "bert_score": float, "bleu": float, "avg_tr": float}
#
# 4. Hyper-parameter tuning:
#    We'll randomly sample values for one parameter while others are fixed at 0.1.
#
# 5. Score computation:
#    Score = 0.35 * ROUGE-L + 0.35 * BERTScore + 0.1 * BLEU - 0.2 * AvgTR
#
# 6. Cross-validation on validation set using K=5.
#    Split val set into folds, train on 4 folds + train set, evaluate on the remaining fold.
#
# NOTE: Replace the placeholders with actual logic as needed.
# --------------------------------------------------------------------------------------

def evaluate_model(model, val_data):
    """
    Placeholder evaluation function.
    In practice, you would run model inference on val_data and compute the metrics.
    """
    # Return random scores for demonstration.
    return {
        "rouge_l": np.random.uniform(0.1, 1.0),
        "bert_score": np.random.uniform(0.1, 1.0),
        "bleu": np.random.uniform(0.1, 1.0),
        "avg_tr": np.random.uniform(0.1, 1.0)
    }

def compute_score(metrics):
    """
    Compute the weighted aggregate score.
    Score = 0.35 * ROUGE-L + 0.35 * BERTScore + 0.1 * BLEU - 0.2 * AvgTR
    """
    return (0.35 * metrics["rouge_l"] 
            + 0.35 * metrics["bert_score"] 
            + 0.1 * metrics["bleu"] 
            - 0.2 * metrics["avg_tr"])

# Load data
with open('data/train_set.txt', 'r') as f:
    train_sentences = [line.strip() for line in f if line.strip()]

with open('data/validation_set.txt', 'r') as f:
    val_sentences = [line.strip() for line in f if line.strip()]

with open('data/test_set.txt', 'r') as f:
    test_sentences = [line.strip() for line in f if line.strip()]

# Default hyper-parameters
default_s1 = 0.1
default_s2 = 0.1
default_r = 0.1
default_l = 0.1

param_range = (0.1, 1.0)
num_samples = 4  # number of random samples for each param
K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)

def cross_validate(s1, s2, r_val, l_val):
    """
    Perform cross-validation for the given hyper-parameters.
    """
    fold_metrics = {
        "rouge_l": [],
        "bert_score": [],
        "bleu": [],
        "avg_tr": []
    }

    val_data = np.array(val_sentences)
    for train_idx, val_idx in kf.split(val_data):
        val_train_split = val_data[train_idx]
        val_val_split = val_data[val_idx]

        # Combine original train_set with the validation fold used as pseudo-train
        combined_train_data = train_sentences + val_train_split.tolist()

        # Setup the options for training (this is a placeholder, adjust to match your trainer.py interface)
        opt = {
            'train_data': combined_train_data,
            's1': s1,
            's2': s2,
            'r': r_val,
            'l': l_val,
            # Add other necessary parameters required by your training script
            # For example:
            # 'batch_size': 32,
            # 'epochs': 5,
            # 'model_save_path': 'path/to/save/model',
            # ...
        }

        # Train model using trainer.py
        model = train(opt)

        # Evaluate on val_val_split
        metrics = evaluate_model(model, val_val_split.tolist())
        for k, v in metrics.items():
            fold_metrics[k].append(v)

    # Compute average metrics across folds
    avg_metrics = {k: float(np.mean(v_list)) for k, v_list in fold_metrics.items()}
    return avg_metrics

def tune_parameter(param_name):
    """
    Tune a single parameter. 
    param_name: one of 's1', 's2', 'r', 'l'
    """
    best_score = -float('inf')
    best_value = None
    results = []

    # Generate random values for the parameter
    sampled_values = [round(random.uniform(param_range[0], param_range[1]), 3) for _ in range(num_samples)]

    for val in sampled_values:
        if param_name == 's1':
            s1 = val; s2 = default_s2; r_val = default_r; l_val = default_l
        elif param_name == 's2':
            s1 = default_s1; s2 = val; r_val = default_r; l_val = default_l
        elif param_name == 'r':
            s1 = default_s1; s2 = default_s2; r_val = val; l_val = default_l
        elif param_name == 'l':
            s1 = default_s1; s2 = default_s2; r_val = default_r; l_val = val
        else:
            raise ValueError("Unknown parameter name!")

        # Cross-validate
        avg_metrics = cross_validate(s1, s2, r_val, l_val)
        score = compute_score(avg_metrics)
        results.append((val, avg_metrics, score))

        if score > best_score:
            best_score = score
            best_value = val

    return best_value, best_score, results

# Tuning each parameter independently
print("Tuning s_1...")
best_s1, best_s1_score, s1_results = tune_parameter('s1')
print("Best s_1:", best_s1, "with score:", best_s1_score)

print("Tuning s_2...")
best_s2, best_s2_score, s2_results = tune_parameter('s2')
print("Best s_2:", best_s2, "with score:", best_s2_score)

print("Tuning r...")
best_r, best_r_score, r_results = tune_parameter('r')
print("Best r:", best_r, "with score:", best_r_score)

print("Tuning l...")
best_l, best_l_score, l_results = tune_parameter('l')
print("Best l:", best_l, "with score:", best_l_score)

# At this point, you have best hyperparameters selected for each parameter independently.
# You could further run final training or do another combined tuning if desired.
