import os
import numpy as np
from dqn import EditorialAgent


recon_stopwords_file = \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "stopwords.txt")
with open(recon_stopwords_file) as f:
    for l in f:
        recon_stopwords = set([l.strip() for l in f])


def get_compression_ratio(item):
    #  Measures the proportion of tokens labeled as REMOVE in item.labels.
    co_ratio = len(
        [i for i in item.labels if i == EditorialAgent.REMOVE]) / len(item.labels)  # Creates a list of labels marked as REMOVE.
    return co_ratio # Ex: item.labels = ["REMOVE", "KEEP", "REMOVE", "KEEP"], then co_ratio = 2 / 4 = 0.5


def get_reconstruction_ratio(item):
    # calculates the reconstruction ratio, which measures how accurately tokens from the original sentence are reconstructed in the top-k predictions.
    # aka, compares tokens from the original sentence with the top-k reconstructed tokens and calculates the proportion of correct matches.
    
    # Initialize Counters: 
    cor, denom = 0, 0 # cor: Tracks the number of correctly reconstructed tokens. denom: Tracks the total number of tokens evaluated as the denominator.
    for i in range(len(item.labels)): # Loops through each token in item.sentence.
        # Skips tokens that are stopwords or unknown ([UNK]).
        if item.sentence[i] in recon_stopwords: 
            continue
        if item.sentence[i] == "[UNK]":
            continue

        # Count Correct Reconstructions:
        cor += item.sentence[i] in item.recon_sent_topk[i] # from the Item class in dqn.py, increments cor if the token appears in the top-k reconstructed tokens.
        denom += 1 # Update Denominator:
    return 0 if denom == 0 else cor / denom


def calculate_comp_recon_rewards(episode_items):
    #  calculates compression, reconstruction, and combined rewards for a list of items. Adds compression, reconstruction, and combined rewards to each episode_item
    for ei in episode_items: # Loops through all items in the episode. Each ei is an Item class in dqn.py. 
        ei.cr = get_compression_ratio(ei)
        ei.rr = get_reconstruction_ratio(ei)
        ei.crr = ei.cr + ei.rr 
    return episode_items    # cr, rr and crr are all None when the item class is initialized


def summary_assessment(episode_items):
    # Evaluates the episode based on reconstruction, compression, likelihood, and similarity metrics. Compute a final reward for the episode and assigns it to all items.
    
    # Updates items with compression, reconstruction, and combined rewards.
    episode_items = calculate_comp_recon_rewards(episode_items) 
    last_item = episode_items[-1]

    llh_boundary = 0.005
    llh_reward = (1 if last_item.comp_llh > llh_boundary else 0)  # Assigns 1 if the likelihood is above a threshold, otherwise 0.
    sim_reward = last_item.comp_sim # Uses the similarity score directly.

    r = last_item.rr * last_item.cr # from the beginning to T-th steps as defined in the step reward paragraph in the paper.
    r += sim_reward * 0.1
    r += llh_reward * 0.1

    for ei in episode_items: # Assigns the final reward to all items.
        ei.reward = r
    return episode_items, episode_items # To maintain compatibility, the function simply duplicates the list, allowing older code to continue working without changes.


def reward(episode_items, min_cr, min_rr): # Minimum compression ratio and reconstruction ratio required at each step.
    # Calculates step-wise rewards (r_SR from the paper) for an episode based on compression, reconstruction, and other metrics.

    nstep = len(episode_items) # Number of steps in the episode.
    min_rr_per_step = 1 - np.cumsum([(1 - min_rr) / nstep for i in range(0, nstep)]) # Decreasing reconstruction ratio thresholds over the steps. 
    # Ex: min_rr_per_step = [0.95, 0.9, 0.85, 0.8] when min_rr = 0.8 and nstep = 4. 

    min_cr_per_step = np.cumsum([(min_cr / nstep) for i in range(0, nstep)]) # Increasing compression ratio thresholds over the steps.
    # Ex: min_cr_per_step = [0.15, 0.3, 0.45, 0.6] when min_cr = 0.6 and nstep = 4
    llh_boundary = 0.005 # Likelihood threshold for a positive likelihood reward.

    episode_items = calculate_comp_recon_rewards(episode_items) # update the cr, rr and crr for each item in episode_items

    # Initialize Variables: 
    learn_items, all_items = [], [] # learn_items: Items that meet the reward criteria. all_items: All processed items.
    prev_n = len(episode_items[0].sentence) # Tracks the number of tokens in the sentence at the previous step.
    failed = False # A flag to indicate whether the episode failed at a certain step.

    for cur_min_rr, cur_min_cr, ei in zip(min_rr_per_step, min_cr_per_step, episode_items): # Loops through each item step, along with the corresponding thresholds for compression and reconstruction ratios.
        step_cr = len(ei.comp_sent) / prev_n # Compression ratio for the current step, based on the change in sentence length from the previous step.
        prev_n = len(ei.comp_sent)

        # Compute Step-Wise Rewards
        cr = (1 - step_cr) if ei.cr >= cur_min_cr else 0 # Reward for meeting the compression threshold. Penalized (0) if below the threshold.
        rr = 1 if ei.rr >= cur_min_rr else -1  # Reward for meeting the reconstruction threshold. Penalized (-1) if below the threshold.

        # Product of reconstruction and compression rewards, initialized to a large penalty (-9.99) if the episode failed.
        r_sr = -9.99
        if not failed:
            r_sr = rr * cr

        # Update Item Rewards: assigns the calculated reward (r_sr) to the current item and adds it to all_items.
        ei.reward = r_sr
        all_items.append(ei)

        # Check for Failure
        if not failed:
            if ei.rr < cur_min_rr or ei.cr < cur_min_cr:  # If reconstruction or compression ratio falls below the respective thresholds, the episode fails:
                ei.reward = -1 # Assigns a penalty (-1) to the current item.
                learn_items.append(ei) # Adds it to learn_items.
                failed = True
            else:
                learn_items.append(ei)

    # Calculate Episode-Wide Rewards: 
    reached_step = len(learn_items) / nstep # Proportion of steps completed successfully.
    last_item = learn_items[-1]

    comp_recon_rate = last_item.rr * last_item.cr # Product of r_C and r_R to get r_SR for the last step.
    sim_reward = last_item.comp_sim #  Similarity reward from the last item.
    llh_reward = (1 if last_item.comp_llh > llh_boundary else 0) # Likelihood reward, based on whether it exceeds the threshold (llh_boundary).

    r_sa = \
        reached_step * (comp_recon_rate + sim_reward * 0.1 + llh_reward * 0.1)
    for li in learn_items:
        li.reward += r_sa

    return learn_items, all_items
