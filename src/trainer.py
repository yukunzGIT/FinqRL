import copy
import json
import os
import shutil
import sys
import torch
import torch.optim as optim

import numpy as np

from itertools import chain

import bertnlp
import dqn
import reward
import util

import matplotlib.pyplot as plt

# Initialize lists to store rewards and rewards_ for plots
#all_rewards = []
all_rewards_ = []



logger = None


def train(opt):

    # File and Path Initialization
    SAVE_DIR = opt.save
    LATEST_MODEL = os.path.join(SAVE_DIR, "latest-model.pt") # Saves the latest model checkpoint.
    BEST_REWARD_MODEL = os.path.join(SAVE_DIR, "best-reward-model.pt")  # Saves the model with the best reward.
    CONFIG_FILE = os.path.join(SAVE_DIR, "config.json") # Stores the training configuration for reproducibility.
    LOG_FILE = os.path.join(SAVE_DIR, "log.txt") # Logs training progress and details.

    global logger # Initializes a global logger to write logs to the file specified by LOG_FILE.
    logger = util.init_logger(LOG_FILE)

    with open(CONFIG_FILE, "w") as f:
        json.dump(vars(opt), f) # Saves the training configuration (opt) as a JSON file to CONFIG_FILE.

    # Initializes the BERT-based NLP model, loading pre-trained weights.
    bertnlp.init() # NOTE

    # Agent and Data Initialization
    agent = dqn.EditorialAgent(layer_num=opt.nlayer, hidden_dim=opt.hdim) # NOTE
    train_sents = load_file(opt.t) # data/FinRAD_13K_gemini_summary.txt

    memory = dqn.Memory(buffer_size=int(opt.memory)) # default=2000
    optimizer = torch.optim.Adam(agent.parameters(), lr=opt.lr) # default=0.001
    sentence_batcher = make_batcher(train_sents, 1) # Prepares a batch generator for sentences. batch_size=1

    # Training Variables: 
    n_sentence_iter = 0 # Tracks the number of times the sentence corpus has been iterated.
    model_update_interval = 1000 # Specifies how often the model should be saved.
    max_avg_reward = -1  # Tracks the highest average reward achieved during training.

    # Learning rate scheduler: ReduceLROnPlateau
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # gradient_clip_value = 1.0  # Clip gradients to this maximum value

    try:
        for epoch in range(int(opt.e)): # default=10000000000000

            # Batch Preparation:
            try:
                cur_sentences = next(sentence_batcher) # Retrieves the next batch of sentences from sentence_batcher
            except StopIteration: # If the batcher is exhausted (StopIteration), increments n_sentence_iter and reinitializes the batcher.
                n_sentence_iter += 1
                logger.info("Corpus Iteration {}".format(n_sentence_iter))
                sentence_batcher = make_batcher(train_sents, 1)
                cur_sentences = next(sentence_batcher)

            # Generate Episode Items: generates RL episode items for the current batch of sentences.
            items, reports = generate_episode_items(bertnlp, agent,
                                                    cur_sentences,
                                                    min_cr=opt.min_cr,
                                                    min_rr=opt.min_rr) # NOTE

            # Store Experience                                                    
            [memory.append(i) for i in chain.from_iterable(items) if not i.is_terminal]  # Flattens the episode items into a single list and stores non-terminal items in the replay memory.

            if memory.size() < opt.batch: # Skips training if the replay memory does not contain enough samples for a batch.
                continue

            loss, reward = agent.replay(memory.sample(opt.batch)) # Samples a batch from memory and computes loss and reward.
            loss_, reward_ = step(loss, reward, optimizer, agent) # Optimizes the agent using the computed loss and reward.

            #all_rewards.append(reward)
            all_rewards_.append(reward_)

            # Logs training progress
            msg = "Report : Epoch={} Reward={:.3f} Loss={:.3f} Eps1={:.3f} Eps2={:.3f}\n".format(
                epoch, reward_, loss_, agent.epsilon, agent.selection_epsilon)
            msg += "=" * 70 + "\n\t" + "\n\t".join(
                [i.report() for i in reports[0]]) + "\n" + "=" * 70
            logger.info(msg)

            # Model Checkpointing
            if epoch != 0 and epoch % model_update_interval == 0:
                logger.info("Update latest model@Iteration {}".format(n_sentence_iter))
                save(agent, LATEST_MODEL) # Saves the latest model checkpoint at regular intervals.
                averaged_reward = memory.averaged_reward()
                if averaged_reward > max_avg_reward: # Compares the current average reward with the best reward.
                    max_avg_reward = averaged_reward # Updates the best model checkpoint if the current average reward is higher.
                    save(agent, BEST_REWARD_MODEL)
                    logger.info("Update best reward model@Iteration{}(Averaged Reward={:.5f})".format(n_sentence_iter, max_avg_reward))

            if epoch != 0 and epoch % opt.decay_interval == 0: # opt.decay_interval, default=10
                agent.apply_epsilon_decay()

    except KeyboardInterrupt:
        logger.info("Terminating process ... ")
        logger.info("done!")


def generate_episode_items(bertnlp, agent, sentences, min_cr=0.5, min_rr=0.5, add_prefix=True, k=10): # add_prefix: Whether to add prefixes during processing (default True).
    # Processes sentences to generate reinforcement learning episodes with compression and reconstruction metrics.
    def run_compression_reconstruction(items):

        # Extracts original sentences (orig_sents) and their corresponding labels (labels) from items.
        orig_sents, labels = zip(*[(i.sentence, i.labels) for i in items])
        orig_sents = copy.deepcopy(orig_sents)
        labels = copy.deepcopy(labels) # Uses deepcopy to avoid modifying the original objects during processing.

        # Compression and Reconstruction
        comp_sents, comp_topk, recon_sents, recon_topk = \
            bertnlp.apply_compression_and_reconstruction(orig_sents, labels, add_prefix, k) # NOTE the apply_compression_and_reconstruction() from bertnlp.py

        # Log-Likelihoods: 
        llhs = []
        for cs in comp_sents:
            llhs.append(bertnlp.mrf_log_prob(cs, tokenized=True)) # NOTE Computes log-likelihoods (llhs) for each compressed sentence using BERT’s masked random field (MRF) probabilities.

        # Similarities
        sims = bertnlp.predict_similarity(orig_sents, comp_sents, tokenized=True) # NOTE Predicts semantic similarities (sims) between the original and compressed sentences.

        # Updates each item: 
        for i, cs, csk, rs, rsl, llh, sim in zip(items, comp_sents, comp_topk,
                                                 recon_sents, recon_topk, llhs,
                                                 sims):
            i.set(cs, csk, rs, rsl, llh, sim)
        return items

    # Tokenization and Hidden States
    tokenized_sents, hiddens = bertnlp.apply_tokenize_and_get_hiddens(sentences, rm_pad=True) # NOTE tokenizes the input sentences and retrieves their hidden states (hiddens) from the BERT model. Removes padding tokens from the hidden states (rm_pad=True).

    # Process Sentences with Agent
    items = [agent.process(s, h, do_exploration=True) for (s, h) in zip(tokenized_sents, hiddens)] # Processes each tokenized sentence (s) and its hidden states (h) using the RL agent. # Enables exploration during processing (do_exploration=True).
    
    # Run Compression and Reconstruction
    items = [run_compression_reconstruction(i) for i in items]  # Applies the nested function run_compression_reconstruction to perform compression and reconstruction on each processed sentence.

    # Computes compression and reconstruction rewards for each episode item.
    train_items, report_items = zip(*[reward.reward(ei, min_cr=min_cr, min_rr=min_rr) for ei in items])
    return train_items, report_items # train_items: Used for training the RL agent. # report_items: Used for logging purpose only.


def load_file(file):
    with open(file, "r") as f:
        return [l.strip().split() for l in f]


def make_batcher(sentences, batch_size, do_shuf=True):
    N = len(sentences)
    idx = list(range(N))
    if do_shuf:
        idx = np.random.permutation(idx)
    for i in range(0, N, batch_size):
        cur_idx = idx[i:i + batch_size]
        yield [sentences[j] for j in cur_idx]


def step(loss, reward, opt, agent):
    opt.zero_grad()
    loss.backward()
    # nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)  # Apply gradient clipping
    torch.nn.utils.clip_grad_value_(agent.parameters(), 1)
    opt.step()
    return loss.detach().cpu().float().numpy(), reward.detach().cpu().float().numpy()


def save(model, save_path):
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser("Training script", add_help=True)

    args.add_argument("-t", help="Training Corpus", required=True, type=str)
    args.add_argument("-e", help="Epoch", default=10, type=int) # default=10000000000000
    args.add_argument("-min_cr", help="Minimum threshold for compression",
                      default=0.3, type=float)
    args.add_argument("-min_rr", help="Minimum threshold for reconstruction",
                      default=0.5, type=float)
    args.add_argument("-lr", help="Learning rate", default=0.001, type=float)
    args.add_argument("-nlayer", help="Number of layer in the agent MLP",
                      default=2, type=int)
    args.add_argument("-hdim",
                      help="Dimension of hidden layers in the agent MLP",
                      default=200, type=int)
    args.add_argument("-batch", help="Batch size", default=1, type=int)
    args.add_argument("-memory", help="Memory buffer size", default=2000,
                      type=int)
    args.add_argument("-decay_interval",
                      help="Epoch interval for e-greedy decay", default=10,
                      type=int)
    args.add_argument("-report_num", help="Number of report items", default=5,
                      type=int)
    args.add_argument("-save", help="Model directory", default="./model",
                      type=str)
    args.add_argument("-stopwords", help="List of stopwords", default=None,
                      type=str)
    # args.add_argument("-dropout", help="Dropout rate", default=0.1, type=float)

    opt = args.parse_args()
    # if os.path.exists(opt.save):
    #     print("{} already exists!!".format(opt.save))
    #     print("Are you sure to overwrite?[Yn]: ", end="")
    #     x = input()
    #     if x == "Y":
    #         shutil.rmtree(opt.save)
    #     else:
    #         sys.exit(1)
    # os.mkdir(opt.save)

    train(opt)

    # After training is completed, plot the curves
    plt.figure()
    #plt.plot(all_rewards, label='Reward')
    plt.plot(all_rewards_, label='Reward_')
    plt.xlabel('Training Iterations')
    plt.ylabel('Reward')
    plt.title('Learning Curves')
    plt.legend()

    plt.savefig('learning_curves.png')
    plt.show()

    




