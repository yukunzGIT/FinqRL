import torch
import torch.nn as nn
import torch.nn.functional as F # Contains functions like activation functions and loss functions.
import numpy as np
import copy # Used for creating deep or shallow copies of objects.
from collections import deque # A double-ended queue for efficient append and pop operations.


from nn import MLP, key_value_attention, DEVICE


# Util functions
def gen_uniform_action_score(act_num, length):
    # Generates a tensor with uniformly distributed scores for a given number of actions.
    # Ex: gen_uniform_action_score(3, 5)
    # Output: Tensor of shape (5, 3) where each row is [0.3333, 0.3333, 0.3333].    
    value = 1/act_num
    return torch.Tensor([value]).repeat(length, act_num)


def init_hiddens_and_scores(hiddens, scores): # ex: hiddens = torch.randn(5, 768)  # Hidden states for 5 samples, each of size 768.
    """ Generate initial representations """
    # Converts hidden states and scores into PyTorch tensors and moves them to the computation device.

    # If scores is not provided, generate a uniform action score using gen_uniform_action_score.
    if scores is None:
        scores = gen_uniform_action_score(3, hiddens.shape[0]) # act_num=3: Number of actions (e.g., Remove, Keep, Replace). # length=hiddens.shape[0]: The number of samples or data points for Hidden states.

    # Checks if hiddens is a NumPy array. If it is, convert it to a PyTorch tensor using torch.Tensor.
    if isinstance(hiddens, np.ndarray):
        hiddens = torch.Tensor(hiddens)

    # checks if scores is a NumPy array. If it is, converts it to a PyTorch tensor.
    if isinstance(scores, np.ndarray):
        scores = torch.Tensor(scores)

    # Moves both hiddens and scores tensors to the computation device (DEVICE). 
    return hiddens.to(DEVICE), scores.to(DEVICE) # This ensures compatibility for GPU acceleration or CPU usage as configured.


class LocalGlobalEncoder(nn.Module):
    # Encodes input states (hidden representations) into local and global representations by incorporating biases and attention mechanisms.
    # Ex: 
    # encoder = LocalGlobalEncoder(action_size=3)
    # hiddens = torch.randn(5, 768)  # Hidden states for 5 samples.
    # labels = torch.tensor([0, 1, 2, 0, 1])  # Action labels.
    # predicted = torch.tensor([1, 0, 1, 1, 0])  # Predicted actions.
    # output = encoder(hiddens, labels=l abels, predicted=predicted)
    # print(output.shape)  # Output: torch.Size([5, 1536])
    
    def __init__(self, action_size=3, size=768): # The size of the hidden state vectors (default: 768, as commonly used in transformer-based models)
        super().__init__()
        self.size = size * 2
        self.bias1 = nn.Embedding(action_size, 1) # Embedding layer with action_size entries. Each entry maps to a single scalar (shape: (action_size, 1)). # with num_embeddings=3, embedding_dim=1)
        self.bias1.weight.data.uniform_(-1.0, 1.0) # Initializes embedding weights to random values uniformly sampled between -1.0 and 1.0.
        self.bias2 = nn.Embedding(2, 1) # Embedding layer with 2 entries. Used for incorporating predicted or actual labels.
        self.bias2.weight.data.uniform_(-1.0, 1.0) # same initialize
        self.has_loss = False # A placeholder attribute for tracking loss computation in derived classes.

    def forward(self, hiddens, scores=None, **kwargs): # kwargs: Additional arguments, such as: 1. labels: Ground truth labels for the actions. 2. predicted: Predicted labels (e.g., from another model).
        hiddens, scores = init_hiddens_and_scores(hiddens, scores) # Ensures that hiddens and scores are in the correct format (PyTorch tensors on the appropriate device).

        action_bias = self.bias1(kwargs.get("labels").to(DEVICE)) # Retrieves the embedding bias corresponding to the labels argument.  kwargs.get("labels"): Extracts labels from kwargs.
        acted_bias = self.bias2(kwargs.get("predicted").to(DEVICE)) # uses the predicted labels to retrieve biases from bias2.
        local_e = hiddens + action_bias + acted_bias # Compute Local Representations. Shapes: hiddens: (batch_size, hidden_size); action_bias and acted_bias: (batch_size, 1); Broadcasting is applied to sum these tensors element-wise.
        # local_e shape remains (batch_size, hidden_size).

        global_e = key_value_attention(local_e, local_e) # Applies an attention mechanism fucntion from nn.py to derive global representations.
        # global_e shape (batch_size, hidden_size).

        return torch.cat([local_e, global_e], dim=1) # Concatenate Representations along the feature dimension (dim=1) # Output: (batch_size, hidden_size * 2).


class EditorialAgent(nn.Module):
    # defines the EditorialAgent

    # possible 3 actions
    REMOVE, KEEP, REPLACE = 0, 1, 2

    def __init__(self, layer_num=2, hidden_dim=200): # By default, number of layers in the MLP model = 2 and Dimensionality of the hidden layers = 200.
        super().__init__() # Calls the constructor of the parent class (torch.nn.Module) to ensure proper initialization.
        self.action_size = 3 # number of possible actions 
        self.state_size = 768 * 2 # the size of the input state from the previous local+ global concatenated embeddings

        # Buffer for up to 2000 experiences.
        self.memory = deque(maxlen=2000) # A double-ended queue as a replay buffer to store past experiences (state, action, reward, next state, done=True/False). This is essential for the agent to learn from past interactions.
        
        self.gamma = 0.95 # Discount factor for future rewards.

        # Parameters for epsilon-greedy exploration:
        self.epsilon = 0.900 # Initial exploration rate.
        self.epsilon_min = 0.03 # Minimum probability of exploration.
        self.epsilon_decay = 0.995 #  Rate at which exploration decreases over time.

        self.temperature = 768 # A scaling factor, used in temperature scaling for logits or embeddings. help control the sharpness of predictions.

        # Parameters for epsilon-greedy selection:
        self.selection_epsilon = 0.900
        self.selection_epsilon_min = 0.03
        self.selection_epsilon_decay = 0.995

        # A neural network that processes input embeddings (local and global representations) and outputs encoded features
        self.encoder = LocalGlobalEncoder()

        # A fully connected neural network that predicts action scores for each input state.
        self.mlp = MLP(in_dim=self.state_size, hid_dim=hidden_dim,
                       out_dim=self.action_size, layer_num=layer_num) # in_dim: Input size, matching state_size 768 * 2.  hid_dim = 200 by default. out_dim: Output size, matching action_size of 3.
        # Ex: If state_size = 1536, hidden_dim = 200, and layer_num = 2, the MLP structure could look like:  Linear(1536 -> 200) -> ReLU -> Linear(200 -> 3)
        self.to(DEVICE) # Moves the model to the specified computation device (GPU or CPU).

    def predict(self, hidden, prev_score, do_exploration=True, **enc_kwargs):
        # Predicts the action scores for the given state. It decides between exploration (random scores) and exploitation (MLP-predicted scores) based on epsilon.
        # hidden: The local representation of the input state (e.g., embeddings for words or sentences). 
        # prev_score: Previous action scores, representing the initial probabilities for each action (e.g., REMOVE, KEEP, REPLACE).
        # do_exploration: If True, the agent may explore by generating random scores.
        # **enc_kwargs: Additional keyword arguments for the encoder (e.g., labels, temperature, etc.).

        # Combine the hidden state and prev_score into a compact feature representation using the encoder.
        inp = self.encoder(hidden, prev_score, **enc_kwargs)

        # Decide whether to explore or exploit based on a random number and the epsilon parameter.
        if np.random.rand() <= self.epsilon and do_exploration: # np.random.rand() generates a random number between 0 and 1.
            # Returns random scores for each action.
            return torch.Tensor(np.random.rand(inp.shape[0], self.action_size)).to(DEVICE) # inp.shape[0]: Number of samples. self.action_size: Number of possible actions=3. Converts the random scores to a PyTorch tensor and moves it to the specified computation device (CPU or GPU)
            # Ex: random_scores = torch.Tensor([  # 2 samples and 3 actions:
                                # [0.67, 0.22, 0.11], ### Scores for REMOVE, KEEP, REPLACE
                                # [0.45, 0.25, 0.30]
                                # ]).to(DEVICE)
        else: # Exploitation. aka if not exploring, the agent uses the MLP model to predict action scores.
            scores = self.mlp(inp)
            return scores  # Ex for one sample: scores = [0.75, 0.15, 0.10]  # REMOVE, KEEP, REPLACE

    def replay(self, items): # items: A batch of experiences, where each experience is an instance of the Item class.
        # performs experience replay, allowing the agent to learn from past experiences. Updates the agent’s model using a batch of past experiences stored in the replay memory.
        # Updates the model by performing experience replay. Uses a batch of past experiences (items) to compute the loss and backpropagate.
        
        # Extract Relevant Data from Items
        # outputs: Predicted action scores for the current state. Uses the predict method without exploration (do_exploration=False). Ex: outputs = [torch.tensor([0.7, 0.2, 0.1])]
        # next_outputs: Maximum predicted score for the next state. default is None. # Ex: next_outputs = [0.8]
        # pred_idxs: Predicted action indices. # Ex: pred_idxs = [0]
        outputs, next_outputs, pred_idxs, actions, rewards, is_terminals = zip(*[(
            self.predict(i.hidden, i.prev_score, do_exploration=False,
                         **{"labels": torch.tensor(i.prev_labels),
                            "predicted": torch.tensor(i.prev_is_predicted),
                            "temperature": self.temperature}),
            i.next_max_score,
            i.pred_idx,
            i.action,
            i.reward,
            i.is_terminal
        ) for i in items]) # Ex: rewards = [1.0]; is_terminals = [False]

        # Select the predicted value corresponding to the predicted action index (pred_idx) from outputs.
        values = [o[i] for (o, i) in zip(outputs, pred_idxs)] # ex: values = [0.7, 0.3] when outputs = [torch.tensor([0.7, 0.2, 0.1]), torch.tensor([0.5, 0.3, 0.2])] AND pred_idxs = [0, 1]

        # Prepare Tensor Data. Converts extracted data (rewards, actions, next_outputs) into PyTorch tensors and moves them to the computation device (CPU or GPU).
        values = torch.stack(values).to(DEVICE)
        rewards = torch.Tensor(rewards).to(DEVICE)
        actions = torch.tensor(np.stack(actions)).to(DEVICE)
        next_max_scores = torch.Tensor(np.stack(next_outputs)).to(DEVICE)

        # Retrieves the predicted Q-values corresponding to the actions taken.
        action_values = values.gather(1, actions.unsqueeze(1)).squeeze(1) # Ex: Q-values for actions 0 and 1: action_values = torch.tensor([0.7, 0.3]) when values = torch.tensor([[0.7, 0.2, 0.1], [0.5, 0.3, 0.2]]) AND actions = torch.tensor([0, 1])

        # Compute Target Q-values based on Bellman equation: 
        next_action_values = next_max_scores
        expected_action_values = rewards + self.gamma * next_action_values

        # Compute Loss and Backpropagate
        loss = F.smooth_l1_loss(action_values, expected_action_values)
        # loss = F.mse_loss(action_values, expected_action_values)
        # loss.backward()

        return loss, rewards.mean()

    def replay_encoder(self, items):
        # Calculates the loss for the encoder using a batch of experiences.
        encoder_loss = torch.stack([self.encoder.loss(i.hidden, i.prev_score) for i in items]).sum() # key is the self.encoder.loss() from LocalGlobalEncoder()
        return encoder_loss.sum() # Stacks individual losses into a tensor and sums them.

    def apply_epsilon_decay(self):
        # update exploration rate. aka reduces the exploration rate (epsilon) over time based on decay factors.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.selection_epsilon_min < self.selection_epsilon:
            self.selection_epsilon *= self.selection_epsilon_decay

    def prepare(self, hidden):
        # Initializes prediction-related tensors for a batch of inputs.
        length = len(hidden) # Determine Batch Length
        is_predicted = torch.tensor([0] * length) # Initialize is_predicted to all 0, aka not predicted.
        labels = torch.tensor([1] * length)  # default action label is 'Keep' for all
        
        # Create One-Hot Labels:
        labels_onehot = torch.FloatTensor(length, 3)
        labels_onehot.zero_()
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1) # Ex: labels_onehot = torch.tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]])

        return is_predicted, labels, labels_onehot

    def get_highest_entropy_idx(self, scores, is_predicted):
        # Finds the index with the highest entropy from the scores tensor, ignoring already predicted indices.
        p = scores / scores.sum(dim=1).unsqueeze(1) # Converts scores into probabilities by normalizing across actions.
        h = (p * p.log()).sum(dim=1) * -1 # Computes entropy based on the formula
        h[is_predicted == 1] = -np.Inf # Ignore Predicted Indices

        # Find Index with Highest Entropy:
        sorted_idx = torch.argsort(h)
        max_h = sorted_idx[-1]
        return max_h

    def get_highest_value_idx(self, scores, is_predicted):
        # Finds the index of the highest value in the scores tensor, ignoring indices that are already predicted (is_predicted).
        maxs = scores.max(dim=1)[0] # Computes the maximum score for each row in the scores tensor.
        maxs[is_predicted == 1] = -np.Inf # Ignore Already Predicted Indices
        sorted_idx = torch.argsort(maxs)
        max_v = sorted_idx[-1]
        return max_v

    def get_ordered_idx_and_score(self, scores, is_predicted):
        # Returns all scores in a flattened view, ordered by their value, along with their row and column indices.

        # Clones and flattens the scores tensor after masking predicted values.
        scores_flat = scores.clone()
        scores_flat[is_predicted == 1] = -np.Inf
        scores_flat = scores_flat.view(-1)

        # Sorts the flattened scores in descending order.
        sorted_idxs = torch.argsort(scores_flat, descending=True)

        # Retrieve Row and Column Indices:
        row_idxs = sorted_idxs / 3
        col_idxs = sorted_idxs % 3
        return [{"idx": r, "action": c, "score": s} for (r, c, s) in zip(row_idxs, col_idxs, scores_flat[sorted_idxs])] 
        # Ex: [{"idx": 1, "action": 1, "score": 0.8}, {"idx": 0, "action": 0, "score": 0.5}, ...] 

    def get_action_at(self, scores, idx):
        # Returns the action with the highest score at a specific index, along with its one-hot encoding.
        action_at = torch.argmax(scores[idx]) # Find the Highest Scoring Action

        # Constructs a one-hot vector where the highest scoring action is marked as 1.
        one_hot = torch.zeros(3)
        one_hot[action_at] = 1
        return action_at, one_hot # Ex: action_at = 1; one_hot = torch.tensor([0, 1, 0])

    def get_next_prev_score(self, scores, is_predicted, onehot_labels):
        # Adjusts the scores based on whether they have been predicted, replacing predicted scores with their corresponding one-hot labels.
        scores_ = scores.clone() # Creates a copy of the scores tensor to avoid modifying the original.
        
        # Replace Predicted Scores:
        for idx, i in enumerate(is_predicted):  # Loops through each row in scores:
            if i == 0:
                scores_[idx] = scores_[idx] # f not predicted (i == 0), keeps the original score.
            else:
                scores_[idx] = onehot_labels[idx]  # If predicted (i == 1), replaces it with the one-hot label.
        return scores_ # Ex: scores_ = torch.tensor([[0.5, 0.2, 0.3], [1.0, 0.0, 0.0]]) when: scores = torch.tensor([[0.5, 0.2, 0.3], [0.1, 0.8, 0.6]]); is_predicted = torch.tensor([0, 1]); onehot_labels = torch.tensor([[0, 1, 0], [1, 0, 0]])

    def npfy(self, x):
        # Converts a PyTorch tensor to a NumPy array, or returns the input directly if it’s not a tensor.
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

    def process(self, sentence, hidden, do_exploration=True): # sentence: The input sentence to process. hidden: The encoded representation (hidden states) of the sentence.
        # Processes a sentence step-by-step to predict actions for each part of the input. It constructs items for each step, useful for reinforcement learning or inference.
        with torch.no_grad(): # Disables gradient calculation to save memory and computation since this method doesn't update the model during execution.
            # Initialization, aka, call the prepare method to initialize:
            is_predicted, labels, labels_onehot = self.prepare(hidden)
            items, labels_per_step = [], []
            prev_score, next_score, prev_labels, prev_is_predicted = None, None, None, None

            for i in range(len(sentence)): # Iterates over each token in the sentence.
                # set previous score, aka updates the previous state variables for the current step.
                prev_score = next_score
                prev_labels = labels.clone()
                prev_is_predicted = is_predicted.clone()

                # compute/predict current scores
                # Calls the predict method to compute action scores for the current step:
                cur_scores = self.predict(hidden, prev_score, do_exploration, **{"labels": prev_labels, "predicted": prev_is_predicted, "temperature": self.temperature}) # temperature: A scaling factor.
                
                # Update Items with Next Max Score
                if i != 0: # Skip First Step since the first step doesn’t have a previous item to update.
                    # For the last item in items, sets:
                    items[-1].next_max_score = self.npfy(torch.max(cur_scores, dim=1)[0][is_predicted != 1].max()) # torch.max(cur_scores, dim=1)[0]: Finds the maximum scores for the current step. is_predicted != 1: Filters out already predicted indices. 
                    # NOTE self.npfy() converts the result to NumPy format.

                # get target index to predict next, aka decides the next index to predict based on exploration or exploitation:
                if do_exploration and np.random.rand() <= self.selection_epsilon:
                    target_idx = self.get_highest_entropy_idx(cur_scores, is_predicted) # Exploration: Chooses randomly based on entropy.
                else:
                    target_idx = self.get_highest_value_idx(cur_scores, is_predicted) # Exploitation: Chooses based on the highest value.

                # update predicted indices
                is_predicted[target_idx] = 1

                # get predicted action at the index in scalar and one-hot formats
                action, action_onehot = self.get_action_at(cur_scores, target_idx)

                # update labels
                labels[target_idx] = action
                labels_onehot[target_idx] = action_onehot
                labels_per_step.append(labels.clone())

                # prepare prev score in next step
                next_score = self.get_next_prev_score(cur_scores, is_predicted, labels_onehot)

                # prepare an episode tuple
                item_args = map(self.npfy, [sentence, hidden, prev_score, cur_scores, next_score, target_idx, action, labels.clone(), prev_labels, prev_is_predicted, False])
                items.append(Item(*item_args))

            # assert all([i == 1 for i in is_predicted])
            items[-1].is_terminal = True
            return items # returns a list of Item objects,


class Item:
    # Represents an individual experience or instance used during training in a reinforcement learning task. Items hold information about states, actions, rewards, and transitions.
    # Ex: item = Item(
    #   sentence="The stock market is volatile.",
    #   hidden=torch.randn(1, 768),
    #   prev_score=torch.tensor([0.3, 0.4, 0.3]),
    #   cur_score=torch.tensor([0.7, 0.2, 0.1]),
    #   next_score=torch.tensor([0.8, 0.15, 0.05]),
    #   pred_idx=0,
    #   action=0,
    #   labels=[1],
    #   prev_labels=[0],
    #   prev_is_predicted=False,
    #   is_terminal=False)

    def __init__(self, sentence, hidden, prev_score, cur_score, next_score,
                 pred_idx, action, labels, prev_labels, prev_is_predicted, is_terminal):
        self.sentence = sentence # The sentence or input data associated with this item.
        self.hidden = hidden # The state representation (e.g., embeddings) for the input.
        self.prev_score = prev_score # The action scores from the previous step.
        self.cur_score = cur_score # The current scores for actions.
        self.next_score = next_score # The predicted scores for the next state.
        self.next_max_score = None  # for fixed target-q, to be fixed
        self.pred_idx = pred_idx # The index of the predicted action.
        self.action = action # The action taken.
        self.labels = labels # Labels associated with the input.
        self.prev_labels = prev_labels # The labels from the previous step.
        self.prev_is_predicted = prev_is_predicted  # Boolean indicating if the previous action was predicted. our tracker u's from the paper.
        self.is_terminal = is_terminal # Boolean indicating if this is a terminal state.
        self.reward = None
        self.cr = None
        self.rr = None
        self.crr = None

        self.comp_sent = None
        self.comp_sent_topk = None
        self.recon_sent = None
        self.recon_sent_topk = None
        self.comp_llh = None
        self.comp_sim = None

    def set(self, cs, csk, rs, rsk, llh, sim):
        # item.set(
        #   cs="Stock market volatile.",
        #   csk=["Market is volatile.", "Stock volatile."],
        #   rs="The stock market is volatile.",
        #   rsk=["Stock market is volatile.", "The market is volatile."],
        #   llh=-2.3,
        #   sim=0.85)
        #   Updates the compressed and reconstructed representations for the sentence.
        self.comp_sent = cs # Compressed sentence.
        self.comp_sent_topk = list(map(set, csk)) # Top-k compressed sentences.
        self.recon_sent = rs # Reconstructed sentence.
        self.recon_sent_topk = list(map(set, rsk)) # Top-k reconstructed sentences.
        self.comp_llh = llh # Log-likelihood of the compressed sentence.
        self.comp_sim = sim  # Similarity score.

    def report(self):
        # The report method provides a human-readable string summarizing the attributes of an Item instance, which is useful for debugging or logging.
        
        # Extracts the scores for the predicted index (self.pred_idx) from self.cur_score.
        cur_score = "[{:+06.2f}, {:+06.2f}, {:+06.2f}]".format(*list(self.cur_score[self.pred_idx])) # cur_score = "[+00.70, +00.20, +00.10]" from self.cur_score = torch.tensor([0.7, 0.2, 0.1])

        return "cr={:.2f}/rr={:.2f}/crr={:.2f}/llh={:.2f}/sim={:.2f}/act={}({:10}, {}) -> {:+.2f} : {}".format(
            self.cr, self.rr, self.cr + self.rr,
            self.comp_llh, self.comp_sim, self.action, self.sentence[self.pred_idx], cur_score,
            self.reward, " ".join(self.comp_sent)) # Ex: "cr=0.50/rr=0.40/crr=0.90/llh=-2.30/sim=0.85/act=REMOVE(stock     , [+00.70, +00.20, +00.10]) -> +1.00 : Stock volatile"

 
    def is_bad(self):
        # The is_bad method determines whether the current Item instance is considered "bad." 
        # This is used for filtering out undesirable experiences during training.
        if self.reward == -1 and np.random.rand() < 0.5:
            return True
        return False


class Memory:
    # replay buffer for storing and managing experiences in reinforcement learning. This implementation uses a double-ended queue (deque) to efficiently handle storage and retrieval of experiences. 
    def __init__(self, buffer_size=2000):
        self.memory = deque() # Initializes an empty double-ended queue for storing experiences. deque is efficient for appending and removing elements from both ends.
        self.buffer_size = buffer_size # Sets the maximum size of the buffer. If the buffer exceeds this size, the oldest experience is removed.
        self.mappend = self.memory.append # Adds an item to the end of the queue.
        self.mpopleft = self.memory.popleft # Removes an item from the front of the queue.

    def __call__(self, *args, **kwargs):
        # Allows the Memory object to be called directly to retrieve the entire buffer.
        # Ex usage: all_experiences = memory()
        return self.memory # Returns the deque object containing all stored experiences.

    def size(self):
        # Ex usage: current_size = memory.size()
        return len(self.memory) # Returns the current number of stored experiences in the buffer.

    def append(self, x): # Adds a new experience (x) to the memory. If the buffer is full, removes the oldest experience before appending.
        # Ex usage: experience = {"state": ..., "action": ..., "reward": ...}
        #           memory.append(experience)
        if len(self.memory) == self.buffer_size:
            self.mpopleft()
        self.mappend(x)

    def get(self):
        # Ex usage: all_items = memory.get()
        return self.memory

    def sample(self, n):
        # Randomly samples n experiences from the memory for training.
        # Ex usage: batch = memory.sample(32)
        idxs = np.random.randint(0, self.size(), n)
        return [self.memory[i] for i in idxs]

    def averaged_reward(self):
        # Calculates the average reward of all stored experiences in the buffer.
        # Ex usage: avg_reward = memory.averaged_reward()
        return sum([i.reward for i in self.memory])/self.size()

