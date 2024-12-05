import semantic_text_similarity.models as smodel # For calculating sentence similarity.
from dqn import EditorialAgent
import torch
from bert_score import score
from pytorch_transformers import BertTokenizer, BertConfig, BertForMaskedLM

PAD_ID = 0
MASK_ID = 103
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
C = "[CLS]"
S = "[SEP]"
M = "[MASK]"
MAX_LENGTH = 512


config, tokenizer, model, sim_model = None, None, None, None


def init(maxlen=512): # maximum input length of 512 tokens.
    # Initializes the BERT model, tokenizer, and configurations, setting up global variables for processing.
    
    # Set Global Variables:
    global config, tokenizer, model, sim_model, MAX_LENGTH # global variables

    # Set Maximum Length:
    MAX_LENGTH = maxlen

    # Load Configurations:
    bert_model_name = 'bert-base-uncased'
    config = BertConfig.from_pretrained(bert_model_name)
    config.output_hidden_states = True # ensures output hidden states.

    # Load Tokenizer and Model: Loads the BERT tokenizer and masked language model.
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    model = BertForMaskedLM.from_pretrained(bert_model_name, config=config)

    # Moves the model to the specified device (e.g., GPU) and sets it to evaluation mode.
    model.to(DEVICE)
    model.eval()

    # Initialize Similarity Model:
    # sim_model = smodel.WebBertSimilarity(device=DEVICE) # Loads a similarity model.


def put_cls_sep(sentences, tokenized=False):
    # Adds [CLS] and [SEP] tokens to sentences for BERT processing.
    if tokenized: # If the input is tokenized (split into tokens), prepends [CLS] and appends [SEP].
        return [[C] + s + [S] for s in sentences]
    else:
        return ["{} {} {}".format(C, s, S) for s in sentences] # Adds [CLS] and [SEP] as strings for non-tokenized input.
        # Ex: ["[CLS] The stock market is volatile. [SEP]", "[CLS] Investors are cautious. [SEP]"]


def pad(token_id_sequences, pad_id=PAD_ID):
    # Pads token ID sequences to the same length, respecting MAX_LENGTH.
    maxlen = max([len(s) for s in token_id_sequences]) # Find Maximum Sequence Length:

    if maxlen > MAX_LENGTH:
        maxlen = MAX_LENGTH # Clamp Length to MAX_LENGTH:

    rtn = [tis + [pad_id] * (maxlen - len(tis)) for tis in token_id_sequences] # Appends padding tokens (pad_id) to shorter sequences.
    return torch.tensor(rtn).to(DEVICE) # Convert to Tensor.  Ex: tensor([[101, 2009, 2003, [pad], [pad]], [101, 2054, [pad], [pad]. [pad]]])


def encode_pad(sentences, add_cls_sep=True, tokenized=False):
    # Encodes sentences into token IDs, adds [CLS] and [SEP], and pads them.
    if add_cls_sep:
        sentences = put_cls_sep(sentences, tokenized) # Add [CLS] and [SEP]:

    if tokenized:
        token_ids = list(map(tokenizer.convert_tokens_to_ids, sentences)) # Convert to Token IDs:
    else:
        token_ids = list(map(tokenizer.encode, sentences))
    padded_token_ids = pad(token_ids) # Pad Token IDs:
    return padded_token_ids.to(DEVICE)


def tokenize_and_put_ids_one(sentence):
    # Processes multiple sentences, tokenizing and converting to token IDs.
    tokenized_sent = tokenizer.tokenize(sentence) # Tokenize Each Sentence:
    tokenized_sent_id = list(map(tokenizer.convert_tokens_to_ids, tokenized_sent))
    return tokenized_sent, tokenized_sent_id # Ex: [(["The", "market", "is", "volatile", "."], [101, 1996, 3006, 2003, 10132, 102]),
                                                #       (["Invest", "cautiously", "."], [101, 5398, 4678, 102])]


def tokenize_and_put_ids(sentences):
    # Tokenizes multiple sentences and converts them into token IDs. 
    return list(zip(*map(tokenize_and_put_ids_one, sentences))) # Calls the helper function tokenize_and_put_ids_one for each sentence.


def run_bert(inputs, encoded=False, add_cls_sep=False, tokenized=True):
    # Runs a BERT model on the input data, optionally encoding and adding [CLS] and [SEP] tokens.
    with torch.no_grad(): # Ensures that gradients are not computed during the forward pass to save memory.
        if not encoded: # Prepares inputs for the BERT model by tokenizing and padding.
            inputs = encode_pad(inputs, add_cls_sep=add_cls_sep, tokenized=tokenized)
        logits, hiddens = model(inputs) # Processes the inputs through the BERT model.
        hiddens = hiddens[-1]  # last layer, retrieves the hidden states from the final layer.
        return logits, hiddens # logits: Predictions for masked tokens.  hiddens: Hidden states for each token.


def apply_tokenize_and_get_hiddens(sentences, rm_pad=True):
    # Tokenizes sentences, runs them through BERT, and retrieves hidden states.
    tokenized_sentences = list(map(tokenizer.convert_ids_to_tokens, map(tokenizer.convert_tokens_to_ids, sentences))) # Tokenize Sentences
    _, hiddens = run_bert(tokenized_sentences, add_cls_sep=True) # Runs the tokenized sentences through the BERT model to retrieve hidden states.

    hiddens = hiddens[:, 1:-1, :] # Remove Special Tokens, aka strips [CLS] and [SEP] tokens from the hidden states.
    if rm_pad: # Trims hidden states to match the lengths of the original sentences.
        hiddens = [h[:len(s), :] for (s, h) in zip(tokenized_sentences, hiddens)] # Remove Padding (Optional):
    return tokenized_sentences, hiddens


def convert_to_masked_input(tokens, labels, comp_or_recon="comp", prefix_tokens=None):
    # Converts tokens into a masked format based on labels and compression/reconstruction mode.
    masked, non_masked = [], []
    # If the mode is reconstruction (recon):
    if comp_or_recon == "recon": # Processes tokens based on their label (REMOVE, KEEP, REPLACE).
        for (t, l) in zip(tokens, labels):
            non_masked.append(t) # Keeps a copy of unmasked tokens in non_masked.
            if l == EditorialAgent.REMOVE:
                masked.append("[MASK]")
            elif l == EditorialAgent.KEEP:
                masked.append(t)
            elif l == EditorialAgent.REPLACE:
                # predict tokens
                masked.append("[MASK]")
            else:
                assert False

    elif comp_or_recon == "comp": # If the mode is compression (comp):
        for (t, l) in zip(tokens, labels):
            if l == EditorialAgent.REMOVE: # Omits tokens labeled as REMOVE.
                continue
            elif l == EditorialAgent.KEEP:
                non_masked.append(t)
                masked.append(t)
            elif l == EditorialAgent.REPLACE:
                non_masked.append(t)
                masked.append("[MASK]")
            else:
                assert False
    else:
        assert False
    if prefix_tokens is not None:
        return prefix_tokens + masked, prefix_tokens + non_masked
    else:
        return masked, non_masked


def remove_prefix(inputs, prefix):
    # Iterates over input and prefix pairs, removing the prefix (by slicing the prefix length) from each input.
    return [i[len(p):] for (i, p) in zip(inputs, prefix)] # Ex: return ["hello", "world"] when inputs = ["prefix_hello", "prefix_world"]; prefix = ["prefix_", "prefix_"]


def apply_compression(sentences, labels, add_prefix=True, k=10):
    # applies compression by masking tokens labeled for removal and predicting replacements using an iterative mask-prediction mechanism.
    comp_masks, comp_nomasks = [], [] # Initializes lists to store masked and non-masked versions of the compressed sentences.
    for sent, label in zip(sentences, labels): # Iterates through the sentences and their corresponding labels:
        comp_mask, comp_nomask = \
            convert_to_masked_input(sent, label,
                                    comp_or_recon="comp",
                                    prefix_tokens=(sent if add_prefix else None)) # If add_prefix is True, the original sentence is used as a prefix.
        comp_masks.append(comp_mask) # Masked version of the sentence with tokens replaced according to labels.
        comp_nomasks.append(comp_nomask) # Unmasked tokens.

    comp_sents, topk_pred = iterative_mask_prediction(comp_masks, comp_nomasks, k) # NOTE Calls the below iterative_mask_prediction() to predict the best tokens to replace the masks, up to k candidates.

    if add_prefix: # If a prefix was added earlier, it is now removed using remove_prefix.
        comp_sents = remove_prefix(comp_sents, sentences)
        topk_pred = remove_prefix(topk_pred, sentences)
        assert len(comp_sents) == len(topk_pred)
    return comp_sents, topk_pred # Returns the compressed sentences (comp_sents) and top-k predictions for replacement tokens.


def apply_reconstruction(sentences, labels, comp_sents, add_prefix=True, k=10):
    # reconstructs the original sentence by predicting masked tokens in the compressed sentence.
    recon_masks = [] # Initializes a list to store reconstruction masks.
    for sent, label, comp in zip(sentences, labels, comp_sents): # NOTE Iterates through sentences, labels, and compressed sentences:
        # Creates a recon_mask using convert_to_masked_input with comp_or_recon="recon"
        recon_mask, _ = \
            convert_to_masked_input(sent, label,
                                    comp_or_recon="recon",
                                    prefix_tokens=(comp if add_prefix else None)) # If add_prefix is True, the compressed sentence is used as a prefix.
        recon_masks.append(recon_mask)

    recon_sents, topk_pred = iterative_mask_prediction(recon_masks, None, k=k) # Predicts the tokens to replace masks in the reconstruction process.

    if add_prefix: # Removes any prefixes added earlier.
        recon_sents = remove_prefix(recon_sents, comp_sents)
        topk_pred = remove_prefix(topk_pred, comp_sents)
        assert len(recon_sents) == len(topk_pred)
    return recon_sents, topk_pred # Returns reconstructed sentences and top-k predictions.


def apply_compression_and_reconstruction(sentences,labels, add_prefix=True, k=10):
    # combines the compression and reconstruction processes, applying compression first and then reconstructing the original sentence.
    comp_sents, comp_topk_pred = apply_compression(sentences, labels, add_prefix, k)

    recon_sents, recon_topk_pred = apply_reconstruction(sentences, labels, comp_sents, add_prefix, k)

    return comp_sents, comp_topk_pred, recon_sents, recon_topk_pred # Returns compressed sentences, top-k predictions for compression, reconstructed sentences, and top-k predictions for reconstruction.


def mrf_log_prob(sentence, tokenized=False):
    # computes the log-probability of a sentence using a Masked Random Field (MRF) approach with the BERT model.
    if len(sentence) == 0:
        return 0 # If the input sentence is empty, it immediately returns a log-probability of 0.
    
    # Prepares the target sentence for processing.
    if not tokenized: # If tokenized is False, the sentence is tokenized and converted into IDs.
        tar_sentence, _ = tokenize_and_put_ids([sentence])
        tar_sentence = sentence * len(sentence[0])
    else:
        tar_sentence = [sentence] * len(sentence) # The target sentence is duplicated n times (n = len(sentence)), one for each word position in the sentence.

    bert_input = encode_pad(tar_sentence, tokenized=True, add_cls_sep=True) # Encodes and pads the tokenized sentence for BERT processing, adding [CLS] and [SEP] tokens.

    # # Extracts the diagonal tokens corresponding to the original sentence words.
    diag_ids = bert_input[:, 1:-1].diag() 
    n = bert_input.shape[0]
    bert_input[range(n), range(1, n+1)] = 103 # Masks one word at a time using the MASK_ID (103).

    # Runs the masked inputs through the BERT model to get logits, then applies log_softmax to compute probabilities distribution. 
    logits, _ = run_bert(bert_input, encoded=True)
    logits = torch.log_softmax(logits, dim=2)

    # Computes the log-probabilities for the masked tokens.
    scores = torch.index_select(logits[:, 1:-1], 2, diag_ids)[:, range(n), range(n)].diag()
    return scores.mean().exp() # Returns the average log-probability (converted back to probability using torch.exp).


def predict_similarity(sents1, sents2, tokenized=True):
    # calculates the similarity between pairs of sentences using a preloaded similarity model.
    if tokenized: # If the sentences are tokenized, joins the tokens into complete sentences
        sents1 = [' '.join(s) for s in sents1]
        sents2 = [' '.join(s) for s in sents2]

    precision, recall, f1 = score(sents1, sents2, lang="en", verbose=False)
    scores = f1
    # scores = sim_model.predict(list(zip(sents1, sents2))) # Predicts similarity scores between each pair of sentences (sents1 and sents2) using the similarity model
    return scores  # Normalizes the score by dividing by 5 (the maximum similarity score).


def iterative_mask_prediction(masked_sentences, non_masked_sentences=None, k=10): # masked_sentences: List of sentences with masked tokens. non_masked_sentences: List of unmasked sentences (optional, used for compression tasks). k: Number of top predictions to return for each masked token.
    # predicts the most probable tokens for masked positions in sentences iteratively, up to k candidates.
    is_compression = non_masked_sentences is not None # Determines if the task is compression (when non_masked_sentences is provided).
    with torch.no_grad(): # Disables gradient computation, as this function is used for inference only.
        inputs = encode_pad(masked_sentences, tokenized=True) # Encodes and pads the masked sentences. This adds [CLS] and [SEP] tokens and ensures the input has consistent dimensions.
        if is_compression: # If compression is enabled, verifies that the shapes of inputs and non_masked_inputs match.
            non_masked_inputs = encode_pad(non_masked_sentences, tokenized=True)
            assert inputs.shape == non_masked_inputs.shape
        
        # Prepare for Top-K Predictions:
        topk_predictions = inputs.repeat([k, 1, 1]).transpose(0, 1).transpose(1, 2) # Duplicates the input k times for predicting the top k candidates for each masked position.
        is_mask = (inputs == MASK_ID).float() # Identifies positions of masked tokens in the input (MASK_ID = 103 for BERT).

        # Iterative Prediction Loop:
        while True:
            logits, _ = run_bert(inputs, encoded=True) # Runs the masked input through the BERT model to get logits (probabilities for each token).
            logits = torch.exp(logits) # Converts logits to probabilities using torch.exp

            # Find Most Probable Masked Tokens:
            max_values = torch.max(logits, dim=2).values * is_mask # Extracts the maximum probability (max_values) for each token position, considering only masked positions (is_mask).
            target_idx = torch.argmax(max_values, dim=1).unsqueeze(1) # Identifies the indices (target_idx) of the most probable tokens for replacement.

            target_index_scores = torch.stack([mat[idx] for (idx, mat) in zip(target_idx, logits)]) # Collects the scores for the predicted tokens at the identified indices (target_idx).

            #  Adjusts the predicted scores for compression tasks:
            if is_compression: # 
                target_index_scores = target_index_scores.squeeze(1) # squeeze(1): Removes the unnecessary singleton dimension.
                non_masked_orig_token_ids = non_masked_inputs.gather(1, target_idx) # gather: Extracts original token IDs from non_masked_inputs at the predicted indices (target_idx).
                target_index_scores.scatter_(1, non_masked_orig_token_ids, torch.zeros_like(non_masked_orig_token_ids).float()) # Sets the scores for original tokens (from non_masked_inputs) to zero, ensuring the model does not re-predict tokens already present in the unmasked input.
                target_index_scores = target_index_scores.unsqueeze(1) # Restores the original dimensionality for further processing.

            pred_token_ids = torch.argmax(target_index_scores, dim=2) # argmax: Extracts the ID of the most probable token for each masked position.
            pred_topk_token_ids = torch.topk(target_index_scores, k=k, dim=2)[1] # topk: Retrieves the top k predicted token IDs for each masked position based on the scores.

            # Update Predictions:
            for tar_i, topk_ids, mat in zip(target_idx, pred_topk_token_ids, topk_predictions): # Iterates through each target_idx (masked position), its top k predictions (topk_ids), and the matrix (mat) storing the predictions.
                mat[tar_i] = topk_ids # Updates the matrix with the top k token IDs at the target indices.

            # Handle Non-Masked Positions:
            orig_tokens = inputs.gather(1, target_idx) # gather(1, target_idx): Collects the original token IDs at the target indices.
            is_mask_in_target_idxs = is_mask.gather(1, target_idx)
            non_masked_target_idxs = is_mask_in_target_idxs != 1 # Identifies positions in target_idx that are not masked (is_mask_in_target_idxs != 1).
            pred_token_ids[non_masked_target_idxs] = orig_tokens[non_masked_target_idxs] # Ensures that predictions for non-masked positions are replaced with the original tokens from inputs.

            # Update Mask Indicators: updates the is_mask tensor to reflect that the current masked positions (target_idx) have been processed.
            is_mask.scatter_(1, target_idx, torch.zeros_like(pred_token_ids).float()) # scatter_: Assigns 0 (indicating no longer masked) to the positions in is_mask corresponding to target_idx.
            
            # Update Input Tokens:
            inputs.scatter_(1, target_idx, pred_token_ids) # Updates the inputs tensor with the newly predicted token IDs (pred_token_ids) at the positions specified by target_idx.
            # This effectively replaces [MASK] tokens with their most probable replacements.

            if is_mask.sum() == 0: # If there are no remaining [MASK] tokens, the loop terminates.
                break

        lengths = [len(s) for s in masked_sentences] # Computes the original lengths of the input sentences (lengths)
        pred_sentences = [tokenizer.convert_ids_to_tokens(ps[1:i+1]) for (i, ps) in zip(lengths, inputs.cpu().numpy())]  # Converts the predicted token IDs (inputs) back into human-readable tokens using tokenizer.convert_ids_to_tokens, slicing out special tokens [CLS] and [SEP].
        
        # Converts the top-k token predictions into readable text format, removing special tokens [CLS] and [SEP].
        topk_predictions = [list(map(tokenizer.convert_ids_to_tokens, s))[1:i+1] for (i, s) in zip(lengths, topk_predictions.cpu().numpy())]
        return pred_sentences, topk_predictions
        # Ex: masked_sentences = [["The", "[MASK]", "sat", "on", "the", "mat"]]; k=2 
        # topk_predictions = [["The", "cat", "sat", "on", "the", "mat"], ["The", "rat", "sat", "on", "the", "mat"]]

