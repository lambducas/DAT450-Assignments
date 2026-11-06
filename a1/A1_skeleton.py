
from base64 import encode


import torch, pickle
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from torch.utils.data import DataLoader
import numpy as np
import sys, time, os

###
### Part 1. Tokenization.
###

# Byte-pair encoding (BPE): the main idea
# ---------------------------------------
# A token is one piece of text (not necessarily a word/subword/"token").
# In this tokenizer, every token is stored as one
# or more bytes. A pair means two tokens that appear next to each other, and a
# merge means joining such a pair to make one longer token.
#
# Training BPE:
#   1. Split every file of the training text into individual UTF-8 bytes.
#   2. Start the vocabulary with the four special tokens and all 256 possible
#      byte values. The vocabulary maps each token to an integer ID.
#   3. Count how often each pair of neighboring tokens occurs.
#   4. Join the most common pair everywhere it occurs. Add the joined token to
#      the vocabulary and save the pair as a merge rule.
#   5. Repeat steps 3 and 4 until the requested vocabulary size is reached 
#      (NB: make sure it is provided).

#
# Encoding new text:
#   1. Remove whitespace from the beginning and end.
#   2. Split the text into individual UTF-8 bytes.
#   3. Apply the saved merge rules in the same order in which they were learned.
#   4. Replace the resulting tokens with their vocabulary IDs.
#   5. Add BOS/EOS IDs, and optionally truncate or pad the sequence.
#
# This simple version treats a whole non-empty file as one sequence. Therefore,
# it may learn tokens that contain spaces or parts of more than one word.

def split_into_bytes(text):
    """Split a string into individual UTF-8 byte tokens.    


    Args:
        text: The input string to encode.

    Returns:
        A list containing one ``bytes`` object for each byte in the text.
    """
    ...
    bytes_list = []
    stripped = text.strip()
    bytes_stripped = bytes(stripped, 'utf-8') # b'text'
    limit = len(bytes_stripped)
    for i in range(limit):
        bytes_list.append(bytes([bytes_stripped[i]])) # bytes_stripped[i] returns an int and bytes(int) returns byte encoding
    return bytes_list



def build_tokenizer(train_file, max_voc_size=None, model_max_length=None,
                    pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>'):
    """ Build a tokenizer from the given file.

        Args:
             train_file:        The name of the file containing the training texts.
             max_voc_size:      The maximally allowed size of the vocabulary.
             model_max_length:  Truncate texts longer than this length.
             pad_token:         The dummy string corresponding to padding.
             unk_token:         The dummy string corresponding to out-of-vocabulary tokens.
             bos_token:         The dummy string corresponding to the beginning of the text.
             eos_token:         The dummy string corresponding to the end the text.
    """


    vocabulary = {}

    for byte_value in range(256):
        vocabulary[bytes([byte_value])] = byte_value

    vocabulary[pad_token] = 256
    vocabulary[unk_token] = 257
    vocabulary[bos_token] = 258
    vocabulary[eos_token] = 259


    sequences = []

    
    with open(train_file, "r", encoding="utf-8") as file:
        text = file.read()
        text_len = len(text)
        sequences = [
        split_into_bytes(line)
        for line in text.splitlines()
        if line.strip()
        ]

    merge_rules = []

    while len(vocabulary) < max_voc_size:
        pair_counts = Counter()

        for sequence in sequences:
            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i + 1])
                pair_counts[pair] += 1

        if not pair_counts:
            break

        most_common_pair, count = pair_counts.most_common(1)[0]
        merged_token = most_common_pair[0] + most_common_pair[1]

        vocabulary[merged_token] = len(vocabulary)
        merge_rules.append(most_common_pair)


        new_sequences = []

        for sequence in sequences:
            new_sequence = []
            i = 0

            while i < len(sequence):
                if (i + 1 < len(sequence)
                    and (sequence[i], sequence[i + 1])
                    == most_common_pair):
                    new_sequence.append(merged_token)
                    i += 2
                else:
                    new_sequence.append(sequence[i])
                    i += 1

            new_sequences.append(new_sequence)

        sequences = new_sequences


    inverse_vocabulary = {
        token_id: token
        for token, token_id in vocabulary.items()
    }

    return A1Tokenizer(
    vocabulary,
    inverse_vocabulary,
    merge_rules,
    model_max_length,
    pad_token,
    unk_token,
    bos_token,
    eos_token)

    # TODO: build the vocabulary, using a byte-level BPE tokenization (explained above)
    # Then return a tokenizer object (implemented below).

class A1Tokenizer:
    """A minimal implementation of a tokenizer similar to tokenizers in the HuggingFace library."""


    def __init__(self, vocabulary, inverse_vocabulary, merge_rules, model_max_length=None,
                 pad_token="<PAD>",unk_token="<UNK>", bos_token="<BOS>", eos_token="<EOS>"):
        self.vocabulary = vocabulary
        self.inverse_vocabulary = inverse_vocabulary
        self.merge_rules = merge_rules
        self.model_max_length = model_max_length
        self.pad_token_id = vocabulary[pad_token]
        self.unk_token_id = vocabulary[unk_token]
        self.bos_token_id = vocabulary[bos_token]
        self.eos_token_id = vocabulary[eos_token]

    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        """Tokenize the given texts and return a BatchEncoding containing the integer-encoded tokens.
           
           Args:
             texts:           The texts to tokenize.
             truncation:      Whether the texts should be truncated to model_max_length.
             padding:         Whether the tokenized texts should be padded on the right side.
             return_tensors:  If None, then return lists; if 'pt', then return PyTorch tensors.

           Returns:
             A BatchEncoding where the field `input_ids` stores the integer-encoded texts.
        """
        if return_tensors and return_tensors != 'pt':
            raise ValueError('Should be pt')
        
        # TODO: Your work here is to split the texts into words and map them to integer values.
        # 
        # - If `truncation` is set to True, the length of the encoded sequences should be 
        #   at most self.model_max_length.
        # - If `padding` is set to True, then all the integer-encoded sequences should be of the
        #   same length. That is: the shorter sequences should be "padded" by adding dummy padding
        #   tokens on the right side.
        # - If `return_tensors` is undefined, then the returned `input_ids` should be a list of lists.
        #   Otherwise, if `return_tensors` is 'pt', then `input_ids` should be a PyTorch 2D tensor.
        if isinstance(texts, str):
           texts = [texts]

        int_sequences = []

        # Token replacement logic using saved merge rules.
        for text in texts:
            tokens = split_into_bytes(text)

            for pair in self.merge_rules:
                merged = pair[0] + pair[1]
                new_tokens = []
                i = 0

                while i < len(tokens):
                    if (
                        i + 1 < len(tokens)
                        and (tokens[i], tokens[i + 1]) == pair
                    ):
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1

                tokens = new_tokens

            ids = [self.vocabulary[token] for token in tokens]
            ids = [self.bos_token_id] + ids + [self.eos_token_id] # Add beginning and end symbols.

            if truncation and len(ids) > self.model_max_length:
                ids = ids[:self.model_max_length]
                ids[-1] = self.eos_token_id

            int_sequences.append(ids)

        if padding:
            max_length = max(len(ids) for ids in int_sequences)

            for ids in int_sequences:
                ids.extend(
                    [self.pad_token_id] * (max_length - len(ids))
                )
        # Make 2d pytorch sensor of ints.
        if return_tensors == "pt":
            int_sequences = torch.tensor(int_sequences)
                    

        # TODO: Return a BatchEncoding where input_ids stores the result of the integer encoding.
        # Optionally, if you want to be 100% HuggingFace-compatible, you should also include an 
        # attention mask of the same shape as input_ids. In this mask, padding tokens correspond
        # to the the value 0 and real tokens to the value 1.

        return BatchEncoding({"input_ids": int_sequences})

    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.vocabulary)

    def decode(self, ids):
        """Return a string decoded from given input ids"""
        return b"".join(
        self.inverse_vocabulary[i]
        for i in ids
        if isinstance(self.inverse_vocabulary[i], bytes) # Do not include one of the 4 special characters, only bytes.
    ).decode("utf-8")
        
    def save(self, filename):
        """Save the tokenizer to the given file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename):
        """Load a tokenizer from the given file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)
   

###
### Part 3. Defining the model.
###

class A1RNNModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the RNN-based language model."""
    def __init__(self, vocab_size=None, embedding_size=None, hidden_size=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

class A1RNNModel(PreTrainedModel):
    """The neural network model that implements a RNN-based language model."""
    config_class = A1RNNModelConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.embedding = ...
        self.rnn = ...
        self.unembedding = ...
        self.post_init() # for compatibility 
        
    def forward(self, labels):
        """The forward pass of the RNN-based language model.
        
           Args:
             X:  The input tensor (2D), consisting of a batch of integer-encoded texts.
           Returns:
             CausalLMOutputWithPast object that stores loss and logits (the output tensor (3D) for all token positions for all vocabulary items.)
        """

        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)

        X = ... # all columns in input_ids except the last one
        Y = ... # all columns in input_ids except the first one

        assert(X.shape == Y.shape)

        embedded = ...
        rnn_out, _ = ...
        out = ...

        #  compute the loss for the model output and Y 
        return CausalLMOutputWithPast(...)


###
### Part 4. Training the language model.
###

## Hint: the following TrainingArguments hyperparameters may be relevant for your implementation:
#
# - optim:            What optimizer to use. You can assume that this is set to 'adamw_torch',
#                     meaning that we use the PyTorch AdamW optimizer.
# - eval_strategy:    You can assume that this is set to 'epoch', meaning that the model should
#                     be evaluated on the validation set after each epoch
# - use_cpu:          Force the trainer to use the CPU; otherwise, CUDA or MPS should be used.
#                     (In your code, you can just use the provided method select_device.)
# - learning_rate:    The optimizer's learning rate.
# - num_train_epochs: The number of epochs to use in the training loop.
# - per_device_train_batch_size: 
#                     The batch size to use while training.
# - per_device_eval_batch_size:
#                     The batch size to use while evaluating.
# - output_dir:       The directory where the trained model will be saved.

class A1Trainer:
    """A minimal implementation similar to a Trainer from the HuggingFace library."""

    def __init__(self, model, args, train_dataset, eval_dataset, processing_class):
        """Set up the trainer.
           
           Args:
             model:          The model to train.
             args:           The training parameters stored in a TrainingArguments object.
             train_dataset:  The dataset containing the training documents.
             eval_dataset:   The dataset containing the validation documents.
             eval_dataset:   The dataset containing the validation documents.
             processing_class:      The tokenizer.
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class

        assert(args.optim == 'adamw_torch')
        assert(args.eval_strategy == 'epoch')

    def select_device(self):
        """Return the device to use for training, depending on the training arguments and the available backends."""
        if self.args.use_cpu:
            return torch.device('cpu')
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
            
    def train(self):
        """Train the model."""
        args = self.args

        device = self.select_device()
        print('Device:', device)
        self.model.to(device)
        
        
        # TODO: Relevant arguments: at least args.learning_rate, but you can optionally also consider
        # other Adam-related hyperparameters here.
        optimizer = torch.optim.AdamW(...)

        # TODO: Relevant arguments: args.per_device_train_batch_size, args.per_device_eval_batch_size
        train_loader = DataLoader(...)
        val_loader = DataLoader(...)
        
        # TODO: Your work here is to implement the training loop.
        #       
        # for each training epoch (use args.num_train_epochs here):
        #   for each batch B in the training set:
        #
        #       PREPROCESSING AND FORWARD PASS:
        #       labels = apply your tokenizer to B
        #       replace padding tokens with value -100 so loss function will ignore them
	    #       put labels onto the GPU (or whatever device you use)
        #       apply the model to labels
        #
        #       BACKWARD PASS AND MODEL UPDATE:
        #       optimizer.zero_grad()
        #       loss.backward() # loss is stored as an attribute of model's output
        #       optimizer.step()

        print(f'Saving to {args.output_dir}.')
        self.model.save_pretrained(args.output_dir)

    
