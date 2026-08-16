# Check if running on Google Colab and GPU availability
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
from tqdm import tqdm
import numpy as np
from torch.utils.data import random_split
import pickle
import matplotlib.pyplot as plt

# Check GPU availability (works on any system)
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"\nGPU detected: {num_gpus} GPU(s) available")
    
    # Print info for each GPU
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    
    # Set default GPU (use GPU 0)
    torch.cuda.set_device(0)
    print(f"\nUsing GPU 0: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
else:
    print("\n⚠ WARNING: No GPU detected!")
    print("  Running on CPU (training will be slower)")


class Vocabulary:
    def __init__(self):
        self.token2idx = {} # create a mapping from token to index
        self.idx2token = {} # create a mapping from index to token
        # Define special tokens
        self.PAD_TOKEN = '<PAD>'
        self.SOS_TOKEN = '<>'
        self.EOS_TOKEN = '<EOS>'
        self.UNK_TOKEN = '<UNK>'

        # Add special tokens
        self.add_token(self.PAD_TOKEN)
        self.add_token(self.SOS_TOKEN)
        self.add_token(self.EOS_TOKEN)
        self.add_token(self.UNK_TOKEN)

    def add_token(self, token):
        # Add a token to the vocabulary if it's not already present
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    def build_from_sequences(self, sequences):
        # Build vocabulary from a list of token sequences
        for seq in sequences:
            for token in seq:
                self.add_token(token)

    def encode(self, tokens):
        # Convert a list of tokens to their corresponding indices
        return [self.token2idx.get(t, self.token2idx[self.UNK_TOKEN]) for t in tokens]

    def decode(self, indices):
        # Convert a list of indices back to their corresponding tokens
        return [self.idx2token[i] for i in indices if i in self.idx2token]

    def __len__(self):
        return len(self.token2idx)
    

class MazeDataset(Dataset):
    def __init__(self, csv_path, vocab=None, is_train=True):
        # Load data from CSV
        self.df = pd.read_csv(csv_path)
        self.is_train = is_train

        # Parse sequences
        self.inputs = [ast.literal_eval(s) for s in self.df['input_sequence']]
        self.outputs = [ast.literal_eval(s) for s in self.df['output_path']]

        # Build or use vocabulary
        if vocab is None:
            self.vocab = Vocabulary()
            self.vocab.build_from_sequences(self.inputs + self.outputs)
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get input and output sequences for the given index
        input_seq = self.inputs[idx]
        output_seq = self.outputs[idx]

        # Encode sequences
        input_encoded = self.vocab.encode(input_seq)
        output_encoded = self.vocab.encode(output_seq)

        return {
            'input': torch.tensor(input_encoded, dtype=torch.long),
            'output': torch.tensor(output_encoded, dtype=torch.long),
            'input_len': len(input_encoded),
            'output_len': len(output_encoded)
        }

def collate_fn(batch):
    # Pad sequences to max length in batch
    inputs = [item['input'] for item in batch]
    outputs = [item['output'] for item in batch]
    input_lens = [item['input_len'] for item in batch]
    output_lens = [item['output_len'] for item in batch]

    # Pad sequences
    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    outputs_padded = nn.utils.rnn.pad_sequence(outputs, batch_first=True, padding_value=0)

    return {
        'input': inputs_padded,
        'output': outputs_padded,
        'input_len': torch.tensor(input_lens),
        'output_len': torch.tensor(output_lens)
    }

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0) # create embedding layer for all tokens
        # RNN layer
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, x, lengths):
        """
        Args:
            x: [batch, seq_len]
            lengths: [batch]
        Returns:
            outputs: [batch, seq_len, hidden_dim]
            hidden: tuple of (h_n, c_n)
        """
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]

        # Pack padded sequences
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        # Forward pass through RNN - implicit, not explicitly called
        outputs, hidden = self.rnn(packed)

        # Unpack the outputs
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs, hidden # returns the series of hidden states and the final hidden state
    
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Initialize layers for attention mechanism
        self.W1 = nn.Linear(hidden_dim, hidden_dim) # for decoder hidden state
        self.W2 = nn.Linear(hidden_dim, hidden_dim) # for encoder outputs
        self.V = nn.Linear(hidden_dim, 1) # to produce attention scores

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        Args:
            decoder_hidden: [batch, hidden_dim]
            encoder_outputs: [batch, seq_len, hidden_dim]
            mask: [batch, seq_len] - True for padding positions
        Returns:
            context: [batch, hidden_dim]
            attention_weights: [batch, seq_len]
        """
        # decoder_hidden: [batch, 1, hidden_dim]
        decoder_hidden = decoder_hidden.unsqueeze(1)

        # Calculate attention scores
        score = self.V(torch.tanh(self.W1(decoder_hidden) + self.W2(encoder_outputs)))
        score = score.squeeze(-1)  # [batch, seq_len]

        # Apply mask if provided
        if mask is not None:
            score = score.masked_fill(mask, -1e9)

        # Calculate attention weights
        attention_weights = F.softmax(score, dim=-1)  # [batch, seq_len]

        # Calculate context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)  # [batch, hidden_dim]

        return context, attention_weights
    

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Initialize layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0) # create embedding layer for all tokens
        self.attention = BahdanauAttention(hidden_dim) # attention mechanism
        self.rnn = nn.RNN(embed_dim + hidden_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0) # RNN layer
        self.fc = nn.Linear(hidden_dim, vocab_size) # final output layer
        self.dropout = nn.Dropout(dropout) # dropout layer

    def forward(self, x, hidden, encoder_outputs, mask=None):
        """
        Args:
            x: [batch, 1] - current token
            hidden: tuple of (h, c) from previous step
            encoder_outputs: [batch, seq_len, hidden_dim]
            mask: [batch, seq_len]
        Returns:
            output: [batch, vocab_size]
            hidden: tuple of (h, c)
            attention_weights: [batch, seq_len]
        """
        embedded = self.dropout(self.embedding(x))  # [batch, 1, embed_dim]

        # Get context vector from attention
        context, attention_weights = self.attention(hidden[-1], encoder_outputs, mask)
        context = context.unsqueeze(1)  # [batch, 1, hidden_dim]

        # Concatenate embedding and context
        rnn_input = torch.cat([embedded, context], dim=2)  # [batch, 1, embed_dim + hidden_dim]

        # Pass through RNN
        output, hidden = self.rnn(rnn_input, hidden)

        # Generate prediction
        output = self.fc(output.squeeze(1))  # [batch, vocab_size]

        return output, hidden, attention_weights
    

# This is the main Seq2Seq model that combines the Encoder and Decoder
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        # Initialize encoder, decoder, and device
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def create_mask(self, lengths, max_len):
        # Create padding mask
        batch_size = lengths.size(0)
        # Ensure lengths tensor is on the correct device for comparison
        mask = torch.arange(max_len, device=self.device).expand(batch_size, max_len) >= lengths.to(self.device).unsqueeze(1)
        return mask
    
    def forward(self, src, src_len, tgt, tgt_len, teacher_forcing_ratio=0.5):
            """
            Args:
                src: [batch, src_seq_len]
                src_len: [batch]
                tgt: [batch, tgt_seq_len]  <-- This is [T1, T2, ..., TN]
                tgt_len: [batch]
                teacher_forcing_ratio: probability of using teacher forcing
            Returns:
                outputs: [batch, tgt_seq_len, vocab_size] <-- This will be [P1, P2, ..., PN]
            """
            batch_size = src.size(0)
            tgt_seq_len = tgt.size(1)
            vocab_size = self.decoder.fc.out_features

            # Encode
            encoder_outputs, hidden = self.encoder(src, src_len)

            # Create mask for attention
            mask = self.create_mask(src_len, src.size(1))

            # Initialize outputs
            outputs = torch.zeros(batch_size, tgt_seq_len, vocab_size).to(self.device)

            # First input is *always* the SOS token (index 1)
            sos_token_idx = 1 # Assuming <SOS> is 1, as in your vocab
            decoder_input = torch.full((batch_size, 1), sos_token_idx, dtype=torch.long).to(self.device)

            # Decode step by step
            # Loop L times (from 0 to L-1) to generate L tokens
            for t in range(tgt_seq_len):
                # output is the prediction for the t-th token
                output, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs, mask)
                outputs[:, t, :] = output # Store prediction at index t

                # Teacher forcing
                teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                top1 = output.argmax(1)

                # If teacher forcing, next input is the *current* target token
                # Otherwise, it's the model's own prediction
                decoder_input = tgt[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)

            return outputs

    def predict(self, src, src_len, max_len=50, start_token=None, end_token=None):
        # Generate prediction without teacher forcing
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)

            # Encode
            encoder_outputs, hidden = self.encoder(src, src_len)

            # Create mask
            mask = self.create_mask(src_len, src.size(1))

            # Start with the provided start token (or default to 1 for <SOS>)
            if start_token is None:
                start_token = 1
            decoder_input = torch.full((batch_size, 1), start_token, dtype=torch.long).to(self.device)

            # Initialize predictions list (do NOT include start token in output)
            predictions = []

            for _ in range(max_len):
                output, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs, mask)
                top1 = output.argmax(1)
                predictions.append(top1)

                # Break if all sequences have generated the end token
                if end_token is not None and (top1 == end_token).all():
                    break

                decoder_input = top1.unsqueeze(1)

            return torch.stack(predictions, dim=1)

def accuracy_metric(predictions, targets, ignore_index=0):
    # Calculates token-level accuracy, ignoring padding tokens.
    # Reshape predictions and targets to (batch_size * seq_len,) if they are not already
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    # Create a mask for non-padding tokens
    mask = (targets != ignore_index)

    # Apply the mask
    predictions = predictions[mask]
    targets = targets[mask]

    if targets.numel() == 0: # Handle case where all are padding
        return torch.tensor(1.0) # Or 0.0 depending on desired behavior for empty sequences

    correct = (predictions == targets).sum().float()
    total = targets.numel()
    return correct / total

def sequence_accuracy_metric(predictions, targets, ignore_index=0):
    # Calculates exact sequence match accuracy.
    batch_size = predictions.size(0)
    correct_sequences = 0

    for i in range(batch_size):
        # Get predicted sequence (remove padding)
        pred_seq = predictions[i]
        target_seq = targets[i]

        # Find actual lengths (non-padding)
        pred_mask = (pred_seq != ignore_index)
        target_mask = (target_seq != ignore_index)

        pred_tokens = pred_seq[pred_mask]
        target_tokens = target_seq[target_mask]

        # Check if sequences match exactly
        if pred_tokens.size(0) == target_tokens.size(0) and torch.all(pred_tokens == target_tokens):
            correct_sequences += 1

    return correct_sequences / batch_size if batch_size > 0 else 0.0

def train_epoch(model, dataloader, optimizer, criterion, device):
    # Train for one epoch with teacher forcing
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        src = batch['input'].to(device)
        tgt = batch['output'].to(device)
        src_len = batch['input_len'].to(device)
        tgt_len = batch['output_len'].to(device)

        optimizer.zero_grad()

        # Forward pass with teacher forcing for training
        output = model(src, src_len, tgt, tgt_len, teacher_forcing_ratio=0.5)

        # Calculate loss
        output_flat = output.reshape(-1, output.size(-1))
        tgt_flat = tgt.reshape(-1)

        loss = criterion(output_flat, tgt_flat)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate_true_metrics(model, dataloader, criterion, device, vocab, desc="Evaluating"):
    """
    Evaluate with TRUE metrics using autoregressive generation (NO teacher forcing).
    Returns: loss, token_accuracy, sequence_accuracy
    """
    model.eval()
    total_loss = 0
    total_correct_tokens = 0
    total_tokens = 0
    exact_matches = 0
    total_samples = 0

    # Get start and end tokens
    start_token = vocab.token2idx.get('<PATH_START>', vocab.token2idx['<>'])
    end_token = vocab.token2idx.get('<PATH_END>', vocab.token2idx['<EOS>'])

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            src = batch['input'].to(device)
            tgt = batch['output'].to(device)
            src_len = batch['input_len'].to(device)
            tgt_len = batch['output_len'].to(device)

            # Calculate loss (using teacher forcing just for loss calculation)
            output = model(src, src_len, tgt, tgt_len, teacher_forcing_ratio=0)
            output_flat = output.reshape(-1, output.size(-1))
            tgt_flat = tgt.reshape(-1)
            loss = criterion(output_flat, tgt_flat)
            total_loss += loss.item()

            # Generate predictions using autoregressive generation (NO teacher forcing)
            predictions = model.predict(src, src_len, max_len=50,
                                       start_token=start_token,
                                       end_token=end_token)

            # Calculate TRUE token and sequence accuracy
            for i in range(src.size(0)):
                total_samples += 1

                # Get predicted sequence
                pred_indices = predictions[i].cpu().tolist()
                # Trim at end token
                try:
                    first_end_idx = pred_indices.index(end_token)
                    pred_indices = pred_indices[:first_end_idx + 1]
                except ValueError:
                    pass

                # Get true sequence (without padding)
                real_len = tgt_len[i].item()
                true_indices = tgt[i].cpu().tolist()[:real_len]

                # Calculate token-level accuracy
                min_len = min(len(pred_indices), len(true_indices))
                for j in range(min_len):
                    if pred_indices[j] == true_indices[j]:
                        total_correct_tokens += 1
                    total_tokens += 1

                # Account for length differences in token accuracy
                total_tokens += abs(len(pred_indices) - len(true_indices))

                # Calculate sequence-level accuracy (exact match)
                if pred_indices == true_indices:
                    exact_matches += 1

    token_accuracy = total_correct_tokens / total_tokens if total_tokens > 0 else 0.0
    seq_accuracy = exact_matches / total_samples if total_samples > 0 else 0.0

    return total_loss / len(dataloader), token_accuracy, seq_accuracy

def parse_openings(tokens):
    # Parses the adjacency list to find all openings between cells.
    openings = set()
    in_adj_list = False
    for i, tok in enumerate(tokens):
        if tok == '<ADJLIST_START>':
            in_adj_list = True
        elif tok == '<ADJLIST_END>':
            in_adj_list = False

        # Check if token is a connection '<-->'
        if in_adj_list and tok == '<-->' and i > 0 and i < len(tokens) - 1:
            try:
                cell1_str = tokens[i-1]
                cell2_str = tokens[i+1]
                # Ensure we are looking at two coordinate tuples
                if isinstance(cell1_str, str) and cell1_str.startswith('(') and \
                   isinstance(cell2_str, str) and cell2_str.startswith('('):

                    c1 = ast.literal_eval(cell1_str)
                    c2 = ast.literal_eval(cell2_str)
                    # Add a sorted tuple to handle (c1, c2) and (c2, c1) identically
                    openings.add(tuple(sorted((c1, c2))))
            except Exception:
                pass # Ignore parsing errors
    return openings

def parse_origin_target(tokens):
    # Extracts origin and target coordinates from the input token list.
    orig = None
    targ = None
    for i, t in enumerate(tokens):
        if t == '<ORIGIN_START>' and i+1 < len(tokens):
            try:
                coord_str = tokens[i+1]
                if isinstance(coord_str, str) and coord_str.startswith('(') and coord_str.endswith(')'):
                    x, y = coord_str.strip('()').split(',')
                    orig = (int(x), int(y))
            except Exception:
                pass
        if t == '<TARGET_START>' and i+1 < len(tokens):
            try:
                coord_str = tokens[i+1]
                if isinstance(coord_str, str) and coord_str.startswith('(') and coord_str.endswith(')'):
                    x, y = coord_str.strip('()').split(',')
                    targ = (int(x), int(y))
            except Exception:
                pass
    return orig, targ

def parse_path(tokens):
    # Extracts a list of (x, y) coordinates from a token list.
    coords = []
    for t in tokens:
        if isinstance(t, str) and t.startswith('(') and t.endswith(')'):
            try:
                x, y = t.strip('()').split(',')
                coords.append((int(x), int(y)))
            except Exception:
                pass
    return coords

def plot_maze_prediction(input_tokens, ground_truth, predicted, title='Maze Path Prediction', size=6):
    # Plots the ground truth and predicted paths on a 6x6 grid.
    orig, targ = parse_origin_target(input_tokens)
    gt_path = parse_path(ground_truth)
    pred_path = parse_path(predicted)
    openings = parse_openings(input_tokens)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    for ax, path, subtitle in [(ax1, gt_path, 'Ground Truth'), (ax2, pred_path, 'Predicted')]:

        # --- WALL DRAWING ---
        # Draw all potential horizontal and vertical walls
        # Note: r = Row (Y-axis), c = Col (X-axis)
        for r in range(size):
            for c in range(size):
                # Wall below cell (r, c)
                if (r < size - 1):
                    wall = tuple(sorted(((r, c), (r+1, c))))
                    if wall not in openings:
                        # Draw horizontal line
                        ax.plot([c, c+1], [r+1, r+1], color='black', linewidth=2)

                # Wall to the right of cell (r, c)
                if (c < size - 1):
                    wall = tuple(sorted(((r, c), (r, c+1))))
                    if wall not in openings:
                        # Draw vertical line
                        ax.plot([c+1, c+1], [r, r+1], color='black', linewidth=2)

        # Draw outer boundary
        ax.plot([0, size], [0, 0], color='black', linewidth=2) # Top
        ax.plot([0, size], [size, size], color='black', linewidth=2) # Bottom
        ax.plot([0, 0], [0, size], color='black', linewidth=2) # Left
        ax.plot([size, size], [0, size], color='black', linewidth=2) # Right
        # --- END WALL DRAWING ---

        # Draw path
        plot_path = path
        # If origin isn't in path, add it to the start for a complete visual
        if orig and (not path or orig != path[0]):
             plot_path = [orig] + path

        if len(plot_path) >= 1:
            # FIX: p[1] is column (x), p[0] is row (y)
            xs = [p[1]+0.5 for p in plot_path]
            ys = [p[0]+0.5 for p in plot_path]
            ax.plot(xs, ys, '-o', color='tab:blue', label='path', linewidth=2, markersize=8)

        # Draw origin and target
        if orig is not None:
            # FIX: orig[1] is x, orig[0] is y
            ax.plot(orig[1]+0.5, orig[0]+0.5, 'go', markersize=15, label='origin')
        if targ is not None:
            # FIX: targ[1] is x, targ[0] is y
            ax.plot(targ[1]+0.5, targ[0]+0.5, 'rx', markersize=15, label='target', markeredgewidth=3)

        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_aspect('equal')
        ax.invert_yaxis() # Invert y-axis to match (row, col) indexing
        ax.set_title(subtitle)
        ax.legend(loc='upper right')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def get_lcs_length(x, y):
        # Compute Longest Common Subsequence length dynamically
        m = len(x)
        n = len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

def calculate_lcs_f1(model, dataloader, device, start_token, end_token):
        model.eval()
        f1_scores = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calculating LCS F1"):
                src = batch['input'].to(device)
                src_len = batch['input_len'].to(device)
                targets = batch['output'].to(device)
                target_lens = batch['output_len']

                predictions = model.predict(src, src_len, max_len=50,
                                            start_token=start_token,
                                            end_token=end_token)

                for i in range(src.size(0)):
                    # Get sequences
                    pred_seq = predictions[i].cpu().tolist()
                    try:
                        pred_seq = pred_seq[:pred_seq.index(end_token) + 1]
                    except ValueError: pass

                    true_len = target_lens[i].item()
                    true_seq = targets[i].cpu().tolist()[:true_len]

                    # Calculate LCS
                    lcs_len = get_lcs_length(pred_seq, true_seq)

                    # Calc F1 based on LCS
                    prec = lcs_len / len(pred_seq) if len(pred_seq) > 0 else 0
                    rec = lcs_len / len(true_seq) if len(true_seq) > 0 else 0

                    if prec + rec > 0:
                        f1 = 2 * prec * rec / (prec + rec)
                    else:
                        f1 = 0
                    f1_scores.append(f1)

        return sum(f1_scores) / len(f1_scores)

if __name__ == "__main__":
    # Intialize Hyperparameters as specified
    EMBED_DIM = 128
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
    DROPOUT = 0.0
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {DEVICE}")

    # Setup DataLoaders with Train/Val Split

    # 1. Load the Full Training Data & Test Data - change the path name as needed
    train_dataset = MazeDataset('train_6x6_mazes.csv')
    test_dataset = MazeDataset('test_6x6_mazes.csv', vocab=train_dataset.vocab, is_train=False)

    # 2. Define Split Ratio (e.g., 80% Train / 20% Validation)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    # 3. Perform the Random Split
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    # 4. Create DataLoaders for all three sets
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    vocab_size = len(train_dataset.vocab)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training samples:   {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    print(f"Test samples:       {len(test_dataset)}")


    # Importing the pickle module to save the vocabulary object
    print("Saving vocabulary to 'vocab_rnn.pkl'...")
    with open('vocab_rnn.pkl', 'wb') as f:
        pickle.dump(train_dataset.vocab, f)

    print("Vocabulary saved!")

    # Initialize model
    encoder = Encoder(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    decoder = Decoder(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Training loop with TRUE metrics (autoregressive generation, NO teacher forcing)
    # Initialize variables if not already defined (e.g., if resume cell wasn't run) - from scratch training always here
    
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_token_accs = []
    val_token_accs = []
    train_seq_accs = []
    val_seq_accs = []
    print("Initialized fresh training variables")

    print("\n" + "="*90)
    print(" "*15 + "TRAINING WITH TRUE METRICS (AUTOREGRESSIVE GENERATION)")
    print("="*90)
    print("Both training and validation metrics calculated using model.predict()")
    print("NO teacher forcing in evaluation - TRUE token and sequence accuracy")
    print("="*90)

    print(" Starting Training...\n")

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*70}")

        # Training step (with teacher forcing for learning)
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)

        # Evaluate TRUE metrics on training set (autoregressive, no teacher forcing)
        print("\nCalculating TRUE metrics on training set...")
        _, train_token_acc, train_seq_acc = evaluate_true_metrics(
            model, train_loader, criterion, DEVICE, train_dataset.vocab, desc="Eval Train Set"
        )

        # Evaluate TRUE metrics on validation set (autoregressive, no teacher forcing)
        print("\nCalculating TRUE metrics on validation set...")
        val_loss, val_token_acc, val_seq_acc = evaluate_true_metrics(
            model, val_loader, criterion, DEVICE, train_dataset.vocab, desc="Eval Val Set"
        )
        # val_loss, val_token_acc, val_seq_acc = evaluate_true_metrics(
        #     model, val_loader, criterion, DEVICE, train_dataset.vocab, desc="Eval Val Set"
        # )

        # Track metrics for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_token_accs.append(train_token_acc)
        val_token_accs.append(val_token_acc)
        train_seq_accs.append(train_seq_acc)
        val_seq_accs.append(val_seq_acc)

        print(f"\n{'='*70}")
        print(f"RESULTS FOR EPOCH {epoch+1}")
        print(f"{'='*70}")
        print(f"  Train Loss: {train_loss:.4f} | Token Acc: {train_token_acc:.4f} | Seq Acc: {train_seq_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Token Acc: {val_token_acc:.4f} | Seq Acc: {val_seq_acc:.4f}")

        # Save best model with complete checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # Save complete checkpoint for resuming training
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_token_accs': train_token_accs,
                'val_token_accs': val_token_accs,
                'train_seq_accs': train_seq_accs,
                'val_seq_accs': val_seq_accs,
            }
            torch.save(checkpoint, 'best_model_checkpoint.pt')

            # Also save just the model weights for easy loading during inference
            torch.save(model.state_dict(), 'rnn.pt')

            print("\n  Saved best model checkpoint!")

        print(f"{'='*70}\n")


    # Plot Loss and Accuracy Curves (3 subplots)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Loss Curves
    axes[0].plot(range(1, len(train_losses)+1), train_losses, 'b-', label='Train Loss', marker='o', linewidth=2)
    axes[0].plot(range(1, len(val_losses)+1), val_losses, 'r-', label='Validation Loss', marker='s', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('RNN Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Token Accuracy Curves (TRUE - Autoregressive)
    axes[1].plot(range(1, len(train_token_accs)+1), train_token_accs, 'b-', label='Train Token Acc', marker='o', linewidth=2)
    axes[1].plot(range(1, len(val_token_accs)+1), val_token_accs, 'r-', label='Val Token Acc', marker='s', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Token Accuracy', fontsize=12)
    axes[1].set_title('RNN Token Accuracy (Autoregressive)', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].text(0.5, 0.02, 'No teacher forcing', transform=axes[1].transAxes,
                ha='center', fontsize=9, style='italic', color='gray')

    # Plot 3: Sequence Accuracy Curves (TRUE - Autoregressive)
    axes[2].plot(range(1, len(train_seq_accs)+1), train_seq_accs, 'b-', label='Train Seq Acc', marker='o', linewidth=2)
    axes[2].plot(range(1, len(val_seq_accs)+1), val_seq_accs, 'r-', label='Val Seq Acc', marker='s', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Sequence Accuracy (Exact Match)', fontsize=12)
    axes[2].set_title('RNN Sequence Accuracy (Autoregressive)', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].text(0.5, 0.02, 'No teacher forcing', transform=axes[2].transAxes,
                ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig('rnn_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n" + "="*70)
    print(f"TRAINING SUMMARY (TRUE METRICS - NO TEACHER FORCING)")
    print(f"="*70)
    print(f"  Best Validation Loss: {best_val_loss:.4f}")
    print(f"\n  Final Train Loss:      {train_losses[-1]:.4f}")
    print(f"  Final Val Loss:        {val_losses[-1]:.4f}")
    print(f"\n  Final Train Token Acc: {train_token_accs[-1]:.4f} (autoregressive)")
    print(f"  Final Val Token Acc:   {val_token_accs[-1]:.4f} (autoregressive)")
    print(f"\n  Final Train Seq Acc:   {train_seq_accs[-1]:.4f} (exact match)")
    print(f"  Final Val Seq Acc:     {val_seq_accs[-1]:.4f} (exact match)")
    print(f"="*70)

    # Load best model and test
    model.load_state_dict(torch.load('rnn.pt'))
    model.eval()

    # Get a sample
    sample = test_dataset[0]
    src = sample['input'].unsqueeze(0).to(DEVICE)
    src_len = torch.tensor([sample['input_len']])

    # Get the correct start and end tokens from vocabulary
    # FIX: Changed '<SOS>' to '<>' to match Vocabulary class
    start_token = test_dataset.vocab.token2idx.get('<PATH_START>', test_dataset.vocab.token2idx['<>'])
    end_token = test_dataset.vocab.token2idx.get('<PATH_END>', test_dataset.vocab.token2idx['<EOS>'])

    print(f"Start token: {start_token}, End token: {end_token}")

    # Predict
    predictions = model.predict(src, src_len, max_len=50, start_token=start_token, end_token=end_token)
    predicted_indices = predictions[0].cpu().tolist()
    predicted_tokens = test_dataset.vocab.decode(predicted_indices)

    # Get original tokens for visualization
    actual_tokens = test_dataset.vocab.decode(sample['output'].tolist())
    input_tokens = test_dataset.vocab.decode(sample['input'].tolist()) # Need this for the plot

    print("Predicted path:", predicted_tokens)
    print("Actual path:", actual_tokens)

    # --- NEW VISUALIZATION CODE ---
    print("\n--- Maze Visualization ---")
    is_match = (predicted_tokens == actual_tokens)
    plot_maze_prediction(input_tokens, actual_tokens, predicted_tokens, title=f'Prediction (Match: {is_match})')

    # Load best model and test
    model.load_state_dict(torch.load('rnn.pt'))
    model.eval()

    # Get the correct start and end tokens from vocabulary
    # FIX: Changed '<SOS>' to '<>' to match Vocabulary class
    start_token = test_dataset.vocab.token2idx.get('<PATH_START>', test_dataset.vocab.token2idx['<>'])
    end_token = test_dataset.vocab.token2idx.get('<PATH_END>', test_dataset.vocab.token2idx['<EOS>'])

    print(f"Using Start token: {start_token}, End token: {end_token}\n")

    # Visualize 5 RANDOM predictions (as per assignment requirement)
    import random
    random_indices = random.sample(range(len(test_dataset)), 5)
    print(f"Visualizing random test samples: {random_indices}\n")

    # Loop for 5 predictions
    for idx, i in enumerate(random_indices):
        print(f"\n" + "="*40)
        print(f"      SAMPLE {idx+1} / 5 (Index: {i})")
        print("="*40)

        # Get a sample
        sample = test_dataset[i]
        src = sample['input'].unsqueeze(0).to(DEVICE)
        src_len = torch.tensor([sample['input_len']])

        # Predict
        predictions = model.predict(src, src_len, max_len=50, start_token=start_token, end_token=end_token)
        predicted_indices = predictions[0].cpu().tolist()
        predicted_tokens = test_dataset.vocab.decode(predicted_indices)

        # Get original tokens for visualization
        actual_tokens = test_dataset.vocab.decode(sample['output'].tolist())
        input_tokens = test_dataset.vocab.decode(sample['input'].tolist())

        print("\nPredicted path:", predicted_tokens)
        print("Actual path:   ", actual_tokens)

        # --- VISUALIZATION ---
        is_match = (predicted_tokens == actual_tokens)
        plot_maze_prediction(input_tokens, actual_tokens, predicted_tokens,
                            title=f'RNN Sample {idx+1} (Test Index: {i}, Match: {is_match})')
        


    # --- 1. Load Model ---
    # Ensure the model architecture is defined (run cells 4-7 first)
    encoder = Encoder(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    decoder = Decoder(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    # Load the saved weights from your best epoch
    try:
        model.load_state_dict(torch.load('rnn.pt'))
        print("Successfully loaded 'rnn.pt'.")
    except FileNotFoundError:
        print("Error: 'rnn.pt' not found. Please run the training cell first.")
        raise FileNotFoundError("Could not find 'rnn.pt'.")
    model.eval() # Set model to evaluation mode

    # --- 2. Get Special Tokens ---
    # *** START OF FIX ***
    # Find the correct START and END tokens from the vocabulary, matching your data
    try:
        start_token = test_dataset.vocab.token2idx['<>'] # Your '<SOS>' token

        # Prioritize '<PATH_END>' since that's what's in your target data
        if '<PATH_END>' in test_dataset.vocab.token2idx:
            end_token_str = '<PATH_END>'
        else:
            end_token_str = '<EOS>'

        end_token = test_dataset.vocab.token2idx[end_token_str]
        print(f"Using Start Token: '<>' (ID: {start_token})")
        print(f"Using End Token:   '{end_token_str}' (ID: {end_token})")

    except KeyError as e:
        print(f"Error: Could not find a required special token in vocabulary: {e}")
        raise e
    # *** END OF FIX ***


    # --- 3. Initialize Counters ---
    exact_matches = 0
    total_samples = 0

    # --- 4. Loop Through Test Loader ---
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Calculating Exact Match Accuracy"):

            # Get data from batch
            src = batch['input'].to(DEVICE)
            src_len = batch['input_len'].to(DEVICE)
            targets = batch['output'].to(DEVICE)
            target_lens = batch['output_len']

            # Get model predictions
            predictions = model.predict(src, src_len, max_len=50,
                                        start_token=start_token,
                                        end_token=end_token)

            # Iterate over each sample in the batch
            for i in range(src.size(0)):
                total_samples += 1

                # --- Get Predicted Sequence ---
                pred_indices = predictions[i].cpu().tolist()

                # *** START OF FIX ***
                # Trim the prediction at the first end_token, *including* the token itself
                try:
                    first_end_idx = pred_indices.index(end_token)
                    pred_indices = pred_indices[:first_end_idx + 1] # Keep the end token
                except ValueError:
                    pass # No end token was found
                # *** END OF FIX ***

                # --- Get Target Sequence ---
                # Get the real target sequence, trimming off padding
                # This list correctly includes the '<PATH_END>' token
                real_len = target_lens[i].item()
                target_indices = targets[i].cpu().tolist()[:real_len]

                # --- Compare ---
                if pred_indices == target_indices:
                    exact_matches += 1

    # --- 5. Report Results ---
    if total_samples > 0:
        accuracy = (exact_matches / total_samples) * 100
        print("\n" + "="*40)
        print("      Exact Sequence Match Results")
        print("="*40)
        print(f"Total Samples:   {total_samples}")
        print(f"Exact Matches:   {exact_matches}")
        print(f"Accuracy:        {accuracy:.2f}%")
    else:
        print("Error: No samples were processed from the test loader.")
    

    # Run LCS F1
    lcs_f1 = calculate_lcs_f1(model, test_loader, DEVICE, start_token, end_token)
    print(f"\nLCS (Order-Aware) F1 Score: {lcs_f1:.4f}")

