import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
import sys
import os
import pickle
from tqdm import tqdm

# --- 1. Vocabulary Class (Must match Notebook exactly) ---
class Vocabulary:
    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}
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
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.idx2token[idx] = token
            self.token2idx[token] = idx

    def build_from_sequences(self, sequences):
        for seq in sequences:
            for token in seq:
                self.add_token(token)

    def encode(self, tokens):
        return [self.token2idx.get(t, self.token2idx[self.UNK_TOKEN]) for t in tokens]

    def decode(self, indices):
        return [self.idx2token[i] for i in indices if i in self.idx2token]

    def __len__(self):
        return len(self.token2idx)

# --- 2. Dataset Class ---
class MazeDatasetRNN(Dataset):
    def __init__(self, csv_path, vocab=None):
        self.df = pd.read_csv(csv_path)
        
        # Parse inputs
        self.inputs = [ast.literal_eval(s) for s in self.df['input_sequence']]
        
        # Handle outputs (might be empty or missing in test files)
        if 'output_path' in self.df.columns:
            self.outputs = []
            for s in self.df['output_path']:
                if pd.isna(s) or (isinstance(s, str) and s.strip() == ''):
                    self.outputs.append([])
                else:
                    try:
                        self.outputs.append(ast.literal_eval(s))
                    except:
                        self.outputs.append([])
        else:
            self.outputs = [[] for _ in range(len(self.inputs))]

        # Vocab is mandatory now
        if vocab is None:
            raise ValueError("Vocabulary must be provided (loaded from vocab_rnn.pkl).")
        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_seq = self.inputs[idx]
        input_encoded = self.vocab.encode(input_seq)
        
        # Encode output if available (just for consistency, not used for prediction)
        output_seq = self.outputs[idx] if idx < len(self.outputs) else []
        output_encoded = self.vocab.encode(output_seq)

        return {
            'input': torch.tensor(input_encoded, dtype=torch.long),
            'output': torch.tensor(output_encoded, dtype=torch.long),
            'input_len': len(input_encoded),
            'output_len': len(output_encoded)
        }

def collate_fn_rnn(batch):
    inputs = [item['input'] for item in batch]
    input_lens = [item['input_len'] for item in batch]
    output_lens = [item['output_len'] for item in batch]

    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    
    outputs = [item['output'] for item in batch if len(item['output']) > 0]
    if outputs:
        outputs_padded = nn.utils.rnn.pad_sequence(outputs, batch_first=True, padding_value=0)
    else:
        outputs_padded = None

    return {
        'input': inputs_padded,
        'output': outputs_padded,
        'input_len': torch.tensor(input_lens),
        'output_len': torch.tensor(output_lens)
    }

# --- 3. Model Classes ---
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        decoder_hidden = decoder_hidden.unsqueeze(1)
        score = self.V(torch.tanh(self.W1(decoder_hidden) + self.W2(encoder_outputs)))
        score = score.squeeze(-1)
        if mask is not None:
            score = score.masked_fill(mask, -1e9)
        attention_weights = F.softmax(score, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        return context, attention_weights

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_dim)
        self.rnn = nn.RNN(embed_dim + hidden_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, encoder_outputs, mask=None):
        embedded = self.dropout(self.embedding(x))
        context, attention_weights = self.attention(hidden[-1], encoder_outputs, mask)
        context = context.unsqueeze(1)
        rnn_input = torch.cat([embedded, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        output = self.fc(output.squeeze(1))
        return output, hidden, attention_weights

class Seq2SeqRNN(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def create_mask(self, lengths, max_len):
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=self.device).expand(batch_size, max_len) >= lengths.to(self.device).unsqueeze(1)
        return mask

    def predict(self, src, src_len, max_len=50, start_token=None, end_token=None):
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            encoder_outputs, hidden = self.encoder(src, src_len)
            mask = self.create_mask(src_len, src.size(1))

            if start_token is None:
                start_token = 1
            decoder_input = torch.full((batch_size, 1), start_token, dtype=torch.long).to(self.device)
            predictions = []

            for _ in range(max_len):
                output, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs, mask)
                top1 = output.argmax(1)
                predictions.append(top1)
                if end_token is not None and (top1 == end_token).all():
                    break
                decoder_input = top1.unsqueeze(1)

            return torch.stack(predictions, dim=1)

# --- 4. Functions ---

def load_rnn_model(model_path, vocab_size, device):
    EMBED_DIM = 128
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
    DROPOUT = 0.0
    
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle Checkpoint Wrapper (if user passes checkpoint instead of weights)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    encoder = EncoderRNN(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    decoder = DecoderRNN(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    model = Seq2SeqRNN(encoder, decoder, device).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    return model

def generate_predictions_rnn(model, dataloader, vocab, device):
    model.eval()
    all_predictions = []
    
    # Automatically find start/end tokens from loaded vocab
    # Defaults to ID 1 and 2 if not found (standard for notebook)
    start_token = vocab.token2idx.get('<PATH_START>', vocab.token2idx.get('<>', 1))
    end_token = vocab.token2idx.get('<PATH_END>', vocab.token2idx.get('<EOS>', 2))
    
    print(f"Using Start Token: {start_token}, End Token: {end_token}")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            src = batch['input'].to(device)
            src_len = batch['input_len'].to(device)
            
            predictions = model.predict(src, src_len, max_len=50,
                                       start_token=start_token,
                                       end_token=end_token)
            
            for i in range(predictions.size(0)):
                pred_indices = predictions[i].cpu().tolist()
                
                # Include end token in output for consistency with training
                try:
                    first_end_idx = pred_indices.index(end_token)
                    pred_indices = pred_indices[:first_end_idx + 1]
                except ValueError:
                    pass
                
                pred_tokens = vocab.decode(pred_indices)
                all_predictions.append(pred_tokens)
    
    return all_predictions

def main():
    if len(sys.argv) != 5:
        print("Usage: python eval.py <model_path> <model_type> <input_csv_path> <output_csv_path>")
        print("Example: python eval.py best_model.pt rnn test.csv submission.csv")
        sys.exit(1)
    
    model_path = sys.argv[1]
    model_type = sys.argv[2].lower()
    input_csv_path = sys.argv[3]
    output_csv_path = sys.argv[4]
    
    # Fixed path for vocabulary
    vocab_path = "vocab_rnn.pkl"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Vocabulary (FIXED: Must exist)
    if not os.path.exists(vocab_path):
        print(f"Error: '{vocab_path}' not found in current directory.")
        print("Please download vocab_rnn.pkl from your notebook and place it here.")
        sys.exit(1)
        
    print(f"Loading vocabulary from {vocab_path}...")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    vocab_size = len(vocab)
    print(f"Loaded Vocabulary size: {vocab_size}")

    # 2. Load Data
    test_dataset = MazeDatasetRNN(input_csv_path, vocab=vocab)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_rnn)
    print(f"Number of test samples: {len(test_dataset)}")

    # 3. Load Model
    if model_type == 'rnn':
        model = load_rnn_model(model_path, vocab_size, device)
    else:
        print("Error: Transformer not implemented.")
        sys.exit(1)

    # 4. Generate
    print("Generating predictions...")
    predictions = generate_predictions_rnn(model, test_loader, vocab, device)

    # 5. Save
    output_df = pd.read_csv(input_csv_path)
    output_df['output_path'] = [str(pred) for pred in predictions]
    
    if 'id' not in output_df.columns:
        output_df.insert(0, 'id', [f'id_{i}' for i in range(len(output_df))])
    
    final_cols = [c for c in ['id', 'input_sequence', 'maze_type', 'output_path'] if c in output_df.columns]
    output_df = output_df[final_cols]
    
    output_df.to_csv(output_csv_path, index=False)
    print(f"✓ Output saved to: {output_csv_path}")

if __name__ == "__main__":
    main()