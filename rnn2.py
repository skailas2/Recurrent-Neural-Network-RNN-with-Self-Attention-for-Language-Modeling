"""
Rewritten solution, based on starter code given to students
"""

import torch
from torch import nn, optim, Tensor
from typing import Optional
import json
from transformers import AutoTokenizer
import time

# DO NOT CHANGE THIS LINE!
# And DO NOT reset the torch seed anywhere else in your code!
torch.manual_seed(10601)


class SentenceDataset:
    def __init__(self, filepath):
        with open(filepath) as f:
            data = json.load(f)
            # Convert each sequence to a tensor
            data = [torch.tensor(seq, dtype=torch.float32) for seq in data]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class RNNCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        input_dim (int): Input dimension of RNN
        hidden_dim (int): Hidden dimension of RNN
        """
        super(RNNCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Initialize weights and biases
        self.i2h = nn.Linear(input_dim, hidden_dim, bias=True)
        self.h2h = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Activation function
        self.activation = nn.ReLU()

    def forward(self, input: Tensor, hidden_state: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): Input at timestep t
                - shape: (batch_size, input_dim)
            hidden_state (Tensor): Hidden state from timestep t-1
                - shape: (batch_size, hidden_dim)

        Returns:
            Tensor: Next hidden state at timestep t
                - shape: (batch_size, hidden_dim)
        """
        # Combine input and hidden state, apply activation
        hidden = self.i2h(input) + self.h2h(hidden_state)
        out = self.activation(hidden)
        return out

class RNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
    ):
        """
        input_dim (int): Input dimension of RNN
        hidden_dim (int): Hidden dimension of RNN
        """
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Initialize the RNNCell Class
        self.cell = RNNCell(input_dim, hidden_dim)

        # Initialize the output transformation (weights for the output layer)
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def step(self, input: Tensor, hidden_prev: Optional[Tensor] = None) -> Tensor:
        """
        Compute hidden and output states for a single timestep

        Args:
            input (Tensor): input at current timestep t
                - shape: (batch_size, input_dim,)
            hidden_prev (Tensor): hidden states of preceding timesteps [1, t-1]
                If there are no previous hidden states (i.e. we are at t=1), then
                this may be None and we will initialize the previous hidden state
                to all zeros.
                - shape: (batch_size, t-1, hidden_dim)

        Returns:
            Tensor: RNN hidden state at current timestep t
                - shape: (batch_size, hidden_dim,)
            Tensor: RNN output at current timestep t
                - shape: (batch_size, hidden_dim,)
        """
        if hidden_prev is None:
            # Initialize the hidden state with zeros if it's the first timestep
            last_hidden_state = torch.zeros(input.size(0), self.hidden_dim).to(input.device)
        else:
            # Use the last hidden state from the previous timestep
            last_hidden_state = hidden_prev[:, -1, :]

        # Call the RNN cell
        next_hidden_state = self.cell(input, last_hidden_state)

        # Transform hidden state to output state
        next_output_state = self.out(next_hidden_state)

        return next_hidden_state, next_output_state

    def forward(self, sequence: Tensor) -> Tensor:
        """
        Compute hidden and output states for all timesteps over input sequence

        Args:
            sequence (Tensor): inputs to RNN over t timesteps
                - shape (batch_size, t, input_dim)

        Returns:
            Tensor: hidden states over t timesteps
                - shape (batch_size, t, hidden_dim)
            Tensor: output states over t timesteps
                - shape (batch_size, t, hidden_dim)
        """
        hidden_states = None
        output_states = []
        b, t, _ = sequence.shape

        for i in range(t):
            # Extract the current input
            inp = sequence[:, i, :]

            # Call step() to get the next hidden/output states
            next_hidden_state, next_output_state = self.step(inp, hidden_states)
            next_hidden_state = next_hidden_state.unsqueeze(1)

            # Concatenate the newest hidden state to all previous ones
            if hidden_states is None:
                hidden_states = next_hidden_state
            else:
                hidden_states = torch.cat((hidden_states, next_hidden_state), dim=1)

            # Append the next output state to the list
            output_states.append(next_output_state)

        # Stack all of the output states over the timestep dimension
        output_states = torch.stack(output_states, dim=1)

        return hidden_states, output_states

class SelfAttention(nn.Module):
    """Scaled dot product attention from original transformers paper"""

    def __init__(self, hidden_dim, key_dim, value_dim):
        """
        hidden_dim (int): Hidden dimension of RNN
        key_dim (int): Dimension of attention key and query vectors
        value_dim (int): Dimension of attention value vectors
        """
        super(SelfAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.value_dim = value_dim

        # Initialize Query, Key, and Value transformations
        self.query_transform = nn.Linear(hidden_dim, key_dim)
        self.key_transform = nn.Linear(hidden_dim, key_dim)
        self.value_transform = nn.Linear(hidden_dim, value_dim)

        # Output projection within the Attention Layer (NOT the LM head)
        self.output_transform = nn.Linear(value_dim, hidden_dim)

    def step(self, y_all: Tensor) -> Tensor:
        """
        Compute attention for **current** timestep t

        Args:
            y_all (Tensor): Predictions up to current timestep t
                - shape (batch_size, t, hidden_dim)

        Returns:
            Tensor: Attention output for current timestep
                - shape (batch_size, hidden_dim,)
        """
        last_hidden_state = y_all[:, -1].unsqueeze(1)

        # Compute the QKV values
        query = self.query_transform(last_hidden_state)
        keys = self.key_transform(y_all)
        values = self.value_transform(y_all)

        scaling = self.key_dim ** 0.5
        query = query / scaling

        # Compute attention weights over values
        # Remember to divide raw attention scores by scaling factor
        # These scores should then be normalized using softmax
        weights = torch.softmax(torch.bmm(query, keys.transpose(1, 2)), dim=-1)

        # Compute weighted sum of values based on attention weights
        output_state = torch.bmm(weights, values)

        # Apply output projection back to hidden dimension
        output_state = self.output_transform(output_state).squeeze(1)

        return output_state

    def forward(self, y_all) -> Tensor:
        """
        Compute attention for all timesteps

        Args:
            y_all (Tensor): Predictions up to current timestep t
                - shape (batch_size, t, hidden_dim)

        Returns:
            Tensor: Attention output for all timesteps
                - shape (batch_size, t, hidden_dim)
        """
        t = y_all.shape[1]
        output_states = []

        for i in range(t):
            # Perform a step of SelfAttention and unsqueeze the result,
            # Then add it to the output states
            output_state = self.step(y_all[:, :i + 1]).unsqueeze(1)
            output_states.append(output_state)

        # Concatenate all of the outputs in the list across the sequence length dimension (t)
        output_states = torch.cat(output_states, dim=1)

        return output_states


class RNNLanguageModel(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        vocab_size,
        key_dim=None,
        value_dim=None,
    ):
        """
        embed_dim (int): Dimension of word embeddings
        hidden_dim (int): Dimension of RNN hidden states
        vocab_size (int): Number of (sub)words in model vocabulary
        """
        super(RNNLanguageModel, self).__init__()

        # Initialize word embeddings
        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        # RNN backbone
        self.rnn = RNN(input_dim=embed_dim, hidden_dim=hidden_dim)

        # Self-Attention Layer
        if key_dim is not None and value_dim is not None:
            self.attention = SelfAttention(hidden_dim, key_dim, value_dim)
        else:
            self.attention = None

        # Final projection from RNN output state to next token logits
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Computes next-token logits and hidden states for each token in tokens

        Args:
            tokens (Tensor): Input tokens IDs
                - shape (batch_size, t,)

        Returns:
            Tensor: Next-token logits for each token from the LM head
                - shape (batch_size, t, vocab_size)
            Tensor: RNN hidden states for each token
                - shape (batch_size, t, hidden_dim)
            Tensor: RNN output states for each token
                - shape (batch_size, t, hidden_dim)
        """
        # Apply embeddings
        embeddings = self.embeddings(tokens)  # (batch_size, t, embed_dim)

        # Apply RNN
        hidden_states, rnn_outputs = self.rnn(embeddings)  # (batch_size, t, hidden_dim)

        # Apply attention if defined
        if self.attention:
            rnn_outputs = self.attention(rnn_outputs)  # (batch_size, t, hidden_dim)

        # Compute logits
        token_logits = self.lm_head(rnn_outputs)  # (batch_size, t, vocab_size)

        return token_logits, hidden_states, rnn_outputs

    def select_token(self, token_logits: Tensor, temperature: float) -> int:
        """
        Selects (or samples) next token from token_logits

        Args:
            token_logits (Tensor): Next token logits
                - shape (batch_size, vocab_size,)
            temperature (float): Sampling temperature. If 0, do greedy decoding.

        Returns:
            index (int): ID of next token selected
        """
        if temperature == 0:
            # Greedy Decoding
            return torch.argmax(token_logits, dim=-1)
        else:
            # Temperature Sampling
            token_logits = token_logits / temperature
            token_probs = torch.softmax(token_logits, dim=-1)
            index = torch.multinomial(token_probs, 1)[0]
            return index

    def generate(self, tokens: Tensor, max_tokens=10, temperature=0.0) -> Tensor:
        """
        Generates new tokens given `tokens` as a prefix.

        Args:
            tokens (Tensor): Input tokens
                - shape: (1, input_length,)
            max_tokens (int): Number of new tokens to generate
            temperature (float): Sampling temperature

        Returns:
            Tensor: generated tokens
                - shape: (max_tokens,)
        """
        # Get hidden states for input tokens by calling forward
        token_logits, hidden_states, attn_inputs = self(tokens)
        next_token_logits = token_logits[0, -1]

        new_tokens = []
        step = 0

        # Start generating new tokens
        while True:
            step += 1

            # Select next token based on next_token_logits
            next_token = self.select_token(next_token_logits, temperature)
            new_tokens.append(next_token.item())

            # Stop generating once we reach max_tokens
            if step >= max_tokens:
                break

            # Get next input embedding
            embed = self.embeddings(next_token.unsqueeze(0))  # (1, embed_dim)

            # Get next hidden state and next attention input state from RNN
            next_hidden_state, next_attn_input = self.rnn.step(embed, hidden_states[:, -1])

            # Update hidden states
            hidden_states = torch.cat(
                [hidden_states, next_hidden_state.unsqueeze(1)], dim=1
            )

            # Update attention inputs
            attn_inputs = torch.cat(
                [attn_inputs, next_attn_input.unsqueeze(1)], dim=1
            )

            # Call attention if defined
            if self.attention:
                next_output_state = self.attention.step(attn_inputs)
            else:
                next_output_state = next_hidden_state

            # Generate the token to be used in the next step of generation
            next_token_logits = self.lm_head(next_output_state)

        return torch.tensor(new_tokens)

def train(lm, train_data, valid_data, loss_fn, optimizer, num_sequences, batch_size):
    """
    Run one epoch of language model training

    Args:
        lm (RNNLanguageModel): RNN language model
        dataset (list[Tensor]): Train dataset
        dataset (list[Tensor]): Validation dataset
        loss_fn: PyTorch cross entropy loss function
        optimizer: PyTorch Adam optimizer
        num_sequences: The total number of sequences to train on
        batch_size: Number of sequences we process in one step

    Returns:
        List: Training losses
        List: Validation Losses
    """
    # Set the model to training mode
    lm.train()
    max_grad_norm = 1.0

    train_batch_losses = []
    train_batch_loss = 0.0
    valid_batch_losses = []

    # DO NOT change the next line
    dataset = train_data
    start_time = time.time()

    # Run validation every time we process around 10% of the training data
    val_frequency = 0.1
    val_index = int(num_sequences * val_frequency) // batch_size
    if val_index == 0:
        val_index = 1

    # Loop over the dataset
    for idx, sequence in enumerate(dataset):
        time_elapsed = round((time.time() - start_time) / 60, 6)

        # Move the sequence to the device
        sequence = sequence.to(device)

        # Stop training when we hit the num_sequences limit
        if idx == num_sequences // batch_size:
            break

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass through model
        token_logits, hidden_states, attn_inputs = lm(sequence)

        # Compute next-token classification loss
        # Permute token_logits to match the shape expected by loss function
        # token_logits: (batch_size, t, vocab_size)
        # sequence: (batch_size, t)
        # Loss is computed between token_logits[:, :-1, :] and sequence[:, 1:]
        loss = loss_fn(token_logits[:, :-1, :].permute(0, 2, 1), sequence[:, 1:])

        # Backward pass through model
        loss.backward()

        # Clip gradient norm to avoid exploding gradients
        nn.utils.clip_grad_norm_(lm.parameters(), max_grad_norm)

        # Update weights
        optimizer.step()

        # DO NOT change this - accumulate training loss
        train_batch_loss += loss.detach().cpu().item()

        if idx % val_index == 0:
            # Calculate train/val loss as normal
            train_batch_loss = (
                round(train_batch_loss / val_index, 6)
                if idx != 0
                else round(train_batch_loss, 6)
            )

            # Append to the batch loss and reset to 0
            train_batch_losses.append(train_batch_loss)
            train_batch_loss = 0.0

            print(f"Batch: {idx} | Sequence Length: {(sequence.shape[1])} | Elapsed time (minutes): {time_elapsed}")

            # Append to the validation loss
            valid_loss = round(validate(lm, valid_data, loss_fn), 6)
            valid_batch_losses.append(valid_loss)

    print(f"Train Batch Losses: {train_batch_losses}")

    return train_batch_losses, valid_batch_losses

@torch.no_grad()
def validate(lm, dataset, loss_fn):
    """
    Args:
        lm (RNNLanguageModel):
        dataset (list[Tensor]): Validation dataset
        loss_fn: PyTorch cross entropy loss function

    Returns:
        float: Average validation loss
    """
    # Set the model to eval mode
    lm.eval()

    mean_loss = 0.0
    num_batches = len(dataset)

    for i, sequence in enumerate(dataset):
        # Move the sequence to the device
        sequence = sequence.to(device)

        # Forward pass through the model
        token_dists, _, _ = lm(sequence)

        # Compute loss (Same as in train)
        loss = loss_fn(token_dists[:, :-1, :].permute(0, 2, 1), sequence[:, 1:])

        # Accumulate the loss
        mean_loss += loss.detach().cpu().item()

    return mean_loss / num_batches

@torch.no_grad()
def complete(prefix: str, num_tokens=64, temperature=0.0):
    """
    Generates text completion from language model given text prefix.

    Args:
        prefix (str): The input prefix string to begin text generation.
        num_tokens (int): Number of new tokens to generate.
        temperature (float): Sampling temperature. 0 for greedy decoding, > 0 for stochastic sampling.

    Returns:
        str: Generated text completion appended to the prefix.
    """
    # Set the language model to evaluation mode
    lm.eval()

    # Tokenize the prefix and move it to the appropriate device
    input_tokens = tokenizer.encode(prefix, add_special_tokens=False, return_tensors="pt")
    input_tokens = input_tokens.to(device)

    # Use the language model's generate method to produce the output tokens
    generated_tokens = lm.generate(input_tokens, max_tokens=num_tokens, temperature=temperature)
    
    # Decode the generated tokens back into a string
    completion = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return completion

if __name__ == "__main__":
    import argparse
    import time
    import torch
    from torch import nn, optim
    from transformers import AutoTokenizer
    from sentence_dataset import SentenceDataset  # Assume this is implemented
    from rnn_language_model import RNNLanguageModel  # Assume this is implemented
    from training_utils import train, validate, complete  # Assume these are implemented

    parser = argparse.ArgumentParser()

    # Add required arguments
    parser.add_argument("--train_data", type=str, required=True, help="Path to the training data")
    parser.add_argument("--val_data", type=str, required=True, help="Path to the validation data")
    parser.add_argument("--metrics_out", type=str, required=True, help="Path to save metrics")
    parser.add_argument("--train_losses_out", type=str, required=True, help="Path to save training losses")
    parser.add_argument("--val_losses_out", type=str, required=True, help="Path to save validation losses")
    parser.add_argument("--embed_dim", type=int, required=True, help="Embedding dimension size")
    parser.add_argument("--hidden_dim", type=int, required=True, help="Hidden dimension size")
    parser.add_argument("--dk", type=int, required=True, help="Key dimension for attention")
    parser.add_argument("--dv", type=int, required=True, help="Value dimension for attention")
    parser.add_argument("--num_sequences", type=int, required=True, help="Number of sequences for training")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")

    args = parser.parse_args()

    # Initialize the device (cuda, mps, or cpu)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("my_tokenizer")
    vocab_size = tokenizer.vocab_size

    # Initialize the RNN language model
    lm = RNNLanguageModel(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        vocab_size=vocab_size,
        key_dim=args.dk,
        value_dim=args.dv,
    ).to(device)

    print(lm)
    print(
        "Number of Parameters: ",
        sum(p.numel() for p in lm.parameters() if p.requires_grad),
    )

    # Load training and validation datasets
    print("Loading data...")
    train_data = SentenceDataset(args.train_data)
    valid_data = SentenceDataset(args.val_data)

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False
    )
    print("Finished Loading Dataset")

    # Initialize loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lm.parameters(), lr=1e-3)

    # Training loop
    print("Starting training...")
    start_time = time.time()
    train_losses, val_losses = train(
        lm,
        train_dataloader,
        valid_dataloader,
        loss_fn,
        optimizer,
        args.num_sequences,
        args.batch_size,
    )
    end_time = time.time()

    # Calculate total training time
    training_time = end_time - start_time

    # Results summary
    results = {
        "Train Losses": train_losses,
        "Valid Losses": val_losses,
        "Final Train Loss": train_losses[-1],
        "Final Valid Loss": val_losses[-1],
        "Time Taken": training_time,
    }

    for key, value in results.items():
        print(f"{key}: {value}")

    print("Training complete.")

    # Save model (comment out for submission if needed)
    # torch.save(lm.state_dict(), "model.pt")

    # Save losses and metrics
    with open(args.train_losses_out, "w") as f:
        for loss in train_losses:
            f.write(f"{loss}\n")

    with open(args.val_losses_out, "w") as f:
        for loss in val_losses:
            f.write(f"{loss}\n")

    with open(args.metrics_out, "w") as f:
        f.write(f"Final Train Loss: {train_losses[-1]}\n")
        f.write(f"Final Valid Loss: {val_losses[-1]}\n")
        f.write(f"Time Taken: {training_time}\n")

    # Example usage of the complete function
    test_prefix = "Once upon a time"
    print("Example text generation:")
    completion = complete(test_prefix, num_tokens=64, temperature=0.3)
    print(f"Prefix: {test_prefix}\nGenerated Text: {completion}")
