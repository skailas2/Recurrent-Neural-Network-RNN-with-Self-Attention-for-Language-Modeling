from rnn import *
import torch
from torch import nn


def run_tests():
    # RNNCell tests
    test_rnn_cell1()
    test_rnn_cell2()
    # RNN tests
    test_rnn1()
    test_rnn2()
    # SelfAttention tests
    test_attention1()
    test_attention2()


# RNNCell Tests
def test_rnn_cell1():
    print("Testing RNNCell Test Case 1...", end="")
    input_dim = 256
    hidden_dim = 128
    batch_size = 1
    perform_rnn_cell_test(input_dim, hidden_dim, batch_size)
    print("Passed")


def test_rnn_cell2():
    print("Testing RNNCell Test Case 2...", end="")
    input_dim = 128
    hidden_dim = 64
    batch_size = 2
    perform_rnn_cell_test(input_dim, hidden_dim, batch_size)
    print("Passed")


def perform_rnn_cell_test(input_dim, hidden_dim, batch_size):
    my_cell = RNNCell(input_dim, hidden_dim)
    cell = nn.RNNCell(input_dim, hidden_dim, nonlinearity="relu", bias=True)

    # Copy weights and biases
    my_cell.i2h.weight.data.copy_(cell.weight_ih.data)
    my_cell.i2h.bias.data.copy_(cell.bias_ih.data)
    my_cell.h2h.weight.data.copy_(cell.weight_hh.data)
    my_cell.h2h.bias.data.copy_(cell.bias_hh.data)

    dummy_input = torch.rand(batch_size, input_dim)
    dummy_hidden = torch.rand(batch_size, hidden_dim)

    my_out = my_cell(dummy_input, dummy_hidden)
    out = cell(dummy_input, dummy_hidden)

    assert torch.allclose(my_out, out, atol=1e-5), "Outputs are not close"


# RNN Tests
def test_rnn1():
    print("Testing RNN Test Case 1...", end="")
    input_size = 256
    hidden_size = 128
    batch_size = 2
    seq_len = 5
    perform_rnn_test(input_size, hidden_size, batch_size, seq_len)
    print("Passed")


def test_rnn2():
    print("Testing RNN Test Case 2...", end="")
    input_size = 128
    hidden_size = 64
    batch_size = 3
    seq_len = 10
    perform_rnn_test(input_size, hidden_size, batch_size, seq_len)
    print("Passed")


def perform_rnn_test(input_size, hidden_size, batch_size, seq_len):
    num_layers = 1
    batch_first = True
    nonlinearity = "relu"

    rnn = nn.RNN(
        input_size,
        hidden_size,
        num_layers=num_layers,
        bias=True,
        nonlinearity=nonlinearity,
        batch_first=batch_first,
    )

    my_rnn = RNN(input_size, hidden_size)

    # Copy weights and biases
    my_rnn.cell.i2h.weight.data.copy_(rnn.weight_ih_l0.data)
    my_rnn.cell.i2h.bias.data.copy_(rnn.bias_ih_l0.data)
    my_rnn.cell.h2h.weight.data.copy_(rnn.weight_hh_l0.data)
    my_rnn.cell.h2h.bias.data.copy_(rnn.bias_hh_l0.data)
    my_rnn.out.weight.data.copy_(torch.eye(hidden_size))
    my_rnn.out.bias.data.zero_()

    input = torch.rand(batch_size, seq_len, input_size)

    my_hidden_states, my_output_states = my_rnn(input)
    my_hidden_states = my_hidden_states[:, -1, :].unsqueeze(0)

    output, hidden = rnn(input)
    output = my_rnn.out(output)  # Use same projection as our RNN

    assert my_hidden_states.shape == hidden.shape, "Hidden states shape mismatch"
    assert my_output_states.shape == output.shape, "Output states shape mismatch"

    assert torch.allclose(
        my_hidden_states, hidden, atol=1e-5
    ), "Hidden states are not close"
    assert torch.allclose(
        my_output_states, output, atol=1e-5
    ), "Output states are not close"


# SelfAttention Tests
def test_attention1():
    print("Testing SelfAttention Test Case 1...", end="")
    embed_dim = 256
    kdim = (
        embed_dim  # Pytorch doesn't like giving us the q bias if the dims are not equal
    )
    vdim = embed_dim
    batch_size = 1
    seq_len = 5
    perform_attention_test(embed_dim, kdim, vdim, batch_size, seq_len)
    print("Passed")


def test_attention2():
    print("Testing SelfAttention Test Case 2...", end="")
    embed_dim = 128
    kdim = embed_dim
    vdim = embed_dim
    batch_size = 2
    seq_len = 10
    perform_attention_test(embed_dim, kdim, vdim, batch_size, seq_len)
    print("Passed")


def perform_attention_test(embed_dim, kdim, vdim, batch_size, seq_len):
    num_heads = 1
    dropout = 0.0
    bias = True

    attention = nn.MultiheadAttention(
        embed_dim,
        num_heads,
        dropout,
        bias=bias,
        kdim=kdim,
        vdim=vdim,
    )

    my_attention = SelfAttention(embed_dim, kdim, vdim)

    in_proj_weight = attention.in_proj_weight
    in_proj_bias = attention.in_proj_bias

    q_weight = in_proj_weight[:embed_dim, :]
    k_weight = in_proj_weight[embed_dim : embed_dim + kdim, :]
    v_weight = in_proj_weight[embed_dim + kdim :, :]

    q_bias = in_proj_bias[:embed_dim]
    k_bias = in_proj_bias[embed_dim : embed_dim + kdim]
    v_bias = in_proj_bias[embed_dim + kdim :]

    my_attention.query_transform.weight.data.copy_(q_weight.data)
    my_attention.key_transform.weight.data.copy_(k_weight.data)
    my_attention.value_transform.weight.data.copy_(v_weight.data)

    if q_bias is not None:
        my_attention.query_transform.bias.data.copy_(q_bias.data)
    if k_bias is not None:
        my_attention.key_transform.bias.data.copy_(k_bias.data)
    if v_bias is not None:
        my_attention.value_transform.bias.data.copy_(v_bias.data)

    my_attention.output_transform.weight.data.copy_(attention.out_proj.weight.data)
    my_attention.output_transform.bias.data.copy_(attention.out_proj.bias.data)

    sequence = torch.rand(batch_size, seq_len, embed_dim)

    t = seq_len - 1
    y_all = sequence[:, : t + 1, :]

    query = sequence[:, t, :].unsqueeze(0)
    key = sequence[:, : t + 1, :].transpose(0, 1)
    value = sequence[:, : t + 1, :].transpose(0, 1)

    my_output_state = my_attention.step(y_all)

    attn_output, _ = attention(query, key, value)
    attn_output = attn_output.squeeze(0)

    assert torch.allclose(
        my_output_state, attn_output, atol=1e-5
    ), "Outputs are not close"


run_tests()
