import torch
import numpy as np
from ..rnn_layers import *


class CaptioningRNN:
    def __init__(
        self,
        word_to_idx,
        input_dim=512,
        wordvec_dim=128,
        hidden_dim=128,
        cell_type="rnn",
        dtype=torch.float32,
    ):
        if cell_type not in {"rnn", "lstm"}:
            raise ValueError(f'Invalid cell_type "{cell_type}"')

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)
        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        # Word embedding
        self.params["W_embed"] = torch.randn(vocab_size, wordvec_dim, dtype=dtype) / 100

        # CNN to hidden state
        self.params["W_proj"] = torch.randn(input_dim, hidden_dim, dtype=dtype) / np.sqrt(input_dim)
        self.params["b_proj"] = torch.zeros(hidden_dim, dtype=dtype)

        # RNN/LSTM weights
        dim_mul = 4 if cell_type == "lstm" else 1
        self.params["Wx"] = torch.randn(wordvec_dim, dim_mul * hidden_dim, dtype=dtype) / np.sqrt(wordvec_dim)
        self.params["Wh"] = torch.randn(hidden_dim, dim_mul * hidden_dim, dtype=dtype) / np.sqrt(hidden_dim)
        self.params["b"] = torch.zeros(dim_mul * hidden_dim, dtype=dtype)

        # Hidden to vocab
        self.params["W_vocab"] = torch.randn(hidden_dim, vocab_size, dtype=dtype) / np.sqrt(hidden_dim)
        self.params["b_vocab"] = torch.zeros(vocab_size, dtype=dtype)

        # Enable gradient tracking
        for k in self.params:
            self.params[k].requires_grad_()

    def loss(self, features, captions):
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=self.dtype)
        if isinstance(captions, np.ndarray):
            captions = torch.tensor(captions, dtype=torch.long)

        device = features.device

        captions_in = captions[:, :-1].to(device)
        captions_out = captions[:, 1:].to(device)
        mask = (captions_out != self._null).type(self.dtype)



        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]
        W_embed = self.params["W_embed"]
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        if self.cell_type == "rnn":
            forward_fn = rnn_forward
            backward_fn = rnn_backward
        else:
            forward_fn = lstm_forward
            backward_fn = lstm_backward

        h0, cache_proj = affine_forward(features, W_proj, b_proj)
        x_embed, cache_embed = word_embedding_forward(captions_in, W_embed)
        h, cache_rnn = forward_fn(x_embed, h0, Wx, Wh, b)
        scores, cache_scores = temporal_affine_forward(h, W_vocab, b_vocab)
        loss, dscores = temporal_softmax_loss(scores, captions_out, mask)

        # Backward
        dh, dW_vocab, db_vocab = temporal_affine_backward(dscores, cache_scores)
        dx, dh0, dWx, dWh, db = backward_fn(dh, cache_rnn)
        dW_embed = word_embedding_backward(dx, cache_embed)
        _, dW_proj, db_proj = affine_backward(dh0, cache_proj)

        grads = {
            "W_proj": dW_proj,
            "b_proj": db_proj,
            "W_embed": dW_embed,
            "Wx": dWx,
            "Wh": dWh,
            "b": db,
            "W_vocab": dW_vocab,
            "b_vocab": db_vocab,
        }

        return loss, grads

    def sample(self, features, max_length=30):
        N = features.shape[0]
        captions = np.full((N, max_length), self._null, dtype=np.int32)

        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]
        W_embed = self.params["W_embed"]
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        h, _ = affine_forward(features, W_proj, b_proj)
        c = torch.zeros_like(h) if self.cell_type == "lstm" else None
        word = torch.full((N,), self._start, dtype=torch.long, device=features.device)

        for t in range(max_length):
            x_embed, _ = word_embedding_forward(word, W_embed)
            if self.cell_type == "rnn":
                h, _ = rnn_step_forward(x_embed, h, Wx, Wh, b)
            else:
                h, c, _ = lstm_step_forward(x_embed, h, c, Wx, Wh, b)
            scores, _ = affine_forward(h, W_vocab, b_vocab)
            word = scores.argmax(dim=1)
            captions[:, t] = word.cpu().numpy()

        return captions
