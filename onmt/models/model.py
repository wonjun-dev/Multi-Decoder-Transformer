""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch.nn.functional as F
import torch


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """
    def __init__(self, encoder, decoder_A, decoder_B, router):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder_A = decoder_A
        self.decoder_B = decoder_B
        self.router = router

    def forward(self,
                src,
                tgt,
                lengths,
                bptt=False,
                latent_input=None,
                segment_input=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths = self.encoder(src, lengths)
        if bptt is False:
            self.decoder_A.init_state(src, memory_bank, enc_state)
            self.decoder_B.init_state(src, memory_bank, enc_state)
        # routing
        avg_emb = torch.mean(memory_bank,
                             dim=0)  # memory_bank (max_len, bs, 256)
        rout_prob = self.router(avg_emb)  # (bs, 2)

        dec_out_a, attns_a = self.decoder_A(tgt,
                                            memory_bank,
                                            memory_lengths=lengths,
                                            latent_input=latent_input,
                                            segment_input=segment_input)

        dec_out_b, attns_b = self.decoder_B(tgt,
                                            memory_bank,
                                            memory_lengths=lengths,
                                            latent_input=latent_input,
                                            segment_input=segment_input)
        return dec_out_a, dec_out_b, attns_a, attns_b, rout_prob

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder_A.update_dropout(dropout)
        self.decoder_B.update_dropout(dropout)


class Router(nn.Module):
    def __init__(self):
        super(Router, self).__init__()
        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 32)
        self.l4 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.dropout(self.l1(x)))
        x = F.relu(self.dropout(self.l2(x)))
        x = F.relu(self.dropout(self.l3(x)))
        x = F.softmax(self.l4(x), dim=-1)
        return x