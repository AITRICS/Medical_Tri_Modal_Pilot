from builder.models.src.transformer.attention import *
from builder.models.src.transformer.encoder import *
from builder.models.src.transformer.module import *


class DecoderInterface(nn.Module):
    def __init__(self):
        super(DecoderInterface, self).__init__()

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        """ Update dropout probability of encoder """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

class BaseDecoder(DecoderInterface):
    """ ASR Decoder Super Class for KoSpeech model implementation """
    def __init__(self):
        super(BaseDecoder, self).__init__()

    def forward(self, targets: Tensor, encoder_outputs: Tensor, **kwargs) -> Tensor:
        """
        Forward propagate a `encoder_outputs` for training.

        Args:
            targets (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
            encoder_outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``

        Returns:
            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
        """
        raise NotImplementedError

    @torch.no_grad()
    def decode(self, encoder_outputs: Tensor, *args) -> Tensor:
        """
        Decode encoder_outputs.

        Args:
            encoder_outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``

        Returns:
            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
        """
        raise NotImplementedError