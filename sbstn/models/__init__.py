"""S-BSTN building blocks.

    sca.SCA               selective cross-attention + SAN pruning (spatial)
    encoder.BiEncoder     per-direction SCA + LSTM encoders
    bita.BiTA             bi-temporal attention, late fusion
    decoder.DecoderLSTM   autoregressive LSTM decoder, linear head
    sbstn.SBSTN           the assembled model; ablations via sbstn.VARIANTS
"""
from .sbstn import SBSTN, VARIANTS, build
from .baselines import Persistence

__all__ = ["SBSTN", "VARIANTS", "build", "Persistence"]
