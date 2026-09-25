import torch
import torch.nn as nn


NUM_EMOTIONS = 7


class TextMLP(nn.Module):
    """RoBERTa mean-pool embeddings -> emotion class. Input dim: 768."""

    def __init__(self, input_dim=768, num_classes=NUM_EMOTIONS):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class AudioMLP(nn.Module):
    """WavLM flat embeddings -> emotion class. Input dim: 19968 (13 layers x 1536)."""

    def __init__(self, input_dim=19968, num_classes=NUM_EMOTIONS):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class FusionMLP(nn.Module):
    """
    Learned weighted sum across WavLM transformer layers, concatenated with
    RoBERTa text embeddings, -> emotion class.

    x_text:  (batch, text_dim)
    x_audio: (batch, num_layers, audio_dim)
    """

    def __init__(self, text_dim=768, audio_dim=1536, num_layers=13, num_classes=NUM_EMOTIONS):
        super().__init__()
        self.layer_w = nn.Parameter(torch.zeros(num_layers))
        self.network = nn.Sequential(
            nn.Linear(text_dim + audio_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, num_classes),
        )

    def forward(self, x_text, x_audio):
        w = torch.softmax(self.layer_w, dim=0)
        pooled = (x_audio * w[None, :, None]).sum(1)
        return self.network(torch.cat([x_text, pooled], dim=1))
