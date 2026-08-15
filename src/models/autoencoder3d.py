from torch import nn


class DefectAutoencoder3D(nn.Module):
    """Small 3D encoder-decoder mapping clean volumes to defect-injected volumes."""

    def __init__(self, latent_channels=32, dropout=0.1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.GELU(),
            nn.MaxPool3d(2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.GELU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, latent_channels, 3, padding=1),
            nn.GELU(),
            nn.Dropout3d(dropout),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_channels, 16, 2, stride=2),
            nn.GELU(),
            nn.Conv3d(16, 16, 3, padding=1),
            nn.GELU(),

            nn.ConvTranspose3d(16, 8, 2, stride=2),
            nn.GELU(),
            nn.Conv3d(8, 8, 3, padding=1),
            nn.GELU(),

            nn.Conv3d(8, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
