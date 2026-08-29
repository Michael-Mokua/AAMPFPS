import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class AnomalyVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super(AnomalyVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

class AnomalyDetector:
    def __init__(self, input_dim):
        self.model = AnomalyVAE(input_dim)
        self.threshold = 0.5 # Placeholder threshold for reconstruction loss

    def calculate_anomaly_score(self, x_tensor):
        """
        Calculates reconstruction error. Higher = More anomalous.
        """
        self.model.eval()
        with torch.no_grad():
            recon, mu, logvar = self.model(x_tensor)
            loss = nn.functional.mse_loss(recon, x_tensor, reduction='none')
            score = torch.mean(loss, dim=1).item()
            
        return score

    def is_suspicious(self, x_tensor):
        score = self.calculate_anomaly_score(x_tensor)
        if score > self.threshold:
            logger.warning(f"Anomaly Detection: Suspicious data pattern (Score: {score:.4f})")
            return True
        return False
