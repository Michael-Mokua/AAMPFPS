import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class MultiTaskPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=2, n_heads=4):
        super(MultiTaskPredictor, self).__init__()
        
        # 1. Sequential Backbone (Transformer Encoder + LSTM)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.2)
        
        # 2. Multi-Task Heads
        # Head A: Match Outcome (3 classes: H/D/A)
        self.out_res = nn.Linear(hidden_dim, 3)
        # Head B: Corners (Regression)
        self.out_corners = nn.Linear(hidden_dim, 1)
        # Head C: Cards (Regression)
        self.out_cards = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        shared = self.relu(self.fc_shared(out))
        
        # Multi-Task Heads
        result_probs = self.softmax(self.head_result(shared))
        expected_corners = self.head_corners(shared)
        expected_cards = self.head_cards(shared)
        
        return result_probs, expected_corners, expected_cards

class DeepLearningEngine:
    def __init__(self):
        self.model = None
        
    def train(self, X_seq, y_results, y_corners, y_cards, epochs=20, lr=0.001):
        """
        Multi-Task Training Loop
        """
        logger.info(f"Training Multi-Task Neural Backbone for {epochs} epochs...")
        _, _, input_size = X_seq.shape
        self.model = MultiTaskPredictor(input_size=input_size)
        
        # Loss functions for different tasks
        criterion_res = nn.CrossEntropyLoss()
        criterion_regression = nn.MSELoss()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            res_p, corn_e, card_e = self.model(X_seq)
            
            loss_res = criterion_res(res_p, y_results)
            loss_corn = criterion_regression(corn_e.squeeze(), y_corners)
            loss_card = criterion_regression(card_e.squeeze(), y_cards)
            
            # Weighted total loss
            total_loss = loss_res + 0.5 * loss_corn + 0.5 * loss_card
            total_loss.backward()
            optimizer.step()
            
            if (epoch+1) % 5 == 0:
                logger.info(f'MTL Epoch [{epoch+1}/{epochs}], Total Loss: {total_loss.item():.4f}')
                
        return self.model

    def predict(self, X_seq):
        """
        Returns: [Home, Draw, Away], Expected Corners, Expected Cards
        """
        if self.model is None:
            return np.array([0.33, 0.33, 0.34]), 10.5, 3.8
            
        self.model.eval()
        with torch.no_grad():
            res_p, corn_e, card_e = self.model(X_seq)
            
        return res_p.numpy()[0], float(corn_e), float(card_e)
