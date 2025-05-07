"""
Module with different regression models and loss functions. Simple, but with some meaningful tricks.
"""

import copy
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from torch.utils.data import DataLoader, TensorDataset


class NormalizedProfitLoss(nn.Module):
    def __init__(self, alpha_base: float = 0.1, base_loss = None):
        """
        Loss function that maximizes Sharpe-like ratio along with minimizing base loss.
        
        Args:
            alpha_base: Weight coefficient for base_loss (0-1)
            base_loss: Base loss function, defaults to MSE if None
        """
        super().__init__()
        self.alpha_base = alpha_base
        self.base_loss = base_loss if base_loss is not None else F.mse_loss
    
    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss as a combination of base loss and negative Sharpe ratio.
        
        Loss = -avg(preds*target)*(1-alpha_base)/std(preds*target) + base_loss*alpha_base
        
        Args:
            preds: Model predictions
            target: Ground truth values
            
        Returns:
            Combined loss value
        """
        # Calculate product of predictions and targets
        prod = preds * target
        
        # Calculate mean and standard deviation
        mean_prod = torch.mean(prod)
        std_prod = torch.std(prod)
        
        # Add small epsilon to prevent division by zero
        eps = 1e-6
        
        # Calculate sharpe-like ratio (negative because we want to maximize it)
        sharpe_term = -mean_prod / (std_prod + eps) * (1 - self.alpha_base)
        
        # Calculate base loss term
        base_term = self.base_loss(preds, target) * self.alpha_base
        
        # Return combined loss
        return sharpe_term + base_term
    

class LinearRegressionL2(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1, weight_decay: float = 0.01, device: str | torch.device = "cpu"):
        """
        Linear regression model with L2 regularization.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (default=1 for standard regression)
            weight_decay: L2 regularization coefficient
            device: Device to run the model on ("cpu", "cuda", or a specific torch device)
        """
        super().__init__()
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.linear = nn.Linear(input_dim, output_dim)
        self.weight_decay = weight_decay
        self.to(self.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Predictions of shape (batch_size, output_dim)
        """
        return self.linear(x)
    
    def l2_penalty(self) -> torch.Tensor:
        """
        Calculate L2 penalty on weights.
        
        Returns:
            L2 norm of weights multiplied by weight_decay
        """
        return self.weight_decay * torch.sum(self.linear.weight ** 2)
    
    def fit(self, 
            X_train: torch.Tensor, 
            y_train: torch.Tensor, 
            loss_fn: Callable = F.mse_loss,
            epochs: int = 100, 
            batch_size: int = 32, 
            lr: float = 0.01,
            verbose: bool = False) -> list[float]:
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            loss_fn: Loss function to minimize
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            verbose: Whether to print progress
            
        Returns:
            List of training losses per epoch
        """
        # Move data to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                # Forward pass
                preds = self(batch_X)
                
                # Calculate loss with L2 regularization
                loss = loss_fn(preds, batch_y) + self.l2_penalty()
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
                
        return losses
    
    def evaluate(self, 
                X_val: torch.Tensor, 
                y_val: torch.Tensor, 
                loss_fn: Callable = F.mse_loss) -> float:
        """
        Evaluate the model on validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
            loss_fn: Loss function to calculate
            
        Returns:
            Validation loss
        """
        # Move data to device
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        
        with torch.no_grad():
            preds = self(X_val)
            val_loss = loss_fn(preds, y_val)
            
        return val_loss.item()


def optimize_l2_regularization(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    input_dim: int,
    output_dim: int = 1,
    loss_fn: Callable = NormalizedProfitLoss(),
    n_trials: int = 20,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.01,
    device: str | torch.device = "cpu"
) -> tuple[float, LinearRegressionL2]:
    """
    Use Optuna to find the optimal L2 regularization coefficient.
    
    Args:
        - `X_train` (torch.Tensor): Training features
        - `y_train` (torch.Tensor): Training targets
        - `X_val` (torch.Tensor): Validation features
        - `y_val` (torch.Tensor): Validation targets
        - `input_dim` (int): Input feature dimension
        - `output_dim` (int): Output dimension
        - `loss_fn` (Callable): Loss function to minimize
        - `n_trials` (int): Number of Optuna trials
        - `epochs` (int): Number of training epochs for each trial
        - `batch_size` (int): Batch size for training
        - `lr` (float): Learning rate
        - `device` (str | torch.device): Device to run the model on ("cpu", "cuda", or a specific torch device)
        
    Returns:
        - tuple[float, LinearRegressionL2]: Tuple of best weight_decay value and the best model
    """
    def objective(trial):
        # Sample weight decay coefficient from log uniform distribution
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1, log=True)
        
        # Create and train model
        model = LinearRegressionL2(input_dim, output_dim, weight_decay, device)
        model.fit(X_train, y_train, loss_fn, epochs, batch_size, lr)
        
        # Evaluate on validation set
        val_loss = model.evaluate(X_val, y_val, loss_fn)
        
        return val_loss
    
    # Create study and optimize
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    # Get best parameters and retrain model
    best_weight_decay = study.best_params["weight_decay"]
    best_model = LinearRegressionL2(input_dim, output_dim, best_weight_decay, device)
    best_model.fit(X_train, y_train, loss_fn, epochs, batch_size, lr)
    
    print(f"Best weight_decay: {best_weight_decay:.6f}")
    print(f"Best validation loss: {study.best_value:.6f}")
    
    return best_weight_decay, best_model
    
