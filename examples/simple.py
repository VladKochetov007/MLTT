# %%
import torch
from MLTT import CapitalAllocator
from MLTT.utils import change, to_weights_matrix

class TimeSeriesMomentum(CapitalAllocator):
    def __init__(self, lookback_period: int = 252):
        """
        Args:
            lookback_period (int): Number of days to look back for momentum calculation
        """
        self.lookback_period = lookback_period
        
    @property
    def min_observations(self) -> int:
        return self.lookback_period + 1
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict portfolio weights based on past returns momentum.
        
        Args:
            x (torch.Tensor): Price tensor with shape (time_steps, n_assets)
            
        Returns:
            torch.Tensor: Portfolio weights with shape (time_steps, n_assets)
        """
        # Calculate returns over lookback period
        returns = change(x, lag=self.lookback_period)

        signs = torch.sign(returns)

        # Normalize to valid portfolio weights
        weights = to_weights_matrix(signs)
        
        return weights

# Generate some random price data
n_assets = 3
n_days = 300
log_prices = torch.randn(n_days, n_assets).cumsum(dim=0)

strategy = TimeSeriesMomentum(lookback_period=60)
weights = strategy(log_prices)
print(f"Portfolio weights: {weights.numpy()}")
import torch
from MLTT.allocation import backtest_model

# Create your strategy
strategy = TimeSeriesMomentum(lookback_period=60)
# %%
result = backtest_model(
    model=strategy,
    prices=log_prices,
    commission=0.01,  # 1% trading commission
    save_weights=True  # Save portfolio weights for analysis
)

# Access equity curve and other metrics
print(f"Final equity: {torch.exp(result.log_equity[-1]).item()}")
print(f"Gross equity: {torch.exp(result.gross_equity[-1]).item()}")
print(f"Total expenses: {result.expenses_log.sum().item()}")

# %%
import matplotlib.pyplot as plt

# Create figure with custom styling
plt.figure(figsize=(12, 6), facecolor='#f7f7f7')

# Plot equity curves with labels and styling
plt.plot(
    result.log_equity, 
    label='Net Equity (with commissions)', 
    linewidth=2,
    color='#1f77b4'
)
plt.plot(
    result.gross_equity, 
    label='Gross Equity (no commissions)', 
    linewidth=2,
    color='#ff7f0e',
    linestyle='--'
)

# Add titles and labels
plt.title('Portfolio Performance', pad=20, fontsize=14)
plt.xlabel('Time', labelpad=10)
plt.ylabel('Log Equity', labelpad=10)

# Add grid and legend
plt.grid(True, alpha=0.3)
plt.legend(framealpha=0.9)

# Adjust layout and show
plt.tight_layout()
plt.show()

# %%
