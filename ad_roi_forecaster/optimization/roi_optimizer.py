import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple
from ad_roi_forecaster.data.schemas import CampaignDataset


def optimize_roi(campaign_data: CampaignDataset, total_budget: float) -> Tuple[List[float], float]:
    """
    Optimize ROI given a set of campaign data and a total budget.

    Args:
        campaign_data: CampaignDataset containing multiple campaign records.
        total_budget: Total budget for the campaigns.

    Returns:
        Tuple consisting of recommended spend allocation vector and expected ROI.
    """
    n = len(campaign_data.records)

    # Define the objective function: negative ROI because we use minimize
    def objective(spend_allocation):
        roi = sum(
            (record.revenue - spend) - (record.spend - spend)
            for record, spend in zip(campaign_data.records, spend_allocation)
        )
        return -roi

    # Constraints
    constraints = (
        {'type': 'eq', 'fun': lambda spend_allocation: np.sum(spend_allocation) - total_budget},  # Total budget constraint
        {'type': 'ineq', 'fun': lambda spend: spend}  # Each spend should be >= 0
    )

    # Initial guess: allocate equally, but handle zero budget case
    if total_budget == 0:
        x0 = [0.0] * n
    else:
        x0 = [total_budget / n] * n

    # Solve optimization problem
    solution = minimize(
        objective, 
        x0,
        method='SLSQP',
        constraints=constraints,
        bounds=[(0, None) for _ in range(n)]
    )

    # Calculate expected ROI
    expected_roi = -solution.fun

    return solution.x, expected_roi

