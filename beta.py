# For double headers, need to estimate variance in order to estimate how much our predicted score will change when we take the best of two.
# Scores are not normally distributed - instead, use a beta distribution which is bounded from 0 to 1 (x = pred_score/10).
# Beta distribution has two parameters: a and b.
# We can derive a and b from mean and variance. Right now we only have mean (pred_score).
# Variance within a gymnast is hard to calculate given sparse data.
# Thus, we will model the relationship between average score and variance from 2025 data, then use the variance corresponding to our pred_score.
# Relationship between standard deviation and mean looks a little better than variance and mean (compare graphs generated below),
# so use the std to calculate variance.

#%%
import pandas as pd
import numpy as np
from scipy import stats
import os


def fit_variance_models(scores_csv="2025 files/scores_long_adjusted.csv"):
    """
    Fit quadratic models to estimate variance based on mean score for each event.
    Model: var = coefficient * (mean - 10)^2

    Returns:
        dict: {event: coefficient} for VT, UB, BB, FX, AA
    """
    df = pd.read_csv(scores_csv)
    coefficients = {}

    for event in ['VT', 'UB', 'BB', 'FX']:
        event_df = df[df['Event'] == event].copy()

        # Calculate each gymnast's mean and variance
        gymnast_stats = event_df.groupby('GymnastID')['score_adj'].agg(['mean', 'std', 'count'])
        gymnast_stats = gymnast_stats[gymnast_stats['count'] >= 3]  # Need 3+ scores
        gymnast_stats = gymnast_stats.dropna()
        gymnast_stats = gymnast_stats[gymnast_stats['mean'] > 9.5]  # Focus on competitive range
        gymnast_stats['var'] = gymnast_stats['std'] ** 2

        # Fit constrained quadratic: var = coef * (mean - 10)^2
        X = (gymnast_stats['mean'] - 10) ** 2
        coef = np.dot(X, gymnast_stats['var']) / np.dot(X, X)
        coef = max(coef, 0.001)  # Enforce positive

        coefficients[event] = coef

    # AA coefficient is sum of individual events (variances add for independent events)
    coefficients['AA'] = sum(coefficients[e] for e in ['VT', 'UB', 'BB', 'FX'])

    return coefficients


def estimate_variance(mean_score, event, var_coefficients):
    """
    Estimate variance for a gymnast based on their mean score and event.
    Model: var = coefficient * (mean - max_score)^2

    Args:
        mean_score: Gymnast's predicted/mean score
        event: Event name (VT, UB, BB, FX, AA)
        var_coefficients: Dict of variance coefficients from fit_variance_models()

    Returns:
        Estimated variance
    """
    max_score = 40 if event == 'AA' else 10
    coef = var_coefficients.get(event, 0.5)

    variance = coef * (mean_score - max_score) ** 2
    return max(variance, 0.0001)  # Ensure positive


def calculate_beta_params(mean_score, variance, event):
    """
    Calculate beta distribution parameters (alpha, beta) from mean and variance.

    Using the derived equations:
        u = mean / max_score  (scaled to 0-1)
        v = variance / max_score^2  (scaled variance)
        a = u^2/v - u^3/v - u
        b = a/u - a

    Args:
        mean_score: Gymnast's predicted score (original scale)
        variance: Estimated variance (original scale)
        event: Event name

    Returns:
        (alpha, beta) parameters for beta distribution
    """
    max_score = 40 if event == 'AA' else 10

    # Scale to 0-1
    u = mean_score / max_score
    v = variance / (max_score ** 2)

    # Prevent division by zero
    v = max(v, 1e-10)

    # Calculate alpha and beta using derived equations
    a = (u * u / v) - (u * u * u / v) - u
    b = (a / u) - a if u > 0 else 1

    # Ensure valid parameters (must be > 0)
    a = max(a, 0.1)
    b = max(b, 0.1)

    return a, b


def simulate_double_header_boost(mean_score, event, var_coefficients, n_sims=10000,
                                  week=None, gymnast_id=None, individual_variances=None,
                                  individual_maxes=None):
    """
    Simulate the expected boost from a double header (max of two performances).

    Args:
        mean_score: Gymnast's predicted score
        event: Event name
        var_coefficients: Dict of variance coefficients (group-level)
        n_sims: Number of simulations
        week: Current week number (if >= 7, use individual variance)
        gymnast_id: Gymnast's ID (for individual variance lookup)
        individual_variances: Dict of {(GymnastID, Event): variance}
        individual_maxes: Dict of {(GymnastID, Event): max_score} for capping

    Returns:
        boost: Expected increase in score (E[max] - mean)
    """
    max_score = 40 if event == 'AA' else 10

    # Determine which variance to use based on week
    if week is not None and week >= 7:
        # Week 7+: use individual variance if available
        if individual_variances is not None and gymnast_id is not None:
            variance = individual_variances.get((gymnast_id, event))
            if variance is None:
                # No individual variance available (< 3 scores) - return 0 boost
                return 0.0
        else:
            # No individual variance data provided - return 0 boost
            return 0.0
    else:
        # Before week 7: use group-level variance model
        variance = estimate_variance(mean_score, event, var_coefficients)

    alpha, beta_param = calculate_beta_params(mean_score, variance, event)

    # Sample from beta distribution (on 0-1 scale)
    samples1 = stats.beta.rvs(alpha, beta_param, size=n_sims)
    samples2 = stats.beta.rvs(alpha, beta_param, size=n_sims)

    # Take max and scale back to original scale
    max_samples = np.maximum(samples1, samples2) * max_score
    expected_max = max_samples.mean()

    boost = expected_max - mean_score

    # Cap boost so adjusted score doesn't exceed historical max
    if individual_maxes is not None and gymnast_id is not None:
        hist_max = individual_maxes.get((gymnast_id, event))
        if hist_max is not None:
            max_boost = max(0, hist_max - mean_score)
            boost = min(boost, max_boost)

    return boost


def get_double_header_boosts(pred_scores_df, var_coefficients, n_sims=5000):
    """
    Calculate double header boosts for a dataframe of predictions.

    Args:
        pred_scores_df: DataFrame with 'pred_score' and 'Event' columns
        var_coefficients: Dict of variance coefficients
        n_sims: Number of simulations per gymnast

    Returns:
        Series of boosts aligned with input dataframe
    """
    boosts = []
    for _, row in pred_scores_df.iterrows():
        boost = simulate_double_header_boost(
            row['pred_score'],
            row['Event'],
            var_coefficients,
            n_sims
        )
        boosts.append(boost)
    return pd.Series(boosts, index=pred_scores_df.index)


# Cache for variance coefficients
_var_coefficients = None


def get_var_coefficients():
    """Get or compute variance coefficients (cached)."""
    global _var_coefficients
    if _var_coefficients is None:
        scores_path = "2025 files/scores_long_adjusted.csv"
        if os.path.exists(scores_path):
            _var_coefficients = fit_variance_models(scores_path)
        else:
            # Fallback defaults
            _var_coefficients = {
                'VT': 0.5, 'UB': 1.0, 'BB': 0.8, 'FX': 0.6, 'AA': 2.9
            }
    return _var_coefficients


def calculate_individual_variances(scores_csv="Files/scores_long_adjusted.csv"):
    """
    Calculate variance for each gymnast on each event from their historical scores.
    Uses MAD-based variance estimation to reduce impact of outliers (e.g., falls).

    Args:
        scores_csv: Path to scores data with GymnastID, Event, and score_adj columns

    Returns:
        dict: {(GymnastID, Event): variance} for gymnasts with 3+ scores
    """
    if not os.path.exists(scores_csv):
        return {}

    df = pd.read_csv(scores_csv)
    variances = {}

    for event in ['VT', 'UB', 'BB', 'FX', 'AA']:
        event_df = df[df['Event'] == event]

        # Group by gymnast and calculate MAD-based variance
        for gymnast_id, group in event_df.groupby('GymnastID'):
            scores = group['score_adj'].values
            if len(scores) < 3:
                continue

            # MAD-based variance: more robust to outliers like falls
            median = np.median(scores)
            mad = np.median(np.abs(scores - median))
            # For normal distribution: σ ≈ MAD / 0.6745
            sigma = mad / 0.6745
            variance = sigma ** 2

            # Ensure minimum variance to avoid division issues
            variance = max(variance, 0.0001)
            variances[(gymnast_id, event)] = variance

    return variances


# Cache for individual variances
_individual_variances = None


def get_individual_variances():
    """Get or compute individual variances (cached)."""
    global _individual_variances
    if _individual_variances is None:
        _individual_variances = calculate_individual_variances()
    return _individual_variances


def calculate_individual_maxes(scores_csv="Files/scores_long_adjusted.csv"):
    """
    Calculate historical max score for each gymnast on each event.

    Args:
        scores_csv: Path to scores data with GymnastID, Event, and score_adj columns

    Returns:
        dict: {(GymnastID, Event): max_score}
    """
    if not os.path.exists(scores_csv):
        return {}

    df = pd.read_csv(scores_csv)
    maxes = {}

    for event in ['VT', 'UB', 'BB', 'FX', 'AA']:
        event_df = df[df['Event'] == event]

        for gymnast_id, group in event_df.groupby('GymnastID'):
            maxes[(gymnast_id, event)] = group['score_adj'].max()

    return maxes


# Cache for individual maxes
_individual_maxes = None


def get_individual_maxes():
    """Get or compute individual max scores (cached)."""
    global _individual_maxes
    if _individual_maxes is None:
        _individual_maxes = calculate_individual_maxes()
    return _individual_maxes


#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Fit and display variance models
    var_coefficients = fit_variance_models()
    print("Variance coefficients (var = coef * (mean - 10)^2):")
    for event, coef in var_coefficients.items():
        print(f"  {event}: {coef:.6f}")

    # Test the implementation
    print("\n" + "=" * 60)
    print("Example boosts for different predicted scores:")
    print("=" * 60)

    for event in ['VT', 'UB', 'BB', 'FX']:
        print(f"\n{event}:")
        for pred in [9.5, 9.7, 9.85, 9.95]:
            var = estimate_variance(pred, event, var_coefficients)
            std = np.sqrt(var)
            alpha, beta_param = calculate_beta_params(pred, var, event)
            boost = simulate_double_header_boost(pred, event, var_coefficients)
            print(f"  pred={pred:.2f}: var={var:.6f}, std={std:.4f}, α={alpha:.1f}, β={beta_param:.1f}, boost=+{boost:.4f}")

    print(f"\nAA:")
    for pred in [38.0, 38.8, 39.4, 39.8]:
        var = estimate_variance(pred, 'AA', var_coefficients)
        std = np.sqrt(var)
        alpha, beta_param = calculate_beta_params(pred, var, 'AA')
        boost = simulate_double_header_boost(pred, 'AA', var_coefficients)
        print(f"  pred={pred:.1f}: var={var:.6f}, std={std:.4f}, α={alpha:.1f}, β={beta_param:.1f}, boost=+{boost:.4f}")

    # Verify the model makes sense - plot example distributions
    print("\n" + "=" * 60)
    print("Generating example distribution plots...")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (event, pred) in enumerate([('VT', 9.85), ('UB', 9.7), ('BB', 9.8), ('FX', 9.75)]):
        ax = axes[idx // 2, idx % 2]

        var = estimate_variance(pred, event, var_coefficients)
        alpha, beta_param = calculate_beta_params(pred, var, event)

        # Plot beta distribution
        x = np.linspace(0.9, 1.0, 1000)
        y = stats.beta.pdf(x, alpha, beta_param)

        ax.plot(x * 10, y / 10, 'b-', lw=2, label=f'Beta(α={alpha:.1f}, β={beta_param:.1f})')
        ax.axvline(pred, color='red', linestyle='--', lw=2, label=f'Predicted: {pred}')
        ax.fill_between(x * 10, y / 10, alpha=0.3)

        # Show expected max
        boost = simulate_double_header_boost(pred, event, var_coefficients)
        ax.axvline(pred + boost, color='purple', linestyle=':', lw=2, label=f'E[max]: {pred + boost:.3f}')

        ax.set_xlabel(f'{event} Score')
        ax.set_ylabel('Density')
        ax.set_title(f'{event}: pred={pred}, boost=+{boost:.4f}')
        ax.legend()
        ax.set_xlim(9.0, 10.0)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Extra/beta_model_verification.png', dpi=150)
    print("Saved beta_model_verification.png")
