import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_probs_results_comparison(final_profits_true, final_profits_estimated, savefig = True):
    """
    Generate comparison plots between true probabilities and estimated probabilities results.

    Parameters:
    - final_profits_true: Array of final profits using true probabilities.
    - final_profits_estimated: Array of final profits using estimated probabilities.
    """
    # Expected profit vs. actual profit - comparison
    expected_profit_true = np.mean(final_profits_true)
    expected_profit_estimated = np.mean(final_profits_estimated)

    plt.figure(figsize=(12, 6))
    # Calculate the range and bins for both datasets
    min_profit = min(final_profits_true.min(), final_profits_estimated.min())
    max_profit = max(final_profits_true.max(), final_profits_estimated.max())
    bins = np.linspace(min_profit, max_profit, 31)  # 31 edges to create 30 bins

    sns.histplot(final_profits_true, bins=bins, kde=False, color='blue', label=f'True Probabilities (Avg: ${expected_profit_true:.2f})', alpha=0.6)
    sns.histplot(final_profits_estimated, bins=bins, kde=False, color='orange', label=f'Estimated Probabilities (Avg: ${expected_profit_estimated:.2f})', alpha=0.6)
    plt.axvline(expected_profit_true, color='blue', linestyle='--')
    plt.axvline(expected_profit_estimated, color='orange', linestyle='--')
    plt.title('Utilizing a DSS with Perfect and Imperfect Estimation of Win Probabilities over 1000 Seasons')
    plt.xlabel('Season End Cumulative Profit ($)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # Risk analysis - comparison
    worst_case_true = np.min(final_profits_true)
    best_case_true = np.max(final_profits_true)
    worst_case_estimated = np.min(final_profits_estimated)
    best_case_estimated = np.max(final_profits_estimated)

    print(f'Using True Probabilities - Worst-case final profit: ${worst_case_true:.2f}')
    print(f'Using True Probabilities - Best-case final profit: ${best_case_true:.2f}')
    print(f'Using Estimated Probabilities - Worst-case final profit: ${worst_case_estimated:.2f}')
    print(f'Using Estimated Probabilities - Best-case final profit: ${best_case_estimated:.2f}')
    if savefig:
        plt.savefig('./data/histogram_comparison_plot_probs.eps', format='eps')
        plt.savefig('./data/histogram_comparison_plot_probs.png', format='png')
        
def plot_dss_results_comparison(final_profits_true, final_profits_estimated, savefig = True):
    """
    Generate comparison plots between true probabilities and estimated probabilities results.

    Parameters:
    - final_profits_true: Array of final profits using true probabilities.
    - final_profits_estimated: Array of final profits using estimated probabilities.
    """
    # Expected profit vs. actual profit - comparison
    expected_profit_true = np.mean(final_profits_true)
    expected_profit_estimated = np.mean(final_profits_estimated)

    plt.figure(figsize=(12, 6))
    # Calculate the range and bins for both datasets
    min_profit = min(final_profits_true.min(), final_profits_estimated.min())
    max_profit = max(final_profits_true.max(), final_profits_estimated.max())
    bins = np.linspace(min_profit, max_profit, 31)  # 31 edges to create 30 bins

    sns.histplot(final_profits_true, bins=bins, kde=False, color='blue', label=f'Using a DSS (Avg: ${expected_profit_true:.2f})', alpha=0.6)
    sns.histplot(final_profits_estimated, bins=bins, kde=False, color='green', label=f'Random Bets (Avg: ${expected_profit_estimated:.2f})', alpha=0.6)
    plt.axvline(expected_profit_true, color='blue', linestyle='--')
    plt.axvline(expected_profit_estimated, color='orange', linestyle='--')
    plt.title('Random Betting vs Using a Decision Support System Over 1000 Seasons')
    plt.xlabel('Season End Cumulative Profit ($)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # Risk analysis - comparison
    worst_case_true = np.min(final_profits_true)
    best_case_true = np.max(final_profits_true)
    worst_case_estimated = np.min(final_profits_estimated)
    best_case_estimated = np.max(final_profits_estimated)

    print(f'Using True Probabilities - Worst-case final profit: ${worst_case_true:.2f}')
    print(f'Using True Probabilities - Best-case final profit: ${best_case_true:.2f}')
    print(f'Using Estimated Probabilities - Worst-case final profit: ${worst_case_estimated:.2f}')
    print(f'Using Estimated Probabilities - Best-case final profit: ${best_case_estimated:.2f}')
    if savefig:
        plt.savefig('./data/histogram_comparison_plot_dss.eps', format='eps')
        plt.savefig('./data/histogram_comparison_plot_dss.png', format='png')

def plot_cumulative_profit_comparison(exp1, exp2, exp3, season = 50, savefig = True):
    # Create a figure and axis
    plt.figure(figsize=(12, 6))

    # Plot cumulative profit for both experiments
    sns.lineplot(data=exp1, color = 'green', x=exp1.index, y='Cumulative Profit ($)', linewidth = 3, label='Random Bets')
    sns.lineplot(data=exp2, color = 'orange', x=exp2.index, y='Cumulative Profit ($)', linewidth = 3, label='DSS w/ True Probabilities')
    sns.lineplot(data=exp3, color = 'blue', x=exp3.index, y='Cumulative Profit ($)', linewidth = 3, label='DSS w/ Estimated Probabilities')
    plt.axhline(y=0, color='grey', linestyle='--', alpha=0.5)

    # Customize the plot
    plt.title(f'Season {season} Cumulative Profit Comparison: True vs Estimated Probabilities')
    plt.xlabel('Game Number')
    plt.ylabel('Cumulative Profit ($)')
    plt.legend()

    # Save the plot as EPS and PNG
    plt.savefig('./data/cumulative_profit_comparison.eps', format='eps')
    plt.savefig('./data/cumulative_profit_comparison.png', format='png')

def plot_all_results_comparison(final_profits_true, final_profits_estimated, final_profits_random, savefig=True):
    """
    Generate comparison plots between true probabilities, estimated probabilities, and random betting results.

    Parameters:
    - final_profits_true: Array of final profits using true probabilities.
    - final_profits_estimated: Array of final profits using estimated probabilities.
    - final_profits_random: Array of final profits using random bets.
    """
    # Expected profit vs. actual profit - comparison
    expected_profit_true = np.mean(final_profits_true)
    expected_profit_estimated = np.mean(final_profits_estimated)
    expected_profit_random = np.mean(final_profits_random)

    plt.figure(figsize=(12, 6))
    # Calculate the range and bins for all datasets
    min_profit = min(final_profits_true.min(), final_profits_estimated.min(), final_profits_random.min())
    max_profit = max(final_profits_true.max(), final_profits_estimated.max(), final_profits_random.max())
    bins = np.linspace(min_profit, max_profit, 31)  # 31 edges to create 30 bins

    sns.histplot(final_profits_true, bins=bins, kde=False, color='blue', label=f'DSS with true probabiltiies (Avg: ${expected_profit_true:.2f})', alpha=0.6)
    sns.histplot(final_profits_estimated, bins=bins, kde=False, color='orange', label=f'DSS with estimated probabilities (Avg: ${expected_profit_estimated:.2f})', alpha=0.6)
    sns.histplot(final_profits_random, bins=bins, kde=False, color='green', label=f'Random Bets (Avg: ${expected_profit_random:.2f})', alpha=0.6)
    plt.axvline(expected_profit_true, color='blue', linestyle='--')
    plt.axvline(expected_profit_estimated, color='orange', linestyle='--')
    plt.axvline(expected_profit_random, color='green', linestyle='--')
    plt.title('Comparison of Profits: Using Decision Support vs. No Decision Support vs. Random Betting Over 1000 Seasons')
    plt.xlabel('Season End Cumulative Profit ($)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # Risk analysis - comparison
    worst_case_true = np.min(final_profits_true)
    best_case_true = np.max(final_profits_true)
    worst_case_estimated = np.min(final_profits_estimated)
    best_case_estimated = np.max(final_profits_estimated)
    worst_case_random = np.min(final_profits_random)
    best_case_random = np.max(final_profits_random)

    print(f'Using True Probabilities - Worst-case final profit: ${worst_case_true:.2f}')
    print(f'Using True Probabilities - Best-case final profit: ${best_case_true:.2f}')
    print(f'Using Estimated Probabilities - Worst-case final profit: ${worst_case_estimated:.2f}')
    print(f'Using Estimated Probabilities - Best-case final profit: ${best_case_estimated:.2f}')
    print(f'Using Random Bets - Worst-case final profit: ${worst_case_random:.2f}')
    print(f'Using Random Bets - Best-case final profit: ${best_case_random:.2f}')
    
    if savefig:
        plt.savefig('./data/histogram_comparison_plot_all.eps', format='eps')
        plt.savefig('./data/histogram_comparison_plot_all.png', format='png')
    
if __name__ == '__main__': 
    # Load the data from CSV file
    df = pd.read_csv('./data/final_profits_comparison.csv')

    # Extract the data from the DataFrame
    final_profits_true = df['True Probabilities'].values
    final_profits_estimated = df['Estimated Probabilities'].values

    # Call the plot_results_comparison function
    plot_probs_results_comparison(final_profits_true, final_profits_estimated)
    
    exp1 = pd.read_csv('experiment1.csv')
    exp2 = pd.read_csv('experiment2.csv')
    plot_cumulative_profit_comparison(exp1, exp2)


    
