import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_results_comparison(final_profits_true, final_profits_estimated, savefig = True):
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

    sns.histplot(final_profits_true, bins=bins, kde=True, color='blue', label=f'True Probabilities (Avg: ${expected_profit_true:.2f})', alpha=0.6)
    sns.histplot(final_profits_estimated, bins=bins, kde=True, color='orange', label=f'Estimated Probabilities (Avg: ${expected_profit_estimated:.2f})', alpha=0.6)
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
        plt.savefig('histogram_comparison_plot.eps', format='eps')
        plt.savefig('histogram_comparison_plot.png', format='png')

def plot_cumulative_profit_comparison(exp1, exp2, season = 50, savefig = True):
    # Create a figure and axis
    plt.figure(figsize=(12, 6))

    # Plot cumulative profit for both experiments
    sns.lineplot(data=exp1, x=exp1.index, y='Cumulative Profit ($)', label='True Probabilities')
    sns.lineplot(data=exp2, x=exp2.index, y='Cumulative Profit ($)', label='Estimated Probabilities')

    # Customize the plot
    plt.title(f'Season {season} Cumulative Profit Comparison: True vs Estimated Probabilities')
    plt.xlabel('Game Number')
    plt.ylabel('Cumulative Profit ($)')
    plt.legend()

    # Save the plot as EPS and PNG
    plt.savefig('cumulative_profit_comparison.eps', format='eps')
    plt.savefig('cumulative_profit_comparison.png', format='png')

    
if __name__ == '__main__': 
    # Load the data from CSV file
    df = pd.read_csv('final_profits_comparison.csv')

    # Extract the data from the DataFrame
    final_profits_true = df['True Probabilities'].values
    final_profits_estimated = df['Estimated Probabilities'].values

    # Call the plot_results_comparison function
    plot_results_comparison(final_profits_true, final_profits_estimated)
    
    exp1 = pd.read_csv('experiment1.csv')
    exp2 = pd.read_csv('experiment2.csv')
    plot_cumulative_profit_comparison(exp1, exp2)


    
