import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class BettingSimulation:
    def __init__(self, num_games=82, estimation_error=0.0, limited_estimates=False, random_seed=None):
        """
        Initialize the betting simulation.

        Parameters:
        - num_games: Number of games in the season.
        - estimation_error: Standard deviation of the estimation error (0 means perfect estimation).
        - limited_estimates: If True, adjust estimated probabilities to 0%, 50%, or 100% based on thresholds.
        - random_seed: Seed for reproducibility.
        """
        self.num_games = num_games
        self.estimation_error = estimation_error
        self.limited_estimates = limited_estimates
        self.random_seed = random_seed
        self.results = None

        if random_seed is not None:
            np.random.seed(random_seed)
        
        # List of NBA teams excluding the Bulls
        self.nba_teams = [
            'Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 'Charlotte Hornets', 'Cleveland Cavaliers',
            'Dallas Mavericks', 'Denver Nuggets', 'Detroit Pistons', 'Golden State Warriors', 'Houston Rockets',
            'Indiana Pacers', 'LA Clippers', 'Los Angeles Lakers', 'Memphis Grizzlies', 'Miami Heat',
            'Milwaukee Bucks', 'Minnesota Timberwolves', 'New Orleans Pelicans', 'New York Knicks',
            'Oklahoma City Thunder', 'Orlando Magic', 'Philadelphia 76ers', 'Phoenix Suns', 'Portland Trail Blazers',
            'Sacramento Kings', 'San Antonio Spurs', 'Toronto Raptors', 'Utah Jazz', 'Washington Wizards'
        ]

    def generate_schedule(self):
        """
        Generate a schedule of opponent teams for the Bulls.
        """
        # For simplicity, randomly select opponent teams from the list, allowing repeats
        opponents = np.random.choice(self.nba_teams, size=self.num_games, replace=True)
        return opponents

    def generate_true_probabilities(self):
        """
        Generate true probabilities of winning each game and assign opponent teams.
        For simplicity, we'll sample from a uniform distribution between 0.3 and 0.9.
        """
        true_probabilities = np.random.choice(np.arange(0, 1.1, 0.1), size=self.num_games)
        opponents = self.generate_schedule()
        return true_probabilities, opponents

    def estimate_probabilities(self, true_probabilities):
        """
        Estimate probabilities with added estimation error.
        """
        estimated_probabilities = true_probabilities + np.random.normal(
            loc=0.0, scale=self.estimation_error, size=self.num_games)
        # Round to nearest 0.1 and ensure probabilities are between 0 and 1
        estimated_probabilities = np.round(estimated_probabilities, decimals=1)
        estimated_probabilities = np.clip(estimated_probabilities, 0, 1)
        return estimated_probabilities

    def adjust_estimated_probabilities(self, estimated_probabilities):
        """
        Adjust estimated probabilities when limited_estimates is True.
        """
        adjusted_probabilities = np.where(
            estimated_probabilities < 0.3, 0.0,
            np.where(
                estimated_probabilities <= 0.7, 0.5, 1.0
            )
        )
        return adjusted_probabilities

    def determine_bets(self, probabilities):
        """
        Determine bet amounts based on probabilities.
        """
        bet_amounts = []
        for p in probabilities:
            if p < 0.5:
                bet_amounts.append(0)
            elif p < 0.6:
                bet_amounts.append(10)
            elif p < 0.7:
                bet_amounts.append(20)
            elif p < 0.8:
                bet_amounts.append(30)
            elif p < 0.9:
                bet_amounts.append(40)
            else:
                bet_amounts.append(50)
        return np.array(bet_amounts)

    def simulate_games(self, true_probabilities):
        """
        Simulate game outcomes based on true probabilities.
        Returns an array of wins (1) and losses (0).
        """
        return np.random.binomial(1, true_probabilities)

    def calculate_profits(self, bet_amounts, game_outcomes):
        """
        Calculate profits for each game.
        Winning returns the bet amount (profit), losing results in losing the bet amount.
        """
        profits = []
        for bet, outcome in zip(bet_amounts, game_outcomes):
            if bet == 0:
                profits.append(0)
            elif outcome == 1:
                profits.append(bet)  # Net profit is the bet amount
            else:
                profits.append(-bet)  # Loss is the bet amount
        return np.array(profits)

    def run_season(self):
        """
        Run a simulation for one season.
        """
        # Generate probabilities and opponents
        true_probabilities, opponents = self.generate_true_probabilities()
        estimated_probabilities = self.estimate_probabilities(true_probabilities)

        # Adjust estimated probabilities if limited_estimates is True
        if self.limited_estimates:
            adjusted_estimated_probabilities = self.adjust_estimated_probabilities(estimated_probabilities)
        else:
            adjusted_estimated_probabilities = estimated_probabilities

        # Determine bets using true probabilities
        bet_amounts_true = self.determine_bets(true_probabilities)

        # Determine bets using estimated probabilities
        bet_amounts_estimated = self.determine_bets(adjusted_estimated_probabilities)

        # Simulate game outcomes
        game_outcomes = self.simulate_games(true_probabilities)

        # Calculate profits using bets from true probabilities
        profits_true = self.calculate_profits(bet_amounts_true, game_outcomes)

        # Calculate profits using bets from estimated probabilities
        profits_estimated = self.calculate_profits(bet_amounts_estimated, game_outcomes)

        # Store results in a DataFrame
        self.results = pd.DataFrame({
            'Game': np.arange(1, self.num_games + 1),
            'Opponent': opponents,
            'True Probability (%)': (true_probabilities*100).astype(int),
            'Estimated Probability (%)': (estimated_probabilities*100).astype(int),
            'Adjusted Estimated Probability (%)': (adjusted_estimated_probabilities*100).astype(int),
            'Bet Amount ($) - True Probabilities': bet_amounts_true,
            'Bet Amount ($) - Estimated Probabilities': bet_amounts_estimated,
            'Game Outcome': game_outcomes,
            'Profit ($) - True Probabilities': profits_true,
            'Profit ($) - Estimated Probabilities': profits_estimated
        }).set_index("Game")

        self.results['Cumulative Profit ($) - True Probabilities'] = self.results['Profit ($) - True Probabilities'].cumsum()
        self.results['Cumulative Profit ($) - Estimated Probabilities'] = self.results['Profit ($) - Estimated Probabilities'].cumsum()
        return self.results
    
def simulate_multiple_seasons(num_seasons=1000, **kwargs):
    """
    Simulate multiple seasons and collect final profits.

    Parameters:
    - num_seasons: Number of seasons to simulate.
    - kwargs: Arguments to pass to BettingSimulation.

    Returns:
    - final_profits_true: Array of final profits using true probabilities.
    - final_profits_estimated: Array of final profits using estimated probabilities.
    - all_results: List of DataFrames containing results for each season.
    """
    final_profits_true = []
    final_profits_estimated = []
    all_results = []
    for i in range(num_seasons):
        sim = BettingSimulation(**kwargs)
        season_results = sim.run_season()
        final_profits_true.append(season_results['Cumulative Profit ($) - True Probabilities'].iloc[-1])
        final_profits_estimated.append(season_results['Cumulative Profit ($) - Estimated Probabilities'].iloc[-1])
        all_results.append(season_results)
    return np.array(final_profits_true), np.array(final_profits_estimated), all_results
