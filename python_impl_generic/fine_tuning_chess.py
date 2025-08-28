import itertools
import subprocess
import optuna
import os

# Define hyperparameter ranges for grid search
value_lr_range = [0.001, 0.0003, 0.0001]
self_learning_lr_range = [0.001, 0.0003, 0.0001]
max_depth_range = [50, 100, 150]
num_q_s_a_calls_range = [10, 20, 50]

# Function to run an experiment
def run_experiment(value_lr, self_learning_lr, max_depth, num_q_s_a_calls):
    """Run a training experiment with specified parameters."""
    command = [
        "python", "go_probs.py",
        "--env", "mychess6x6",
        "--cmd", "train",
        "--value_lr", str(value_lr),
        "--self_learning_lr", str(self_learning_lr),
        "--max_depth", str(max_depth),
        "--num_q_s_a_calls", str(num_q_s_a_calls),
        "--n_high_level_iterations", "10",  # Test iteration
        "--evaluate_n_games", "10"  # Reduced for faster tuning
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    return result.stdout

# Function to parse performance metrics
def parse_performance(output):
    """
    Extract performance metrics (e.g., win/loss rates) from experiment output.
    Adjust this function to match your training script's output format.
    """
    wins, losses, draws = 0, 0, 0
    for line in output.split("\n"):
        if "wins" in line.lower():  
            parts = line.split(",")
            wins = int(parts[0].split(":")[1].strip())
            losses = int(parts[1].split(":")[1].strip())
            draws = int(parts[2].split(":")[1].strip())
    total_games = wins + losses + draws
    win_rate = wins / total_games if total_games > 0 else 0.0
    return wins, losses, draws, win_rate

# Function for grid search
def grid_search():
    """Perform grid search over hyperparameter ranges."""
    param_combinations = itertools.product(value_lr_range, self_learning_lr_range, max_depth_range, num_q_s_a_calls_range)
    results = []
    for value_lr, self_learning_lr, max_depth, num_q_s_a_calls in param_combinations:
        print(f"Testing: value_lr={value_lr}, self_learning_lr={self_learning_lr}, max_depth={max_depth}, num_q_s_a_calls={num_q_s_a_calls}")
        output = run_experiment(value_lr, self_learning_lr, max_depth, num_q_s_a_calls)
        wins, losses, draws, win_rate = parse_performance(output)
        results.append({
            "value_lr": value_lr,
            "self_learning_lr": self_learning_lr,
            "max_depth": max_depth,
            "num_q_s_a_calls": num_q_s_a_calls,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate
        })
    return results

# Optuna objective function for hyperparameter tuning
def objective(trial):
    """Define objective for Optuna hyperparameter optimization."""
    value_lr = trial.suggest_loguniform("value_lr", 1e-4, 1e-2)
    self_learning_lr = trial.suggest_loguniform("self_learning_lr", 1e-4, 1e-2)
    max_depth = trial.suggest_int("max_depth", 50, 150)
    num_q_s_a_calls = trial.suggest_int("num_q_s_a_calls", 10, 50)

    # Run the experiment
    output = run_experiment(value_lr, self_learning_lr, max_depth, num_q_s_a_calls)
    wins, losses, draws, win_rate = parse_performance(output)  # Extract metrics
    return win_rate  # Maximize win rate

# Main entry point
if __name__ == "__main__":
    # Choose mode: grid search or automated tuning
    mode = input("Choose mode: 'grid' for Grid Search or 'optuna' for Automated Tuning: ").strip().lower()
    
    if mode == "grid":
        # Perform grid search
        results = grid_search()
        # Save results to a file
        with open("fine_tuning_results.json", "w") as f:
            import json
            json.dump(results, f, indent=4)
        print("Grid search completed. Results saved to 'fine_tuning_results.json'.")
    
    elif mode == "optuna":
        # Perform automated tuning with Optuna
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)
        
        # Print and save the best parameters
        print("Best Parameters:", study.best_params)
        with open("optuna_best_params.json", "w") as f:
            import json
            json.dump(study.best_params, f, indent=4)
        print("Optuna tuning completed. Best parameters saved to 'optuna_best_params.json'.")
    
    else:
        print("Invalid mode. Please choose 'grid' or 'optuna'.")
