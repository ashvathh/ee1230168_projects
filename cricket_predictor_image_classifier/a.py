import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decision_tree import DecisionTree, accuracy, count_nodes


def run_experiments_depth(train_csv='train.csv', test_csv='test.csv', target_col='result', depths=[5,10,15,20], min_samples_split=2, output_folder=None):
    # Load data
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Basic safety: check target col exists
    if target_col not in train_df.columns:
        raise ValueError(f"target_col '{target_col}' not found in train csv columns: {train_df.columns.tolist()}")
    if target_col not in test_df.columns:
        raise ValueError(f"target_col '{target_col}' not found in test csv columns: {test_df.columns.tolist()}")

    train_accs = []
    test_accs = []
    training_times = []
    models = {}
    for d in depths:
        print(f"\nTraining Decision Tree with max_depth = {d}")
        dt = DecisionTree(max_depth=d, min_samples_split=min_samples_split, target_col=target_col)
        
        start_time = time.time()
        dt.fit(train_df)
        training_time = time.time() - start_time
        training_times.append(training_time)
        
        y_train = train_df[target_col].values
        y_test = test_df[target_col].values

        preds_train = dt.predict(train_df)
        preds_test = dt.predict(test_df)

        a_train = accuracy(y_train, preds_train)
        a_test = accuracy(y_test, preds_test)
        n_nodes = count_nodes(dt.root)
        print(f"Train acc: {a_train:.4f} | Test acc: {a_test:.4f} | Nodes: {n_nodes} | Training time: {training_time:.3f}s")
        train_accs.append(a_train)
        test_accs.append(a_test)
        models[d] = dt

    # Plotting
    plt.figure(figsize=(8,5))
    plt.plot(depths, train_accs, marker='o', label='Train Accuracy')
    plt.plot(depths, test_accs, marker='o', label='Test Accuracy')
    plt.xlabel('Max Tree Depth')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Accuracy vs Max Depth')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot if output_folder is provided
    if output_folder:
        plot_filename = "a_plot.png"
        plot_path = os.path.join(output_folder, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
    
    plt.show()
    plt.close()

    # return results and models
    return {
        'depths': depths,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'training_times': training_times,
        'models': models
    }



if __name__ == "__main__":
    # Parse command-line arguments
    if len(sys.argv) != 5:
        print("Usage: python a.py <train_data_path> <val_data_path> <test_data_path> <output_file_path>")
        sys.exit(1)
    
    train_csv = sys.argv[1]
    valid_csv = sys.argv[2]
    test_csv = sys.argv[3]
    output_path = sys.argv[4]
    
    # Handle both folder path and file path
    if output_path.endswith('.csv'):
        output_file = output_path
        output_folder = os.path.dirname(output_path)
    else:
        output_folder = output_path
        output_file = os.path.join(output_path, 'result.csv')
    
    # Create output folder if it doesn't exist
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    # Load data
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)
    test_df = pd.read_csv(test_csv)
    target_col1 = 'result'   
    
    
    #normal decision tree experiments
    print("Running Decision Tree experiments with Entropy splitting...\n")
    depths = [5, 10, 15, 20]  # experiment depths as asked
    results = run_experiments_depth(train_csv=train_csv, test_csv=test_csv, target_col=target_col1, depths=depths, output_folder=output_folder)

    # Print a small summary table
    
    print("\nSummary:")
    for d, ta, te, tt in zip(results['depths'], results['train_accs'], results['test_accs'], results['training_times']):
        print(f"Depth {d:2d} | Train acc: {ta:.4f} | Test acc: {te:.4f} | Time: {tt:.3f}s")

    # Pick best test accuracy and save model if desired
    best_idx = np.argmax(results['test_accs'])
    best_depth = results['depths'][best_idx]
    best_model = results['models'][best_depth]
    print(f"\nBest test accuracy = {results['test_accs'][best_idx]:.4f} at depth = {best_depth}")
    
    # Make predictions on test data using the best model
    test_predictions = best_model.predict(test_df)
    
    # Save predictions to output file
    output_df = pd.DataFrame({'result': test_predictions})
    output_df.to_csv(output_file, index=False)
    print(f"\nTest predictions saved to: {output_file}")

    