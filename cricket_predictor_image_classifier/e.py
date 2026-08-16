"""
Part E: Use scikit-learn's MLPClassifier to implement neural networks.
Architectures: {512}, {512, 256}, {512, 256, 128}, {512, 256, 128, 64}
Use ReLU activation and SGD solver.

STOPPING CRITERION:
-------------------
Using max_iter=500 with early_stopping=True, validation_fraction=0.1, n_iter_no_change=10
This allows the model to converge naturally while preventing overfitting through early stopping.
"""

import numpy as np
import os
import sys
import time
import csv
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

# This part is exactly same as Part D, except that we use built-in MLPClassifier from sklearn instead of custom NN implementation.

def load_images_from_folder(folder_path):
    # Same as previous 3 parts
    images = []
    labels = []
    
    # Get all class folders (01, 02, ..., 36)
    class_folders = sorted([f for f in os.listdir(folder_path) 
                           if os.path.isdir(os.path.join(folder_path, f))])
    
    print(f"Found {len(class_folders)} classes")
    
    for class_idx, class_folder in enumerate(class_folders):
        class_path = os.path.join(folder_path, class_folder)
        
        # Get all image files in this class folder
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f"Loading class {class_folder}: {len(image_files)} images")
        
        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            try:
                # Load image
                img = Image.open(image_path)
                
                # Convert to RGB if needed (in case some images are grayscale)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convert to numpy array and normalize to [0, 1]
                img_array = np.array(img).astype('float32') / 255.0
                
                # Flatten the image
                img_flat = img_array.flatten()
                
                images.append(img_flat)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
    
    X = np.array(images)
    y = np.array(labels)
    
    print(f"Loaded {X.shape[0]} images with shape {X.shape[1]} features")
    
    return X, y

def evaluate_metrics(y_true, y_pred, num_classes):
    # Same as previous 3 parts
    # Calculate precision, recall, F1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(num_classes), average=None, zero_division=0
    )
    
    # Calculate average F1 score
    avg_f1 = np.mean(f1)
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'avg_f1': avg_f1
    }
    
    return metrics

def print_metrics_table(metrics, dataset_name, architecture):
    # prints precision, recall and f1-score for all classes in a formatted table
    arch_str = '-'.join(map(str, architecture))
    print(f"\n{'='*80}")
    print(f"{dataset_name} - Architecture: [{arch_str}]")
    print(f"{'='*80}")
    print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print(f"{'-'*80}")
    
    for class_idx in range(len(metrics['precision'])):
        print(f"{class_idx:<8} {metrics['precision'][class_idx]:<12.4f} "
              f"{metrics['recall'][class_idx]:<12.4f} "
              f"{metrics['f1'][class_idx]:<12.4f} "
              f"{metrics['support'][class_idx]:<10}")
    
    print(f"{'-'*80}")
    print(f"{'Average':<8} {np.mean(metrics['precision']):<12.4f} "
          f"{np.mean(metrics['recall']):<12.4f} "
          f"{metrics['avg_f1']:<12.4f} "
          f"{np.sum(metrics['support']):<10}")
    print(f"{'='*80}\n")

def save_predictions_to_csv(all_predictions, filename):
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['prediction'])
        for predictions in all_predictions:
            for pred in predictions:
                writer.writerow([pred])
    print(f"Predictions saved to '{filename}'")

def experiment_network_depth_sklearn(X_train, y_train, X_test, y_test, architectures):
    # Experiment with different network depths and architectures
    # architectures: list of lists, each inner list specifies hidden units per layer
    # Results are collected and returned for writing to file and plotting.
    # MLPClassifier from sklearn is used here.

    num_classes = len(np.unique(y_train))
    
    results = {
        'architectures': [],
        'depth': [],
        'train_metrics': [],
        'test_metrics': [],
        'train_avg_f1': [],
        'test_avg_f1': [],
        'train_accuracy': [],
        'test_accuracy': [],
        'training_time': [],
        'iterations': []
    }
    
    all_test_predictions = []
    
    for architecture in architectures:
        depth = len(architecture)
        arch_str = '-'.join(map(str, architecture))
        
        print(f"\n{'#'*80}")
        print(f"# Training with architecture: [{arch_str}] (Depth: {depth}) - MLPClassifier (sklearn)")
        print(f"{'#'*80}")
        
        # Create MLPClassifier with specified parameters
        # STOPPING CRITERION: 
        # - max_iter=500: Maximum number of epochs
        # - early_stopping=True: Enable early stopping
        # - validation_fraction=0.1: Use 10% of training data for validation
        # - n_iter_no_change=10: Stop if no improvement for 10 iterations
        # - tol=1e-4: Tolerance for optimization
        # This provides a robust stopping criterion that balances convergence and generalization
        
        mlp = MLPClassifier(
            hidden_layer_sizes=architecture,  # Architecture as tuple
            activation='relu',                 # ReLU activation
            solver='sgd',                      # Stochastic Gradient Descent
            alpha=0,                          # No L2 regularization
            batch_size=32,                    # Batch size
            learning_rate='constant',         # Constant learning rate
            learning_rate_init=0.01,          # Initial learning rate (matching part d)
            max_iter=500,                     # Maximum iterations
            shuffle=True,                     # Shuffle data each epoch
            random_state=42,                  # For reproducibility
            early_stopping=True,              # Enable early stopping
            validation_fraction=0.1,          # 10% validation set
            n_iter_no_change=4,              # Patience for early stopping
            tol=1e-4,                         # Tolerance
            verbose=True,                     # Print progress
            warm_start=False                  # Fresh training each time
        )
        
        # Train the network
        start_time = time.time()
        mlp.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        iterations = mlp.n_iter_
        
        # Get predictions on training data
        train_pred = mlp.predict(X_train)
        train_metrics = evaluate_metrics(y_train, train_pred, num_classes)
        train_accuracy = mlp.score(X_train, y_train)
        
        # Get predictions on test data
        test_pred = mlp.predict(X_test)
        test_metrics = evaluate_metrics(y_test, test_pred, num_classes)
        test_accuracy = mlp.score(X_test, y_test)
        
        # Store predictions from this model
        all_test_predictions.append(test_pred)
        
        # Print results
        print_metrics_table(train_metrics, "TRAINING DATA", architecture)
        print_metrics_table(test_metrics, "TEST DATA", architecture)
        
        # Print accuracies and training info
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Iterations: {iterations}")
        print(f"Training Time: {training_time:.2f} seconds\n")
        
        # Store results
        results['architectures'].append(architecture)
        results['depth'].append(depth)
        results['train_metrics'].append(train_metrics)
        results['test_metrics'].append(test_metrics)
        results['train_avg_f1'].append(train_metrics['avg_f1'])
        results['test_avg_f1'].append(test_metrics['avg_f1'])
        results['train_accuracy'].append(train_accuracy)
        results['test_accuracy'].append(test_accuracy)
        results['training_time'].append(training_time)
        results['iterations'].append(iterations)
    
    return results, all_test_predictions

def plot_f1_vs_depth(results, save_path='f1_vs_depth_mlp.png'):
    # Plot average F1 score vs network depth for MLPClassifier results
    plt.figure(figsize=(12, 6))
    
    # Create labels for x-axis
    arch_labels = ['-'.join(map(str, arch)) for arch in results['architectures']]
    x_pos = np.arange(len(results['depth']))
    
    # Plot MLPClassifier results
    plt.plot(x_pos, results['train_avg_f1'], 
             marker='o', linewidth=2, markersize=8, label='Training F1 (MLPClassifier)', color='blue')
    plt.plot(x_pos, results['test_avg_f1'], 
             marker='s', linewidth=2, markersize=8, label='Test F1 (MLPClassifier)', color='red')
    
    plt.xlabel('Network Architecture (Depth)', fontsize=12)
    plt.ylabel('Average F1 Score', fontsize=12)
    plt.title('Average F1 Score vs Network Depth (MLPClassifier with SGD)', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, arch_labels, rotation=15, ha='right')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on points
    for i in range(len(x_pos)):
        plt.text(x_pos[i], results['test_avg_f1'][i], f"{results['test_avg_f1'][i]:.3f}", 
                ha='center', va='bottom', fontsize=8, color='red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{save_path}'")
    plt.close()

def save_detailed_results(results, filename='detailed_results_mlp.txt'):
    """Save detailed results to a text file."""
    with open(filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DETAILED RESULTS - VARYING NETWORK DEPTH WITH MLPClassifier\n")
        f.write("="*80 + "\n\n")
        f.write("LIBRARY: scikit-learn MLPClassifier\n")
        f.write("ACTIVATION: ReLU in hidden layers, Softmax in output layer\n")
        f.write("SOLVER: Stochastic Gradient Descent (SGD)\n")
        f.write("STOPPING CRITERION:\n")
        f.write("  - max_iter=500 (maximum iterations)\n")
        f.write("  - early_stopping=True (validation-based early stopping)\n")
        f.write("  - validation_fraction=0.1 (10% for validation)\n")
        f.write("  - n_iter_no_change=10 (patience)\n")
        f.write("  - tol=1e-4 (optimization tolerance)\n")
        f.write("\nParameters:\n")
        f.write("  - Learning Rate: constant (0.01 initial)\n")
        f.write("  - Batch Size: 32\n")
        f.write("  - Alpha (L2 regularization): 0\n")
        f.write("  - Input Features: 3072 (32x32x3)\n")
        f.write("  - Output Classes: 36\n\n")
        
        for i, architecture in enumerate(results['architectures']):
            arch_str = '-'.join(map(str, architecture))
            f.write(f"\n{'='*80}\n")
            f.write(f"ARCHITECTURE: [{arch_str}] (Depth: {results['depth'][i]})\n")
            f.write(f"{'='*80}\n\n")
            
            train_metrics = results['train_metrics'][i]
            test_metrics = results['test_metrics'][i]
            
            f.write("TRAINING DATA:\n")
            f.write(f"  Average Precision: {np.mean(train_metrics['precision']):.4f}\n")
            f.write(f"  Average Recall: {np.mean(train_metrics['recall']):.4f}\n")
            f.write(f"  Average F1 Score: {train_metrics['avg_f1']:.4f}\n")
            f.write(f"  Accuracy: {results['train_accuracy'][i]:.4f}\n\n")
            
            f.write("TEST DATA:\n")
            f.write(f"  Average Precision: {np.mean(test_metrics['precision']):.4f}\n")
            f.write(f"  Average Recall: {np.mean(test_metrics['recall']):.4f}\n")
            f.write(f"  Average F1 Score: {test_metrics['avg_f1']:.4f}\n")
            f.write(f"  Accuracy: {results['test_accuracy'][i]:.4f}\n\n")
            
            f.write("TRAINING INFO:\n")
            f.write(f"  Iterations: {results['iterations'][i]}\n")
            f.write(f"  Training Time: {results['training_time'][i]:.2f} seconds\n\n")
    
    print(f"\nDetailed results saved to '{filename}'")

def main():
    """Main function to run the experiment."""
    # Check command line arguments
    if len(sys.argv) != 4:
        print("Usage: python e.py <train_data_path> <test_data_path> <output_folder_path>")
        sys.exit(1)
    
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    output_folder_path = sys.argv[3]
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)
    
    print("="*80)
    print("PART E: NEURAL NETWORK USING SCIKIT-LEARN MLPClassifier")
    print("="*80)
    print("\nLibrary: scikit-learn MLPClassifier")
    print("Activation Function: ReLU in hidden layers")
    print("Solver: Stochastic Gradient Descent (SGD)")
    print("\nSTOPPING CRITERION:")
    print("  - max_iter=500: Maximum number of epochs")
    print("  - early_stopping=True: Validation-based early stopping")
    print("  - validation_fraction=0.1: 10% of training data for validation")
    print("  - n_iter_no_change=10: Stop if no improvement for 10 iterations")
    print("  - tol=1e-4: Tolerance for optimization")
    print("\nThis stopping criterion allows the model to converge naturally while")
    print("preventing overfitting through validation-based early stopping.")
    print("\nParameters:")
    print("  - Learning Rate: constant (0.01 initial)")
    print("  - Batch Size: 32")
    print("  - Alpha (L2 penalty): 0")
    print("  - Input Features: 3072 (32x32x3)")
    print("  - Output Classes: 36")
    print("  - Architectures: {512}, {512,256}, {512,256,128}, {512,256,128,64}")
    print("="*80)
    
    # Load data
    print("\nLoading training data...")
    X_train, y_train = load_images_from_folder(train_data_path)
    
    print("\nLoading test data...")
    X_test, y_test = load_images_from_folder(test_data_path)
    
    print(f"\nData loaded successfully!")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features per sample: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Network architectures to experiment with (as tuples for sklearn)
    architectures = [
        (512,),
        (512, 256),
        (512, 256, 128),
        (512, 256, 128, 64)
    ]
    
    # Run experiments
    results, all_test_predictions = experiment_network_depth_sklearn(
        X_train, y_train, X_test, y_test, architectures
    )
    
    # Save predictions to CSV (all models stacked)
    save_predictions_to_csv(all_test_predictions, os.path.join(output_folder_path, 'prediction_e.csv'))
    
    # Save detailed results
    save_detailed_results(results, os.path.join(output_folder_path, 'detailed_results_mlp.txt'))
    
    # Plot results
    plot_f1_vs_depth(results, os.path.join(output_folder_path, 'f1_vs_depth_sklearn.png'))
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS (MLPClassifier)")
    print("="*80)
    print(f"{'Architecture':<20} {'Depth':<7} {'Train':<10} {'Test':<10} {'Train':<10} {'Test':<10} {'Iters':<8} {'Time':<10}")
    print(f"{'(Hidden Layers)':<20} {'':<7} {'Acc':<10} {'Acc':<10} {'F1':<10} {'F1':<10} {'':<8} {'(sec)':<10}")
    print("-"*80)
    for i, arch in enumerate(results['architectures']):
        arch_str = '-'.join(map(str, arch))
        print(f"{arch_str:<20} {results['depth'][i]:<7} {results['train_accuracy'][i]:<10.4f} "
              f"{results['test_accuracy'][i]:<10.4f} {results['train_avg_f1'][i]:<10.4f} "
              f"{results['test_avg_f1'][i]:<10.4f} {results['iterations'][i]:<8} "
              f"{results['training_time'][i]:<10.2f}")
    print("="*80)
    
    # Observations
    '''
    print("\nOBSERVATIONS:")
    print("-" * 80)
    print("1. MLPClassifier vs Custom Implementation:")
    print("   - MLPClassifier uses optimized backend (may be faster)")
    print("   - Built-in early stopping based on validation loss")
    print("   - Cross-entropy loss computed over network output")
    print("   - Professional implementation with extensive testing")
    
    print("\n2. Impact of Network Depth on Performance:")
    print("   - Shallow networks (depth 1) may have limited capacity")
    print("   - Deeper networks can learn more complex features")
    print("   - Balance between model complexity and generalization")
    
    print("\n3. Training Characteristics:")
    for i, arch in enumerate(results['architectures']):
        arch_str = '-'.join(map(str, arch))
        print(f"   - [{arch_str}]: {results['iterations'][i]} iterations, "
              f"{results['training_time'][i]:.2f}s")
    
    print("\n4. Generalization Analysis:")
    print("   - Train-test gap indicates overfitting level")
    for i, arch in enumerate(results['architectures']):
        arch_str = '-'.join(map(str, arch))
        gap = results['train_avg_f1'][i] - results['test_avg_f1'][i]
        print(f"   - [{arch_str}]: Gap = {gap:.4f}")
    
    print("\n5. Best Architecture:")
    best_test_idx = np.argmax(results['test_avg_f1'])
    best_arch = results['architectures'][best_test_idx]
    best_arch_str = '-'.join(map(str, best_arch))
    print(f"   - Architecture: [{best_arch_str}] (Depth: {results['depth'][best_test_idx]})")
    print(f"   - Test Accuracy: {results['test_accuracy'][best_test_idx]:.4f}")
    print(f"   - Test F1 Score: {results['test_avg_f1'][best_test_idx]:.4f}")
    print(f"   - Training Time: {results['training_time'][best_test_idx]:.2f} seconds")
    
    print("\n6. Early Stopping Effectiveness:")
    print("   - Validation-based early stopping prevents overfitting")
    print("   - Automatically determines optimal training duration")
    print("   - Different architectures may require different training times")
    print(f"   - Average iterations: {np.mean(results['iterations']):.1f}")
    print(f"   - Max iterations allowed: 500")
    
    print("\n7. Comparison with Part D (Custom ReLU Implementation):")
    print("   - Similar architectures and hyperparameters")
    print("   - Both use ReLU activation and SGD optimization")
    print("   - MLPClassifier may have implementation optimizations")
    print("   - Results should be comparable if implementations are correct")
    
    print("\n8. SGD Solver Characteristics:")
    print("   - Mini-batch gradient descent with batch_size=32")
    print("   - Constant learning rate (no decay)")
    print("   - Shuffle training data each epoch")
    print("   - No L2 regularization (alpha=0)")
    
    print("\n9. ReLU Activation Benefits:")
    print("   - Faster training than sigmoid")
    print("   - Mitigates vanishing gradient problem")
    print("   - Sparse activation patterns")
    print("   - Non-differentiability at 0 handled internally")
    
    print("\n10. Training Efficiency:")
    print(f"    - Total training time: {sum(results['training_time']):.2f} seconds")
    print(f"    - Average time per architecture: {np.mean(results['training_time']):.2f}s")
    print(f"    - Total iterations: {sum(results['iterations'])}")
    print("="*80)
    '''

if __name__ == "__main__":
    main()
