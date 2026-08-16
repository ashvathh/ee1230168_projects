import numpy as np
import os
import sys
import time
import csv
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork


def load_images_from_folder(folder_path):
    # Loads images from a folder which has classes as subfolders (01, 02, ..., 36)
    # returns the data and labels as numpy arrays.
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
    
    X = np.array(images) # array of shape (num_samples, 3072) - images flattened to 32*32*3
    y = np.array(labels) # array of shape (num_samples,) labels as integers
    
    print(f"Loaded {X.shape[0]} images with shape {X.shape[1]} features")
    
    return X, y

def evaluate_metrics(y_true, y_pred, num_classes):
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

def print_metrics_table(metrics, dataset_name, hidden_units):
    # prints precision, recall and f1-score for all classes in a formatted table
    print(f"\n{'='*80}")
    print(f"{dataset_name} - Hidden Units: {hidden_units}")
    print(f"{'='*80}")
    print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print(f"{'-'*80}")
    
    for class_idx in range(len(metrics['precision'])):
        print(f"{class_idx:<8} {metrics['precision'][class_idx]:<12.4f} "
              f"{metrics['recall'][class_idx]:<12.4f} "
              f"{metrics['f1'][class_idx]:<12.4f} "
              f"{metrics['support'][class_idx]:<10}")
    
    print(f"{'-'*80}")

    # Printing average metrics
    print(f"{'Average':<8} {np.mean(metrics['precision']):<12.4f} "
          f"{np.mean(metrics['recall']):<12.4f} "
          f"{metrics['avg_f1']:<12.4f} "
          f"{np.sum(metrics['support']):<10}")
    print(f"{'='*80}\n")

def experiment_hidden_units(X_train, y_train, X_test, y_test, hidden_units_list):
    # Experiment with different numbers of hidden units in a single hidden layer.
    # Results are collected and returned for writing to file and plotting.

    num_features = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    results = {
        'hidden_units': [],
        'train_metrics': [],
        'test_metrics': [],
        'train_avg_f1': [],
        'test_avg_f1': [],
        'train_accuracy': [],
        'test_accuracy': [],
        'training_time': [],
        'epochs_trained': []
    }
    # Each is an array corresponding to each hidden unit setting

    # Train neural networks with different hidden units
    for hidden_units in hidden_units_list:
        print(f"\n{'#'*80}")
        print(f"# Training with {hidden_units} hidden units")
        print(f"{'#'*80}")
        
        # Create neural network with single hidden layer
        nn = NeuralNetwork(
            num_features=num_features,
            hidden_layers=[hidden_units],  # Single hidden layer
            num_classes=num_classes,
            learning_rate=0.01,
            batch_size=32
        )


        # STOPPING CRITERION for early stopping: Loss change < 0.001 or max 250 epochs
        start_time = time.time()
        history = nn.train(X_train, y_train, max_epochs=250, track_progress=True, 
                          min_loss_change=0.001, patience=1)
        training_time = time.time() - start_time
        
        epochs_trained = history['epochs_trained']

        # Get predictions on training data
        train_pred, train_class_probs = nn.predict(X_train)
        train_metrics = evaluate_metrics(y_train, train_pred, num_classes)
        train_accuracy = history['accuracy'][-1]
        
        # Get predictions on test data
        test_pred, test_class_probs = nn.predict(X_test)
        test_metrics = evaluate_metrics(y_test, test_pred, num_classes)
        test_accuracy = nn.evaluate(X_test, y_test)
        
        # Print precision recall f1 tables for all classes
        print_metrics_table(train_metrics, "TRAINING DATA", hidden_units)
        print_metrics_table(test_metrics, "TEST DATA", hidden_units)
        
        # Print accuracies and training info
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Epochs Trained: {epochs_trained}")
        print(f"Training Time: {training_time:.2f} seconds\n")
        
        # Store results
        results['hidden_units'].append(hidden_units)
        results['train_metrics'].append(train_metrics)
        results['test_metrics'].append(test_metrics)
        results['train_avg_f1'].append(train_metrics['avg_f1'])
        results['test_avg_f1'].append(test_metrics['avg_f1'])
        results['train_accuracy'].append(train_accuracy)
        results['test_accuracy'].append(test_accuracy)
        results['training_time'].append(training_time)
        results['epochs_trained'].append(epochs_trained)
        results.setdefault('test_predictions', []).append(test_pred) # for saving predictions later
        # has nested list of predictions for each hidden unit setting
    
    return results

def save_predictions_to_csv(all_predictions, filename):
    # saves predictions to a CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['prediction'])
        for predictions in all_predictions:
            for pred in predictions:
                writer.writerow([pred])
    print(f"\nPredictions saved to '{filename}'")

def plot_f1_vs_hidden_units(results, save_path='f1_vs_hidden_units.png'):
    # Plot average F1 score vs number of hidden units.
    plt.figure(figsize=(10, 6))
    
    plt.plot(results['hidden_units'], results['train_avg_f1'], 
             marker='o', linewidth=2, markersize=8, label='Training F1')
    plt.plot(results['hidden_units'], results['test_avg_f1'], 
             marker='s', linewidth=2, markersize=8, label='Test F1')
    
    plt.xlabel('Number of Hidden Units', fontsize=12)
    plt.ylabel('Average F1 Score', fontsize=12)
    plt.title('Average F1 Score vs Number of Hidden Units', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')  # Log scale for better visualization
    
    # Add value labels on points
    for i, hu in enumerate(results['hidden_units']):
        plt.text(hu, results['train_avg_f1'][i], f"{results['train_avg_f1'][i]:.3f}", 
                ha='center', va='bottom', fontsize=9)
        plt.text(hu, results['test_avg_f1'][i], f"{results['test_avg_f1'][i]:.3f}", 
                ha='center', va='top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{save_path}'")
    plt.close()

def save_detailed_results(results, filename='detailed_results.txt'):
    # Saving detailed results to a text file
    with open(filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DETAILED RESULTS - VARYING HIDDEN LAYER UNITS\n")
        f.write("="*80 + "\n\n")
        f.write("STOPPING CRITERION: Loss change < 0.001 or max 250 epochs\n")
        f.write("Learning Rate: 0.01\n")
        f.write("Batch Size: 32\n")
        f.write("Input Features: 3072 (32x32x3)\n")
        f.write("Output Classes: 36\n\n")
        
        for i, hidden_units in enumerate(results['hidden_units']):
            f.write(f"\n{'='*80}\n")
            f.write(f"HIDDEN UNITS: {hidden_units}\n")
            f.write(f"{'='*80}\n\n")
            
            train_metrics = results['train_metrics'][i]
            test_metrics = results['test_metrics'][i]
            
            # Train metrics
            f.write("TRAINING DATA:\n")
            f.write(f"  Average Precision: {np.mean(train_metrics['precision']):.4f}\n")
            f.write(f"  Average Recall: {np.mean(train_metrics['recall']):.4f}\n")
            f.write(f"  Average F1 Score: {train_metrics['avg_f1']:.4f}\n")
            f.write(f"  Accuracy: {results['train_accuracy'][i]:.4f}\n\n")
            
            # Test metrics
            f.write("TEST DATA:\n")
            f.write(f"  Average Precision: {np.mean(test_metrics['precision']):.4f}\n")
            f.write(f"  Average Recall: {np.mean(test_metrics['recall']):.4f}\n")
            f.write(f"  Average F1 Score: {test_metrics['avg_f1']:.4f}\n")
            f.write(f"  Accuracy: {results['test_accuracy'][i]:.4f}\n\n")
            
            f.write("TRAINING INFO:\n")
            f.write(f"  Epochs Trained: {results['epochs_trained'][i]}\n")
            f.write(f"  Training Time: {results['training_time'][i]:.2f} seconds\n\n")
    
    print(f"\nDetailed results saved to '{filename}'")

def main():
    # Main function to run the experiment.
    # Check command line arguments
    if len(sys.argv) != 4:
        print("Usage: python b.py <train_data_path> <test_data_path> <output_folder_path>")
        sys.exit(1)
    
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    output_folder_path = sys.argv[3]
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)
    
    print("="*80)
    print("PART B: NEURAL NETWORK WITH VARYING HIDDEN LAYER UNITS")
    print("="*80)
    print("\nSTOPPING CRITERION: Loss change < 0.001 or max 250 epochs")
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
    
    # Hidden layer units to experiment with
    hidden_units_list = [1, 5, 10, 50, 100]
    
    # Run experiments
    results = experiment_hidden_units(X_train, y_train, X_test, y_test, hidden_units_list)
    
    # Save predictions to CSV in output folder
    prediction_file = os.path.join(output_folder_path, 'prediction_b.csv')
    save_predictions_to_csv(results['test_predictions'], prediction_file)
    
    # Save detailed results in output folder
    results_file = os.path.join(output_folder_path, 'detailed_results_b.txt')
    save_detailed_results(results, results_file)
    
    # Plot results and save in output folder
    plot_file = os.path.join(output_folder_path, 'f1_vs_hidden_units.png')
    plot_f1_vs_hidden_units(results, plot_file)
    
    # Print summary
    '''
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    print(f"{'Hidden':<8} {'Train':<10} {'Test':<10} {'Train':<10} {'Test':<10} {'Epochs':<8} {'Time':<10}")
    print(f"{'Units':<8} {'Acc':<10} {'Acc':<10} {'F1':<10} {'F1':<10} {'Trained':<8} {'(sec)':<10}")
    print("-"*80)
    for i, hu in enumerate(results['hidden_units']):
        print(f"{hu:<8} {results['train_accuracy'][i]:<10.4f} {results['test_accuracy'][i]:<10.4f} "
              f"{results['train_avg_f1'][i]:<10.4f} {results['test_avg_f1'][i]:<10.4f} "
              f"{results['epochs_trained'][i]:<8} {results['training_time'][i]:<10.2f}")
    print("="*80)
    
    # Findings
    print("\nFINDINGS:")
    print("-" * 80)
    print("1. Impact of Hidden Units on Performance:")
    print("   - With very few hidden units (1), the network has limited capacity to")
    print("     learn complex patterns, resulting in lower F1 scores.")
    print("   - As hidden units increase (5, 10, 50), the network capacity grows,")
    print("     typically improving both training and test performance.")
    print("   - At 100 hidden units, the network may show signs of overfitting if")
    print("     train F1 is significantly higher than test F1.")
    
    print("\n2. Training vs Test Performance:")
    print("   - The gap between training and test F1 scores indicates generalization.")
    print("   - Larger gaps suggest overfitting; smaller gaps suggest better generalization.")
    
    print("\n3. Optimal Hidden Units:")
    best_test_idx = np.argmax(results['test_avg_f1'])
    best_hidden_units = results['hidden_units'][best_test_idx]
    print(f"   - Best test performance: {best_hidden_units} hidden units")
    print(f"     (Test Accuracy: {results['test_accuracy'][best_test_idx]:.4f}, Test F1: {results['test_avg_f1'][best_test_idx]:.4f})")
    print(f"     (Trained for {results['epochs_trained'][best_test_idx]} epochs in {results['training_time'][best_test_idx]:.2f} seconds)")
    
    print("\n4. Training Efficiency:")
    print(f"   - Total training time: {sum(results['training_time']):.2f} seconds")

    print("="*80)
    '''

if __name__ == "__main__":
    main()
