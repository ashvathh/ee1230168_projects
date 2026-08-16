import numpy as np
import os
import sys
import time
import csv
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork

# This part is exactly the same as Part C, except using ReLU activation in hidden layers.

def load_images_from_folder(folder_path):
    # Same as in Part B
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
    
    X = np.array(images)
    y = np.array(labels)
    
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

    # Printing average metrics
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

def experiment_network_depth_relu(X_train, y_train, X_test, y_test, architectures):
    # Experiment with different network depths and architectures
    # architectures: list of lists, each inner list specifies hidden units per layer
    # Results are collected and returned for writing to file and plotting.

    num_features = X_train.shape[1]
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
        'epochs_trained': []
    }
    
    all_test_predictions = []
    
    # Keep track of the last model for saving weights for transfer learning in Part F
    last_model = None
    
    # Iterate over each architecture
    for architecture in architectures:
        depth = len(architecture)
        arch_str = '-'.join(map(str, architecture))
        
        print(f"\n{'#'*80}")
        print(f"# Training with architecture: [{arch_str}] (Depth: {depth}) - ReLU Activation")
        print(f"{'#'*80}")
        
        # Create neural network with ReLU activation
        nn = NeuralNetwork(
            num_features=num_features,
            hidden_layers=architecture,
            num_classes=num_classes,
            learning_rate=0.01,
            batch_size=32,
            activation='relu'  # Use ReLU activation
        )
        
        # Train the network with early stopping
        # STOPPING CRITERION: Loss change < 0.001 or max 250 epochs
        start_time = time.time()
        history = nn.train(X_train, y_train, max_epochs=250, track_progress=True, 
                          min_loss_change=0.001, patience=1)
        training_time = time.time() - start_time
        
        epochs_trained = history['epochs_trained']

        # Get predictions on training data
        train_pred, _ = nn.predict(X_train)
        train_metrics = evaluate_metrics(y_train, train_pred, num_classes)
        train_accuracy = nn.evaluate(X_train, y_train)
        
        # Get predictions on test data
        test_pred, _ = nn.predict(X_test)
        test_metrics = evaluate_metrics(y_test, test_pred, num_classes)
        test_accuracy = nn.evaluate(X_test, y_test)
        
        # Store predictions from this model
        all_test_predictions.append(test_pred)
        
        # Print results
        print_metrics_table(train_metrics, "TRAINING DATA", architecture)
        print_metrics_table(test_metrics, "TEST DATA", architecture)
        
        # Print accuracies and training info
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Epochs Trained: {epochs_trained}")
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
        results['epochs_trained'].append(epochs_trained)
        
        # Keep reference to the last model
        last_model = nn
    
    return results, all_test_predictions, last_model

def plot_f1_vs_depth(results, save_path='f1_vs_depth_relu.png', results_sigmoid=None):
    # Plot average F1 score vs network depth for ReLU NN
    plt.figure(figsize=(12, 6))
    
    # Create labels for x-axis
    arch_labels = ['-'.join(map(str, arch)) for arch in results['architectures']]
    x_pos = np.arange(len(results['depth']))
    
    # Plot ReLU results
    plt.plot(x_pos, results['train_avg_f1'], 
             marker='o', linewidth=2, markersize=8, label='Training F1 (ReLU)', color='blue')
    plt.plot(x_pos, results['test_avg_f1'], 
             marker='s', linewidth=2, markersize=8, label='Test F1 (ReLU)', color='red')
    
    # Plot sigmoid results if provided
    if results_sigmoid is not None:
        plt.plot(x_pos, results_sigmoid['train_avg_f1'], 
                 marker='o', linewidth=2, markersize=8, label='Training F1 (Sigmoid)', 
                 color='blue', linestyle='--', alpha=0.6)
        plt.plot(x_pos, results_sigmoid['test_avg_f1'], 
                 marker='s', linewidth=2, markersize=8, label='Test F1 (Sigmoid)', 
                 color='red', linestyle='--', alpha=0.6)
    
    plt.xlabel('Network Architecture (Depth)', fontsize=12)
    plt.ylabel('Average F1 Score', fontsize=12)
    plt.title('Average F1 Score vs Network Depth (ReLU vs Sigmoid)', fontsize=14, fontweight='bold')
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

def save_detailed_results(results, filename='detailed_results_depth_relu.txt'):
    # Save detailed results to a text file.
    with open(filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DETAILED RESULTS - VARYING NETWORK DEPTH WITH ReLU ACTIVATION\n")
        f.write("="*80 + "\n\n")
        f.write("ACTIVATION: ReLU in hidden layers, Softmax in output layer\n")
        f.write("STOPPING CRITERION: Loss change < 0.001 or max 250 epochs\n")
        f.write("Learning Rate: 0.01\n")
        f.write("Batch Size: 32\n")
        f.write("Input Features: 3072 (32x32x3)\n")
        f.write("Output Classes: 36\n\n")
        
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
            f.write(f"  Epochs Trained: {results['epochs_trained'][i]}\n")
            f.write(f"  Training Time: {results['training_time'][i]:.2f} seconds\n\n")
    
    print(f"\nDetailed results saved to '{filename}'")

def save_model_weights(model, filename='model_weights_d.npz'):
    # Saving the model weights and biases to a .npz file. So that they can be loaded later for transfer learning.
    # Create a dictionary to store weights and biases
    weights_dict = {}
    
    for i, (w, b) in enumerate(zip(model.weights, model.biases)):
        weights_dict[f'weight_{i}'] = w
        weights_dict[f'bias_{i}'] = b
    
    # Save additional model information
    weights_dict['num_features'] = model.num_features
    weights_dict['hidden_layers'] = np.array(model.hidden_layers)
    weights_dict['num_classes'] = model.num_classes
    weights_dict['learning_rate'] = model.learning_rate
    weights_dict['batch_size'] = model.batch_size
    weights_dict['activation'] = model.activation
    
    np.savez(filename, **weights_dict)
    print(f"Model weights saved to '{filename}'")

def main():
    """Main function to run the experiment."""
    # Check command line arguments
    if len(sys.argv) != 4:
        print("Usage: python d.py <train_data_path> <test_data_path> <output_folder_path>")
        sys.exit(1)
    
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    output_folder_path = sys.argv[3]
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)
    
    print("="*80)
    print("PART D: NEURAL NETWORK WITH RELU ACTIVATION")
    print("="*80)
    print("\nActivation Function: ReLU in hidden layers, Softmax in output layer")
    print("STOPPING CRITERION: Loss change < 0.001 or max 250 epochs")
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
    
    # Network architectures to experiment with
    architectures = [
        [512],
        [512, 256],
        [512, 256, 128],
        [512, 256, 128, 64]
    ]
    
    # Run experiments
    results, all_test_predictions, last_model = experiment_network_depth_relu(X_train, y_train, X_test, y_test, architectures)
    
    # Save predictions to CSV in output folder
    prediction_file = os.path.join(output_folder_path, 'prediction_d.csv')
    save_predictions_to_csv(all_test_predictions, prediction_file)
    
    # Save the weights of the last model (4-layer architecture) for transfer learning in Part F
    weights_file = os.path.join(output_folder_path, 'model_weights_d.npz')
    save_model_weights(last_model, weights_file)
    
    # Save detailed results in output folder
    results_file = os.path.join(output_folder_path, 'detailed_results_depth_relu.txt')
    save_detailed_results(results, results_file)
    
    # Plot results and save in output folder
    plot_file = os.path.join(output_folder_path, 'f1_vs_depth_relu.png')
    plot_f1_vs_depth(results, plot_file)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS (ReLU ACTIVATION)")
    print("="*80)
    print(f"{'Architecture':<20} {'Depth':<7} {'Train':<10} {'Test':<10} {'Train':<10} {'Test':<10} {'Epochs':<8} {'Time':<10}")
    print(f"{'(Hidden Layers)':<20} {'':<7} {'Acc':<10} {'Acc':<10} {'F1':<10} {'F1':<10} {'':<8} {'(sec)':<10}")
    print("-"*80)
    for i, arch in enumerate(results['architectures']):
        arch_str = '-'.join(map(str, arch))
        print(f"{arch_str:<20} {results['depth'][i]:<7} {results['train_accuracy'][i]:<10.4f} "
              f"{results['test_accuracy'][i]:<10.4f} {results['train_avg_f1'][i]:<10.4f} "
              f"{results['test_avg_f1'][i]:<10.4f} {results['epochs_trained'][i]:<8} "
              f"{results['training_time'][i]:<10.2f}")
    print("="*80)
    
    # Observations
    '''
    print("\nOBSERVATIONS:")
    print("-" * 80)
    print("1. ReLU Activation Benefits:")
    print("   - ReLU helps mitigate vanishing gradient problem")
    print("   - Faster training due to simpler gradient computation")
    print("   - Sparse activation (many neurons output 0)")
    
    print("\n2. Impact of Network Depth on Performance:")
    print("   - Shallow networks (depth 1) may have limited representational capacity")
    print("   - Deeper networks can learn more complex hierarchical features")
    print("   - Very deep networks may face optimization challenges")
    
    print("\n3. Training Characteristics:")
    print("   - ReLU typically converges faster than sigmoid")
    print("   - Dead neurons (always outputting 0) may occur with ReLU")
    for i, arch in enumerate(results['architectures']):
        arch_str = '-'.join(map(str, arch))
        print(f"   - [{arch_str}]: {results['epochs_trained'][i]} epochs, "
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
    
    print("\n6. Comparison with Sigmoid (Part C):")
    print("   - ReLU typically trains faster than sigmoid")
    print("   - ReLU may achieve better or comparable performance")
    print("   - ReLU is less prone to vanishing gradients")
    print("   - Sigmoid provides smooth gradients everywhere")
    print("   - Load results from Part C to make direct comparison")
    
    print("\n7. ReLU-Specific Observations:")
    print("   - Non-differentiability at z=0 handled using sub-gradients")
    print("   - Dying ReLU problem: neurons that always output 0")
    print("   - Unbounded activation (unlike sigmoid which is bounded [0,1])")
    print("   - Sparse representations due to thresholding at 0")
    
    print("\n8. Training Efficiency:")
    print(f"   - Total training time: {sum(results['training_time']):.2f} seconds")
    print(f"   - Average epochs per architecture: {np.mean(results['epochs_trained']):.1f}")
    print("="*80)
    '''

if __name__ == "__main__":
    main()
