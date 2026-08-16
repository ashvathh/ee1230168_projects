"""
Part F: Transfer Learning from Consonants to Digits.
Compare training from scratch vs transfer learning approach.

Architecture: [512, 256, 128, 64] with ReLU activation
Both models trained for 20 epochs on digits dataset (10 classes)
"""

import numpy as np
import os
import sys
import time
import csv
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork

# load_images_from_folder, evaluate_metrics same as in parts B, C, D, E

def load_images_from_folder(folder_path):
    # Same as in part B, C, D, but for digits dataset (classes 0-9 mapped from folders 37-46)
    images = []
    labels = []
    
    # Get all class folders (37, 38, ..., 46 for digits)
    class_folders = sorted([f for f in os.listdir(folder_path) 
                           if os.path.isdir(os.path.join(folder_path, f))])
    
    print(f"Found {len(class_folders)} digit classes: {class_folders}")
    
    for new_class_idx, class_folder in enumerate(class_folders):
        class_path = os.path.join(folder_path, class_folder)
        
        # Get all image files in this class folder
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f"Loading class {class_folder} (remapped to {new_class_idx}): {len(image_files)} images")
        
        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            try:
                # Load image
                img = Image.open(image_path)
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convert to numpy array and normalize to [0, 1]
                img_array = np.array(img).astype('float32') / 255.0
                
                # Flatten the image
                img_flat = img_array.flatten()
                
                images.append(img_flat)
                # Remap labels to 0-9
                labels.append(new_class_idx)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
    
    X = np.array(images)
    y = np.array(labels)
    
    print(f"Loaded {X.shape[0]} images with shape {X.shape[1]} features")
    print(f"Labels range from {y.min()} to {y.max()}")
    
    return X, y

def evaluate_metrics(y_true, y_pred, num_classes):
    # Calculate precision, recall, F1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(num_classes), average=None, zero_division=0
    )
    
    avg_f1 = np.mean(f1)
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'avg_f1': avg_f1
    }
    
    return metrics

def save_predictions_to_csv(scratch_predictions, transfer_predictions, filename):
    # Save predictions from both models to a CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['prediction'])
        
        # Write predictions from scratch model
        for pred in scratch_predictions:
            writer.writerow([pred])
        
        # Write predictions from transfer learning model
        for pred in transfer_predictions:
            writer.writerow([pred])
    
    print(f"\nPredictions saved to '{filename}'")
    print(f"  - Scratch model predictions: {len(scratch_predictions)} samples")
    print(f"  - Transfer learning predictions: {len(transfer_predictions)} samples")
    print(f"  - Total predictions: {len(scratch_predictions) + len(transfer_predictions)}")

def load_pretrained_weights(filename='model_weights_d.npz'):
    # Function to load pre-trained weights from consonant model - {512, 256, 128, 64} relu model
    print(f"\nLoading pre-trained weights from '{filename}'...")
    data = np.load(filename)
    
    # Determine number of layers
    num_layers = 0
    while f'weight_{num_layers}' in data:
        num_layers += 1
    
    print(f"Found {num_layers} layers in pre-trained model")
    
    weights = []
    biases = []
    
    # Load weights and biases (excluding the last output layer)
    for i in range(num_layers - 1):  # Exclude output layer
        w = data[f'weight_{i}']
        b = data[f'bias_{i}']
        weights.append(w)
        biases.append(b)
        print(f"  Layer {i}: Weight shape {w.shape}, Bias shape {b.shape}")
    
    print(f"Loaded {len(weights)} hidden layers (output layer will be re-initialized)")
    
    return weights, biases

def train_from_scratch(X_train, y_train, X_test, y_test, num_classes, epochs=20):
    
    print("\n" + "="*80)
    print("TRAINING FROM SCRATCH ON DIGITS")
    print("="*80)
    
    num_features = X_train.shape[1]
    architecture = [512, 256, 128, 64]
    
    # Create neural network with random initialization
    nn = NeuralNetwork(
        num_features=num_features,
        hidden_layers=architecture,
        num_classes=num_classes,
        learning_rate=0.01,
        batch_size=32,
        activation='relu'
    )
    
    print(f"Architecture: {architecture}")
    print(f"Input features: {num_features}")
    print(f"Output classes: {num_classes}")
    print(f"Training for {epochs} epochs...")
    
    # Track metrics per epoch
    train_f1_scores = []
    test_f1_scores = []
    train_accuracies = []
    test_accuracies = []
    
    start_time = time.time()
    
    # we train one epoch at a time for better tracking of metrics
    for epoch in range(epochs):
        # Train for one epoch
        history = nn.train(X_train, y_train, max_epochs=1, track_progress=False)
        
        # Evaluate on training set
        train_pred, _ = nn.predict(X_train)
        train_metrics = evaluate_metrics(y_train, train_pred, num_classes)
        train_accuracy = nn.evaluate(X_train, y_train)
        
        # Evaluate on test set
        test_pred, _ = nn.predict(X_test)
        test_metrics = evaluate_metrics(y_test, test_pred, num_classes)
        test_accuracy = nn.evaluate(X_test, y_test)
        
        train_f1_scores.append(train_metrics['avg_f1'])
        test_f1_scores.append(test_metrics['avg_f1'])
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Acc: {train_accuracy:.4f}, Train F1: {train_metrics['avg_f1']:.4f} | "
              f"Test Acc: {test_accuracy:.4f}, Test F1: {test_metrics['avg_f1']:.4f}")
    
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Final Train Accuracy: {train_accuracies[-1]:.4f}, Train F1: {train_f1_scores[-1]:.4f}")
    print(f"Final Test Accuracy: {test_accuracies[-1]:.4f}, Test F1: {test_f1_scores[-1]:.4f}")
    
    history = {
        'train_f1': train_f1_scores,
        'test_f1': test_f1_scores,
        'train_accuracy': train_accuracies,
        'test_accuracy': test_accuracies,
        'training_time': training_time
    }
    
    return nn, history

def transfer_learning(X_train, y_train, X_test, y_test, num_classes, 
                     pretrained_weights, pretrained_biases, epochs=20):
    # In this function, we initialize hidden layers with pre-trained weights
    # Then we fine-tune the entire network on digits dataset
    print("\n" + "="*80)
    print("TRANSFER LEARNING: FINE-TUNING PRE-TRAINED MODEL ON DIGITS")
    print("="*80)
    
    num_features = X_train.shape[1]
    architecture = [512, 256, 128, 64]
    
    # Create neural network
    nn = NeuralNetwork(
        num_features=num_features,
        hidden_layers=architecture,
        num_classes=num_classes,
        learning_rate=0.01,
        batch_size=32,
        activation='relu'
    )
    
    print(f"Architecture: {architecture}")
    print(f"Input features: {num_features}")
    print(f"Output classes: {num_classes}")
    print(f"Initializing hidden layers with pre-trained weights...")
    
    # Replace hidden layer weights with pre-trained weights
    for i in range(len(pretrained_weights)):
        nn.weights[i] = pretrained_weights[i].copy() # done instead of random initialization - for transfer learning
        nn.biases[i] = pretrained_biases[i].copy() # done instead of random initialization- for transfer learning
        print(f"  Loaded layer {i}: Weight shape {nn.weights[i].shape}")
    
    # Output layer is randomly initialized (already done in __init__)
    print(f"  Output layer randomly initialized: Weight shape {nn.weights[-1].shape}")
    
    print(f"\nFine-tuning entire network for {epochs} epochs...")
    
    # Track metrics per epoch
    train_f1_scores = []
    test_f1_scores = []
    train_accuracies = []
    test_accuracies = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Train for one epoch
        history = nn.train(X_train, y_train, max_epochs=1, track_progress=False)
        
        # Evaluate on training set
        train_pred, _ = nn.predict(X_train)
        train_metrics = evaluate_metrics(y_train, train_pred, num_classes)
        train_accuracy = nn.evaluate(X_train, y_train)
        
        # Evaluate on test set
        test_pred, _ = nn.predict(X_test)
        test_metrics = evaluate_metrics(y_test, test_pred, num_classes)
        test_accuracy = nn.evaluate(X_test, y_test)
        
        train_f1_scores.append(train_metrics['avg_f1'])
        test_f1_scores.append(test_metrics['avg_f1'])
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Acc: {train_accuracy:.4f}, Train F1: {train_metrics['avg_f1']:.4f} | "
              f"Test Acc: {test_accuracy:.4f}, Test F1: {test_metrics['avg_f1']:.4f}")
    
    training_time = time.time() - start_time
    
    print(f"\nFine-tuning completed in {training_time:.2f} seconds")
    print(f"Final Train Accuracy: {train_accuracies[-1]:.4f}, Train F1: {train_f1_scores[-1]:.4f}")
    print(f"Final Test Accuracy: {test_accuracies[-1]:.4f}, Test F1: {test_f1_scores[-1]:.4f}")
    
    history = {
        'train_f1': train_f1_scores,
        'test_f1': test_f1_scores,
        'train_accuracy': train_accuracies,
        'test_accuracy': test_accuracies,
        'training_time': training_time
    }
    
    return nn, history

def plot_comparison(scratch_history, transfer_history, save_path='transfer_learning_comparison.png'):
    
    epochs = range(1, len(scratch_history['train_f1']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot F1 scores
    ax1 = axes[0]
    ax1.plot(epochs, scratch_history['train_f1'], 'b-o', label='Scratch - Train F1', linewidth=2)
    ax1.plot(epochs, scratch_history['test_f1'], 'b--s', label='Scratch - Test F1', linewidth=2)
    ax1.plot(epochs, transfer_history['train_f1'], 'r-o', label='Transfer - Train F1', linewidth=2)
    ax1.plot(epochs, transfer_history['test_f1'], 'r--s', label='Transfer - Test F1', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_title('F1 Score Comparison: From Scratch vs Transfer Learning', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot Accuracy
    ax2 = axes[1]
    ax2.plot(epochs, scratch_history['train_accuracy'], 'b-o', label='Scratch - Train Acc', linewidth=2)
    ax2.plot(epochs, scratch_history['test_accuracy'], 'b--s', label='Scratch - Test Acc', linewidth=2)
    ax2.plot(epochs, transfer_history['train_accuracy'], 'r-o', label='Transfer - Train Acc', linewidth=2)
    ax2.plot(epochs, transfer_history['test_accuracy'], 'r--s', label='Transfer - Test Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy Comparison: From Scratch vs Transfer Learning', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved as '{save_path}'")
    plt.close()

def save_detailed_results(scratch_history, transfer_history, filename='transfer_learning_results.txt'):
    """Save detailed comparison results to a text file."""
    with open(filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TRANSFER LEARNING: FROM SCRATCH VS PRE-TRAINED MODEL\n")
        f.write("="*80 + "\n\n")
        f.write("Task: Digit Classification (10 classes: digits 0-9 in Devanagari)\n")
        f.write("Architecture: [512, 256, 128, 64] with ReLU activation\n")
        f.write("Training: 20 epochs, batch_size=32, learning_rate=0.01\n\n")
        
        f.write("="*80 + "\n")
        f.write("TRAINING FROM SCRATCH\n")
        f.write("="*80 + "\n")
        f.write("All weights randomly initialized\n\n")
        f.write(f"{'Epoch':<8} {'Train Acc':<12} {'Test Acc':<12} {'Train F1':<12} {'Test F1':<12}\n")
        f.write("-"*80 + "\n")
        for i in range(len(scratch_history['train_f1'])):
            f.write(f"{i+1:<8} {scratch_history['train_accuracy'][i]:<12.4f} "
                   f"{scratch_history['test_accuracy'][i]:<12.4f} "
                   f"{scratch_history['train_f1'][i]:<12.4f} "
                   f"{scratch_history['test_f1'][i]:<12.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("TRANSFER LEARNING (PRE-TRAINED ON CONSONANTS)\n")
        f.write("="*80 + "\n")
        f.write("Hidden layers initialized with pre-trained weights from consonant model\n")
        f.write("Output layer randomly initialized for 10-class digit classification\n\n")
        f.write(f"{'Epoch':<8} {'Train Acc':<12} {'Test Acc':<12} {'Train F1':<12} {'Test F1':<12}\n")
        f.write("-"*80 + "\n")
        for i in range(len(transfer_history['train_f1'])):
            f.write(f"{i+1:<8} {transfer_history['train_accuracy'][i]:<12.4f} "
                   f"{transfer_history['test_accuracy'][i]:<12.4f} "
                   f"{transfer_history['train_f1'][i]:<12.4f} "
                   f"{transfer_history['test_f1'][i]:<12.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("FINAL COMPARISON\n")
        f.write("="*80 + "\n")
        f.write(f"\nFrom Scratch:\n")
        f.write(f"  Final Test Accuracy: {scratch_history['test_accuracy'][-1]:.4f}\n")
        f.write(f"  Final Test F1 Score: {scratch_history['test_f1'][-1]:.4f}\n")
        f.write(f"  Training Time: {scratch_history['training_time']:.2f} seconds\n")
        f.write(f"\nTransfer Learning:\n")
        f.write(f"  Final Test Accuracy: {transfer_history['test_accuracy'][-1]:.4f}\n")
        f.write(f"  Final Test F1 Score: {transfer_history['test_f1'][-1]:.4f}\n")
        f.write(f"  Training Time: {transfer_history['training_time']:.2f} seconds\n")
        f.write(f"\nImprovement:\n")
        acc_improvement = transfer_history['test_accuracy'][-1] - scratch_history['test_accuracy'][-1]
        f1_improvement = transfer_history['test_f1'][-1] - scratch_history['test_f1'][-1]
        time_difference = transfer_history['training_time'] - scratch_history['training_time']
        f.write(f"  Accuracy: {acc_improvement:+.4f} ({acc_improvement*100:+.2f}%)\n")
        f.write(f"  F1 Score: {f1_improvement:+.4f} ({f1_improvement*100:+.2f}%)\n")
        f.write(f"  Training Time: {time_difference:+.2f} seconds\n")
    
    print(f"Detailed results saved to '{filename}'")

def main():
    """Main function to run transfer learning experiment."""
    # Check command line arguments
    if len(sys.argv) != 4:
        print("Usage: python f.py <train_data_path> <test_data_path> <output_folder_path>")
        sys.exit(1)
    
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    output_folder_path = sys.argv[3]
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)
    
    print("="*80)
    print("PART F: TRANSFER LEARNING - CONSONANTS TO DIGITS")
    print("="*80)
    print("\nExperiment: Compare training from scratch vs transfer learning")
    print("Task: Digit classification (10 classes)")
    print("Architecture: [512, 256, 128, 64] with ReLU activation")
    print("Training: 20 epochs for both approaches")
    print("="*80)
    
    # Load digits data
    print("\nLoading training data (digits)...")
    X_train, y_train = load_images_from_folder(train_data_path)
    
    print("\nLoading test data (digits)...")
    X_test, y_test = load_images_from_folder(test_data_path)
    
    print(f"\nData loaded successfully!")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features per sample: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    num_classes = 10  # Digits 0-9
    
    # Part 1: Train from scratch
    scratch_model, scratch_history = train_from_scratch(
        X_train, y_train, X_test, y_test, num_classes, epochs=20
    )
    
    # Part 2: Transfer learning
    # Load pre-trained weights from consonant model (from output folder where d.py saved it)
    pretrained_weights_path = os.path.join(output_folder_path, 'model_weights_d.npz')
    pretrained_weights, pretrained_biases = load_pretrained_weights(pretrained_weights_path)
    
    transfer_model, transfer_history = transfer_learning(
        X_train, y_train, X_test, y_test, num_classes,
        pretrained_weights, pretrained_biases, epochs=20
    )
    
    # Get final predictions on test set from both models
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS ON TEST SET")
    print("="*80)
    scratch_predictions, _ = scratch_model.predict(X_test)
    transfer_predictions, _ = transfer_model.predict(X_test)
    print(f"Scratch model predictions: {scratch_predictions.shape}")
    print(f"Transfer learning predictions: {transfer_predictions.shape}")
    
    # Save predictions to CSV
    save_predictions_to_csv(scratch_predictions, transfer_predictions, os.path.join(output_folder_path, 'prediction_f.csv'))
    
    # Compare results
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(f"\n{'Metric':<25} {'From Scratch':<15} {'Transfer Learning':<20} {'Improvement':<15}")
    print("-"*80)
    print(f"{'Final Test Accuracy':<25} {scratch_history['test_accuracy'][-1]:<15.4f} "
          f"{transfer_history['test_accuracy'][-1]:<20.4f} "
          f"{transfer_history['test_accuracy'][-1] - scratch_history['test_accuracy'][-1]:+.4f}")
    print(f"{'Final Test F1 Score':<25} {scratch_history['test_f1'][-1]:<15.4f} "
          f"{transfer_history['test_f1'][-1]:<20.4f} "
          f"{transfer_history['test_f1'][-1] - scratch_history['test_f1'][-1]:+.4f}")
    print(f"{'Final Train Accuracy':<25} {scratch_history['train_accuracy'][-1]:<15.4f} "
          f"{transfer_history['train_accuracy'][-1]:<20.4f} "
          f"{transfer_history['train_accuracy'][-1] - scratch_history['train_accuracy'][-1]:+.4f}")
    print(f"{'Final Train F1 Score':<25} {scratch_history['train_f1'][-1]:<15.4f} "
          f"{transfer_history['train_f1'][-1]:<20.4f} "
          f"{transfer_history['train_f1'][-1] - scratch_history['train_f1'][-1]:+.4f}")
    print(f"{'Training Time (sec)':<25} {scratch_history['training_time']:<15.2f} "
          f"{transfer_history['training_time']:<20.2f} "
          f"{transfer_history['training_time'] - scratch_history['training_time']:+.2f}")
    print("="*80)
    
    # Save results and plots
    save_detailed_results(scratch_history, transfer_history, os.path.join(output_folder_path, 'transfer_learning_results.txt'))
    plot_comparison(scratch_history, transfer_history, os.path.join(output_folder_path, 'transfer_learning_comparison.png'))

if __name__ == "__main__":
    main()
