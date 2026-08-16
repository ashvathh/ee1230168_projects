import numpy as np


class NeuralNetwork:
    '''
    This class is an implementation of Neural networks from first principles.
    It supports multimple hidden layers with configurable number of neurons,
    Sigmoid or ReLU activation functions, mini-batch gradient descent,
    It uses cross-entropy loss and softmax output layer for multi-class classification.

    It is implemented only using numpy which makes the computations needed for neural networks really efficient.

    weights and biases are initialized using Xavier/He initialization.
    forward propagation and backward propagation are implemented as per the equations provided in the description.

    '''
     
    def __init__(self, num_features, hidden_layers, num_classes, learning_rate=0.01, batch_size=32, activation='sigmoid'):
        
        self.num_features = num_features # number of i/p features
        self.hidden_layers = hidden_layers # list of hidden layer sizes
        self.num_classes = num_classes # number of o/p classes
        self.learning_rate = learning_rate # learning rate for gradient descent
        self.batch_size = batch_size # mini-batch size (M)
        self.activation = activation # activation function (sigmoid or relu)
        
        # Initialize weights and biases
        self.weights = [] #stores matrices of weights for each layer
        self.biases = [] #stores bias vectors for each layer
        

        layer_sizes = [num_features] + hidden_layers + [num_classes] #list of sizes of all nn layers including input and output
        np.random.seed(42)  # For reproducibility
        # Initialize weights and biases for each layer
        for i in range(len(layer_sizes) - 1):
            # Xavier/He initialization
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            #weight of shape (size of current layer, size of next layer)
            b = np.zeros((1, layer_sizes[i + 1])) #bias of shape (1, size of next layer)
            self.weights.append(w)
            self.biases.append(b)

    
    def softmax(self, z):
        # Subtract max for numerical stability
        return np.exp(z - np.max(z, axis=1, keepdims=True)) / np.sum(np.exp(z - np.max(z, axis=1, keepdims=True)), axis=1, keepdims=True)

    def forward_propagation(self, X):

        activations = [X]
        net_inputs = []
        
        current_activation = X
        
        # Forward through hidden layers with specified activation
        for i in range(len(self.weights) - 1):
            # calculate the value after weight multiplication and adding bias
            net = np.dot(current_activation, self.weights[i]) + self.biases[i]
            net_inputs.append(net)
            
            # pass through activation function
            if self.activation == 'relu':
                current_activation = np.maximum(0, net) # applying relu
            else:  # sigmoid
                net.clip(-500, 500)  # prevent overflow
                current_activation = 1 / (1 + np.exp(-net)) # applying sigmoid
            
            activations.append(current_activation)
        
        # Output layer with softmax activation
        net_inputs.append(np.dot(current_activation, self.weights[-1]) + self.biases[-1]) # storing net input of output layer

        activations.append(self.softmax(np.dot(current_activation, self.weights[-1]) + self.biases[-1])) # softmax output
        
        return activations, net_inputs
    
    def compute_loss(self, y_pred, y_true):
        # Compute cross-entropy loss.
        batch_size = y_pred.shape[0]
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Cross-entropy loss: -sum(1{k=true_k} * log(o_k))
        loss = 0
        for i in range(batch_size):
            loss += -np.log(y_pred[i, y_true[i]])
        
        return loss / batch_size
    
    def backward_propagation(self, X, y_true, activations, net_inputs):

        batch_size = X.shape[0]
        num_layers = len(self.weights)
        
        weight_gradients = [None] * num_layers
        bias_gradients = [None] * num_layers
        
        # Output layer gradient (using equation 2 from description)
        # ∂J/∂net(L)_k = (o_k - 1) if k = true_k, else o_k
        output_activation = activations[-1]  # Softmax output
        delta = output_activation.copy()
        
        for i in range(batch_size):
            delta[i, y_true[i]] -= 1
        
        delta = delta / batch_size
        
        # Compute gradients for output layer
        weight_gradients[-1] = np.dot(activations[-2].T, delta)
        bias_gradients[-1] = np.sum(delta, axis=0, keepdims=True)
        
        # Backpropagate through hidden layers
        for layer in range(num_layers - 2, -1, -1):
            # Propagate delta backwards
            delta = np.dot(delta, self.weights[layer + 1].T)
            
            # Apply derivative of activation function
            if self.activation == 'relu':
                # For ReLU, use derivative based on net input (before activation)
                delta = delta * (net_inputs[layer] > 0).astype(float) # relu derivative is taken as 0 at 0
            else:  # sigmoid
                # For sigmoid, use derivative based on activation
                delta = delta * activations[layer + 1] * (1 - activations[layer + 1])
            
            # Compute gradients
            weight_gradients[layer] = np.dot(activations[layer].T, delta)
            bias_gradients[layer] = np.sum(delta, axis=0, keepdims=True)
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients, bias_gradients):
        # Update weights and biases using gradients and learning rate
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def train(self, X_train, y_train, max_epochs=10, track_progress=True, min_loss_change=0.001, patience=1):
        # trains nn with mini batch gradient descent
        # stops when loss change is below threshold for patience epochs
        
        num_samples = X_train.shape[0]
        num_batches = int(np.ceil(num_samples / self.batch_size))
        
        history = {
            'loss': [],
            'accuracy': [],
            'epochs_trained': 0
        }
        # needed for early stopping

        patience_counter = 0 # counter for early stopping
        np.random.seed(42) # for reproducibility
        for epoch in range(max_epochs):
            # Shuffle data using indices only (more memory efficient)
            indices = np.random.permutation(num_samples)
            
            epoch_loss = 0
            
            # Mini-batch training
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, num_samples)
                
                # Use index slicing instead of creating shuffled copies
                batch_indices = indices[start_idx:end_idx]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                
                # Forward propagation
                activations, net_inputs = self.forward_propagation(X_batch)
                
                # Compute loss
                batch_loss = self.compute_loss(activations[-1], y_batch)
                epoch_loss += batch_loss
                
                # Backward propagation
                weight_gradients, bias_gradients = self.backward_propagation(
                    X_batch, y_batch, activations, net_inputs
                )
                
                # Update parameters
                self.update_parameters(weight_gradients, bias_gradients)
            
            # average loss over all batches in one epoch
            avg_loss = epoch_loss / num_batches
            
            # Compute training accuracy
            train_accuracy = self.evaluate(X_train, y_train)
            
            history['loss'].append(avg_loss)
            history['accuracy'].append(train_accuracy)
            history['epochs_trained'] = epoch + 1
            
            if track_progress: # helps track training progress
                print(f"Epoch {epoch + 1}/{max_epochs} - Loss: {avg_loss:.4f} - Accuracy: {train_accuracy:.4f}")
            
            # Early stopping based on loss change
            if len(history['loss']) > 1:
                loss_change = abs(history['loss'][-2] - history['loss'][-1]) # change in most recent loss
                if loss_change < min_loss_change:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if track_progress:
                            print(f"Early stopping: Loss change ({loss_change:.6f}) < {min_loss_change}")
                        break
                else:
                    patience_counter = 0
        
        return history
    
    def predict(self, X):
        # makes predictions on input data and returns predicted class labels and probabilities
        activations, _ = self.forward_propagation(X)
        probabilities = activations[-1]
        predictions = np.argmax(probabilities, axis=1)
        return predictions, probabilities
    
    def evaluate(self, X, y_true):
        # evaluates accuracy on given data 
        predictions, _ = self.predict(X)
        accuracy = np.mean(predictions == y_true)
        return accuracy


if __name__ == "__main__":
    # Example usage to check the working of the NeuralNetwork class
    print("Neural Network Implementation")
    print("=" * 50)
    
    # Create a simple example with random data
    np.random.seed(42)
    
    # Generate random data
    num_samples = 1000
    num_features = 32 * 32 * 3  # For 32x32 RGB images
    num_classes = 36
    
    X_train = np.random.randn(num_samples, num_features)
    y_train = np.random.randint(0, num_classes, num_samples)
    
    # Create neural network
    nn = NeuralNetwork(
        num_features=num_features,
        hidden_layers=[128, 64],  # Two hidden layers
        num_classes=num_classes,
        learning_rate=0.01,
        batch_size=32
    )
    
    print(f"\nNetwork Architecture:")
    print(f"Input features: {num_features}")
    print(f"Hidden layers: {nn.hidden_layers}")
    print(f"Output classes: {num_classes}")
    print(f"Batch size: {nn.batch_size}")
    print(f"Learning rate: {nn.learning_rate}")
    
    # Train the network
    print("\nTraining...")
    history = nn.train(X_train, y_train, max_epochs=5, track_progress=True)
    
    # Evaluate
    accuracy = nn.evaluate(X_train, y_train)
    print(f"\nFinal Training Accuracy: {accuracy:.4f}")
