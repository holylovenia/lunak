class LunakArtificialNeuralNetwork:
    def __init__(self, loss='root_mean_squared', optimizer='sgd'):
        assert loss == 'root_mean_squared', 'loss function not supported'
        assert optimizer == 'sgd', 'optimizer not supported'
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)
        
    def feed_forward(self, X_instance):
        # Calculate output with the first hidden layer
        output_list = self.layers[0].calculate_output(X_instance)
        # Calculate output with the second until the last layer
        for layer in self.layers[1:]:
            next_output_list = layer.calculate_output(output_list)
            output_list = next_output_list
        return output_list.copy()
            
    def backpropagation(self, y_instance):
        # Calculate local gradient for output layer
        next_local_gradient_list = self.layers[-1].calculate_local_gradient_output_layer([y_instance])
        next_layer_weight_matrix = self.layers[-1].weight_matrix

        # Calculate local gradient for hidden layer(s)
        for layer_idx, layer in enumerate(reversed(self.layers[0:-1])):
            next_local_gradient_list = layer.calculate_local_gradient_hidden_layer(next_local_gradient_list, next_layer_weight_matrix)
            next_layer_weight_matrix = layer.weight_matrix
            
    def calculate_delta_weight(self, X_instance, lr, momentum):
        # Update delta weight for first hidden layer
        self.layers[0].update_delta_weight(lr, X_instance, momentum)
        
        # Update delta weight for other layers
        for layer_idx, layer in enumerate(self.layers[1:]):
            layer.update_delta_weight(lr, self.layers[layer_idx].output_list, momentum)
    
    def fit(self, X, y, epochs, lr, momentum=None, batch_size=None, val_data=None, val_size=0):
        assert X.shape[1] == self.layers[0].input_dim, 'Input dimension must be same with the column'
        self.classes_ = np.unique(y)
        
        if batch_size == None:
            batch_size = len(X)
            
        if val_data is None:
            val_size = 0.1
            X, X_val, y, y_val = train_test_split(X, y, test_size=val_size)
        else:
            X_val = val_data[0]
            y_val = val_data[1]
            
        print('Train on {} samples, validate on {} samples'.format(len(X), len(X_val)))
        
        if val_data is not None and val_size != 0:
            print('Validation data will be used instead of val_size.')
            
        for epoch in range(epochs):
            delta = batch_size
            
            with tnrange(0, len(X), delta, desc='Epoch {}'.format(epoch + 1)) as pbar:
                for start in pbar:
                    X_batch = X[start:start+delta]
                    y_batch = y[start:start+delta]

                    for idx, X_instance in enumerate(X_batch):
                        self.feed_forward(X_instance)
                        self.backpropagation(y_batch[idx][0])
                        self.calculate_delta_weight(X_instance, lr, momentum)

                    for layer in self.layers:
                        layer.update_weight()
                        layer.init_delta_weight_zero()

                    pred = self.predict(X)
                    pred_val = self.predict(X_val)
                    
                    pred_proba = self.predict_proba(X)
                    pred_proba_val = self.predict_proba(X_val)

                    acc = self.calculate_accuracy(y, pred)
                    val_acc = self.calculate_accuracy(y_val, pred_val)
                    loss = mean_squared_error(y, pred_proba)
                    val_loss = mean_squared_error(y_val, pred_proba_val)

                    postfix = {
                        'loss': loss,
                        'acc': acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc
                    }
                    pbar.set_postfix(postfix, refresh=True)
    
    def predict_proba(self, X):
        predictions = []
        for idx, X_instance in enumerate(X):
            X_pred = self.feed_forward(X_instance)
            predictions.append([np.mean(X_pred.copy())])
        return predictions
    
    def predict(self, X):
        predictions = []
        for idx, X_instance in enumerate(X):
            X_pred_proba = self.feed_forward(X_instance)
            X_pred = min(self.classes_, key=lambda pred_class:abs(pred_class - np.mean(X_pred_proba)))
            predictions.append([X_pred])
        return predictions
    
    def calculate_accuracy(self, y_true, y_pred):
        if len(confusion_matrix(y_true, y_pred).ravel()) > 1:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        else:
            tp = confusion_matrix(y_true, y_pred).ravel()[0]
            fp = 0
            fn = 0
            tn = 0
        return (tp + tn) / (tp + tn + fp + fn)
