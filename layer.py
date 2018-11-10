class LunakDense:
    def __init__(self, units, activation, input_dim, init, use_bias=False, seed=None):
        self.units = units
        self.input_dim = input_dim
        
        if activation == 'sigmoid':
            self.activation_function = self.sigmoid
        else:
            print('Activation function not supported')
        
        np.random.seed(seed)
        
        if init == 'uniform':
            self.weight_matrix = np.random.uniform(-0.05, 0.05, size=(self.units, input_dim)) 
        elif init == 'random':
            self.weight_matrix = np.random.random(size=(self.units, input_dim))
        else:
            print('Init function not supported')
        
        self.delta_weight_matrix_before = np.zeros((self.units, input_dim))
        self.delta_weight_matrix = np.zeros((self.units, input_dim))
        
        self.use_bias = use_bias
        if self.use_bias:
            bias = np.zeros((units, 1))
            self.weight_matrix = np.hstack((self.weight_matrix, bias))
            self.delta_weight_matrix_before = np.hstack((self.delta_weight_matrix_before, np.zeros((units, 1))))
            self.delta_weight_matrix = np.hstack((self.delta_weight_matrix, np.zeros((units, 1))))
            
    def init_delta_weight_zero(self):
        for idx_unit, unit in enumerate(self.delta_weight_matrix):
            for idx_source, source in enumerate(unit):
                self.delta_weight_matrix[idx_unit] = 0
                self.delta_weight_matrix_before[idx_unit] = 0
                
    def calculate_sigma(self, input_list):
        if self.use_bias:
            input_list = np.append(input_list, 1)
        
        result_list = np.array([])
        for weight_neuron in self.weight_matrix:
            result_list = np.append(result_list, np.dot(weight_neuron, input_list))
        return np.array(result_list)
    
    def calculate_output(self, input_list):
        output_list = np.array([])
        for sigma_neuron in self.calculate_sigma(input_list):
            output_list = np.append(output_list, self.activation_function(sigma_neuron))
        self.output_list = output_list
        return output_list.copy()
    
    def calculate_local_gradient_output_layer(self, target_list):
        """
        Use this if the layer is output layer
        """
        result_list = np.array([])
        for index, output in enumerate(self.output_list):
            local_gradient = output * (1 - output) * (target_list[index] - output)
            result_list = np.append(result_list, local_gradient)  
        self.local_gradient = result_list
        return result_list.copy()
    
    def calculate_local_gradient_hidden_layer(self, local_gradient_output_list, output_layer_weight_matrix):
        """
        Use this if the layer is hidden layer
        """
        result_list = np.array([])
        for index, output in enumerate(self.output_list):
            sigma_local_gradient_output = 0
            for unit_number, local_gradient in enumerate(local_gradient_output_list):
                sigma_local_gradient_output += output_layer_weight_matrix[unit_number][index] * local_gradient
            error_hidden = output * (1 - output) * sigma_local_gradient_output
            result_list = np.append(result_list, error_hidden)
        self.local_gradient = result_list
        return result_list.copy()
    
    def update_delta_weight(self, lr, input_list, momentum=None):
        """
        Function to update delta weight
        """
        if self.use_bias:
            input_list = np.append(input_list, 1)
        if momentum == None:
            for j, unit in enumerate(self.weight_matrix): #j  
                for i, source in enumerate(unit): #i
                    delta_weight = self.delta_weight_matrix[j][i] + lr * self.local_gradient[j] * input_list[i]
                    self.delta_weight_matrix[j][i] = delta_weight.copy()
        else:
            for j, unit in enumerate(self.weight_matrix): #j  
                for i, source in enumerate(unit): #i
                    delta_weight = self.delta_weight_matrix[j][i] + lr * self.local_gradient[j] * input_list[i] + momentum * self.delta_weight_matrix_before[j][i]
                    
                    # Update Delta Weight
                    self.delta_weight_matrix_before[j][i] = delta_weight.copy()
            
            # Copy Last Update of Weight Matrix Before (Equal to Last Weight Matrix)
            for j, unit in enumerate(self.delta_weight_matrix_before):
                for i, source in enumerate(unit):
                    self.delta_weight_matrix[j][i] = self.delta_weight_matrix_before[j][i].copy()
            
    def update_weight(self):
        """
        Function to update weight
        """
        for j, unit in enumerate(self.delta_weight_matrix_before):
            for i, source in enumerate(unit):
                self.weight_matrix[j][i] += self.delta_weight_matrix[j][i]
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))