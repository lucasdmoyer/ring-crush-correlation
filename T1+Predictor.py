
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


# In[2]:

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        self.activation_function = lambda x : 1/(1 + np.exp(-x))   # Replace 0 with your sigmoid calculation.
        
                    
    
    def train(self, features, targets):

        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            hidden_inputs = np.dot(X,self.weights_input_to_hidden) # signals into hidden layer
            hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
            
            final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output) # signals into final output layer
            final_outputs = final_inputs # signals from final output layer'this
            
            #### Implement the backward pass here ####
            ### Backward pass ###

            error = y - final_outputs # Output layer error is the difference between desired target and actual output.

            
            output_error_term = error * 1

            hidden_error = np.dot(self.weights_hidden_to_output, error)
            hidden_error_term = hidden_error * hidden_outputs * (1- hidden_outputs)

            delta_weights_i_h += hidden_error_term * X[:,None]
         
            # Weight step (hidden to output)
            hidden_outputs = hidden_outputs[:,None]
            delta_weights_h_o += output_error_term * hidden_outputs
            #print('delta hidden to out: ' + str(delta_weights_h_o))
        self.weights_hidden_to_output += self.lr * delta_weights_h_o/n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h/n_records # update input-to-hidden weights with gradient descent step
 
    def run(self, features):

        hidden_inputs = np.dot(features,self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output) # signals into final output layer
        final_outputs = (final_inputs) # signals from final output layer 
        
        return final_outputs


# In[3]:

learning_rate = 0.00
hidden_nodes = 3200
output_nodes = 1

N_i = 6
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)


# In[4]:

import json

weights_in = []
with open('weight_in_no_grades', 'rb') as f:
    weights_in = pickle.load(f)
    
weights_out = []
with open('weight_out_no_grades', 'rb') as f:
    weights_out = pickle.load(f)
scaled_features = {}    
with open('variables.json', 'r') as f:
    try:
        scaled_features = json.load(f)
    # if the file is empty the ValueError will be thrown
    except ValueError:
        scaled_features = {}
        
network.weights_input_to_hidden = weights_in
network.weights_hidden_to_output = weights_out


# In[5]:



basisweight = float(input('Enter a basisweight: '))
caliper = float(input('Enter a caliper: '))
cull = float(input('Enter a cull low: '))
moisture = float(input('Enter a moisture: '))
stfi = float(input('Enter a stfi: '))
tsi = float(input('Enter a tsi: '))
basismean, basisstd = scaled_features['basisweight']
calipermean, caliperstd = scaled_features['caliper']
cullmean, cullstd = scaled_features['cull']
moisturemean, moisturestd = scaled_features['moisture']
stfimean, stfistd = scaled_features['stfi']
tsimean, tsistd = scaled_features['tsi']
inbasis = (basisweight - basismean)/basisstd
incaliper = (caliper - calipermean)/caliperstd
incull = (cull - cullmean)/cullstd
inmoisture = (moisture - moisturemean)/moisturestd
instfi = (stfi - stfimean)/stfistd
intsi = (tsi - tsimean)/tsistd
row = [intsi, instfi, incaliper, inmoisture, inbasis, incull]
columns = ['tsi','stfi', 'caliper', 'moisture', 'basisweight', 'cull']
df = pd.DataFrame(columns=columns)
'''
df['tsi'] = intsi
df['stfi'] = instfi
df['caliper'] = incaliper
df['moisture'] = inmoisture
df['basisweight'] = inbasis
df['cull'] = incull '''
df.loc[1] = row
mean, std = scaled_features['rct']
prediction = network.run(df.loc[1]).T*std+mean
print('Predicted Rct :' + str(prediction))


# In[ ]:




# In[ ]:




# In[ ]:



