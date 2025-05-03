#!/usr/bin/env python3

import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import math
from MarrAlbusModel import MarrAlbusModel 
from ffnn import ffnn
def read_mat_file(path):
    """
    Load a .mat file and return its contents as a dict.
    """
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    return mat

if __name__ == "__main__":
    mat_path = "mixoutALL_shifted.mat"  # replace with your .mat filename
    data = read_mat_file(mat_path)
    print((data.keys()))
    print(len(data['mixout'][0][1]))

             
    t = np.array([t for t in range(len(data['mixout'][1][0]))])
    t = t/len(t)
    t = t - .5
    
    x = data['mixout'][1][0]
    y = data['mixout'][1][1]
    
   # print(x,y)

    '''        
    y = np.array([math.sin(x/10+1) for x in range(100)])
    x = np.array([x for x in range(100)])*.01
    t = x - .5
    ''' 


    
    X = t.reshape(-1, 1)
   # Reshape output Y to match X (here it's just the y-values, but this could be more complex)
    Y = np.column_stack((x, y))  # Shape: (n, 2) for x and y coordinates
    #print(Y)
    #print(Y)


    # Initialize the Marr-Albus model
    input_size = X.shape[1]  # 1 (time)
    hidden_size = 10  # Number of hidden units, tune as needed
    output_size = 2  # We predict a single coordinate (x,y-coordinate)
    learning_rate = 0.015

    # Instantiate the model

    print(input_size)    
    model = MarrAlbusModel(input_size, 400*input_size, output_size, learning_rate)

   # model = ffnn(d_in=input_size, d_hidden=2*input_size, d_out=output_size, lr=learning_rate)
  
    # Train the model on the trajectory data
    model.train(X, Y, epochs=1000000000)

    # Test the model with some predictions
    predictions = model.predict(X)
    x_p = [coor[0] for coor in predictions]
    y_p = [coor[1] for coor in predictions]
   # print(predictions)

    # Plot the original trajectory and the predictions
    plt.plot(x, y, label='True Trajectory', marker='o', linestyle='-')
    plt.plot(x_p, y_p, label='Predicted Trajectory', marker='x', linestyle='--')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Marr-Albus(400 unit .15lr and e-7 conv) Trajectory Prediction')
    plt.legend()
    # Save to file before showing (or instead of showing)
    plt.savefig('SL_fit_MA_400_e7_.15lr.png', dpi=300, bbox_inches='tight')



    
    
