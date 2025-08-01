from config import *
import pandas as pd
import numpy as np

def apu_trajgen_fixed_k(mdl,
                        X_t,
                        test_traj_seq_lengths,
                        SEQ_LENGTH,
                        NUM_FEATS,
                        k_steps=1
                        ):
    """
        Works for both stateful and non-stateful models
        Works for any batch size
        Predictions are made point-by-point
        Every k_steps, the model uses the real location
        Returns the predictions of the model on the test data
    """
    mdl.layers[0].reset_states()
    
    Y_preds = [0.0] * len(X_t)
 
    for i in range(len(X_t)):
        X_t[i] = X_t[i].reshape(-1, SEQ_LENGTH, NUM_FEATS)
    
    ## For each trajectory
    for i in range(len(X_t)):
        
        print("Processing trajectory:", i)
        
        # Use k_counter to keep track of the number of steps we have predicted wihtout using the real location
        k_counter = 0
        
        # Initialize Y_preds[i] as an empty list to store predictions step by step
        Y_preds[i] = []
        
        # For each point in the trajectory, we need to predict the next point
        for j in range(test_traj_seq_lengths[i]):
            
            # Reshape X_t to keep only the i, j position
            X_t_ij = X_t[i][j].reshape(1, SEQ_LENGTH, NUM_FEATS)  # Reshape to (1, 1, NUM_FEATS)
            
            # print("X_t_ij shape:", X_t_ij.shape)
        
            # If we are at the first point (k_counter == 0), we need to use the real location
            # or if we have predicted k steps, we need to use the real location
            # Otherwise, we use the predicted location
            if k_counter == 0 or k_counter >= k_steps:
                y_pred = mdl.predict(X_t_ij, batch_size = 1, verbose = 0)
                # Set k_counter to 1, to force the model to use the predicted location
                # for the next prediction (if k_steps > 1)
                k_counter = 1
            else:
                # Use the predicted location (from the previous step) for the next prediction
                y_p = X_t_ij
                y_p[:,:,0:2] = Y_preds[i][j-1][:,:]

                y_pred = mdl.predict(y_p.reshape(1, SEQ_LENGTH, NUM_FEATS), batch_size = 1, verbose = 0)
                # Increment k_counter
                k_counter += 1
        
            # Add the predicted location to the trajectory ("release the predicted location")
            Y_preds[i].append(y_pred)
        
        # Convert Y_preds[i] from a list to a NumPy array
        Y_preds[i] = np.array(Y_preds[i]).reshape(-1, NUM_OUTPUTS)
        
        mdl.layers[0].reset_states()
        
    return Y_preds