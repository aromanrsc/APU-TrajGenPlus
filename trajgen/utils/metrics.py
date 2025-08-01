import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import haversine_distances
from math import radians

def haversine_distance(lat1, lon1, lat2, lon2):
    """
        Compute great-circle (Haversine) distance between two lat/lon points in km.
        Returns the distance in km.
    """

    # Do this check if one argument is NaN
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return 0 # 
        
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    return haversine_distances([[lat1, lon1], [lat2, lon2]])[0, 1] * 6371  #Earth radius in km


def compute_point_to_point_haversine_distances(traj1, traj2):
    """
    Compute the haversine point-to-point distance in meters between two trajectories.
    
    Parameters:
        traj1 (array-like): First trajectory as a list or array of [latitude, longitude] pairs.
        traj2 (array-like): Second trajectory as a list or array of [latitude, longitude] pairs.
    
    Returns:
        list: A list of distances in meters between corresponding points in the two trajectories.
    """
    if len(traj1) != len(traj2):
        raise ValueError("Trajectories must have the same number of points.")
    
    distances = []
    for (lat1, lon1), (lat2, lon2) in zip(traj1, traj2):
        # Convert degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        # Compute haversine distance in kilometers and convert to meters
        distance = haversine_distances([[lat1, lon1], [lat2, lon2]])[0, 1] * 6371000  # Earth radius in meters
        distances.append(distance)
    
    return distances

# ML metrics on normalized data
def compute_trajectory_metrics(X_test, Y_pred):
    
    """
        Compute prediction metrics for the model
        Returns a dictionary with the following metrics:
        - IE: Individual Error
        - ISE: Individual Squared Error
        - MSE: Mean Squared Error
        - ED: Energy Distance, averaged per features
    """
    
    errors = [0.0] * len(X_test)
    squared_errors = [0.0] * len(X_test)
    mses = [0.0] * len(X_test)
    eds = [0.0] * len(X_test)
    
    for i in range(len(X_test)):
        
        err = X_test[i] - Y_pred[i]
        squared_errors[i] = err**2
        errors[i] = err
        mses[i] = np.mean(err**2)
        
        eds1= float(energy_distance(X_test[i][:,0], Y_pred[i][:,0]))
        eds2 = float(energy_distance(X_test[i][:,1], Y_pred[i][:,1]))
        eds[i] = np.mean([eds1, eds2])
        
    results = {"IE" : errors, "ISE" : squared_errors, "MSE" : mses, "ED" : eds}
    
    return results