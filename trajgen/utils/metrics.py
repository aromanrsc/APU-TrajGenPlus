import pandas as pd

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