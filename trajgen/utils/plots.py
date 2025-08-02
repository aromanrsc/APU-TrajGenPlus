import tilemapbase
import statistics
import numpy as np
import pandas as pd

from config import *
import matplotlib.pyplot as plt

def plot_trajectory(real_traj, protected_traj, dists = None, dist_min=0, dist_max=0, 
                    ks = None, k=1, traj_id = 0, show = True, save= True, adaptive= False, aspect_ratio=1.0):
        
    font = PLOT_FONT
    SMALL_SIZE = PLOT_FONT_SMALL_SIZE
    MEDIUM_SIZE = PLOT_FONT_MEDIUM_SIZE
    N_LABELS_X = 9
    N_LABELS_Y = 5

    plt.close('all')
    plt.figure()
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    
    predicted = pd.DataFrame()
    actual = pd.DataFrame()
    
    predicted["lats"] = protected_traj[:, 0]
    predicted["lons"] = protected_traj[:, 1] 
    
    actual["lats"] = real_traj[:, 0]
    actual["lons"] = real_traj[:, 1]
    
        
    # Compute the min/max values of the lons/lats
    # to set the limits of the plot           
    lons_min = min(predicted.lons.min(),actual.lons.min())
    lons_max = max(predicted.lons.max(),actual.lons.max())
    lats_min = min(predicted.lats.min(),actual.lats.min())
    lats_max = max(predicted.lats.max(),actual.lats.max())
    
    axs = plt.subplots(2,1, figsize=(10,10))[1]
    
    # Initialise the tilemapbase    
    tilemapbase.init(create=True)
    tiles = tilemapbase.tiles.build_OSM()
    
    # Extend the map to include a small margin around the trajectory
    # This is done to avoid the trajectory points to be cut off
    # when the trajectory is close to the map border
    extent_factor=0.005
    
    lons_min_ext = lons_min - extent_factor
    lons_max_ext = lons_max + extent_factor
    lats_min_ext = lats_min - extent_factor
    lats_max_ext = lats_max + extent_factor
    extent = tilemapbase.Extent.from_lonlat(
        lons_min_ext,
        lons_max_ext,
        lats_min_ext,
        lats_max_ext
    )
    
    extent = extent.to_aspect(aspect_ratio, False)
    
    trip_pred = predicted.apply(
        lambda x: tilemapbase.project(x.lons, x.lats), axis=1
    ).apply(pd.Series)
    trip_pred.columns = ["x", "y"]
    
    trip_real = actual.apply(
        lambda x: tilemapbase.project(x.lons, x.lats), axis=1
    ).apply(pd.Series)
    trip_real.columns = ["x", "y"]
    
    plotter = tilemapbase.Plotter(extent, tiles, width=1200)
    
    plotter.plot(axs[0], tiles, alpha=0.3)

    # Plot the trajectory points for both the real (red) and protected (green) trajectories
    axs[0].scatter(trip_real.x, trip_real.y,  linewidth=0.35, color="red", alpha=0.5)
    axs[0].scatter(trip_pred.x, trip_pred.y,  linewidth=0.35, color="green", alpha=0.5)
    
    for i in range(0,len(trip_real.x)):
        axs[0].text(trip_real.x[i], trip_real.y[i], '$a_{'+str(i+1)+'}$', horizontalalignment="left", fontdict=font)
        axs[0].text(trip_pred.x[i], trip_pred.y[i], '$p_{'+str(i+1)+'}$', horizontalalignment="left", fontdict=font)

    # Set the axis labels for the upper sub-plot
    axs[0].set_xlabel('Longitude')
    axs[0].set_ylabel('Latitude')
    if (adaptive==True):
        axs[0].set_title("Real vs. protected trajectory (adaptive k)")
    else:
        axs[0].set_title("Real vs. protected trajectory (k = " + str(k) + ")")
    axs[0].legend(["Real", "Protected" ])
    
    # Plot the axis labels for the map
    
    # n_xlabels = len(axs[0].xaxis.get_major_ticks())
    # n_ylabels = len(axs[0].yaxis.get_major_ticks())
    
    # # Ensure the number of tick labels matches the number of ticks
    # n_xlabels = len(axs[0].get_xticks())
    # n_ylabels = len(axs[0].get_yticks())
    
    # axs[0].set_xticklabels(np.round(np.linspace(lons_min_ext, lons_max_ext, n_xlabels),3))
    # axs[0].set_yticklabels(np.round(np.linspace(lats_min_ext, lats_max_ext, n_ylabels),3))
    
    # Dynamically get the current ticks
    x_ticks = axs[0].xaxis.get_major_ticks()
    y_ticks = axs[0].yaxis.get_major_ticks()

    # Ensure the number of labels matches the number of ticks
    # axs[0].set_xticks(x_ticks)  # Explicitly set the ticks
    axs[0].set_xticklabels(np.round(np.linspace(lons_min_ext, lons_max_ext, len(x_ticks)), 3))

    # axs[0].set_yticks(y_ticks)  # Explicitly set the ticks
    axs[0].set_yticklabels(np.round(np.linspace(lats_min_ext, lats_max_ext, len(y_ticks)), 3))
    
    # axs[0].set_ticks(np.round(np.linspace(lons_min_ext, lons_max_ext, N_LABELS_X),3))
    
    if (dists is not None):
        # Plot the point-to-point distance between the real and protected trajectory
        axs[1].plot(range(1, len(dists) + 1), dists, "g.-",linewidth=0.9, label = "point-to-point distance ($\mathcal{T}_r$ vs. $\mathcal{T}_p$)")
        
        # Plot the interval if the values (dist_min, dist_max) are provided 
        if (dist_min > 0) and (dist_max > 0):
            axs[1].axhline(y = statistics.mean(dists), color = 'g', alpha=1,linestyle = '--',linewidth=1, label = "mean distance ($\mathcal{T}_r$ vs. $\mathcal{T}_p$)")
            axs[1].axhline(y = dist_min, color = 'k', alpha=0.5,linestyle = '-',linewidth=0.5)
            axs[1].axhline(y = dist_max, color = 'k', alpha=0.5,linestyle = '-',linewidth=0.5)
        else:
            axs[1].axhline(y = statistics.mean(dists), color = 'g', alpha=1,linestyle = '--',linewidth=1.5, label = "mean distance ($\mathcal{T}_r$ vs. $\mathcal{T}_p$)")

        # Set the axis labels for the bottom sub-plot
        axs[1].set_xlabel('Point number')
        axs[1].set_ylabel('Distance (m)')
        axs[1].legend(loc="lower right")
    else:   
        axs[1].remove()
    
    # Plot the values of k if provided
    if (ks is not None):
        axsk = axs[1].twinx()
        axsk.plot(range(1, len(dists) + 1), ks, "b-",linewidth=0.9, label = "k")
        axsk.legend(loc="lower right")
    
    # Save the figure if save paramenter is True
    if (save==True):
        plt.savefig(PLOT_FOLDER + "aputraj_" + str(traj_id) + "_k" + str(k) + ".pdf")
    
    if show == True:
        plt.show()
        
    return lons_min_ext, lons_max_ext, lats_min_ext, lats_max_ext


def plot_trajectory_adaptive_k(real_traj, protected_traj, dists = None, dist_min=0, dist_max=0, 
                    ks=None, k=1, traj_id = 0, show=True, save=True, adaptive=False, aspect_ratio=1.0,
                    columns_indexes = None, idx_reset = None,
                    savePath = None):
        
    font = PLOT_FONT
    SMALL_SIZE = PLOT_FONT_SMALL_SIZE
    MEDIUM_SIZE = PLOT_FONT_MEDIUM_SIZE
    N_LABELS_X = 9
    N_LABELS_Y = 5

    plt.close('all')
    plt.figure()
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
   
    predicted = pd.DataFrame()
    actual = pd.DataFrame()
    
    # Check if the columns indexes are provided
    # If not, use the default indexes
    # If the columns indexes are provided, use them to extract the latitude and longitude
    
    predicted["lats"] = protected_traj[:, 0]
    predicted["lons"] = protected_traj[:, 1] 
    
    actual["lats"] = real_traj[:, 0]
    actual["lons"] = real_traj[:, 1]
        
    # Compute the min/max values of the lons/lats
    # to set the limits of the plot           
    lons_min = min(predicted.lons.min(),actual.lons.min())
    lons_max = max(predicted.lons.max(),actual.lons.max())
    lats_min = min(predicted.lats.min(),actual.lats.min())
    lats_max = max(predicted.lats.max(),actual.lats.max())
    
    axs = plt.subplots(2,1, figsize=(10,10))[1]
    
    # Initialise the tilemapbase    
    tilemapbase.init(create=True)
    tiles = tilemapbase.tiles.build_OSM()
    
    # Extend the map to include a small margin around the trajectory
    # This is done to avoid the trajectory points to be cut off
    # when the trajectory is close to the map border
    extent_factor=0.005
    
    lons_min_ext = lons_min - extent_factor
    lons_max_ext = lons_max + extent_factor
    lats_min_ext = lats_min - extent_factor
    lats_max_ext = lats_max + extent_factor
    extent = tilemapbase.Extent.from_lonlat(
        lons_min_ext,
        lons_max_ext,
        lats_min_ext,
        lats_max_ext
    )
    
    extent = extent.to_aspect(aspect_ratio, False)
    
    trip_pred = predicted.apply(
        lambda x: tilemapbase.project(x.lons, x.lats), axis=1
    ).apply(pd.Series)
    trip_pred.columns = ["x", "y"]
    
    trip_real = actual.apply(
        lambda x: tilemapbase.project(x.lons, x.lats), axis=1
    ).apply(pd.Series)
    trip_real.columns = ["x", "y"]
    
    plotter = tilemapbase.Plotter(extent, tiles, width=1200)
    
    plotter.plot(axs[0], tiles, alpha=0.3)

    # Plot the trajectory points for both the real (red) and protected (green) trajectories
    axs[0].scatter(trip_real.x, trip_real.y,  linewidth=0.35, color="red", alpha=0.5)
    axs[0].scatter(trip_pred.x, trip_pred.y,  linewidth=0.35, color="green", alpha=0.5)
    
    for i in range(0,len(trip_real.x)):
        axs[0].text(trip_real.x[i], trip_real.y[i], '$a_{'+str(i+1)+'}$', horizontalalignment="left", fontdict=font)
        axs[0].text(trip_pred.x[i], trip_pred.y[i], '$p_{'+str(i+1)+'}$', horizontalalignment="left", fontdict=font)

    # Set the axis labels for the upper sub-plot
    axs[0].set_xlabel('Longitude')
    axs[0].set_ylabel('Latitude')
    if (adaptive==True):
        axs[0].set_title("Real vs. protected trajectory (adaptive k)")
    else:
        axs[0].set_title("Real vs. protected trajectory (k = " + str(k) + ")")
    axs[0].legend(["Real", "Protected" ])
    
    # Plot the axis labels for the map
    
    # n_xlabels = len(axs[0].xaxis.get_major_ticks())
    # n_ylabels = len(axs[0].yaxis.get_major_ticks())
    
    # # Ensure the number of tick labels matches the number of ticks
    # n_xlabels = len(axs[0].get_xticks())
    # n_ylabels = len(axs[0].get_yticks())
    
    # axs[0].set_xticklabels(np.round(np.linspace(lons_min_ext, lons_max_ext, n_xlabels),3))
    # axs[0].set_yticklabels(np.round(np.linspace(lats_min_ext, lats_max_ext, n_ylabels),3))
    
    # Dynamically get the current ticks
    x_ticks = axs[0].xaxis.get_major_ticks()
    y_ticks = axs[0].yaxis.get_major_ticks()

    # Ensure the number of labels matches the number of ticks
    # axs[0].set_xticks(x_ticks)  # Explicitly set the ticks
    axs[0].set_xticklabels(np.round(np.linspace(lons_min_ext, lons_max_ext, len(x_ticks)), 2))

    # axs[0].set_yticks(y_ticks)  # Explicitly set the ticks
    axs[0].set_yticklabels(np.round(np.linspace(lats_min_ext, lats_max_ext, len(y_ticks)), 3))
    
    # axs[0].set_ticks(np.round(np.linspace(lons_min_ext, lons_max_ext, N_LABELS_X),3))
    
    if (dists is not None):
        # Plot the point-to-point distance between the real and protected trajectory
        axs[1].plot(range(1, len(dists) + 1), dists, "g.-",linewidth=0.9, label = "point-to-point distance ($\mathcal{T}_r$ vs. $\mathcal{T}_p$)")
        
        # Plot the interval if the values (dist_min, dist_max) are provided 
        if (dist_min > 0) and (dist_max > 0):
            axs[1].axhline(y = statistics.mean(dists), color = 'g', alpha=1,linestyle = '--',linewidth=1, label = "mean distance ($\mathcal{T}_r$ vs. $\mathcal{T}_p$)")
            axs[1].axhline(y = dist_min, color = 'k', alpha=0.5,linestyle = '-',linewidth=0.5)
            axs[1].axhline(y = dist_max, color = 'k', alpha=0.5,linestyle = '-',linewidth=0.5)
        else:
            axs[1].axhline(y = statistics.mean(dists), color = 'g', alpha=1,linestyle = '--',linewidth=1.5, label = "mean distance ($\mathcal{T}_r$ vs. $\mathcal{T}_p$)")

        # Set the axis labels for the bottom sub-plot
        axs[1].set_xlabel('Point number')
        axs[1].set_ylabel('Distance (m)')
        axs[1].legend(loc="upper right")
    else:   
        axs[1].remove()
    
    # Plot the values of k if provided
    if (ks is not None):
        axsk = axs[1].twinx()
        axsk.plot(range(1, len(dists) + 1), ks, "b.-",linewidth=0.9, label = "k")
        axsk.legend(loc="lower right")
    
    if (idx_reset is not None):    
        axsk.plot(range(1, len(idx_reset) + 1), idx_reset, "r.-",linewidth=0.9, label = "k")
        
    
    # Save the figure if save paramenter is True
    if (save==True):
        plt.savefig(PLOT_FOLDER + "aputraj_" + savePath + ".pdf")
    
    if show == True:
        plt.show()
        
    return lons_min_ext, lons_max_ext, lats_min_ext, lats_max_ext

