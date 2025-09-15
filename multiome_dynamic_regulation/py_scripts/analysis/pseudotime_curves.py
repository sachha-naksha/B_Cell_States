import gc
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Optional, Tuple, Union

import dictys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dictys.net import stat
from dictys.utils.numpy import ArrayLike
from numpy.typing import NDArray
from joblib import Memory
from scipy import stats
from scipy.stats import hypergeom
from tqdm import tqdm

from utils_custom import *


class SmoothedCurves:
    """
    provides methods to compute expression and regulation curves,
    and calculate regulatory forces.
    """
    
    def __init__(self, dictys_dynamic_object=None,
        trajectory_range=(1, 3),
        num_points=40,
        dist=0.001,
        sparsity=0.01,
        mode="regulation",
        ):
        """
        initialize the analyzer
        """
        self.dictys_dynamic_object = dictys_dynamic_object
        self.trajectory_range = trajectory_range
        self.num_points = num_points
        self.dist = dist
        self.sparsity = sparsity
        self.mode = mode
    
    def get_smoothed_curves(
        self
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        compute expression (lcpm) and regulation (ltarget_count) curves over pseudotime for one branch.
        
        args:
            start: Start node on the trajectory
            stop: Stop node on the trajectory
            num: Number of equispaced points to sample (default: 100)
            dist: Pseudotime distance to apply gaussian smoothing over (default: 1.5)
            mode: Data to retrieve - "regulation", "weighted_regulation", "tf_expression", or "expression"
            sparsity: Sparsity threshold for the network (default: 0.01)
            dictys_dynamic_object: Override the class instance object
            
        returns:
            tuple of (curves_dataframe, pseudotime_series)
        """
        obj = dictys_dynamic_object or self.dictys_dynamic_object
        if obj is None:
            raise ValueError("No dictys_dynamic_object provided")

        # sample equispaced points and instantiate smoothing function    
        pts, fsmooth = obj.linspace(self.trajectory_range[0], self.trajectory_range[1], self.num_points, self.dist)
        
        if mode == "regulation":
            # Log number of targets
            stat1_net = fsmooth(stat.net(obj))
            stat1_netbin = stat.fbinarize(stat1_net, sparsity=self.sparsity)
            stat1_y = stat.flnneighbor(stat1_netbin)
        elif mode == "weighted_regulation":
            # Log weighted outdegree
            stat1_net = fsmooth(stat.net(obj))
            stat1_y = stat.flnneighbor(stat1_net, weighted_sparsity=self.sparsity)
        elif mode == "tf_expression":
            stat1_y = fsmooth(stat.lcpm_tf(obj, cut=0))
        elif mode == "expression":
            stat1_y = fsmooth(stat.lcpm(obj, cut=0))
        else:
            raise ValueError(f"Unknown mode {mode}.")
            
        # Pseudo time values (x axis)
        stat1_x = stat.pseudotime(obj, pts)
        tmp_y = stat1_y.compute(pts)
        tmp_x = stat1_x.compute(pts)
        dy = pd.DataFrame(tmp_y, index=stat1_y.names[0])
        dx = pd.Series(tmp_x[0])  # first gene's pseudotime is returned as all genes have the same pseudotime over the pseudo-bulked cells in the window
        
        return dy, dx
    
    @staticmethod
    def calculate_auc(dx: NDArray[float], dy: NDArray[float]) -> NDArray[float]:
        """
        computes area under the curves using trapezoidal rule.
        
        args:
            dx: X-axis values (must be increasing)
            dy: Y-axis values (2D array where each row is a curve)
            
        returns:
            array of AUC values for each curve
        """
        if len(dx) < 2 or not (dx[1:] > dx[:-1]).all():
            raise ValueError("dx must be increasing and have at least 2 values.")
        dxdiff = dx[1:] - dx[:-1]
        dymean = (dy[:, 1:] + dy[:, :-1]) / 2
        ans = dymean @ dxdiff
        return ans
    
    def calculate_transient_logfc(
        self, 
        dx: NDArray[float], 
        dy: NDArray[float]
    ) -> NDArray[float]:
        """
        computes transient log fold change for curves.
        
        args:
            dx: Pseudotime values
            dy: Expression/regulation values
            
        returns:
            Transient logFC values for each curve
        """
        n = dy.shape[1]
        dx = (dx - dx[0]) / (dx[-1] - dx[0])
        dy = dy - np.median(
            [dy, np.repeat(dy[:, [0]], n, axis=1), np.repeat(dy[:, [-1]], n, axis=1)],
            axis=0,
        )
        return self.calculate_auc(dx, dy)
    
    def calculate_switching_time(
        self, 
        dx: NDArray[float], 
        dy: NDArray[float]
    ) -> NDArray[float]:
        """
        measures when the main transition occurs by calculating the
        normalized area under the curve relative to the total change.
        
        args:
            dx: Pseudotime values
            dy: Expression/regulation values
            
        returns:
            switching time values for each curve
        """
        n = dy.shape[1]
        dx = (dx - dx[0]) / (dx[-1] - dx[0])
        dy = np.median(
            [dy, np.repeat(dy[:, [0]], n, axis=1), np.repeat(dy[:, [-1]], n, axis=1)],
            axis=0,
        )
        return (self.calculate_auc(dx, (dy.T - dy[:, -1]).T)) / (dy[:, 0] - dy[:, -1] + 1e-300)
    
    @staticmethod
    def calculate_terminal_logfc(
        dx: NDArray[float], 
        dy: NDArray[float]
    ) -> NDArray[float]:
        """
        computes difference between final and initial values.
        
        args:
            dx: Pseudotime values (must be increasing)
            dy: Expression/regulation values
            
        returns:
            terminal logFC values for each curve
        """
        if len(dx) < 2 or not (dx[1:] > dx[:-1]).all():
            raise ValueError("dx must be increasing and have at least 2 values.")
        return dy[:, -1] - dy[:, 0]

    @staticmethod
    def calculate_force_curves(
        beta_curves: pd.DataFrame, 
        tf_expression: pd.Series
    ) -> pd.DataFrame:
        """
        calculates regulatory force curves using log transformation.
        
        force is calculated as beta * tf_expression
        
        args:
            beta_curves: DataFrame with regulatory coefficients (multi-indexed by tf and target)
            tf_expression: Series with tf expression values
            
        returns:
            dataframe with calculated force curves
        """
        # Count number of targets per tf from beta_curves multi-index
        targets_per_tf = beta_curves.index.get_level_values(0).value_counts()
        
        # Create a DataFrame with repeated tf expression values for each target
        expanded_tf_expr = pd.DataFrame(
            np.repeat(
                tf_expression.values, targets_per_tf.values, axis=0
            ),
            index=beta_curves.index,
            columns=beta_curves.columns,
        )
        
        # convert to numpy arrays for calculations
        beta_array = beta_curves.to_numpy()
        tf_array = expanded_tf_expr.to_numpy()
        
        # add small epsilon to avoid log(0)
        epsilon = 1e-10
        log_beta = np.log10(np.abs(beta_array) + epsilon)
        log_tf = np.log10(tf_array + epsilon)
        
        # peserve signs from original beta values
        signs = np.sign(beta_array)
        
        # calculate forces
        force_array = signs * np.exp(log_beta + log_tf)
        
        # convert back to DataFrame with original index/columns
        force_curves = pd.DataFrame(
            force_array, 
            index=beta_curves.index, 
            columns=beta_curves.columns
        )
        
        return force_curves

    def curve_characteristics(
        self,
        dx: NDArray[float],
        dy: NDArray[float],
        include_metrics: list = None
    ) -> pd.DataFrame:
        """
        compute multiple characteristics for the given curves.
        
        args:
            dx: Pseudotime values
            dy: Expression/regulation values  
            include_metrics: List of metrics to compute. Options: 
                           ['transient_logfc', 'switching_time', 'terminal_logfc', 'auc']
                           if None, computes all metrics.
                           
        returns:
            dataframe with computed characteristics for each curve
        """
        if include_metrics is None:
            include_metrics = ['transient_logfc', 'switching_time', 'terminal_logfc', 'auc']
        
        results = {}
        
        if 'transient_logfc' in include_metrics:
            results['transient_logfc'] = self.calculate_transient_logfc(dx, dy)
        if 'switching_time' in include_metrics:
            results['switching_time'] = self.calculate_switching_time(dx, dy)
        if 'terminal_logfc' in include_metrics:
            results['terminal_logfc'] = self.calculate_terminal_logfc(dx, dy)
        if 'auc' in include_metrics:
            results['auc'] = self.calculate_auc(dx, dy)
        
        return pd.DataFrame(results)

    def classify_tf_global_activity(self, dx, dy, terminal_col, transient_col):
        """
        add tf activity class to dataframe based on z-score normalized logfc comparison
        """
        df = self.curve_characteristics(dx, dy)
        # Z-score normalize both columns
        terminal_zscore = stats.zscore(df[terminal_col])
        transient_zscore = stats.zscore(df[transient_col])

        # classification function
        def get_class_name(terminal_z, transient_z):
            if abs(terminal_z) >= abs(transient_z):
                # terminal effect dominates
                return "Cumulative" if terminal_z > 0 else "Reductive"
            else:
                # transient effect dominates
                return "Bell wave" if transient_z > 0 else "U-shaped"

        # add class name column
        df["tf_class"] = [
            get_class_name(t_z, tr_z)
            for t_z, tr_z in zip(terminal_zscore, transient_zscore)
        ]
        # add z score columns
        df["terminal_z"] = terminal_zscore
        df["transient_z"] = transient_zscore
        df["terminal_rank"] = (
            df["terminal_z"].abs().rank(method="dense", ascending=False).astype(int)
        )
        df["transient_rank"] = (
            df["transient_z"].abs().rank(method="dense", ascending=False).astype(int)
        )
        return df

    def get_top_k_tfs_by_class(self, dx, dy, k=20):
        """
        get top k tfs from each class based on their relevant ranks
        """
        df = self.classify_tf_global_activity(dx, dy, "terminal_logfc", "transient_logfc")
        # determine which rank to use for each tf based on their class
        def get_relevant_rank(row):
            # if terminal effect dominates (Activating/Inactivating or similar), use terminal_rank
            # if transient effect dominates, use transient_rank
            if abs(row["terminal_z"]) >= abs(row["transient_z"]):
                return row["terminal_rank"]
            else:
                return row["transient_rank"]

        df["relevant_rank"] = df.apply(get_relevant_rank, axis=1)

        # get unique classes
        classes = df["tf_class"].unique()

        # dictionary to store top k tfs for each class
        top_tfs_dict = {}

        for class_name in classes:
            class_df = df[df["tf_class"] == class_name].copy()
            # sort by relevant rank and take top k
            top_k = class_df.nsmallest(k, "relevant_rank")
            # extract tf names (assuming index contains tf names)
            top_tfs_dict[class_name] = top_k.index.tolist()

        # create result dataframe with classes as columns
        # pad shorter lists with None to make all columns same length
        max_len = max(len(v) for v in top_tfs_dict.values())

        for class_name in top_tfs_dict:
            while len(top_tfs_dict[class_name]) < max_len:
                top_tfs_dict[class_name].append(None)

        result_df = pd.DataFrame(top_tfs_dict)

        return result_df

class AlignTimeScales:
    """
    standardize the window, sampled points, episode time-scales to S0 pseudotime values.
    """
    def __init__(self, dictys_dynamic_object=None,
        trajectory_range=(1, 3),
        num_points=40,
        dist=0.001,
        sparsity=0.01,
        total_episodes=8,
        ):
        """
        initialize the aligner with an optional dictys dynamic object.
        """
        self.dictys_dynamic_object = dictys_dynamic_object
        self.trajectory_range = trajectory_range
        self.num_points = num_points
        self.dist = dist
        self.sparsity = sparsity
        self.total_episodes = total_episodes

    def pseudotime_of_windows(self):
        """
        returns - numpy array (of window centroids' pseudotime values)
        S0 or S1 pseudotime values depending on the trajectory range.
        """
        # get 2d array of all nodes' pseudotimes per window
        node_specific_pseudotimes_per_window = self.dictys_dynamic_object.point['s'].dist
        # if traj range starts from 0 return the 0th pseudotime out of 0,1,2,3 and if it starts from 1 return the 1st pseudotime out of 0,1,2,3
        idx_for_querying_pseudotimes = self.trajectory_range[0]
        # return the pseudotime values
        return node_specific_pseudotimes_per_window[:, idx_for_querying_pseudotimes]

    def pseudotime_of_sampled_points(self):
        """
        returns - numpy array of sampled points' pseudotime values.
        S0 or S1 pseudotime values depending on the trajectory range.
        """
        # sample equi-spaced points and instantiate smoothing function    
        pts, fsmooth = self.dictys_dynamic_object.linspace(self.trajectory_range[0], self.trajectory_range[1], self.num_points, self.dist)
        # pseudo time values (x axis)
        stat1_x = stat.pseudotime(self.dictys_dynamic_object, pts)
        tmp_x = stat1_x.compute(pts)
        dx = pd.Series(tmp_x[0])
        # return numpy array of pseudotime values
        return dx.to_numpy()


