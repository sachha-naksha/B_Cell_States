####
# 1. Gets data over pseudotime on all scales (window, gaussian smoothed points, episode).
# 2. Classifies TFs based on their global activity.
####

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
    A class for generating smoothed curves over pseudotime.
    
    This class provides methods to compute expression and regulation curves,
    calculate regulatory forces.
    """
    
    def __init__(self, dictys_dynamic_object=None):
        """
        Initialize the analyzer with an optional dictys dynamic object.
        
        Args:
            dictys_dynamic_object: The dictys dynamic object containing network data
        """
        self.dictys_dynamic_object = dictys_dynamic_object
    
    def get_smoothed_curves(
        self,
        start: int,
        stop: int,
        num: int = 100,
        dist: float = 1.5,
        mode: str = "regulation",
        sparsity: float = 0.01,
        dictys_dynamic_object=None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Compute expression (lcpm) and regulation (ltarget_count) curves over pseudotime for one branch.
        
        Args:
            start: Start node on the trajectory
            stop: Stop node on the trajectory
            num: Number of equispaced points to sample (default: 100)
            dist: Pseudotime distance to apply gaussian smoothing over (default: 1.5)
            mode: Data to retrieve - "regulation", "weighted_regulation", "TF_expression", or "expression"
            sparsity: Sparsity threshold for the network (default: 0.01)
            dictys_dynamic_object: Override the class instance object
            
        Returns:
            Tuple of (curves_dataframe, pseudotime_series)
        """
        obj = dictys_dynamic_object or self.dictys_dynamic_object
        if obj is None:
            raise ValueError("No dictys_dynamic_object provided")

        # sample equispaced points and instantiate smoothing function    
        pts, fsmooth = obj.linspace(start, stop, num, dist)
        
        if mode == "regulation":
            # Log number of targets
            stat1_net = fsmooth(stat.net(obj))
            stat1_netbin = stat.fbinarize(stat1_net, sparsity=sparsity)
            stat1_y = stat.flnneighbor(stat1_netbin)
        elif mode == "weighted_regulation":
            # Log weighted outdegree
            stat1_net = fsmooth(stat.net(obj))
            stat1_y = stat.flnneighbor(stat1_net, weighted_sparsity=sparsity)
        elif mode == "TF_expression":
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
        Computes area under the curves using trapezoidal rule.
        
        Args:
            dx: X-axis values (must be increasing)
            dy: Y-axis values (2D array where each row is a curve)
            
        Returns:
            Array of AUC values for each curve
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
        Computes transient log fold change for curves.
        
        This measures the transient behavior by normalizing pseudotime and
        subtracting baseline (median of curve, start, and end values).
        
        Args:
            dx: Pseudotime values
            dy: Expression/regulation values
            
        Returns:
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
        Computes switching time for curves.
        
        This measures when the main transition occurs by calculating the
        normalized area under the curve relative to the total change.
        
        Args:
            dx: Pseudotime values
            dy: Expression/regulation values
            
        Returns:
            Switching time values for each curve
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
        Computes terminal log fold change for curves.
        
        This is simply the difference between final and initial values.
        
        Args:
            dx: Pseudotime values (must be increasing)
            dy: Expression/regulation values
            
        Returns:
            Terminal logFC values for each curve
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
        Calculate regulatory force curves using log transformation.
        
        Force is calculated as beta * TF_expression
        
        Args:
            beta_curves: DataFrame with regulatory coefficients (multi-indexed by TF and target)
            tf_expression: Series with TF expression values
            
        Returns:
            DataFrame with calculated force curves
        """
        # Count number of targets per TF from beta_curves multi-index
        targets_per_tf = beta_curves.index.get_level_values(0).value_counts()
        
        # Create a DataFrame with repeated TF expression values for each target
        expanded_tf_expr = pd.DataFrame(
            np.repeat(
                tf_expression.values, targets_per_tf.values, axis=0
            ),
            index=beta_curves.index,
            columns=beta_curves.columns,
        )
        
        # Convert to numpy arrays for calculations
        beta_array = beta_curves.to_numpy()
        tf_array = expanded_tf_expr.to_numpy()
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        log_beta = np.log10(np.abs(beta_array) + epsilon)
        log_tf = np.log10(tf_array + epsilon)
        
        # Preserve signs from original beta values
        signs = np.sign(beta_array)
        
        # Calculate forces
        force_array = signs * np.exp(log_beta + log_tf)
        
        # Convert back to DataFrame with original index/columns
        force_curves = pd.DataFrame(
            force_array, 
            index=beta_curves.index, 
            columns=beta_curves.columns
        )
        
        return force_curves

##################################### Curve characteristics ############################################

def curve_characteristics(dcurve, dtime):
    """
    Compute curve characteristics for one branch.
    Switching time, Terminal logFC, Transient logFC per TF
    """
    charlist = {
        "Terminal logFC": _dynamic_network_char_terminal_logfc_,
        "Transient logFC": _dynamic_network_char_transient_logfc_,
        "Switching time": _dynamic_network_char_switching_time_,
    }
    # Compute curve characteristics
    dchar = {}
    for xj in charlist:
        dchar[xj] = charlist[xj](dtime.values, dcurve.values)
    dchar = pd.DataFrame.from_dict(dchar)
    dchar.set_index(dcurve.index, inplace=True, drop=True)
    return dchar


def classify_tf_global_activity(df, terminal_col, transient_col):
    """
    Add TF activity class to dataframe based on z-score normalized logFC comparison
    """
    # Z-score normalize both columns
    terminal_zscore = stats.zscore(df[terminal_col])
    transient_zscore = stats.zscore(df[transient_col])

    # Classification function
    def get_class_name(terminal_z, transient_z):
        if abs(terminal_z) >= abs(transient_z):
            # Terminal effect dominates
            return "Cumulative" if terminal_z > 0 else "Reductive"
        else:
            # Transient effect dominates
            return "Bell wave" if transient_z > 0 else "U-shaped"

    # Add class name column
    df["tf_class"] = [
        get_class_name(t_z, tr_z)
        for t_z, tr_z in zip(terminal_zscore, transient_zscore)
    ]
    # Add z score columns
    df["terminal_z"] = terminal_zscore
    df["transient_z"] = transient_zscore
    df["terminal_rank"] = (
        df["terminal_z"].abs().rank(method="dense", ascending=False).astype(int)
    )
    df["transient_rank"] = (
        df["transient_z"].abs().rank(method="dense", ascending=False).astype(int)
    )
    return df


def get_top_k_tfs_by_class(df, k=20):
    """
    Get top k TFs from each class based on their relevant ranks
    """

    # Determine which rank to use for each TF based on their class
    def get_relevant_rank(row):
        # If terminal effect dominates (Activating/Inactivating or similar), use terminal_rank
        # If transient effect dominates, use transient_rank
        if abs(row["terminal_z"]) >= abs(row["transient_z"]):
            return row["terminal_rank"]
        else:
            return row["transient_rank"]

    df["relevant_rank"] = df.apply(get_relevant_rank, axis=1)

    # Get unique classes
    classes = df["tf_class"].unique()

    # Dictionary to store top k TFs for each class
    top_tfs_dict = {}

    for class_name in classes:
        class_df = df[df["tf_class"] == class_name].copy()
        # Sort by relevant rank and take top k
        top_k = class_df.nsmallest(k, "relevant_rank")
        # Extract TF names (assuming index contains TF names)
        top_tfs_dict[class_name] = top_k.index.tolist()

    # Create result dataframe with classes as columns
    # Pad shorter lists with None to make all columns same length
    max_len = max(len(v) for v in top_tfs_dict.values())

    for class_name in top_tfs_dict:
        while len(top_tfs_dict[class_name]) < max_len:
            top_tfs_dict[class_name].append(None)

    result_df = pd.DataFrame(top_tfs_dict)

    return result_df
