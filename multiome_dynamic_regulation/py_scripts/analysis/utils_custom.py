import os
from typing import Optional, Tuple, Union

import dictys
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dictys.net import stat
from dictys.plot.dynamic import _compute_chars_
import ipdb

PATH_TO_CELL_LABELS = "/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added/data/clusters.csv"

def load_dynamic_object(dynamic_object_path):
    """
    Load the dynamic object from the given path
    """
    dynamic_object = dictys.net.dynamic_network.from_file(dynamic_object_path)
    return dynamic_object

######################### Indices retrieval #########################

def get_tf_indices(dictys_dynamic_object, tf_list):
    """
    Get the indices of transcription factors from a list, if present in ndict and nids[0].
    """
    gene_hashmap = dictys_dynamic_object.ndict
    tf_mappings_to_gene_hashmap = dictys_dynamic_object.nids[0]
    
    tf_indices = []
    gene_indices = []
    missing_tfs = []

    for gene in tf_list:
        # Check if the gene is in the gene_hashmap
        if gene in gene_hashmap:
            gene_index = gene_hashmap[gene]  # Get the index in gene_hashmap
            # Check if the gene index is present in tf_mappings_to_gene_hashmap
            match = np.where(tf_mappings_to_gene_hashmap == gene_index)[0]
            if match.size > 0:  # If a match is found
                tf_indices.append(int(match[0]))  # Append the position of the match
                gene_indices.append(int(gene_index))  # Also append the gene index
            else:
                missing_tfs.append(gene)  # Gene exists but not as a TF
        else:
            missing_tfs.append(gene)  # Gene not found at all
    return tf_indices, gene_indices, missing_tfs

def get_target_indices(dictys_dynamic_object, target_list):
    """
    Get the indices of target genes from a list, if present in ndict and nids[1].
    """
    gene_hashmap = dictys_dynamic_object.ndict
    target_indices = []
    for gene in target_list:
        if gene in gene_hashmap:
            target_indices.append(gene_hashmap[gene])
    return target_indices

def get_all_gene_names(dictys_dynamic_object):
    """
    Get all possible target gene names from the dynamic object's ndict
    """
    # Get all gene names from ndict
    gene_names = list(dictys_dynamic_object.ndict.keys())
    gene_names.sort()  # Sort for consistency
    return gene_names

def get_pseudotime_of_windows(dictys_dynamic_object, window_indices):
    """
    Get the pseudotime of specific windows for x-axis in plots
    """
    pseudotime_relative_to_bifurcation = dictys_dynamic_object.point[
        "s"
    ].locs  # Access via dictionary keys
    branch_pseudotime = [
        float(pseudotime_relative_to_bifurcation[idx]) for idx in window_indices
    ]
    return branch_pseudotime

######################### Edge weights retrieval #########################

def get_one_hop_grn_weights(dictys_dynamic_object, tf_indices=None, gene_indices=None, window_indices=None, nonzero_fraction_threshold=0.1):
    """
    Get the weights matrix for both specified TFs and TFs regulating specified targets (mostly local GRN of LF),
    after filtering the global GRN by sparsity across pseudotime
    """
    # Get the weights array
    global_grn_weights = dictys_dynamic_object.prop["es"]["w_n"]

    # Apply sparsity filter to global GRN first
    nonzero_edge_fraction_across_pseudotime = (global_grn_weights != 0).mean(axis=2)  # mean of a boolean, shape is total TFs x total targets
    edge_sparsity_filter = nonzero_edge_fraction_across_pseudotime >= nonzero_fraction_threshold

    # Convert query indices to lists if they're not already
    tf_indices = list(tf_indices) if tf_indices is not None else []
    gene_indices = list(gene_indices) if gene_indices is not None else []

    # Augment the TF indices that regulate the specified target genes (considering sparsity)
    if gene_indices:
        # Only consider edges that pass sparsity filter
        tf_mask = np.any(edge_sparsity_filter[:, gene_indices], axis=1)
        regulating_tf_indices = np.where(tf_mask)[0]
        # Combine with specified TF indices, removing duplicates while preserving order
        all_tf_indices = list(dict.fromkeys(tf_indices + list(regulating_tf_indices)))
    else:
        all_tf_indices = tf_indices
    
    # Augment target gene indices that are regulated by the specified TFs (considering sparsity)
    if tf_indices:
        # Only consider edges that pass sparsity filter
        gene_mask = np.any(edge_sparsity_filter[tf_indices, :], axis=0)
        regulated_gene_indices = np.where(gene_mask)[0]
        # Combine with specified gene indices, removing duplicates while preserving order
        all_gene_indices = list(dict.fromkeys(gene_indices + list(regulated_gene_indices)))
    else:
        all_gene_indices = gene_indices

    # Extract the combined weights matrix using the computed indices
    combined_weights = global_grn_weights[np.ix_(all_tf_indices, all_gene_indices, window_indices)]
    # Apply sparsity mask to final weights (in case non-augmented edges don't meet threshold)
    final_sparsity_mask = edge_sparsity_filter[np.ix_(all_tf_indices, all_gene_indices)]
    combined_weights = np.where(final_sparsity_mask[:, :, np.newaxis], combined_weights, 0)
    return combined_weights, all_tf_indices, all_gene_indices

def get_grn_weights_for_tfs(dictys_dynamic_object, tf_indices=None, window_indices=None, nonzero_fraction_threshold=0.1):
    """
    Get the weights matrix for specified TFs over specific windows
    """
    # Get the weights array
    global_grn_weights = dictys_dynamic_object.prop["es"]["w_n"]
    
    # Convert indices to numpy arrays and ensure they're integers
    tf_indices = np.array(tf_indices, dtype=int) if tf_indices is not None else np.array([], dtype=int)
    window_indices = np.array(window_indices, dtype=int) if window_indices is not None else np.array([], dtype=int)
    
    # Get dimensions
    n_targets = global_grn_weights.shape[1]
    
    # Calculate non-zero fractions across time for the specified TFs
    nonzero_edge_fraction = np.mean(global_grn_weights[tf_indices] != 0, axis=2)  # shape: (n_selected_tfs, n_targets)
    
    # Create sparsity mask
    sparsity_mask = nonzero_edge_fraction >= nonzero_fraction_threshold  # shape: (n_selected_tfs, n_targets)
    
    # Extract weights for specified TFs and windows
    weights = global_grn_weights[tf_indices, :, :]  # shape: (n_selected_tfs, n_targets, n_windows)
    weights = weights[:, :, window_indices]  # Select specific windows
    
    # Apply sparsity mask
    weights = np.where(sparsity_mask[:, :, np.newaxis], weights, 0)
    
    print(f"Output shape: {weights.shape} (n_tfs={len(tf_indices)}, n_targets={n_targets}, n_windows={len(window_indices)})")
    return weights

def get_grn_weights_for_tf_target_pairs(
    dictys_dynamic_object, tf_indices=None, target_indices=None, window_indices=None, nonzero_fraction_threshold=0.1
):
    """
    Get the weights for specific TF-target pairs over specific windows.
    """
    # Get the weights array
    global_grn_weights = dictys_dynamic_object.prop["es"]["w_n"]
    # Apply sparsity filter to global GRN first
    nonzero_edge_fraction_across_pseudotime = (global_grn_weights != 0).mean(axis=2)  # mean of a boolean, shape is total TFs x total targets
    edge_sparsity_filter = nonzero_edge_fraction_across_pseudotime >= nonzero_fraction_threshold
    
    # Get weights of the specified TF-target pairs over the specified windows
    weights_of_tf_target = global_grn_weights[np.ix_(tf_indices, target_indices, window_indices)]
    # Apply sparsity mask to final weights (in case any edges don't meet threshold)
    final_sparsity_mask = edge_sparsity_filter[np.ix_(tf_indices, target_indices)]
    weights_of_tf_target = np.where(final_sparsity_mask[:, :, np.newaxis], weights_of_tf_target, 0)
    return weights_of_tf_target

def get_indirect_grn_weights_for_tfs(
    dictys_dynamic_object, tf_indices=None, window_indices=None, nonzero_fraction_threshold=0.1
):
    """
    Get the indirect weights of TFs over specific windows for x-axis in plots
    """
    # Get the weights array
    global_grn_weights = dictys_dynamic_object.prop["es"]["w_in"]
    # Apply sparsity filter to global GRN first
    nonzero_edge_fraction_across_pseudotime = (global_grn_weights != 0).mean(axis=2)  # mean of a boolean, shape is total TFs x total targets
    edge_sparsity_filter = nonzero_edge_fraction_across_pseudotime >= nonzero_fraction_threshold
    
    # Convert indices to lists if they're not already
    tf_indices = list(tf_indices) if tf_indices is not None else []
    window_indices = list(window_indices) if window_indices is not None else []
    
    # Get the indirect weights of the specified TFs over the specified windows
    indirect_weights_of_tf_target = global_grn_weights[np.ix_(tf_indices, slice(None), window_indices)]
    # Apply sparsity mask to final weights (in case any edges don't meet threshold)
    final_sparsity_mask = edge_sparsity_filter[np.ix_(tf_indices, slice(None))]
    indirect_weights_of_tf_target = np.where(final_sparsity_mask[:, :, np.newaxis], indirect_weights_of_tf_target, 0)
    return indirect_weights_of_tf_target

##################################### Similarity metrics ############################################

def get_sign_switching_pairs(pb_weights, gc_weights, tf_names, target_names, 
                           edge_presence_threshold=0.3,
                           mean_weight=0.3,
                           spread_weight=0.2,
                           min_sign_changes=2):
    """
    Find TF-target pairs that show regulation sign changes between PB and GC branches,
    prioritizing by number of sign changes and considering mean and spread differences.
    
    Parameters:
    -----------
    pb_weights, gc_weights : np.ndarray
        Weight matrices for PB and GC branches (shape: n_tfs x n_targets x n_windows)
    tf_names, target_names : list
        Lists of TF and target gene names
    edge_presence_threshold : float
        Minimum fraction of non-zero values required in each branch
    mean_weight : float
        Weight for mean difference contribution to final score
    spread_weight : float
        Weight for spread difference contribution to final score
    min_sign_changes : int
        Minimum number of sign changes required to consider a pair
    
    Returns:
    --------
    filtered_pairs : list of tuples
        List of (TF, target) pairs that pass filtering criteria
    pair_scores : list of floats
        Corresponding scores for each pair
    pair_metrics : list of dicts
        Detailed metrics for each pair
    """
    # Verify dimensions
    assert len(tf_names) == pb_weights.shape[0], "Number of TF names doesn't match weight matrix"
    assert len(target_names) == pb_weights.shape[1], "Number of target names doesn't match weight matrix"
    
    n_windows = min(pb_weights.shape[2], gc_weights.shape[2])
    
    # Calculate edge presence in each branch
    pb_edge_presence = np.mean(pb_weights[:, :, :n_windows] != 0, axis=2)
    gc_edge_presence = np.mean(gc_weights[:, :, :n_windows] != 0, axis=2)
    edge_present = (pb_edge_presence > edge_presence_threshold) & \
                  (gc_edge_presence > edge_presence_threshold)
    
    # Initialize arrays to store metrics
    sign_changes = np.zeros((pb_weights.shape[0], pb_weights.shape[1]))
    mean_diffs = np.zeros_like(sign_changes)
    spread_diffs = np.zeros_like(sign_changes)
    
    # Calculate metrics for each TF-target pair
    for i in range(pb_weights.shape[0]):
        for j in range(pb_weights.shape[1]):
            if edge_present[i, j]:
                pb_vals = pb_weights[i, j, :n_windows]
                gc_vals = gc_weights[i, j, :n_windows]
                
                # Count sign changes
                sign_diff = np.sign(pb_vals) != np.sign(gc_vals)
                sign_changes[i, j] = np.sum(sign_diff & (pb_vals != 0) & (gc_vals != 0))
                
                # Calculate mean difference
                pb_mean = np.mean(pb_vals)
                gc_mean = np.mean(gc_vals)
                mean_diffs[i, j] = np.abs(pb_mean - gc_mean)
                
                # Calculate spread difference (using standard deviation)
                pb_spread = np.std(pb_vals)
                gc_spread = np.std(gc_vals)
                spread_diffs[i, j] = np.abs(pb_spread - gc_spread)
    
    # Normalize mean and spread differences to [0, 1] range
    mean_diffs_norm = mean_diffs / np.max(mean_diffs) if np.max(mean_diffs) > 0 else mean_diffs
    spread_diffs_norm = spread_diffs / np.max(spread_diffs) if np.max(spread_diffs) > 0 else spread_diffs
    
    # Filter pairs with more than minimum sign changes
    sufficient_changes = sign_changes >= min_sign_changes
    
    # Calculate composite scores
    # Note: sign_changes are not normalized as they are the primary filter
    composite_scores = (sign_changes + 
                       mean_weight * mean_diffs_norm + 
                       spread_weight * spread_diffs_norm)
    
    # Get indices of pairs that pass filters
    tf_idx, target_idx = np.where(sufficient_changes & edge_present)
    
    # Get scores and sort
    scores = composite_scores[tf_idx, target_idx]
    sort_idx = np.argsort(-scores)
    
    # Create output lists
    filtered_pairs = []
    pair_scores = []
    pair_metrics = []
    
    for i in sort_idx:
        tf_i = tf_idx[i]
        target_i = target_idx[i]
        
        filtered_pairs.append((tf_names[tf_i], target_names[target_i]))
        pair_scores.append(scores[i])
        
        # Store detailed metrics
        pair_metrics.append({
            'sign_changes': sign_changes[tf_i, target_i],
            'mean_diff': mean_diffs[tf_i, target_i],
            'spread_diff': spread_diffs[tf_i, target_i],
            'pb_pattern': pb_weights[tf_i, target_i, :n_windows],
            'gc_pattern': gc_weights[tf_i, target_i, :n_windows]
        })
    
    # Print summary statistics
    print(f"\nSign-Switching Edge Statistics:")
    print(f"Total edges examined: {np.sum(edge_present):,}")
    print(f"Edges with ≥{min_sign_changes} sign changes: {np.sum(sufficient_changes):,}")
    print(f"Final pairs: {len(filtered_pairs):,}")
    
    if filtered_pairs:
        print("\nTop switching pairs:")
        for i in range(min(10, len(filtered_pairs))):
            tf, target = filtered_pairs[i]
            metrics = pair_metrics[i]
            print(f"\n{tf} -> {target}")
            print(f"Sign changes: {metrics['sign_changes']:.0f}")
            print(f"Mean difference: {metrics['mean_diff']:.3f}")
            print(f"Spread difference: {metrics['spread_diff']:.3f}")
    
    return filtered_pairs, pair_scores, pair_metrics

##################################### Utils ############################################

def get_state_labels_in_window(dictys_dynamic_object, cell_labels):
    """
    Creates a mapping of window indices to their constituent cells' labels
    """
    # get cell assignment matrix from dictys_dynamic_object.prop['sc']['w']
    cell_assignment_matrix = dictys_dynamic_object.prop['sc']['w']
    state_labels_in_window = {}
    for window_idx in range(cell_assignment_matrix.shape[0]):
        indices_of_cells_present_in_window = np.where(cell_assignment_matrix[window_idx] == 1)[0]
        state_labels_in_window[window_idx] = [cell_labels[idx] for idx in indices_of_cells_present_in_window]
    return state_labels_in_window

def get_state_total_counts(cell_labels):
    """
    Get total number of cells for each state in the dataset
    """
    state_counts = {}
    for label in cell_labels:
        state_counts[label] = state_counts.get(label, 0) + 1
    return state_counts

def get_top_k_fraction_labels(dictys_dynamic_object, window_idx, cell_labels, k=2):
    """
    Returns the k labels with both fractions for a given window
    """
    # Get state labels for all windows
    state_labels_dict = get_state_labels_in_window(dictys_dynamic_object, cell_labels)
    # Get window labels for specified window
    window_labels = state_labels_dict[window_idx]
    # Get total counts across all states
    state_total_counts = get_state_total_counts(cell_labels)
    # Count cells per state in this window
    window_counts = {}
    for label in window_labels:
        window_counts[label] = window_counts.get(label, 0) + 1
    total_cells_in_window = len(window_labels)
    # Calculate both fractions for each state
    state_metrics = {}
    for state in window_counts:
        window_composition = window_counts[state] / total_cells_in_window
        state_distribution = window_counts[state] / state_total_counts[state]
        state_metrics[state] = (window_composition, state_distribution)
    # Sort primarily by state_distribution, then by window_composition
    sorted_states = sorted(state_metrics.items(),
                         key=lambda x: (x[1][1], x[1][0]),
                         reverse=True)
    return sorted_states[:k]

def cluster_regulations(dnet, regulations, method='ward'):
    """
    Perform hierarchical clustering on regulation data.
    """
    from scipy.cluster.hierarchy import linkage, dendrogram
    
    # Only cluster if we have multiple rows
    if dnet.shape[0] <= 1:
        return dnet, regulations, None
        
    # Compute linkage matrix
    row_linkage = linkage(dnet, method=method)
    
    # Get the order of rows after clustering
    row_order = dendrogram(row_linkage, no_plot=True)['leaves']
    
    # Reorder the data and regulation labels
    dnet_clustered = dnet[row_order]
    regulations_clustered = [regulations[i] for i in row_order]
    
    return dnet_clustered, regulations_clustered, row_linkage

def cluster_regulations_by_extremum(dnet, regulations, use_max=True):
    """
    Sort regulations based on when their extremum occurs in pseudotime.
    """
    # Only sort if we have multiple rows
    if dnet.shape[0] <= 1:
        return dnet, regulations, None
    
    # Find position of extremum for each row
    extremum_positions = []
    for row in dnet:
        abs_row = np.abs(row)
        if use_max:
            extremum_idx = np.argmax(abs_row)
        else:
            extremum_idx = np.argmin(abs_row)
        extremum_positions.append(extremum_idx)
    
    # Get sorting order based on extremum positions
    row_order = np.argsort(extremum_positions)
    
    # Reorder the data and regulation labels
    dnet_sorted = dnet[row_order]
    regulations_sorted = [regulations[i] for i in row_order]
    
    return dnet_sorted, regulations_sorted, None

def create_tf_target_pairs(dictys_dynamic_object, tf_pairs_df):
    """
    Create a list of TF-Target tuples from combinatorial control data,
    checking presence of both TFs and targets in the dynamic object.
    """
    tf_target_pairs = []

    for _, row in tf_pairs_df.iterrows():
        # Get TF pair and targets
        tf1, tf2 = eval(row["TF"])
        common_targets = eval(row["common"])

        # Check if TFs are present in dynamic object
        present_tfs = check_tf_presence(dictys_dynamic_object, [tf1, tf2])

        # Check if targets are present in dynamic object
        present_targets = check_gene_presence(dictys_dynamic_object, common_targets)

        # Create pairs for present TFs and targets
        for tf in present_tfs:
            for target in present_targets:
                tf_target_pairs.append((tf, target))

    # Remove any duplicates
    tf_target_pairs = list(set(tf_target_pairs))

    print(f"Created {len(tf_target_pairs)} unique TF-target pairs")
    return tf_target_pairs

def compute_chars(
    self,
    start: int,
    stop: int,
    num: int = 100,
    dist: float = 1.5,
    mode: str = "regulation",
    sparsity: float = 0.01,
) -> pd.DataFrame:
    """
    Compute curve characteristics for one branch.
    """
    pts, fsmooth = self.linspace(start, stop, num, dist)
    if mode == "regulation":
        # Log number of targets
        stat1_net = fsmooth(stat.net(self))
        stat1_netbin = stat.fbinarize(stat1_net, sparsity=sparsity)
        stat1_y = stat.flnneighbor(stat1_netbin)
    elif mode == "expression":
        stat1_y = fsmooth(stat.lcpm(self, cut=0))
    else:
        raise ValueError(f"Unknown mode {mode}.")
    # Pseudo time
    stat1_x = stat.pseudotime(self, pts)
    tmp_y = stat1_y.compute(pts)
    tmp_x = stat1_x.compute(pts)
    dy = pd.DataFrame(tmp_y, index=stat1_y.names[0])
    dx = pd.Series(tmp_x[0]) #first gene's pseudotime is taken as all genes have the same pseudotime
    return dy, dx


##################################### Plotting ############################################

def fig_regulation_heatmap_clustered(
    network: dictys.net.dynamic_network,
    start: int,
    stop: int,
    regulations: list[Tuple[str, str]],
    num: int = 100,
    dist: float = 1.5,
    ax: Optional[matplotlib.axes.Axes] = None,
    cmap: Union[str, matplotlib.cm.ScalarMappable] = "coolwarm",
    figsize: Tuple[float, float] = (2, 0.15),  # Reduced row height from 0.22 to 0.15
    vmax: Optional[float] = None,
    cluster_rows: bool = True,
    use_max: bool = True  # New parameter
) -> Tuple[matplotlib.pyplot.Figure, matplotlib.axes.Axes, matplotlib.cm.ScalarMappable]:
    """
    Draws clustered pseudo-time dependent heatmap of regulation strengths.
    """
    #ipdb.set_trace()
    # Get dynamic network edge strength
    pts, fsmooth = network.linspace(start, stop, num, dist)
    stat1_net = fsmooth(stat.net(network))
    stat1_x = stat.pseudotime(network, pts)
    tmp = stat1_x.compute(pts)[0]
    dx = pd.Series(tmp)
    
    # Test regulation existence and extract regulations
    tdict = [dict(zip(x, range(len(x)))) for x in stat1_net.names]
    t1 = [[x[y] for x in regulations if x[y] not in tdict[y]] for y in range(2)]
    if len(t1[0]) > 0:
        raise ValueError("Regulator gene(s) {} not found in network.".format("/".join(t1[0])))
    if len(t1[1]) > 0:
        raise ValueError("Target gene(s) {} not found in network.".format("/".join(t1[1])))
    
    # Extract regulations to draw
    dnet = stat1_net.compute(pts)
    t1 = np.array([[tdict[0][x[0]], tdict[1][x[1]]] for x in regulations]).T
    dnet = dnet[t1[0], t1[1]]
    
    # Perform clustering on rows if requested
    if cluster_rows:
        dnet, regulations, _ = cluster_regulations_by_extremum(dnet, regulations, use_max=use_max)
    
    # Create figure and axes
    if ax is None:
        figsize = (figsize[0], figsize[1] * dnet.shape[0])
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        if figsize is not None:
            raise ValueError("figsize should not be set if ax is set.")
        fig = ax.get_figure()
        figsize = fig.get_size_inches()
    
    aspect = (figsize[1] / dnet.shape[0]) / (figsize[0] / dnet.shape[1])
    
    # Determine and apply colormap
    if isinstance(cmap, str):
        if vmax is None:
            vmax = np.quantile(np.abs(dnet).ravel(), 0.95)
        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax), 
            cmap=cmap
        )
    elif vmax is not None:
        raise ValueError("vmax should not be set if cmap is a matplotlib.cm.ScalarMappable.")
    
    if hasattr(cmap, "to_rgba"):
        im = ax.imshow(cmap.to_rgba(dnet), aspect=aspect, interpolation='none')
    else:
        im = ax.imshow(dnet, aspect=aspect, interpolation='none', cmap=cmap)
        plt.colorbar(im, label="Regulation strength")
    
    # Set pseudotime labels as x axis labels
    ax.set_xlabel("Pseudotime")
    num_ticks = 10
    tick_positions = np.linspace(0, dnet.shape[1] - 1, num_ticks, dtype=int)
    tick_labels = dx.iloc[tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{x:.6f}" for x in tick_labels], rotation=45, ha="right")
    
    # Set regulation pair labels
    ax.set_yticks(list(range(len(regulations))))
    ax.set_yticklabels(["-".join(x) for x in regulations])
    
    # Add grid lines to separate rows
    ax.set_yticks(np.arange(dnet.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)
    
    return fig, ax, cmap

def fig_regulation_heatmap(
    network: dictys.net.dynamic_network,
    start: int,
    stop: int,
    regulations: list[Tuple[str, str]],
    num: int = 100,
    dist: float = 1.5,
    ax: Optional[matplotlib.axes.Axes] = None,
    cmap: Union[str, matplotlib.cm.ScalarMappable] = "coolwarm",
    figsize: Tuple[float, float] = (2, 0.15),
    vmax: Optional[float] = None
) -> Tuple[matplotlib.pyplot.Figure, matplotlib.axes.Axes, matplotlib.cm.ScalarMappable]:
    """
    Draws pseudo-time dependent heatmap of regulation strengths without clustering.
    Maintains the original order of regulation pairs as provided in the input list.
    """
    # Get dynamic network edge strength
    pts, fsmooth = network.linspace(start, stop, num, dist)
    stat1_net = fsmooth(stat.net(network))
    stat1_x = stat.pseudotime(network, pts)
    tmp = stat1_x.compute(pts)[0]
    dx = pd.Series(tmp)
    
    # Test regulation existence and extract regulations
    tdict = [dict(zip(x, range(len(x)))) for x in stat1_net.names]
    t1 = [[x[y] for x in regulations if x[y] not in tdict[y]] for y in range(2)]
    if len(t1[0]) > 0:
        raise ValueError("Regulator gene(s) {} not found in network.".format("/".join(t1[0])))
    if len(t1[1]) > 0:
        raise ValueError("Target gene(s) {} not found in network.".format("/".join(t1[1])))
    
    # Extract regulations to draw
    dnet = stat1_net.compute(pts)
    t1 = np.array([[tdict[0][x[0]], tdict[1][x[1]]] for x in regulations]).T
    dnet = dnet[t1[0], t1[1]]
    
    # Create figure and axes
    if ax is None:
        figsize = (figsize[0], figsize[1] * dnet.shape[0])
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        if figsize is not None:
            raise ValueError("figsize should not be set if ax is set.")
        fig = ax.get_figure()
        figsize = fig.get_size_inches()
    
    aspect = (figsize[1] / dnet.shape[0]) / (figsize[0] / dnet.shape[1])
    
    # Determine and apply colormap
    if isinstance(cmap, str):
        if vmax is None:
            vmax = np.quantile(np.abs(dnet).ravel(), 0.95)
        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax), 
            cmap=cmap
        )
    elif vmax is not None:
        raise ValueError("vmax should not be set if cmap is a matplotlib.cm.ScalarMappable.")
    
    if hasattr(cmap, "to_rgba"):
        im = ax.imshow(cmap.to_rgba(dnet), aspect=aspect, interpolation='none')
    else:
        im = ax.imshow(dnet, aspect=aspect, interpolation='none', cmap=cmap)
        plt.colorbar(im, label="Regulation strength")
    
    # Set pseudotime labels as x axis labels
    ax.set_xlabel("Pseudotime")
    num_ticks = 10
    tick_positions = np.linspace(0, dnet.shape[1] - 1, num_ticks, dtype=int)
    tick_labels = dx.iloc[tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{x:.6f}" for x in tick_labels], rotation=45, ha="right")
    
    # Set regulation pair labels
    ax.set_yticks(list(range(len(regulations))))
    ax.set_yticklabels(["-".join(x) for x in regulations])
    
    # Add grid lines to separate rows
    ax.set_yticks(np.arange(dnet.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)
    
    return fig, ax, cmap

def fig_expression_gradient_heatmap(
    network: dictys.net.dynamic_network,
    start: int,
    stop: int,
    genes_or_regulations: Union[list[str], list[Tuple[str, str]]],
    num: int = 100,
    dist: float = 1.5,
    ax: Optional[matplotlib.axes.Axes] = None,
    cmap: Union[str, matplotlib.cm.ScalarMappable] = "coolwarm",
    figsize: Tuple[float, float] = (2, 0.15)
) -> Tuple[matplotlib.pyplot.Figure, matplotlib.axes.Axes, matplotlib.cm.ScalarMappable]:
    """
    Draws pseudo-time dependent heatmap of expression gradients.
    
    Args:
        network: Dynamic network object
        start, stop: Start and stop points in pseudotime
        genes_or_regulations: Either a list of gene names or a list of (tf, target) tuples
        num: Number of points to evaluate
        dist: Distance parameter for smoothing
        ax: Optional matplotlib axes
        cmap: Colormap for heatmap
        figsize: Base figure size (width, height per row)
    """
    # Get expression data
    dy, dx = compute_chars(network, start, stop, num, dist, mode='expression')
    
    # Determine if input is gene list or regulation list
    if isinstance(genes_or_regulations[0], tuple):
        # Extract target genes from regulations
        target_genes = [target for _, target in genes_or_regulations]
        # Remove duplicates while preserving order
        target_genes = list(dict.fromkeys(target_genes))
    else:
        # Use gene list directly
        target_genes = list(dict.fromkeys(genes_or_regulations))
    
    # Calculate gradients for target genes
    gradients = np.vstack([
        np.gradient(dy.loc[gene].values, dx.values) 
        for gene in target_genes
    ])
    
    # Create figure and axes
    if ax is None:
        figsize = (figsize[0], figsize[1] * len(target_genes))
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        if figsize is not None:
            raise ValueError("figsize should not be set if ax is set.")
        fig = ax.get_figure()
        figsize = fig.get_size_inches()
    
    aspect = (figsize[1] / len(target_genes)) / (figsize[0] / gradients.shape[1])
    
    # Determine and apply colormap
    if isinstance(cmap, str):
        vmax = np.quantile(np.abs(gradients).ravel(), 0.95)
        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax), 
            cmap=cmap
        )
    
    if hasattr(cmap, "to_rgba"):
        im = ax.imshow(cmap.to_rgba(gradients), aspect=aspect, interpolation='none')
    else:
        im = ax.imshow(gradients, aspect=aspect, interpolation='none', cmap=cmap)
        plt.colorbar(im, label="Expression gradient (Δ Log CPM/Δ Pseudotime)")
    
    # Set pseudotime labels as x axis labels
    ax.set_xlabel("Pseudotime")
    num_ticks = 10
    tick_positions = np.linspace(0, gradients.shape[1] - 1, num_ticks, dtype=int)
    tick_labels = dx.iloc[tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{x:.6f}" for x in tick_labels], rotation=45, ha="right")
    
    # Set target gene labels
    ax.set_yticks(list(range(len(target_genes))))
    ax.set_yticklabels(target_genes)
    
    # Add grid lines to separate rows
    ax.set_yticks(np.arange(len(target_genes) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)
    
    return fig, ax, cmap 
 

# if __name__ == "__main__":
#     data_file = '/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added/output/dynamic.h5'
#     output_folder = '/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added/output'
#     dictys_dynamic_object = load_dynamic_object(data_file)
#     print('loaded dynamic object')
#     traj_object = dictys_dynamic_object.point['s'].p
#     debug_call = traj_linspace(traj_object,1,2,100)
#     print(debug_call)
