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

######################### Data retrieval #########################

def load_dynamic_object(dynamic_object_path):
    """
    Load the dynamic object from the given path
    """
    dynamic_object = dictys.net.dynamic_network.from_file(dynamic_object_path)
    return dynamic_object


def get_tf_indices(dictys_dynamic_object, tf_list):
    """
    Get the indices of transcription factors from a list, if present in ndict and nids[0].
    """
    gene_hashmap = dictys_dynamic_object.ndict
    tf_mappings_to_gene_hashmap = dictys_dynamic_object.nids[0]
    tf_indices = []
    gene_indices = []
    for gene in tf_list:
        # Check if the gene is in the gene_hashmap
        if gene in gene_hashmap:
            gene_index = gene_hashmap[gene]  # Get the index in gene_hashmap
            # Check if the gene index is present in tf_mappings_to_gene_hashmap
            match = np.where(tf_mappings_to_gene_hashmap == gene_index)[0]
            if match.size > 0:  # If a match is found
                tf_indices.append(int(match[0]))  # Append the position of the match
                gene_indices.append(int(gene_index))  # Also append the gene index
    return tf_indices, gene_indices


def get_target_gene_indices(dictys_dynamic_object, target_list):
    """
    Get the indices of target genes from a list, if present in ndict and nids[1].
    """
    gene_hashmap = dictys_dynamic_object.ndict
    target_indices = []
    for gene in target_list:
        if gene in gene_hashmap:
            target_indices.append(gene_hashmap[gene])
    return target_indices

def get_all_target_names(dictys_dynamic_object):
    """
    Get all possible target gene names from the dynamic object's ndict
    """
    # Get all gene names from ndict
    target_names = list(dictys_dynamic_object.ndict.keys())
    target_names.sort()  # Sort for consistency
    return target_names

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

def get_grn_weights_across_windows(dictys_dynamic_object, tf_indices=None, gene_indices=None, window_indices=None):
    """
    Get the weights matrix for both specified TFs and TFs regulating specified targets
    """
    # Get the weights array
    weights = dictys_dynamic_object.prop["es"]["w_n"]  
    # Convert indices to lists if they're not already
    tf_indices = list(tf_indices) if tf_indices is not None else []
    gene_indices = list(gene_indices) if gene_indices is not None else []
    # Get all TF indices that regulate the specified target genes
    if gene_indices:
        # weights shape is (n_tfs, n_genes, n_windows)
        tf_mask = np.any(np.any(weights[:, gene_indices, :] != 0, axis=2), axis=1) #shape (n_tfs,)
        regulating_tf_indices = np.where(tf_mask)[0]
        # Combine with specified TF indices, removing duplicates while preserving order
        all_tf_indices = list(dict.fromkeys(tf_indices + list(regulating_tf_indices)))
    else:
        all_tf_indices = tf_indices

    # Get all gene indices that are regulated by the specified TFs
    if tf_indices:
        gene_mask = np.any(np.any(weights[tf_indices, :, :] != 0, axis=2), axis=0) #shape (n_genes,)
        regulated_gene_indices = np.where(gene_mask)[0]
        
        # Combine with specified gene indices, removing duplicates while preserving order
        all_gene_indices = list(dict.fromkeys(gene_indices + list(regulated_gene_indices)))
    else:
        all_gene_indices = gene_indices

    # Extract the combined weights matrix using the computed indices
    combined_weights = weights[np.ix_(all_tf_indices, all_gene_indices, window_indices)]
    return combined_weights, all_tf_indices, all_gene_indices

def get_weights_for_tf_target_pairs(
    dictys_dynamic_object, tf_indices, target_indices, window_indices
):
    """
    Get the weights for specific TF-target pairs over specific windows.
    """
    # Access the 3-D array of weights using the correct dimensions
    weights_of_tf_target = dictys_dynamic_object.prop["es"]["w_n"][
        np.ix_(tf_indices, target_indices, window_indices)
    ]
    return weights_of_tf_target

def get_indirect_weights_across_windows(
    dictys_dynamic_object, tf_indices, window_indices
):
    """
    Get the indirect weights of TFs over specific windows for x-axis in plots
    """
    array_shape = dictys_dynamic_object.prop["es"]["w_in"].shape
    n_targets = array_shape[1]

    indirect_weights_of_tf_target = dictys_dynamic_object.prop["es"]["w_in"][
        np.ix_(tf_indices, range(n_targets), window_indices)
    ]
    return indirect_weights_of_tf_target


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

def check_tf_presence(dictys_dynamic_object, tf_list):
    """
    Check if the TFs are present in the dynamic network object.
    """
    gene_hashmap = dictys_dynamic_object.ndict
    tf_mappings_to_gene_hashmap = dictys_dynamic_object.nids[0]

    tfs_present_in_dynamic_object = []
    for tf in tf_list:
        # Check if the TF is in the gene_hashmap
        if tf in gene_hashmap:
            gene_index = gene_hashmap[tf]
            # Check if the gene index is present in tf_mappings_to_gene_hashmap
            if np.any(tf_mappings_to_gene_hashmap == gene_index):
                tfs_present_in_dynamic_object.append(tf)

    return tfs_present_in_dynamic_object

def check_gene_presence(dictys_dynamic_object, gene_list):
    """
    Check if the genes are present as targets in the dynamic network object.
    """
    gene_hashmap = dictys_dynamic_object.ndict
    target_mappings_to_hashmap = dictys_dynamic_object.nids[1]

    target_genes_present_in_dynamic_object = []
    for gene in gene_list:
        # Check if the gene is in the gene_hashmap
        if gene in gene_hashmap:
            gene_index = gene_hashmap[gene]
            # Check if the gene index is present in target_mappings_to_hashmap
            if np.any(target_mappings_to_hashmap == gene_index):
                target_genes_present_in_dynamic_object.append(gene)

    return target_genes_present_in_dynamic_object

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
    tmp_pt_0 = pts[[1]]
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
 

