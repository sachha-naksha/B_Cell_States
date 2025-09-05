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
from dictys.utils.numpy import ArrayLike, NDArray
from joblib import Memory
from scipy import stats
from scipy.stats import hypergeom
from tqdm import tqdm

######################### Indices retrieval #########################

def get_tf_indices(dictys_dynamic_object, tf_list):
    """
    Get the indices of transcription factors from a list, if present in ndict and nids[0].
    """
    gene_hashmap = dictys_dynamic_object.ndict
    tf_mappings_to_gene_hashmap = dictys_dynamic_object.nids[0]
    tf_indices = []
    tf_gene_indices = []
    missing_tfs = []
    for gene in tf_list:
        # Check if the gene is in the gene_hashmap
        if gene in gene_hashmap:
            gene_index = gene_hashmap[gene]  # Get the index in gene_hashmap
            # Check if the gene index is present in tf_mappings_to_gene_hashmap
            match = np.where(tf_mappings_to_gene_hashmap == gene_index)[0]
            if match.size > 0:  # If a match is found
                tf_indices.append(int(match[0]))  # Append the position of the match
                tf_gene_indices.append(int(gene_index))  # Also append the gene index
            else:
                missing_tfs.append(gene)  # Gene exists but not as a TF
        else:
            missing_tfs.append(gene)  # Gene not found at all
    return tf_indices, tf_gene_indices, missing_tfs


def get_gene_indices(dictys_dynamic_object, gene_list):
    """
    Get the indices of target genes from a list, if present in ndict and nids[1].
    """
    gene_hashmap = dictys_dynamic_object.ndict
    gene_indices = []
    for gene in gene_list:
        if gene in gene_hashmap:
            gene_indices.append(gene_hashmap[gene])
    return gene_indices


def check_if_gene_in_ndict(dictys_dynamic_object, gene_name, return_index=False):
    """
    Check if a gene is in the ndict of the dynamic object.
    """
    # Input validation
    if not hasattr(dictys_dynamic_object, "ndict"):
        raise AttributeError("Dynamic object does not have ndict attribute")
    # Handle single gene case
    if isinstance(gene_name, str):
        is_present = gene_name in dictys_dynamic_object.ndict
        if return_index:
            return dictys_dynamic_object.ndict.get(gene_name, None)
        return is_present
    # Handle list of genes case
    elif isinstance(gene_name, (list, tuple, set)):
        results = {
            "present": [],
            "missing": [],
            "indices": {} if return_index else None,
        }
        for gene in gene_name:
            if gene in dictys_dynamic_object.ndict:
                results["present"].append(gene)
                if return_index:
                    results["indices"][gene] = dictys_dynamic_object.ndict[gene]
            else:
                results["missing"].append(gene)
        # Add summary statistics
        results["stats"] = {
            "total_genes": len(gene_name),
            "found": len(results["present"]),
            "missing": len(results["missing"]),
            "percent_found": (len(results["present"]) / len(gene_name) * 100),
        }
        return results
    else:
        raise TypeError("gene_name must be a string or a list-like object of strings")

##################################### Window labels ############################################

def get_state_labels_in_window(dictys_dynamic_object, cell_labels):
    """
    Creates a mapping of window indices to their constituent cells' labels
    """
    # get cell assignment matrix from dictys_dynamic_object.prop['sc']['w']
    cell_assignment_matrix = dictys_dynamic_object.prop["sc"]["w"]
    state_labels_in_window = {}
    for window_idx in range(cell_assignment_matrix.shape[0]):
        indices_of_cells_present_in_window = np.where(
            cell_assignment_matrix[window_idx] == 1
        )[0]
        state_labels_in_window[window_idx] = [
            cell_labels[idx] for idx in indices_of_cells_present_in_window
        ]
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
    sorted_states = sorted(
        state_metrics.items(), key=lambda x: (x[1][1], x[1][0]), reverse=True
    )
    return sorted_states[:k]

def window_labels_to_count_df(window_labels_dict):
    """
    Converts a dictionary of window indices to cell labels into a DataFrame
    with counts of each label per window.
    """
    from collections import Counter

    import pandas as pd

    # Get all unique labels
    all_labels = set()
    for labels in window_labels_dict.values():
        all_labels.update(labels)

    # Sort labels for consistency
    all_labels = sorted(all_labels)

    # Get all window indices
    window_indices = sorted(window_labels_dict.keys())

    # Initialize DataFrame with zeros
    count_df = pd.DataFrame(0, index=all_labels, columns=window_indices)

    # Fill in the counts for each window
    for window_idx, labels in window_labels_dict.items():
        # Count occurrences of each label in this window
        label_counts = Counter(labels)

        # Update the DataFrame
        for label, count in label_counts.items():
            count_df.loc[label, window_idx] = count

    return count_df

##################################### Plotting ############################################


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
    vmax: Optional[float] = None,
) -> Tuple[
    matplotlib.pyplot.Figure, matplotlib.axes.Axes, matplotlib.cm.ScalarMappable
]:
    """
    Draws pseudo-time dependent heatmap of regulation strengths without clustering.
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
        raise ValueError(
            "Regulator gene(s) {} not found in network.".format("/".join(t1[0]))
        )
    if len(t1[1]) > 0:
        raise ValueError(
            "Target gene(s) {} not found in network.".format("/".join(t1[1]))
        )
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
            norm=matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax), cmap=cmap
        )
    elif vmax is not None:
        raise ValueError(
            "vmax should not be set if cmap is a matplotlib.cm.ScalarMappable."
        )
    if hasattr(cmap, "to_rgba"):
        im = ax.imshow(cmap.to_rgba(dnet), aspect=aspect, interpolation="none")
    else:
        im = ax.imshow(dnet, aspect=aspect, interpolation="none", cmap=cmap)
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
    return fig, ax, dnet


def cluster_heatmap(
    d,
    optimal_ordering=True,
    method="ward",
    metric="euclidean",
    dshow=None,
    fig=None,
    cmap="coolwarm",
    aspect=0.1,
    figscale=0.02,
    dtop=0.3,
    dright=0,
    wcolorbar=0.03,
    wedge=0.03,
    xselect=None,
    yselect=None,
    xtick=False,
    ytick=True,
    vmin=None,
    vmax=None,
    inverty=True,
):
    """
    Draw 2-D hierachical clustering of pandas.DataFrame, with optional hierachical clustering on both axes.
    X/Y axis of figure corresponds to columns/rows of the dataframe.
    d:		Pandas.DataFrame 2D data with index & column names for clustering.
    optimal_ordering: passed to scipy.cluster.hierarchy.dendrogram
    method: Method of hierarchical clustering, passed to scipy.cluster.hierarchy.linkage.
                    Accepts single strs or a tuple of two strings for different method options for x and y.
    metric:	Metric to compute pariwise distance for clustering, passed to scipy.spatial.distance.pdist.
                    Accepts single strs or a tuple of two strings for different metric options for x and y.
    dshow:	Pandas.DataFrame 2D data with index & column names to draw heatmap. Defaults to d.
    fig:	Figure to plot on.
    cmap:	Colormap
    aspect:	Aspect ratio
    figscale:	Scale of figure compared to font.
    dtop,
    dright:	Top and right dendrogram size. Value from 0 to 1 values as proportion.
                    If 0, do not cluster on given axis.
    wcolorbar: Width of colorbar. Value from 0 to 1 values as proportion.
    wedge:	Width of edges and between colorbar and main figure.
                    Value from 0 to 1 values as proportion.
    xselect,
    yselect:np.array(bool) of coordinates to draw. Current only selected coordinates are used for clustering.
    xtick,
    ytick:	Whether to show ticks.
    vmin,
    vmax:	Minimum/maximum values of heatmap.
    inverty:Whether to invert direction of y.
    Return:
    figure:	plt.Figure drawn on
    x:		column IDs included
    y:		index IDs included.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.cluster.hierarchy import dendrogram, linkage

    assert isinstance(xtick, bool) or (
        isinstance(xtick, list) and len(xtick) == d.shape[1]
    )
    assert isinstance(ytick, bool) or (
        isinstance(ytick, list) and len(ytick) == d.shape[0]
    )
    if isinstance(method, str):
        method = [method, method]
    if len(method) != 2:
        raise ValueError(
            'Parameter "method" must have size 2 for x and y respectively.'
        )
    if isinstance(metric, str):
        metric = [metric, metric]
    if metric is not None and len(metric) != 2:
        raise ValueError(
            'Parameter "metric" must have size 2 for x and y respectively.'
        )
    if metric is None:
        assert d.ndim == 2 and d.shape[0] == d.shape[1]
        assert (d.index == d.columns).all()
        assert method[0] == method[1]
        if xselect is not None:
            assert yselect is not None
            assert (xselect == yselect).all()
        else:
            assert yselect is None
    if dshow is None:
        dshow = d
    assert (
        dshow.shape == d.shape
        and (dshow.index == d.index).all()
        and (dshow.columns == d.columns).all()
    )
    xt0 = d.columns if isinstance(xtick, bool) else xtick
    yt0 = d.index if isinstance(ytick, bool) else ytick
    # Genes to highlight
    d2 = d.copy()
    if xselect is not None:
        d2 = d2.loc[:, xselect]
        dshow = dshow.loc[:, xselect]
        xt0 = [xt0[x] for x in np.nonzero(xselect)[0]]
    if yselect is not None:
        d2 = d2.loc[yselect]
        dshow = dshow.loc[yselect]
        yt0 = [yt0[x] for x in np.nonzero(yselect)[0]]

    wtop = dtop / (1 + d2.shape[0] / 8)
    wright = dright / (1 + d2.shape[1] * aspect / 8)
    iscolorbar = wcolorbar > 0
    t1 = np.array(d2.T.shape)
    t1 = t1 * figscale
    t1[1] /= aspect
    t1[1] /= 1 - wedge * 2 - wtop
    t1[0] /= 1 - wedge * (2 + iscolorbar) - wright - wcolorbar
    if fig is None:
        fig = plt.figure(figsize=t1)
    d3 = dshow.copy()
    if metric is not None:
        # Right dendrogram
        if dright > 0:
            ax1 = fig.add_axes(
                [
                    1 - wedge * (1 + iscolorbar) - wright - wcolorbar,
                    wedge,
                    wright,
                    1 - 2 * wedge - wtop,
                ]
            )
            tl1 = linkage(
                d2,
                method=method[1],
                metric=metric[1],
                optimal_ordering=optimal_ordering,
            )
            td1 = dendrogram(tl1, orientation="right")
            ax1.set_xticks([])
            ax1.set_yticks([])
            d3 = d3.iloc[td1["leaves"], :]
            yt0 = [yt0[x] for x in td1["leaves"]]
        else:
            ax1 = None
        # Top dendrogram
        if dtop > 0:
            ax2 = fig.add_axes(
                [
                    wedge,
                    1 - wedge - wtop,
                    1 - wedge * (2 + iscolorbar) - wright - wcolorbar,
                    wtop,
                ]
            )
            tl2 = linkage(
                d2.T,
                method=method[0],
                metric=metric[0],
                optimal_ordering=optimal_ordering,
            )
            td2 = dendrogram(tl2)
            ax2.set_xticks([])
            ax2.set_yticks([])
            d3 = d3.iloc[:, td2["leaves"]]
            xt0 = [xt0[x] for x in td2["leaves"]]
        else:
            ax2 = None
    else:
        if dright > 0 or dtop > 0:
            from scipy.spatial.distance import squareform

            tl1 = linkage(
                squareform(d2), method=method[0], optimal_ordering=optimal_ordering
            )
            # Right dendrogram
            if dright > 0:
                ax1 = fig.add_axes(
                    [
                        1 - wedge * (1 + iscolorbar) - wright - wcolorbar,
                        wedge,
                        wright,
                        1 - 2 * wedge - wtop,
                    ]
                )
                td1 = dendrogram(tl1, orientation="right")
                ax1.set_xticks([])
                ax1.set_yticks([])
            else:
                ax1 = None
                td1 = None
            # Top dendrogram
            if dtop > 0:
                ax2 = fig.add_axes(
                    [
                        wedge,
                        1 - wedge - wtop,
                        1 - wedge * (2 + iscolorbar) - wright - wcolorbar,
                        wtop,
                    ]
                )
                td2 = dendrogram(tl1)
                ax2.set_xticks([])
                ax2.set_yticks([])
            else:
                ax2 = None
                td2 = None
            td0 = td1["leaves"] if td1 is not None else td2["leaves"]
            d3 = d3.iloc[td0, :].iloc[:, td0]
            xt0, yt0 = [[y[x] for x in td0] for y in [xt0, yt0]]
    axmatrix = fig.add_axes(
        [
            wedge,
            wedge,
            1 - wedge * (2 + iscolorbar) - wright - wcolorbar,
            1 - 2 * wedge - wtop,
        ]
    )
    ka = {"aspect": 1 / aspect, "origin": "lower", "cmap": cmap}
    if vmin is not None:
        ka["vmin"] = vmin
    if vmax is not None:
        ka["vmax"] = vmax
    im = axmatrix.matshow(d3, **ka)
    if not isinstance(xtick, bool) or xtick:
        t1 = list(zip(range(d3.shape[1]), xt0))
        t1 = list(zip(*list(filter(lambda x: x[1] is not None, t1))))
        axmatrix.set_xticks(t1[0])
        axmatrix.set_xticklabels(t1[1], minor=False, rotation=90)
    else:
        axmatrix.set_xticks([])
    if not isinstance(ytick, bool) or ytick:
        t1 = list(zip(range(d3.shape[0]), yt0))
        t1 = list(zip(*list(filter(lambda x: x[1] is not None, t1))))
        axmatrix.set_yticks(t1[0])
        axmatrix.set_yticklabels(t1[1], minor=False)
    else:
        axmatrix.set_yticks([])
    axmatrix.tick_params(
        top=False,
        bottom=True,
        labeltop=False,
        labelbottom=True,
        left=True,
        labelleft=True,
        right=False,
        labelright=False,
    )
    if inverty:
        if ax1 is not None:
            ax1.set_ylim(ax1.get_ylim()[::-1])
        axmatrix.set_ylim(axmatrix.get_ylim()[::-1])
    if wcolorbar > 0:
        cax = fig.add_axes(
            [1 - wedge - wcolorbar, wedge, wcolorbar, 1 - 2 * wedge - wtop]
        )
        fig.colorbar(im, cax=cax)
    return fig, d3.columns, d3.index


def fig_expression_gradient_heatmap(
    network: dictys.net.dynamic_network,
    start: int,
    stop: int,
    genes_or_regulations: Union[list[str], list[Tuple[str, str]]],
    num: int = 100,
    dist: float = 1.5,
    ax: Optional[matplotlib.axes.Axes] = None,
    cmap: Union[str, matplotlib.cm.ScalarMappable] = "coolwarm",
    figsize: Tuple[float, float] = (2, 0.15),
) -> Tuple[
    matplotlib.pyplot.Figure, matplotlib.axes.Axes, matplotlib.cm.ScalarMappable
]:
    """
    Draws pseudo-time dependent heatmap of expression gradients.
    """
    # Get expression data
    pts, fsmooth = network.linspace(start, stop, num, dist)
    stat1_y = fsmooth(stat.lcpm(network, cut=0))
    stat1_x = stat.pseudotime(network, pts)
    dy = pd.DataFrame(stat1_y.compute(pts), index=stat1_y.names[0])
    dx = pd.Series(
        stat1_x.compute(pts)[0]
    )  # gene1's pseudotime is used as all genes have the same pseudotime
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
    gradients = np.vstack(
        [np.gradient(dy.loc[gene].values, dx.values) for gene in target_genes]
    )
    # Create figure and axes
    if ax is None:
        figsize = (figsize[0], figsize[1] * len(target_genes))
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
        figsize = fig.get_size_inches()
    aspect = (figsize[1] / len(target_genes)) / (figsize[0] / gradients.shape[1])
    # Determine and apply colormap
    if isinstance(cmap, str):
        vmax = np.quantile(np.abs(gradients).ravel(), 0.95)
        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax), cmap=cmap
        )
    if hasattr(cmap, "to_rgba"):
        im = ax.imshow(cmap.to_rgba(gradients), aspect=aspect, interpolation="none")
    else:
        im = ax.imshow(gradients, aspect=aspect, interpolation="none", cmap=cmap)
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


def fig_expression_linear_heatmap(
    network: dictys.net.dynamic_network,
    start: int,
    stop: int,
    genes_or_regulations: Union[list[str], list[Tuple[str, str]]],
    num: int = 100,
    dist: float = 1.5,
    ax: Optional[matplotlib.axes.Axes] = None,
    cmap: Union[str, matplotlib.cm.ScalarMappable] = "coolwarm",
    figsize: Tuple[float, float] = (2, 0.15),
) -> Tuple[
    matplotlib.pyplot.Figure, matplotlib.axes.Axes, matplotlib.cm.ScalarMappable
]:
    """
    Draws pseudo-time dependent heatmap of linear expression values.
    """
    # Get expression data
    pts, fsmooth = network.linspace(start, stop, num, dist)
    stat1_y = fsmooth(stat.lcpm(network, cut=0))
    stat1_x = stat.pseudotime(network, pts)
    # Get log2 expression values and convert to linear
    dy = pd.DataFrame(stat1_y.compute(pts), index=stat1_y.names[0])
    dx = pd.Series(stat1_x.compute(pts)[0])
    dy_linear = dy.apply(lambda x: 2**x - 1)
    # Get target genes
    if isinstance(genes_or_regulations[0], tuple):
        target_genes = [target for _, target in genes_or_regulations]
        target_genes = list(dict.fromkeys(target_genes))
    else:
        target_genes = list(dict.fromkeys(genes_or_regulations))
    # Stack expression values
    expression_matrix = np.vstack([dy_linear.loc[gene].values for gene in target_genes])
    # Create figure and axes
    if ax is None:
        figsize = (figsize[0], figsize[1] * len(target_genes))
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
        figsize = fig.get_size_inches()
    aspect = (figsize[1] / len(target_genes)) / (
        figsize[0] / expression_matrix.shape[1]
    )
    # Create heatmap
    if isinstance(cmap, str):
        vmax = np.quantile(expression_matrix.ravel(), 0.95)
        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=0, vmax=vmax), cmap=cmap
        )
    if hasattr(cmap, "to_rgba"):
        im = ax.imshow(
            cmap.to_rgba(expression_matrix), aspect=aspect, interpolation="none"
        )
    else:
        im = ax.imshow(
            expression_matrix, aspect=aspect, interpolation="none", cmap=cmap
        )
        plt.colorbar(im, label="Expression (CPM)")
    # Set pseudotime labels with rounded values
    ax.set_xlabel("Pseudotime")
    num_ticks = 10
    tick_positions = np.linspace(
        0, expression_matrix.shape[1] - 1, num_ticks, dtype=int
    )
    tick_labels = dx.iloc[tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [f"{x:.3f}" for x in tick_labels], rotation=45, ha="right"
    )  # Round to 1 decimal place
    # Set gene labels
    ax.set_yticks(list(range(len(target_genes))))
    ax.set_yticklabels(target_genes)
    # Add grid lines
    ax.set_yticks(np.arange(len(target_genes) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)
    return fig, ax, cmap


def plot_force_heatmap(
    force_df: pd.DataFrame,
    dtime: pd.Series,
    regulations=None,
    tf_to_targets_dict=None,
    ax: Optional[matplotlib.axes.Axes] = None,
    cmap: Union[str, matplotlib.cm.ScalarMappable] = "coolwarm",
    figsize: Tuple[float, float] = (10, 4),
    vmax: Optional[float] = None,
) -> Tuple[matplotlib.pyplot.Figure, matplotlib.axes.Axes, np.ndarray]:
    """
    Draws pseudo-time dependent heatmap of force values.
    """
    # Process input parameters to generate regulation pairs
    reg_pairs = []
    reg_labels = []
    # Case 1: Dictionary of TF -> targets provided
    if tf_to_targets_dict is not None:
        for tf, targets in tf_to_targets_dict.items():
            for target in targets:
                reg_pairs.append((tf, target))
                reg_labels.append(f"{tf}->{target}")
    # Case 2: List of regulation pairs or list of targets for a single TF
    elif regulations is not None:
        # Check if first item is a string (target) or tuple/list (regulation pair)
        if regulations and isinstance(regulations[0], str):
            # It's a list of targets for a single TF
            # Extract TF name from the calling context (not ideal but works for the notebook)
            for key, value in locals().items():
                if (
                    isinstance(value, dict)
                    and "PRDM1" in value
                    and value["PRDM1"] == regulations
                ):
                    tf = "PRDM1"  # Found the TF
                    break
            else:
                # If we can't determine the TF, use the first item in regulations as TF
                # and the rest as targets (this is a fallback and might not be correct)
                tf = regulations[0]
                regulations = regulations[1:]

            for target in regulations:
                reg_pairs.append((tf, target))
                reg_labels.append(f"{tf}->{target}")
        else:
            # It's a list of regulation pairs
            reg_pairs = regulations
            reg_labels = [f"{tf}->{target}" for tf, target in regulations]
    # If no regulations provided, use non-zero regulations from force_df
    if not reg_pairs:
        non_zero_mask = (force_df != 0).any(axis=1)
        force_df_filtered = force_df[non_zero_mask]
        reg_pairs = list(force_df_filtered.index)
        reg_labels = [f"{tf}->{target}" for tf, target in reg_pairs]
    # Extract force values for the specified regulations
    force_values = []
    for pair in reg_pairs:
        tf, target = pair
        try:
            force_values.append(force_df.loc[(tf, target)].values)
        except KeyError:
            raise ValueError(f"Regulation {tf}->{target} not found in force DataFrame")
    # Convert to numpy array
    dnet = np.array(force_values)
    # Create figure and axes
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    # Determine and apply colormap
    if isinstance(cmap, str):
        if vmax is None:
            vmax = np.quantile(np.abs(dnet).ravel(), 0.95)
        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax), cmap=cmap
        )
    elif vmax is not None:
        raise ValueError(
            "vmax should not be set if cmap is a matplotlib.cm.ScalarMappable."
        )
    if hasattr(cmap, "to_rgba"):
        im = ax.imshow(cmap.to_rgba(dnet), aspect="auto", interpolation="none")
    else:
        im = ax.imshow(dnet, aspect="auto", interpolation="none", cmap=cmap)
        plt.colorbar(im, label="Force")
    # Set pseudotime labels
    ax.set_xlabel("Pseudotime")
    num_ticks = 10
    tick_positions = np.linspace(0, dnet.shape[1] - 1, num_ticks, dtype=int)
    tick_labels = dtime.iloc[tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{x:.4f}" for x in tick_labels], rotation=45, ha="right")
    # Set regulation pair labels
    ax.set_yticks(list(range(len(reg_labels))))
    ax.set_yticklabels(reg_labels)
    # Add grid lines
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)
    plt.tight_layout()
    return fig, ax, dnet


def plot_expression_for_multiple_genes(
    targets_in_lf, lcpm_dcurve, dtime, ncols=3, figsize=(18, 15)
):
    """
    Plots expression curves for multiple target genes in a single figure.
    """
    # Calculate number of rows needed
    n_targets = len(targets_in_lf)
    nrows = math.ceil(n_targets / ncols)

    # Create figure and subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_targets > 1 else [axes]  # Handle case of single subplot

    # Loop through each target gene
    for i, gene in enumerate(targets_in_lf):
        ax = axes[i]

        # Check if gene exists in lcpm_dcurve
        if gene in lcpm_dcurve.index:
            # Plot expression curve
            line = ax.plot(dtime, lcpm_dcurve.loc[gene], linewidth=2, color="green")

            # Add label at the end of the line
            ax.text(
                dtime.iloc[-1],
                lcpm_dcurve.loc[gene].iloc[-1],
                f" {gene}",
                color="green",
                verticalalignment="center",
            )

            # Set title and labels
            ax.set_title(gene)
            ax.set_xlabel("Pseudotime")
            ax.set_ylabel("Log CPM")

            # Remove top and right spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        else:
            ax.text(
                0.5,
                0.5,
                f"{gene} not found",
                horizontalalignment="center",
                verticalalignment="center",
            )
            ax.axis("off")
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    # Adjust layout
    plt.tight_layout()
    plt.suptitle("Expression Curves for Target Genes", fontsize=16, y=1.02)
    return fig
