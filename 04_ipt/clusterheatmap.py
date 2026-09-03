import numpy as np
import pandas as pd

from cooltools.lib import numutils, runlength

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


DEFAULT_CONFIG = {"type": "scalar", "options": {}}


def colorlist_to_rgba(clist) -> np.ndarray:
    return np.array([mpl.colors.to_rgba(c) for c in clist]).reshape(1, -1, 4)


def get_track_height_ratios(
    layout: dict[str, list[str]],
    trackconfs: dict[str, dict],
) -> list[int]:
    height_ratios = []
    for block in layout.values():
        for name in block:
            conf = trackconfs.get(name, DEFAULT_CONFIG.copy())
            if "multivec" in conf:
                height_ratios.append(len(conf["multivec"]) * conf.get("height", 1))
            else:
                height_ratios.append(conf.get("height", 1))
    return height_ratios


def prepare_track(name, conf, bins, order, coarse_factor):
    # Make the data matrix (n_tracks, n_loci)
    if "multivec" in conf:
        X = np.concatenate(
            [[bins[col].values] for col in conf["multivec"]],
            axis=0
        )
        names = conf["multivec"]
    else:
        X = np.array([bins[name].values])
        names = [name]

    # Sort the tracks by the cluster labels and sub-orderings
    X = X[:, order]

    # from sklearn.preprocessing import scale
    # X = scale(X, axis=1)

    # Convert to imshow-compatible input and options
    imshow_kwargs = conf.get("options", {}).copy()
    match conf["type"]:
        case 'scalar':
            imshow_kwargs.setdefault('cmap', 'Reds')
            imshow_kwargs.setdefault('vmin', 0)
            data = numutils.coarsen(
                np.nanmean,
                X,
                {1: coarse_factor},
                trim_excess=True
            )
        case 'divergent':
            imshow_kwargs.setdefault('cmap', 'RdBu_r')
            if 'vmax' not in imshow_kwargs:
                vopt = np.nanpercentile(np.nanmax(np.abs(X)), 98)
                imshow_kwargs['vmin'] = -vopt
                imshow_kwargs['vmax'] = vopt
            data = numutils.coarsen(
                np.nanmean,
                X,
                {1: coarse_factor},
                trim_excess=True
            )
        case 'category' | 'categorical':
            if "color_dict" not in imshow_kwargs:
                categories = set(X.ravel().tolist())
                pal = sns.color_palette("tab20", n_colors=len(categories))
                color_dict = {cat: pal[i] for i, cat in enumerate(categories)}
            else:
                color_dict = imshow_kwargs.pop("color_dict")
            data = np.concatenate([
                colorlist_to_rgba([color_dict.get(x, "#000000") for x in row])
                for row in X
            ], axis=0)
        case 'colorlist':
            data = np.concatenate([
                colorlist_to_rgba(row) for row in X
            ], axis=0)
        case _:
            raise ValueError(
                f"Unknown track type for '{name}': '{conf['type']}'"
            )

    return names, data, imshow_kwargs


def clustermap(
    bins: pd.DataFrame,
    group_by: str,
    sort_by: str | list[str],
    layout: dict[str, list[str]],
    trackconfs: dict[str, dict],
    coarse_factor: int = 32,
    figsize: tuple[int] = (24, 20),
    block_gap: float = 0.5,
) -> mpl.figure.Figure:
    """
    Render tracks of binned genomic data.

    Data are grouped by custom labels and a sorted by a custom ordering.

    Parameters
    ----------
    bins : pd.DataFrame
        A DataFrame with all required binned tracks.

    group_by : str
        The column name to group the bins by.

    sort_by : str | list[str]
        The column name(s) to sort the bins in each group by.

    layout : dict[str, list[str]]
        A dictionary with the layout of the tracks by name. Divided into
        "blocks", which aren't currently meaningful. Use dummy block names.
        ``{block_name: [track_name, ...], ...}``.

    trackconfs : dict[str, dict]
        A dictionary with the configuration dictionary of each track.
        See notes for the structure of the configuration dictionary.
        ``{track_name: track_config, ...}``.

    coarse_factor : int
        The factor by which to coarsen scalar data.

    figsize : tuple[int]
        The figure size.

    Returns
    -------
    mpl.figure.Figure
        The rendered figure.

    Notes
    -----
    The track configuration dictionary has the following structure:

    - ``type`` : str
        The type of the track. One of ``'scalar'``, ``'divergent'``,
        ``'category'``, ``'categorical'``, or ``'colorlist'``.
    - ``multivec`` : list[str], optional
        The list of columns to concatenate for a multivec track.
        If not provided, the track will be a single vector track.
    - ``height`` : int, optional [default: 1]
        The height of the track.
    - ``options`` : dict
        A dictionary with additional options to pass to imshow/matshow.
        Use this to set the colormap, vmin, vmax, etc.
        For categorical/colorlist data, you can also pass a ``color_dict`` here.
    """
    labels = bins[group_by].values
    n_loci = labels.shape[0]

    # Sort bins by cluster label, then by user-specified fields
    if isinstance(sort_by, str):
        sort_by = [sort_by]
    sorting_arrays = [labels] + [bins[col].values for col in sort_by]
    order = np.lexsort(sorting_arrays[::-1])

    # Get the partition of the sorted labels (start of each block + the final end,
    # so the LAST cluster isn't cropped out by the xlim below). Callers should drop
    # any unassigned "dud" bins upstream rather than relying on cropping.
    partition = [run[0] for run in runlength.iterruns(labels[order])] + [n_loci]

    # Set up the figure. A blank spacer row is inserted between layout blocks so
    # each group lands on a visually separated set of axes (easy to regroup/move
    # in Illustrator); tracks within a block stay flush.
    lo, hi = 0, partition[-1]
    height_ratios, plan = [], []   # plan: track name per grid row, None = spacer
    for bi, block in enumerate(layout.values()):
        if bi > 0 and block_gap > 0:
            height_ratios.append(block_gap)
            plan.append(None)
        for name in block:
            conf = trackconfs.get(name, DEFAULT_CONFIG.copy())
            if "multivec" in conf:
                height_ratios.append(len(conf["multivec"]) * conf.get("height", 1))
            else:
                height_ratios.append(conf.get("height", 1))
            plan.append(name)

    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(nrows=len(plan), ncols=1, height_ratios=height_ratios, hspace=0)

    # Render the tracks
    ax0 = None
    for r, name in enumerate(plan):
        if name is None:        # spacer row -> left blank
            continue
        ax = plt.subplot(gs[r]) if ax0 is None else plt.subplot(gs[r], sharex=ax0)
        if ax0 is None:
            ax0 = ax

        conf = trackconfs.get(name, DEFAULT_CONFIG.copy())
        names, data, kwargs = prepare_track(name, conf, bins, order, coarse_factor)
        nrows = data.shape[0]

        # Use edge coordinates ([0, N] / [0, nrows]) so images, category
        # rectangles, partition lines, and the axes bounds all align exactly on
        # integers (the -0.5 pixel-center convention left the grid half a bin off).
        if conf.get("type") in ("category", "categorical", "colorlist"):
            # draw categorical strips as VECTOR rectangles (one per run) -- a
            # rasterized RGBA image strip does not import cleanly into Illustrator
            for ri in range(nrows):
                rc = data[ri]                              # (n_loci, 4) RGBA
                chg = np.where(np.any(np.diff(rc, axis=0) != 0, axis=1))[0] + 1
                starts = np.concatenate([[0], chg])
                ends = np.concatenate([chg, [n_loci]])
                for s, e in zip(starts, ends):
                    ax.add_patch(Rectangle((s, ri), e - s, 1,
                                           facecolor=rc[s], edgecolor='none'))
        else:
            im = ax.matshow(
                data,
                rasterized=True,
                interpolation='none',
                extent=[0, n_loci, 0, nrows],
                origin='lower',
                **kwargs
            )
        ax.xaxis.set_visible(False)
        ax.set_aspect('auto')
        ax.set_xlim(lo, hi)
        ax.set_ylim(nrows, 0)
        ax.set_yticks(np.arange(nrows) + 0.5)
        ax.set_yticklabels(names)
        ax.vlines(partition, 0, nrows, lw=1, color='k')

    return fig
