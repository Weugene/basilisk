import json
import os
import re
import numpy as np
import pandas as pd
from logging import Logger
from types import ModuleType
from matplotlib import pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

class Line:
    def __init__(self, x, y, name, style, color, marker=None, secondary_axis=False, **kwargs):
        self.x: list = x
        self.y: list = y
        self.name: str = name
        self.style: str = style
        self.marker: str = marker
        self.color: str = color
        self.secondary_axis = secondary_axis
        self.kwargs = kwargs

    def get_plot_configs(self):
        args = (self.x, self.y)
        kwargs = {
            "linestyle": self.style,
            "marker": self.marker,
            "color": self.color,
            "label": self.name,
            "linewidth": self.kwargs.get("linewidth", 1)
        }
        return args, kwargs

    def get_text_configs(self):
        configs = self.kwargs.get("text")
        args = (configs["x"], configs["y"], configs["s"])
        kwargs = self.extract_specific_keys_values(["x", "y", "s"], input_dict=configs, reverse=True)
        print("get_text_configs", args, kwargs)
        return args, kwargs

    def extract_specific_keys_values(self, keys, input_dict=None, reverse=False):
        """
        Extracts specific keys and their values from the given dictionary.

        Args:
        input_dict (dict): The dictionary to extract data from.
        keys (list): A list of keys to extract from the dictionary.

        Returns:
        dict: A dictionary containing only the specified keys and their values.
        """
        if not input_dict:
            input_dict = self.kwargs
        if reverse:
            return {key: input_dict[key] for key in input_dict if key not in keys}
        else:
            return {key: input_dict[key] for key in input_dict if key in input_dict}

def set_ax_design(
        ax,
        xlabel: dict = None,
        ylabel: dict = None,
        xaxis: dict = None,
        yaxis: dict = None,
        xrange: list = None,
        yrange: list = None,
        xtick_params: dict = None,
        ytick_params: dict = None,
        set_xticks: dict = None,
        set_yticks: dict = None,
        set_xtick_labels: list = None,
        set_ytick_labels: list = None,
        axis_visibility: dict = None
):
    # Setting labels
    if xlabel:
        ax.set_xlabel(**xlabel)
    if ylabel:
        ax.set_ylabel(**ylabel)

    # Setting the axis labels
    if xaxis and xaxis.get("set_label_coords"):
        ax.xaxis.set_label_coords(*xaxis.get("set_label_coords"))
    if yaxis and yaxis.get("set_label_coords"):
        ax.yaxis.set_label_coords(*yaxis.get("set_label_coords"))

    # Setting ticks
    if xtick_params:
        ax.tick_params(axis='x', **xtick_params)
    if ytick_params:
        ax.tick_params(axis='y', **ytick_params)
    if set_xticks:
        ax.set_xticks(**set_xticks)
    if set_yticks:
        ax.set_yticks(**set_yticks)
    if set_xtick_labels is not None:
        ax.xaxis.set_ticklabels(set_xtick_labels)
    if set_ytick_labels is not None:
        ax.yaxis.set_ticklabels(set_ytick_labels)


    # Setting the range for the x-axis
    if xrange:
        ax.axis(xmin=xrange[0], xmax=xrange[1])  # Set the x-axis to display

    # Setting the range for the y-axis
    if yrange:
        ax.axis(ymin=yrange[0], ymax=yrange[1])  # Set the y-axis to display

    # Removing upper and right spines
    if axis_visibility:
        for direction, visibility in axis_visibility.items():
            ax.spines[direction].set_visible(visibility)

def plot_matplotlib(
        lines: list[Line],
        xlabel: dict,
        ylabel: dict,
        xaxis: dict,
        yaxis: dict,
        xtick_params: dict,
        ytick_params: dict,
        xrange: list,
        yrange: list,
        axis_visibility: dict,
        image_name: str,
        path: str,
        title: str = None,
        vertical: list = None,
        horizontal: list = None,
        legend_props: dict | None = None,
        secondary_axis_needed: bool = False,
        second_axis: dict = None,
        grid: dict = None,
        savefig: dict = None,
        aspect: str = "auto",
        set_xticks: dict = None,
        set_yticks: dict = None,
        set_xtick_labels: list = None,
        set_ytick_labels: list = None,
        set_xscale: dict = None,
        set_yscale: dict = None,
        patches: list[dict] = None,
        texts: list[dict] = None
):
    fig, ax1 = plt.subplots()
    ax1.set_aspect(aspect)
    set_ax_design(ax1, xlabel=xlabel, ylabel=ylabel, xaxis=xaxis, yaxis=yaxis, xtick_params=xtick_params, ytick_params=ytick_params, xrange=xrange, yrange=yrange, axis_visibility=axis_visibility, set_xticks=set_xticks, set_yticks=set_yticks, set_xtick_labels=set_xtick_labels, set_ytick_labels=set_ytick_labels)
    if secondary_axis_needed:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_aspect(aspect)
        set_ax_design(ax2, **second_axis)

    # Plotting the lines
    for line in lines:
        ax = ax2 if line.secondary_axis else ax1
        args, kwargs = line.get_plot_configs()
        ax.plot(*args, **kwargs)
        if line.kwargs.get("text"):
            args, kwargs = line.get_text_configs()
            ax.text(*args, **kwargs)

    # Adding vertical lines and filling the area between them
    if vertical and len(vertical) == 2 and horizontal and len(horizontal) == 2:
        # ax1.axvline(x=vertical[0], color='red', linestyle='-')
        # ax1.axvline(x=vertical[1], color='red', linestyle='-')
        ax.axvspan(*vertical, alpha=0.3, color='red', lw=0, zorder=-10)
    if grid is not None:
        grid['linestyle'] = '--'
        ax1.grid(**grid)
    if title:
        fig.title(title)
    if legend_props is not None:
        fig.legend(**legend_props)
    if patches:
        for patch in patches:
            ax1.add_patch(patch)
    if texts:
        for text in texts:
            ax1.text(**text)
    if set_xscale:
        ax1.set_xscale(**set_xscale)
    if set_yscale:
        ax1.set_yscale(**set_yscale)
    fig.tight_layout()
    #fig.patch.set_facecolor('xkcd:mint green') # change background color

    # Saving the plot as a PDF file
    original_image_name = os.path.join(path, "output", f"original_{image_name}")
    compressed_image_name = os.path.join(path, "output", image_name)
    print(f"Saved image {original_image_name}")
    if savefig is None:
        savefig = {
            "bbox_inches": 'tight',
            "format": "pdf"
        }
    fig.savefig(original_image_name, **savefig)

    # Show plot
    #fig.show()
    # Compress image
    command = f"gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.5 -dPDFSETTINGS=/screen -dNOPAUSE -dBATCH -sOutputFile={compressed_image_name} {original_image_name}"
    os.system(command)

def sort_names(image_files):
    file_names = [os.path.basename(string) for string in image_files]
    times = [(float(re.findall(r"\d+\.\d+", string)[0]), string) for string in file_names]
    times = sorted(times, key=lambda x: x[0])
    print(f"Time range: {times[0]} -- {times[-1]}")
    image_files = [t[1] for t in times]
    return image_files

def give_time(file) -> float:
    filename = os.path.basename(file)
    time = re.findall(r"\d+\.\d+", filename)[0]
    return float(time)

def interpolate_by_ngb(arr, mask_val=None, fun=None):
    mask = np.isnan(arr) if mask_val is None else arr == mask_val
    if fun is not None:
        mask = fun(arr) | mask
    arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
    return arr

def signal_filtering(x_original, y_original, intervals: list[list[int, int]], **kwargs):
    x_in = x_original.copy()
    y_in = y_original.copy()
    for xmin, xmax in intervals:
        ind = (x_in > xmin) & (x_in < xmax)
        xx = x_in[ind]
        yy = y_in[ind]
        sm_x, sm_y = sm_lowess(yy, xx, **kwargs).T
        # Apply the combined mask to x_in and y_in
        x_in[ind] = sm_x
        y_in[ind] = sm_y
    return x_in, y_in

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return "DataFrame"
        elif isinstance(obj, pd.Series):
            return "Series"
        elif isinstance(obj, ModuleType):
            return "ModuleType"
        elif isinstance(obj, Logger):
            return "Logger"
        # print(obj)
        try:
            return json.JSONEncoder.default(self, obj)
        except:
            return None