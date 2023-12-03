from sfepy.scripts.resview import (
    read_mesh,
    pv_plot,
    FieldOptsToListAction,
    StoreNumberAction,
    OptsToListAction,
    make_title,
)
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import pyvista as pv
import os
import numpy as np

helps = {
    "fields": 'fields to plot, options separated by ":" are possible:\n'
    '"cX" - plot only Xth field component; '
    '"e" - print edges; '
    '"fX" - scale factor for warp/glyphs, see --factor option; '
    '"g - glyphs (for vector fields only), scale by factor; '
    '"iX" - plot X isosurfaces; '
    '"tX" - plot X streamlines, gradient employed for scalar fields; '
    '"mX" - plot cells with mat_id=X; '
    '"oX" - set opacity to X; '
    '"pX" - plot in slot X; '
    '"r" - recalculate cell data to point data; '
    '"sX" - plot data in step X; '
    '"vX" - plotting style: s=surface, w=wireframe, p=points; '
    '"wX" - warp mesh by vector field X, scale by factor',
    "fields_map": "map fields and cell groups, e.g. 1:u1,p1 2:u2,p2",
    "outline": "plot mesh outline",
    "warp": "warp mesh by vector field",
    "factor": "scaling factor for mesh warp and glyphs."
    ' Append "%%" to scale relatively to the minimum bounding box size.',
    "edges": "plot cell edges",
    "isosurfaces": "plot isosurfaces [default: %(default)s]",
    "opacity": "set opacity [default: %(default)s]",
    "color_map": "set color_map, e.g. hot, cool, bone, etc. [default: %(default)s]",
    "axes_options": 'options for directional axes, e.g. xlabel="z1" ylabel="z2",'
    ' zlabel="z3"',
    "no_axes": "hide orientation axes",
    "no_scalar_bars": "hide scalar bars",
    "grid_vector1": 'define positions of plots along grid axis 1 [default: "0, 0, 1.6"]',
    "grid_vector2": 'define positions of plots along grid axis 2 [default: "0, 1.6, 0"]',
    "max_plots": "maximum number of plots along grid axis 1" " [default: 4]",
    "view": "camera azimuth, elevation angles, and optionally zoom factor"
    ' [default: "225,75,0.9"]',
    "camera_position": "define camera position",
    "window_size": "define size of plotting window",
    "animation": "create animation, mp4 file type supported",
    "framerate": "set framerate for animation",
    "screenshot": "save screenshot to file",
    "off_screen": "off screen plots, e.g. when screenshotting",
    "no_labels": "hide plot labels",
    "label_position": 'define position of plot labels [default: "-1, -1, 0, 0.2"]',
    "scalar_bar_size": 'define size of scalar bars [default: "0.15, 0.05"]',
    "scalar_bar_position": 'define position of scalar bars [default: "0.8, 0.02, 0, 1.5"]',
    "step": "select data in a given time step",
    "2d_view": "2d view of XY plane",
}


def plot(
    filenames,
    fields=[],
    fields_map=[],
    step=0,
    outline=False,
    isosurfaces=0,
    show_edges=False,
    warp=None,
    factor=1.0,
    opacity=1.0,
    color_map="binary",
    axes_options=[],
    axes_visibility=False,
    grid_vector1=None,
    grid_vector2=None,
    max_plots=4,
    show_labels=True,
    label_position=[-1, -1, 0, 0.2],
    show_scalar_bars=False,
    scalar_bar_size=[0.15, 0.05],
    scalar_bar_position=[0.8, 0.02, 0, 1.5],
    camera=None,
    camera_position=None,
    window_size=pv.global_theme.window_size,
    anim_output_file=None,
    framerate=2.5,
    screenshot=None,
    off_screen=True,
    view_2d=True,
    scalar_bar_range=None,
):
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-f",
        "--fields",
        metavar="field_spec",
        action=FieldOptsToListAction,
        nargs="+",
        dest="fields",
        default=fields,
        help=helps["fields"],
    )
    parser.add_argument(
        "--fields-map",
        metavar="map",
        action=FieldOptsToListAction,
        nargs="+",
        dest="fields_map",
        default=fields_map,
        help=helps["fields_map"],
    )
    parser.add_argument(
        "-s",
        "--step",
        metavar="step",
        action=StoreNumberAction,
        dest="step",
        default=step,
        help=helps["step"],
    )
    parser.add_argument(
        "-l",
        "--outline",
        action="store_true",
        dest="outline",
        default=outline,
        help=helps["outline"],
    )
    parser.add_argument(
        "-i",
        "--isosurfaces",
        action="store",
        dest="isosurfaces",
        default=isosurfaces,
        help=helps["isosurfaces"],
    )
    parser.add_argument(
        "-e",
        "--edges",
        action="store_true",
        dest="show_edges",
        default=show_edges,
        help=helps["edges"],
    )
    parser.add_argument(
        "-w",
        "--warp",
        metavar="field",
        action="store",
        dest="warp",
        default=warp,
        help=helps["warp"],
    )
    parser.add_argument(
        "--factor",
        metavar="factor",
        action=StoreNumberAction,
        dest="factor",
        default=factor,
        help=helps["factor"],
    )
    parser.add_argument(
        "--opacity",
        metavar="opacity",
        action=StoreNumberAction,
        dest="opacity",
        default=opacity,
        help=helps["opacity"],
    )
    parser.add_argument(
        "--color-map",
        metavar="cmap",
        action="store",
        dest="color_map",
        default=color_map,
        help=helps["color_map"],
    )
    parser.add_argument(
        "--axes-options",
        metavar="options",
        action=OptsToListAction,
        nargs="+",
        dest="axes_options",
        default=axes_options,
        help=helps["axes_options"],
    )
    parser.add_argument(
        "--no-axes",
        action="store_false",
        dest="axes_visibility",
        default=axes_visibility,
        help=helps["no_axes"],
    )
    parser.add_argument(
        "--grid-vector1",
        metavar="grid_vector1",
        action=StoreNumberAction,
        dest="grid_vector1",
        default=grid_vector1,
        help=helps["grid_vector1"],
    )
    parser.add_argument(
        "--grid-vector2",
        metavar="grid_vector2",
        action=StoreNumberAction,
        dest="grid_vector2",
        default=grid_vector2,
        help=helps["grid_vector2"],
    )
    parser.add_argument(
        "--max-plots",
        action=StoreNumberAction,
        dest="max_plots",
        default=max_plots,
        help=helps["max_plots"],
    )
    parser.add_argument(
        "--no-labels",
        action="store_false",
        dest="show_labels",
        default=show_labels,
        help=helps["no_labels"],
    )
    parser.add_argument(
        "--label-position",
        metavar="position",
        action=StoreNumberAction,
        dest="label_position",
        default=label_position,
        help=helps["label_position"],
    )
    parser.add_argument(
        "--no-scalar-bars",
        action="store_false",
        dest="show_scalar_bars",
        default=show_scalar_bars,
        help=helps["no_scalar_bars"],
    )
    parser.add_argument(
        "--scalar-bar-size",
        metavar="size",
        action=StoreNumberAction,
        dest="scalar_bar_size",
        default=scalar_bar_size,
        help=helps["scalar_bar_size"],
    )
    parser.add_argument(
        "--scalar-bar-position",
        metavar="position",
        action=StoreNumberAction,
        dest="scalar_bar_position",
        default=scalar_bar_position,
        help=helps["scalar_bar_position"],
    )
    parser.add_argument(
        "-v",
        "--view",
        metavar="position",
        action=StoreNumberAction,
        dest="camera",
        default=camera,
        help=helps["view"],
    )
    parser.add_argument(
        "--camera-position",
        metavar="camera_position",
        action=StoreNumberAction,
        dest="camera_position",
        default=camera_position,
        help=helps["camera_position"],
    )
    parser.add_argument(
        "--window-size",
        metavar="window_size",
        action=StoreNumberAction,
        dest="window_size",
        default=window_size,
        help=helps["window_size"],
    )
    parser.add_argument(
        "-a",
        "--animation",
        metavar="output_file",
        action="store",
        dest="anim_output_file",
        default=anim_output_file,
        help=helps["animation"],
    )
    parser.add_argument(
        "-r",
        "--frame-rate",
        metavar="rate",
        action=StoreNumberAction,
        dest="framerate",
        default=framerate,
        help=helps["framerate"],
    )
    parser.add_argument(
        "-o",
        "--screenshot",
        metavar="output_file",
        action="store",
        dest="screenshot",
        default=screenshot,
        help=helps["screenshot"],
    )
    parser.add_argument(
        "--off-screen",
        action="store_true",
        dest="off_screen",
        default=off_screen,
        help=helps["off_screen"],
    )
    parser.add_argument(
        "-2",
        "--2d-view",
        action="store_true",
        dest="view_2d",
        default=view_2d,
        help=helps["2d_view"],
    )

    parser.add_argument(
        "--filenames", nargs="+", dest="filenames", action="store", default=filenames
    )
    options = parser.parse_args()

    pv.set_plot_theme("document")
    plotter = pv.Plotter(
        off_screen=options.off_screen, title=make_title(options.filenames)
    )

    if options.anim_output_file:
        raise NotImplementedError("Animation not implemented in this custom version")
    else:
        # _, n_steps = read_mesh(options.filenames, ret_n_steps=True)
        # # dry run
        # scalar_bar_limits = None
        # if options.axes_visibility:
        #     plotter.add_axes(**dict(options.axes_options))
        # for step in range(n_steps):
        #     plotter.clear()
        #     plotter, sb_limits = pv_plot(options.filenames, options,
        #                                     plotter=plotter, step=step,
        #                                     ret_scalar_bar_limits=True)
        #     if scalar_bar_limits is None:
        #         scalar_bar_limits = {k: [] for k in sb_limits.keys()}

        #     for k, v in sb_limits.items():
        #         scalar_bar_limits[k].append(v)

        # plotter.view_xy()

        # for k in scalar_bar_limits.keys():
        #     lims = scalar_bar_limits[k]
        #     clim = (np.min([v[0] for v in lims]),
        #             np.max([v[1] for v in lims]))
        #     scalar_bar_limits[k] = clim

        # plotter.clear()
        # plotter : pv.Plotter = pv_plot(options.filenames, options, plotter=plotter, step=step, scalar_bar_limits=scalar_bar_limits)
        # if options.axes_visibility:
        #         plotter.add_axes(**dict(options.axes_options))
        # # if scalar_bar_range is not None:
        # #     plotter.update_scalar_bar_range(scalar_bar_range)

        # plotter.show(screenshot=options.screenshot,
        #              window_size=options.window_size)

        # if options.screenshot is not None and os.path.exists(options.screenshot):
        #     print(f'saved: {options.screenshot}')

        # plotter.close()
        plotter.clear()
        plotter: pv.Plotter = pv_plot(
            options.filenames, options, plotter=plotter, use_cache=False
        )
        if options.axes_visibility:
            plotter.add_axes(**dict(options.axes_options))

        plotter.view_xy()

        if scalar_bar_range is not None:
            plotter.update_scalar_bar_range(scalar_bar_range)

        plotter.show(screenshot=options.screenshot, window_size=options.window_size)

        if options.screenshot is not None and os.path.exists(options.screenshot):
            print(f"saved: {options.screenshot}")

        plotter.close()
