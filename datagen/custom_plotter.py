from sfepy.scripts.resview import read_mesh, pv_plot, make_title
from argparse import Namespace
import pyvista as pv
import os
# import numpy as np

helps = {
    'fields':
        'fields to plot, options separated by ":" are possible:\n'
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
    'fields_map':
        'map fields and cell groups, e.g. 1:u1,p1 2:u2,p2',
    'outline':
        'plot mesh outline',
    'warp':
        'warp mesh by vector field',
    'factor':
        'scaling factor for mesh warp and glyphs.'
        ' Append "%%" to scale relatively to the minimum bounding box size.',
    'edges':
        'plot cell edges',
    'isosurfaces':
        'plot isosurfaces [default: %(default)s]',
    'opacity':
        'set opacity [default: %(default)s]',
    'color_map':
        'set color_map, e.g. hot, cool, bone, etc. [default: %(default)s]',
    'axes_options':
        'options for directional axes, e.g. xlabel="z1" ylabel="z2",'
        ' zlabel="z3"',
    'no_axes':
        'hide orientation axes',
    'no_scalar_bars':
        'hide scalar bars',
    'grid_vector1':
        'define positions of plots along grid axis 1 [default: "0, 0, 1.6"]',
    'grid_vector2':
        'define positions of plots along grid axis 2 [default: "0, 1.6, 0"]',
    'max_plots':
        'maximum number of plots along grid axis 1'
        ' [default: 4]',
    'view':
        'camera azimuth, elevation angles, and optionally zoom factor'
        ' [default: "225,75,0.9"]',
    'camera_position':
        'define camera position',
    'window_size':
        'define size of plotting window',
    'animation':
        'create animation, mp4 file type supported',
    'framerate':
        'set framerate for animation',
    'screenshot':
        'save screenshot to file',
    'off_screen':
        'off screen plots, e.g. when screenshotting',
    'no_labels':
        'hide plot labels',
    'label_position':
        'define position of plot labels [default: "-1, -1, 0, 0.2"]',
    'scalar_bar_size':
        'define size of scalar bars [default: "0.15, 0.05"]',
    'scalar_bar_position':
        'define position of scalar bars [default: "0.8, 0.02, 0, 1.5"]',
    'step':
        'select data in a given time step',
    '2d_view':
        '2d view of XY plane',
}

def plot(
        filenames, 
        fields = [], 
        fields_map = [], 
        step = 0, 
        outline = False, 
        isosurfaces = 0, 
        show_edges = False, 
        warp = None, 
        factor = 1., 
        opacity = 1., 
        color_map = 'binary', 
        axes_options = [], 
        axes_visibility = False, 
        grid_vector1 = None, 
        grid_vector2 = None, 
        max_plots = 4, 
        show_labels = True, 
        label_position = [-1, -1, 0, 0.2], 
        show_scalar_bars = False, 
        scalar_bar_size = [0.15, 0.05], 
        scalar_bar_position = [0.8, 0.02, 0, 1.5], 
        camera = None, 
        camera_position = None, 
        window_size = pv.global_theme.window_size, 
        anim_output_file = None, 
        framerate = 2.5, 
        screenshot = None, 
        off_screen = True, 
        view_2d = True, 
        scalar_bar_range = None):
    options = Namespace(
        filenames = filenames,
        fields = fields,
        fields_map = fields_map,
        step = step,
        outline = outline,
        isosurfaces = isosurfaces,
        show_edges = show_edges,
        warp = warp,
        factor = factor,
        opacity = opacity,
        color_map = color_map,
        axes_options = axes_options,
        axes_visibility = axes_visibility,
        grid_vector1 = grid_vector1,
        grid_vector2 = grid_vector2,
        max_plots = max_plots,
        show_labels = show_labels,
        label_position = label_position,
        show_scalar_bars = show_scalar_bars,
        scalar_bar_size = scalar_bar_size,
        scalar_bar_position = scalar_bar_position,
        camera = camera,
        camera_position = camera_position,
        window_size = window_size,
        anim_output_file = anim_output_file,
        framerate = framerate,
        screenshot = screenshot,
        off_screen = off_screen,
        view_2d = view_2d,
    )
    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=options.off_screen,
                         title=make_title(options.filenames))

    if options.anim_output_file:
        raise NotImplementedError('Animation not implemented in this custom version')
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
        plotter : pv.Plotter = pv_plot(options.filenames, options, plotter=plotter, use_cache=False)
        if options.axes_visibility:
            plotter.add_axes(**dict(options.axes_options))

        plotter.view_xy()
        
        if scalar_bar_range is not None:
            plotter.update_scalar_bar_range(scalar_bar_range)

        plotter.show(screenshot=options.screenshot,
                     window_size=options.window_size)

        if options.screenshot is not None and os.path.exists(options.screenshot):
            print(f'saved: {options.screenshot}')

        plotter.close()