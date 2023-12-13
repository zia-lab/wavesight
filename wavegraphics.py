#!/usr/bin/env python3

from matplotlib.patches import PathPatch
from matplotlib.path import Path

def frame_patch(xy, width, height, border_width, facecolor='none', hatch = '/', edgecolor = 'white'):
    '''
    Parameters
    ----------
    xy : tuple
        The coordinates of the bottom left corner of the rectangle.
    width : float
        The width of the outer rectangle.
    height : float
        The height of the outer rectangle.
    border_width : float
        The width of the frame.
    facecolor : str, optional
        The color of the rectangle.
    hatch : str, optional
        The hatch pattern of the rectangle.
    
    Returns
    -------
    patch : matplotlib.patches.PathPatch
        The patch to be added to the axes.
    '''
    path = Path([(xy[0]        , xy[1]), # moveto
                 (xy[0]        , xy[1] + height), # lineto
                 (xy[0] + width, xy[1] + height), # lineto
                 (xy[0] + width, xy[1]), # lineto
                 (0,0), # closepoly
                 (xy[0] + border_width        , xy[1] + border_width), # moveto
                 (xy[0] + width - border_width, xy[1] + border_width), # lineto
                 (xy[0] + width - border_width, xy[1] + height - border_width), # lineto
                 (xy[0] + border_width        , xy[1] + height - border_width), # lineto
                 (0,0) # closepoly
                 ],
                [Path.MOVETO,
                  Path.LINETO,
                  Path.LINETO,
                  Path.LINETO,
                  Path.CLOSEPOLY,
                 Path.MOVETO,
                  Path.LINETO,
                  Path.LINETO,
                  Path.LINETO,
                 Path.CLOSEPOLY])
    patch = PathPatch(path,
                      facecolor=facecolor,
                      hatch=hatch,
                      edgecolor=edgecolor)
    return patch