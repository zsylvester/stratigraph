<img src="https://github.com/zsylvester/stratigraph/blob/main/stratigraph_logo.png" width="300">

## stratigraph

'stratigraph' is a Python package for visualizing and analyzing stratigraphic models. These models ideally have the topographic surfaces through time, not just the stratigraphy, although it is possible to tweak the code so that stratigraphy-onpy models are visualized. The 3D visualizations rely on [Mayavi](https://docs.enthought.com/mayavi/mayavi/).

Stratigraphic data can be visualized in time or in space. In the time domain, 'stratigraph' can be used create time-elevation (Barrell) plots and chronostratigraphic (Wheeler) diagrams. For example, here is the animated version of Joseph Barrell's time-elavation plot from 1918:

<img src="https://github.com/zsylvester/stratigraph/blob/main/Barrell_original_plot_animation.gif" width="500">

## Requirements

- matplotlib
- numpy
- mayavi
- PIL
- scipy
- tqdm