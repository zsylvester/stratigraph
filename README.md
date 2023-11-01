<img src="https://github.com/zsylvester/stratigraph/blob/main/stratigraph_logo.png" width="400">

## stratigraph

'stratigraph' is a Python package for visualizing and analyzing stratigraphic models. These models ideally have the topographic surfaces through time, not just the stratigraphy, although it is possible to tweak the code so that stratigraphy-only models are visualized. The 3D visualizations rely on [Mayavi](https://docs.enthought.com/mayavi/mayavi/).

Stratigraphic data can be visualized in time or in space. In the time domain, 'stratigraph' can be used create time-elevation (Barrell) plots and chronostratigraphic (Wheeler) diagrams. For example, here is the 'stratigraph' version of Joseph Barrell's time-elevation plot from 1918:

<img src="https://github.com/zsylvester/stratigraph/blob/main/barrell_fig_100.png" width="600">

Wheeler's first chronostratigraphic diagram (Wheeler, 1964) might look something like this:

<img src="https://github.com/zsylvester/stratigraph/blob/main/wheeler_reconstruction.png" width="600">

'stratigraph' is best suited for more complex datasets, for example the ones derived from experiments where the surface topography is carefully tracked through time. For example, this is a dip section through the deposits of the XES-02 experiment, which was run at St. Anthony Falls Laboratory, University of Minnesota:

<img src="https://github.com/zsylvester/stratigraph/blob/main/XES02_3D_dip_section_130.png" width="600">

The cross section in the upper panel is colored by water depth (not grain size); the lower panel shows the corresponding Wheeler diagram.

'stratigraph' can also be used to display stratigraphic models in 3D; for example, this is a meandering river model (created with 'meanderpy'):

<img src="https://github.com/zsylvester/stratigraph/blob/main/meanderpy_model_1_v_small.jpg" width="600">

If you want to show more of the stratigraphy, you can create an 'exploded view':

<img src="https://github.com/zsylvester/stratigraph/blob/main/TWDB_17_1_exploded_view.png" width="600">

The 4 jupyter notebooks in [this folder](https://github.com/zsylvester/stratigraph/tree/main/stratigraph) illustrate the ways 'stratigraph' can be used to visualize stratigraphy.

## Requirements

- matplotlib
- numpy
- pandas
- mayavi
- scipy
- scikit-learn
- scikit-image
- Pillow
- shapely
- tqdm