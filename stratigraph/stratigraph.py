import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mayavi import mlab
from scipy.ndimage import map_coordinates
from scipy import signal, interpolate
from PIL import Image, ImageDraw
from matplotlib.colors import ListedColormap
from tqdm import tqdm, trange

def create_block_diagram(strat, prop, facies, dx, ve, xoffset, yoffset, scale, ci, plot_strat, plot_contours, plot_sides, color_mode, bottom, topo_min, topo_max, export, opacity):
    """function for creating a 3D block diagram in Mayavi
    strat - input array with stratigraphic surfaces
    facies - property or facies array
    dx - size of gridcells in the horizontal direction in 'strat'
    ve - vertical exaggeration
    offset - offset in the y-direction relative to 0
    scale - scaling factor
    ci - contour interval
    strat_switch - 1 if you want to plot stratigraphy on the sides; 0 otherwise
    contour_switch - 1 if you want to plot contours on the top surface; 0 otherwise
    bottom - elevation value for the bottom of the block"""

    r,c,ts = np.shape(strat)

    # if z is increasing downward:
    if np.max(strat[:, :, -1] - strat[:, :, 0]) < 0:
        strat = -1 * strat

    z = scale*strat[:,:,ts-1].T
    if plot_strat:
        z1 = strat[:,:,0].T
    else:
        z1 = strat[:,:,-1].T

    X1 = scale*(xoffset + np.linspace(0,c-1,c)*dx) # x goes with c and y with r
    Y1 = scale*(yoffset + np.linspace(0,r-1,r)*dx)
    X1_grid , Y1_grid = np.meshgrid(X1, Y1)

    if export == 1:
        surf = mlab.surf(X1,Y1,z,warp_scale=ve,colormap='gist_earth',vmin=scale*topo_min,vmax=scale*topo_max, opacity = opacity)
        # cmapf = matplotlib.cm.get_cmap('Blues_r',256)
        BluesBig = matplotlib.cm.get_cmap('Blues_r', 512)
        newcmp = ListedColormap(BluesBig(np.linspace(0.0, 1.0, 256)))
        normf = matplotlib.colors.Normalize(vmin=scale*topo_min,vmax=scale*topo_max)
        z_range = np.linspace(scale*topo_min, scale*topo_max, 256)
        surf.module_manager.scalar_lut_manager.lut.table = (np.array(newcmp(normf(z_range)))*255).astype('uint8')
    else:
        # if color_mode == 'property':
        #     mlab.mesh(X1_grid, Y1_grid, ve*z, scalars = prop[:, :, -1], colormap='YlOrBr', vmin=0, vmax=1, opacity = opacity) 
        #     if not plot_sides:
        #         mlab.mesh(X1_grid, Y1_grid, ve*scale*strat[:,:,0].T, scalars = facies[:, :, 0], colormap='YlOrBr', vmin=0, vmax=1, opacity = opacity)
        # else:
        mlab.surf(X1, Y1, z, warp_scale=ve, colormap='gist_earth', opacity = opacity) #, line_width=5.0, vmin=scale*topo_min,vmax=scale*topo_max, representation='wireframe')
        if not plot_sides:
            mlab.surf(X1, Y1, scale*strat[:,:,0].T, warp_scale=ve, colormap='gist_earth', opacity=opacity) #colormap='gist_earth',vmin=scale*topo_min,vmax=scale*topo_max, opacity = opacity)
    if plot_contours:
        vmin = scale * topo_min #np.min(strat[:,:,-1])
        vmax = scale * topo_max #np.max(strat[:,:,-1])
        contours = list(np.arange(vmin, vmax, ci*scale)) # list of contour values
        mlab.contour_surf(X1, Y1, z, contours=contours, warp_scale=ve, color=(0,0,0), line_width=1.0)

    if plot_sides:
        gray = (0.6,0.6,0.6) # color for plotting sides
        
        # updip side:
        vertices, triangles = create_section(z1[:,0],dx,bottom) 
        x = scale*(xoffset + vertices[:,0])
        y = scale*(yoffset + np.zeros(np.shape(vertices[:,0])))
        z = scale*ve*vertices[:,1]
        mlab.triangular_mesh(x, y, z, triangles, color=gray, opacity = opacity)
        
        # downdip side:
        vertices, triangles = create_section(z1[:,-1],dx,bottom) 
        x = scale*(xoffset + vertices[:,0])
        y = scale*(yoffset + (r-1)*dx*np.ones(np.shape(vertices[:,0])))
        z = scale*ve*vertices[:,1]
        mlab.triangular_mesh(x, y, z, triangles, color=gray, opacity = opacity)

        # left edge (looking downdip):
        vertices, triangles = create_section(z1[0,:],dx,bottom) 
        x = scale*(xoffset + np.zeros(np.shape(vertices[:,0])))
        y = scale*(yoffset + vertices[:,0])
        z = scale*ve*vertices[:,1]
        mlab.triangular_mesh(x, y, z, triangles, color=gray, opacity = opacity)
        
        # right edge (looking downdip):
        vertices, triangles = create_section(z1[-1,:],dx,bottom) 
        x = scale*(xoffset + (c-1)*dx*np.ones(np.shape(vertices[:,0])))
        y = scale*(yoffset + vertices[:,0])
        z = scale*ve*vertices[:,1]
        mlab.triangular_mesh(x, y, z, triangles, color=gray, opacity = opacity)
        
        # bottom face of block:
        vertices = dx*np.array([[0,0],[c-1,0],[c-1,r-1],[0,r-1]])
        triangles = [[0,1,3],[1,3,2]]
        x = scale*(xoffset + vertices[:,0])
        y = scale*(yoffset + vertices[:,1])
        z = scale*bottom*np.ones(np.shape(vertices[:,0]))
        mlab.triangular_mesh(x, y, ve*z, triangles, color=gray, opacity = opacity)

def add_stratigraphy_to_block_diagram(strat, prop, facies, dx, ve, xoffset, yoffset, scale, plot_surfs, color_mode, colors, colormap, line_thickness, export, opacity):
    """function for adding stratigraphy to the sides of a block diagram
    colors layers by relative age
    strat - input array with stratigraphic surfaces
    facies - 1D array of facies codes for layers
    h - channel depth (height of point bar)
    thalweg_z - array of thalweg elevations for each layer
    dx - size of gridcells in the horizontal direction in 'strat'
    ve - vertical exaggeration
    offset - offset in the y-direction relative to 0
    scale - scaling factor
    plot_surfs - if equals 1, stratigraphic boundaries will be plotted on the sides as black lines
    color_mode - determines what kind of plot is created; can be 'property', 'time', or 'facies'
    colors - colors scheme for facies (list of RGB values)
    line_thickness - tube radius for plotting layers on the sides
    export - if equals 1, the display can be saved as a VRML file for use in other programs (e.g., 3D printing)""" 
    r,c,ts=np.shape(strat)
    if color_mode == 'time':
        norm = matplotlib.colors.Normalize(vmin=0.0, vmax=ts-1)
        cmap = matplotlib.cm.get_cmap(colormap)
    if (color_mode == 'property') | (color_mode == 'facies'):
        norm = matplotlib.colors.Normalize(vmin=0.0, vmax=0.35)
        cmap = matplotlib.cm.get_cmap(colormap)

    for layer_n in trange(ts-1): # main loop
        vmin = 0.0
        vmax = 0.35
        top = strat[:,0,layer_n+1]  # updip side
        base = strat[:,0,layer_n]
        if color_mode == "property":
            props = prop[:,0,layer_n]
        if plot_surfs:
            Y1 = scale*(yoffset + dx*np.arange(0,r))
            X1 = scale*(xoffset + np.zeros(np.shape(base)))
            Z1 = ve*scale*base
            mlab.plot3d(X1,Y1,Z1,color=(0,0,0),tube_radius=line_thickness)
        if np.max(top-base)>0:
            Points,Inds = triangulate_layers(top,base,dx)
            for i in range(len(Points)):
                vertices = Points[i]
                triangles, scalars = create_triangles(vertices)
                Y1 = scale*(yoffset + vertices[:,0])
                X1 = scale*(xoffset + dx*0*np.ones(np.shape(vertices[:,0])))
                Z1 = scale*vertices[:,1]
                if color_mode == "property":
                    scalars = props[Inds[i]]
                else:
                    scalars = []
                plot_layers_on_one_side(layer_n, facies, color_mode, colors, X1, Y1, Z1, ve, triangles, vertices, scalars, colormap, norm, vmin, vmax, export, opacity)

        top = strat[:,-1,layer_n+1]  # downdip side
        base = strat[:,-1,layer_n]
        if color_mode == "property":
            props = prop[:,-1,layer_n]
        if plot_surfs:
            Y1 = scale*(yoffset + dx*np.arange(0,r))
            X1 = scale*(xoffset + dx*(c-1)*np.ones(np.shape(base)))
            Z1 = ve*scale*base
            mlab.plot3d(X1,Y1,Z1,color=(0,0,0),tube_radius=line_thickness)
        if np.max(top-base)>0:
            Points,Inds = triangulate_layers(top,base,dx)
            for i in range(len(Points)):
                vertices = Points[i]
                triangles, scalars = create_triangles(vertices)
                Y1 = scale*(yoffset + vertices[:,0])
                X1 = scale*(xoffset + dx*(c-1)*np.ones(np.shape(vertices[:,0])))
                Z1 = scale*vertices[:,1]
                if color_mode == "property":
                    scalars = props[Inds[i]]
                else:
                    scalars = []
                plot_layers_on_one_side(layer_n, facies, color_mode, colors, X1, Y1, Z1, ve, triangles, vertices, scalars, colormap, norm, vmin, vmax, export, opacity)

        top = strat[0,:,layer_n+1]  # left edge (looking downdip)
        base = strat[0,:,layer_n]
        if color_mode == "property":
            props = prop[0,:,layer_n]
        if plot_surfs:
            Y1 = scale*(yoffset + np.zeros(np.shape(base)))
            X1 = scale*(xoffset + dx*np.arange(0,c))
            Z1 = ve*scale*base
            mlab.plot3d(X1,Y1,Z1,color=(0,0,0),tube_radius=line_thickness)
        if np.max(top-base)>0:
            Points,Inds = triangulate_layers(top,base,dx)
            for i in range(len(Points)):
                vertices = Points[i]
                triangles, scalars = create_triangles(vertices)
                Y1 = scale*(yoffset + dx*0*np.ones(np.shape(vertices[:,0])))
                X1 = scale*(xoffset + vertices[:,0])
                Z1 = scale*vertices[:,1]
                if color_mode == "property":
                    scalars = props[Inds[i]]
                else:
                    scalars = []
                plot_layers_on_one_side(layer_n, facies, color_mode, colors, X1, Y1, Z1, ve, triangles, vertices, scalars, colormap, norm, vmin, vmax, export, opacity)

        top = strat[-1,:,layer_n+1] # right edge (looking downdip)
        base = strat[-1,:,layer_n]
        if color_mode == "property":
            props = prop[-1,:,layer_n]
        if plot_surfs:
            Y1 = scale*(yoffset + dx*(r-1)*np.ones(np.shape(base)))
            X1 = scale*(xoffset + dx*np.arange(0,c))
            Z1 = ve*scale*base
            mlab.plot3d(X1,Y1,Z1,color=(0,0,0),tube_radius=line_thickness)
        if np.max(top-base)>0:
            Points,Inds = triangulate_layers(top,base,dx)
            for i in range(len(Points)):
                vertices = Points[i]
                triangles, scalars = create_triangles(vertices)
                Y1 = scale*(yoffset + dx*(r-1)*np.ones(np.shape(vertices[:,0])))
                X1 = scale*(xoffset + vertices[:,0])
                Z1 = scale*vertices[:,1]
                if color_mode == "property":
                    scalars = props[Inds[i]]
                else:
                    scalars = []
                plot_layers_on_one_side(layer_n, facies, color_mode, colors, X1, Y1, Z1, ve, triangles, vertices, scalars, colormap, norm, vmin, vmax, export, opacity)

def create_exploded_view(strat, prop, facies, x0, y0, nx, ny, gap, dx, ve, scale, plot_strat, plot_surfs, plot_contours, plot_sides, color_mode, colors, colormap, line_thickness, bottom,export, topo_min, topo_max, ci, opacity):
    """function for creating an exploded-view block diagram
    inputs:
    strat - stack of stratigraphic surfaces
    facies - 1D array of facies codes for layers
    topo - stack of topographic surfaces
    nx - number of blocks in x direction
    ny - number of blocks in y direction
    gap - gap between blocks (number of gridcells)
    dx - gridcell size
    ve - vertical exaggeration
    scale - scaling factor (for whole model)
    strat_switch - if equals 1, the stratigraphy will be plotted on the sides of the blocks
    plot_surfs - if equals 1, the stratigraphic surfaces will be plotted on the sides (adds a lot of triangles - not good for 3D printing)
    contour_swicth - if equals 1, contours will be plotted on the top surface
    color_mode - determines what kind of plot is created; can be 'property', 'time', or 'facies'
    colors - colors scheme for facies (list of RGB values)
    line_thickness - - tube radius for plotting layers on the sides
    bottom - elevation value for the bottom of the block
    export - if equals 1, the display can be saved as a VRML file for use in other programs (e.g., 3D printing)"""
    r,c,ts = np.shape(strat)
    count = 0
    for i in range(nx):
        for j in range(ny):
            x1 = i * int(c/nx)
            x2 = (i+1) * int(c/nx)
            y1 = j * int(r/ny)
            y2 = (j+1) * int(r/ny)
            xoffset = x0 + (x1+i*gap)*dx
            yoffset = y0 + (y1+j*gap)*dx
            if color_mode == "property":
                create_block_diagram(strat[y1:y2,x1:x2,:], prop[y1:y2,x1:x2,:], facies[y1:y2,x1:x2,:], dx, ve, xoffset, yoffset, scale, ci, plot_strat, plot_contours, plot_sides, color_mode, bottom, topo_min, topo_max, export, opacity)
                if plot_strat:
                    add_stratigraphy_to_block_diagram(strat[y1:y2,x1:x2,:], prop[y1:y2,x1:x2,:], facies[y1:y2,x1:x2,:], dx, ve, xoffset, yoffset, scale, plot_surfs, color_mode, colors, colormap, line_thickness, export, opacity)
            else:
                create_block_diagram(strat[y1:y2,x1:x2,:], prop, facies[y1:y2,x1:x2,:], dx, ve, xoffset, yoffset, scale, ci, plot_strat, plot_contours, plot_sides, color_mode, bottom, topo_min, topo_max, export, opacity)
                if plot_strat:
                    add_stratigraphy_to_block_diagram(strat[y1:y2,x1:x2,:], prop, facies[y1:y2,x1:x2,:], dx, ve, xoffset, yoffset, scale, plot_surfs, color_mode, colors, colormap, line_thickness, export, opacity)
            count = count+1
            print("block "+str(count)+" done, out of "+str(nx*ny)+" blocks")

def create_fence_diagram(strat, prop, facies, x0, y0, nx, ny, dx, ve, scale, plot_surfs, plot_sides, color_mode, colors, colormap, line_thickness, bottom, export, opacity):
    """function for creating a fence diagram
    inputs:
    strat - stack of stratigraphic surfaces
    facies - 1D array of facies codes for layers
    topo - stack of topographic surfaces
    nx - number of strike sections
    ny - number of dip sections
    dx - gridcell size
    ve - vertical exaggeration
    scale - scaling factor (for whole model)
    plot_surfs - if equals 1, the stratigraphic surfaces will be plotted on the sides (adds a lot of triangles - not good for 3D printing)
    color_mode - determines what kind of plot is created; can be 'property', 'time', or 'facies'
    colors - colors scheme for facies (list of RGB values)
    line_thickness - - tube radius for plotting layers on the sides
    bottom - elevation value for the bottom of the block
    export - if equals 1, the display can be saved as a VRML file for use in other programs (e.g., 3D printing)"""
    r,c,ts=np.shape(strat)
    gray = (0.6,0.6,0.6)
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=ts-1)
    cmap = matplotlib.cm.get_cmap(colormap)
    vmin = np.min(prop)
    vmax = np.max(prop)

    gray = (0.6,0.6,0.6) # color for plotting sides

    z = scale*strat[:,:,ts-1].T
    z1 = strat[:,:,0].T
    xoffset = 0; yoffset = 0
    
    # updip side:
    vertices, triangles = create_section(z1[:,0],dx,bottom) 
    x = scale*(xoffset + vertices[:,0])
    y = scale*(yoffset + np.zeros(np.shape(vertices[:,0])))
    z = scale*ve*vertices[:,1]
    mlab.triangular_mesh(x, y, z, triangles, color=gray, opacity = opacity)
    
    # downdip side:
    vertices, triangles = create_section(z1[:,-1],dx,bottom) 
    x = scale*(xoffset + vertices[:,0])
    y = scale*(yoffset + (r-1)*dx*np.ones(np.shape(vertices[:,0])))
    z = scale*ve*vertices[:,1]
    mlab.triangular_mesh(x, y, z, triangles, color=gray, opacity = opacity)

    # left edge (looking downdip):
    vertices, triangles = create_section(z1[0,:],dx,bottom) 
    x = scale*(xoffset + np.zeros(np.shape(vertices[:,0])))
    y = scale*(yoffset + vertices[:,0])
    z = scale*ve*vertices[:,1]
    mlab.triangular_mesh(x, y, z, triangles, color=gray, opacity = opacity)
    
    # right edge (looking downdip):
    vertices, triangles = create_section(z1[-1,:],dx,bottom) 
    x = scale*(xoffset + (c-1)*dx*np.ones(np.shape(vertices[:,0])))
    y = scale*(yoffset + vertices[:,0])
    z = scale*ve*vertices[:,1]
    mlab.triangular_mesh(x, y, z, triangles, color=gray, opacity = opacity)
    
    # bottom face of block:
    vertices = dx*np.array([[0,0],[c-1,0],[c-1,r-1],[0,r-1]])
    triangles = [[0,1,3],[1,3,2]]
    x = scale*(xoffset + vertices[:,0])
    y = scale*(yoffset + vertices[:,1])
    z = scale*bottom*np.ones(np.shape(vertices[:,0]))
    mlab.triangular_mesh(x, y, ve*z, triangles, color=gray, opacity = opacity)

    section_inds = np.hstack((0, int(c/(nx+1)) * np.arange(1, nx+1), c-1))
    for x1 in tqdm(section_inds): # strike sections
        if plot_sides:
            vertices, triangles = create_section(strat[:,x1,0],dx,bottom) 
            y = y0 + scale*(vertices[:,0])
            x = x0 + scale*(x1*dx+np.zeros(np.shape(vertices[:,0])))
            z = scale*ve*vertices[:,1]
            mlab.triangular_mesh(x,y,z,triangles,color=gray)
        for layer_n in range(ts-1): # main loop
            top = strat[:,x1,layer_n+1]  
            base = strat[:,x1,layer_n]
            if color_mode == 'property':
                props = prop[:,x1,layer_n]
            if plot_surfs:
                Y1 = y0 + scale*(dx*np.arange(0,r))
                X1 = x0 + scale*(x1*dx+np.zeros(np.shape(base)))
                Z1 = ve*scale*base
                mlab.plot3d(X1,Y1,Z1,color=(0,0,0),tube_radius=line_thickness)
            if np.max(top-base)>0:
                Points,Inds = triangulate_layers(top,base,dx)
                for i in range(len(Points)):
                    vertices = Points[i]
                    triangles, scalars = create_triangles(vertices)
                    Y1 = y0 + scale*(vertices[:,0])
                    X1 = x0 + scale*(x1*dx+dx*0*np.ones(np.shape(vertices[:,0])))
                    Z1 = scale*vertices[:,1]
                    if color_mode == 'property':
                        scalars = props[Inds[i]]
                    else:
                        scalars = []
                    # plot_layers_on_one_side(layer_n,facies,color_mode,colors,X1,Y1,Z1,ve,triangles,vertices,scale*scalars,cmap,norm,vmin,vmax,export)
                    plot_layers_on_one_side(layer_n, facies, color_mode, colors, X1, Y1, Z1, ve, triangles, vertices, scalars, colormap, norm, vmin, vmax, export, opacity)

    section_inds = np.hstack((0, int(r/(ny+1)) * np.arange(1, ny+1), r-1))
    for y1 in tqdm(section_inds): # dip sections
        if plot_sides:
            vertices, triangles = create_section(strat[y1,:,0],dx,bottom) 
            y = y0 + scale*(y1*dx+np.zeros(np.shape(vertices[:,0])))
            x = x0 + scale*(vertices[:,0])
            z = scale*ve*vertices[:,1]
            mlab.triangular_mesh(x,y,z,triangles,color=gray)
        for layer_n in range(ts-1): # main loop
            top = strat[y1,:,layer_n+1]  
            base = strat[y1,:,layer_n]
            if color_mode == 'property':
                props = prop[y1,:,layer_n]
            if plot_surfs:
                Y1 = y0 + scale*(y1*dx+np.zeros(np.shape(base)))
                X1 = x0 + scale*(dx*np.arange(0,c))
                Z1 = ve*scale*base
                mlab.plot3d(X1,Y1,Z1,color=(0,0,0),tube_radius=line_thickness)
            if np.max(top-base)>0:
                Points,Inds = triangulate_layers(top,base,dx)
                for i in range(len(Points)):
                    vertices = Points[i]
                    triangles, scalars = create_triangles(vertices)
                    Y1 = y0 + scale*(y1*dx + dx*0*np.ones(np.shape(vertices[:,0])))
                    X1 = x0 + scale*(vertices[:,0])
                    Z1 = scale*vertices[:,1]
                    if color_mode == 'property':
                        scalars = props[Inds[i]]
                    else:
                        scalars = []
                    # plot_layers_on_one_side(layer_n,facies,color_mode,colors,X1,Y1,Z1,ve,triangles,vertices,scale*scalars,cmap,norm,vmin,vmax,export)
                    plot_layers_on_one_side(layer_n, facies, color_mode, colors, X1, Y1, Z1, ve, triangles, vertices, scalars, colormap, norm, vmin, vmax, export, opacity)
        # print('done with section '+str(nsec)+' of '+str(ny)+' dip sections')
        r,c = np.shape(strat[:,:,-1])
        Y1 = scale*(np.linspace(0,r-1,r)*dx)
        X1 = scale*(np.linspace(0,c-1,c)*dx)
        topo_min = np.min(strat[:,:,-1])
        topo_max = np.max(strat[:,:,-1])
        mlab.surf(X1, Y1, scale*strat[:,:,-1].T, warp_scale=ve, colormap='gist_earth', vmin=scale*topo_min, vmax=scale*topo_max, opacity=0.15)

def triangulate_layers(top,base,dx):
    """function for creating vertices of polygons that describe one layer"""
    x = dx * np.arange(0,len(top))
    ind1 = np.argwhere(top-base>0).flatten()
    ind2 = np.argwhere(np.diff(ind1)>1)
    ind2 = np.vstack((np.array([[-1]]),ind2))
    ind2 = np.vstack((ind2,np.array([[len(top)]])))
    Points = [] # list for points to be triangulated
    Inds = []
    for i in range(len(ind2)-1):
        ind3 = ind1[int(ind2[i])+1:int(ind2[i+1])+1]
        if (ind3[0] != 0) & (ind3[-1] != len(top)-1):
            ind3 = np.hstack((ind3[0]-1,ind3))
            ind3 = np.hstack((ind3,ind3[-1]+1)) 
            top1 = top[ind3][:-1]
            base1 = base[ind3][1:]
            x1 = np.concatenate((x[ind3][:-1], x[ind3][::-1][:-1]))
            inds = np.concatenate((ind3[:-1], ind3[::-1][:-1]))
        if (ind3[0] == 0) & (ind3[-1] != len(top)-1):
            ind3 = np.hstack((ind3,ind3[-1]+1))
            top1 = top[ind3][:-1]
            base1 = base[ind3]
            x1 = np.concatenate((x[ind3][:-1], x[ind3][::-1]))
            inds = np.concatenate((ind3[:-1], ind3[::-1]))
        if (ind3[0] != 0) & (ind3[-1] == len(top)-1):
            ind3 = np.hstack((ind3[0]-1,ind3))
            top1 = top[ind3]
            base1 = base[ind3][1:]
            x1 = np.concatenate((x[ind3], x[ind3][::-1][:-1]))
            inds = np.concatenate((ind3, ind3[::-1][:-1]))
        if (ind3[0] == 0) & (ind3[-1] == len(top)-1):
            top1 = top[ind3]
            base1 = base[ind3]
            x1 = np.concatenate((x[ind3], x[ind3][::-1]))
            inds = np.concatenate((ind3, ind3[::-1]))
        npoints = len(top1)+len(base1)
        y = np.hstack((top1,base1[::-1]))
        vertices = np.vstack((x1,y)).T
        Points.append(vertices)
        Inds.append(inds)
    return Points,Inds

def create_triangles(vertices):
    """function for creating list of triangles from vertices
    inputs:
    vertices - 2 x n array with coordinates of polygon
    returns:
    triangles - indices of the 'vertices' array that from triangles (for triangular mesh)
    scalars - 'fake' elevation values for each vertex of the polygon, used for coloring (relies on the base of the polygon)"""
    n = len(vertices[:,0])
    Z1 = vertices[:,1]
    triangles = []
    if (np.mod(n,2)==0) & (vertices[int((n-1)/2),0] != vertices[int((n-1)/2+1),0]): # if polygon is in the interior of the block
        triangles.append([0,1,n-1])
        for i in range(1,int(n/2-1)):
            triangles.append([i,i+1,n-i])
            triangles.append([i+1,n-i,n-i-1])
        triangles.append([int(n/2-1),int(n/2),int(n/2+1)])
        scalars = np.hstack((Z1[0],Z1[int(n/2):][::-1],Z1[int(n/2)+1:]))
    if (np.mod(n,2)==0) & (vertices[int((n-1)/2),0] == vertices[int((n-1)/2+1),0]): # if polygon touches both sides of the block
        for i in range(0,int(n/2-1)):
            triangles.append([i,i+1,n-i-1])
            triangles.append([i+1,n-i-1,n-i-2])
        scalars = np.hstack((Z1[int(n/2):][::-1],Z1[int(n/2):]))
    if np.mod(n,2)!=0: # if polygon has one segment on the side of the block
        if vertices[int((n-1)/2),0] == vertices[int((n-1)/2+1),0]: # if polygon touches the right side of the block
            triangles.append([0,1,n-1])
            for i in range(1,int((n-1)/2)):
                triangles.append([i,i+1,n-i])
                triangles.append([i+1,n-i,n-i-1])
            scalars = np.hstack((Z1[0],Z1[int((n+1)/2):][::-1],Z1[int((n+1)/2):]))
        else:
            for i in range(0,int((n-1)/2)-1): # if polygon touches the left side of the block
                triangles.append([i,i+1,n-i-1])
                triangles.append([i+1,n-i-1,n-i-2])
            triangles.append([int((n-1)/2-1),int((n-1)/2),int((n-1)/2+1)])
            scalars = np.hstack((Z1[int((n+1)/2)-1:][::-1],Z1[int((n+1)/2):]))
    return triangles, scalars

def create_section(profile,dx,bottom):
    """function for creating a cross section from a top surface
    inputs:
    profile - elevation data for top surface
    dx - gridcell size
    bottom - elevation value for the bottom of the block
    returns:
    vertices - coordinates of vertices
    triangles - indices of the 'vertices' array that from triangles (for triangular mesh)
    """
    x1 = dx*np.linspace(0, len(profile)-1, len(profile))
    x = np.hstack((x1, x1[::-1]))
    y = np.hstack((profile, bottom*np.ones(np.shape(x1))))
    vertices = np.vstack((x, y)).T
    n = len(x)
    triangles = []
    for i in range(0,int((n-1)/2)):
        triangles.append([i,i+1,n-i-1])
        triangles.append([i+1,n-i-1,n-i-2])
    return vertices, triangles

def plot_layers_on_one_side(layer_n, facies, color_mode, colors, X1, Y1, Z1, ve, triangles, vertices, scalars, colormap, norm, vmin, vmax, export, opacity):
    """function for plotting layers on one side of a block
    inputs:
    layer_n - layer number
    facies - 1D array of facies codes for layers
    color_mode - determines what kind of plot is created; can be 'property', 'time', or 'facies'
    colors - list of RGB values used if color_mode is 'facies'
    X1,Y1,Z1 - coordinates of mesh vertices
    ve - vertical exaggeration
    triangles - indices of triangles used in mesh
    vertices - coordinates of the vertices
    scalars - scalars used for coloring the mesh in 'property' mode (= z-value of the base of current layer)
    cmap - colormap used for layers in 'time' mode
    norm - color normalization function used in 'time' mode
    export - if equals 1, the display can be saved as a VRML file for use in other programs (e.g., 3D printing)
    """
    if color_mode == 'time':
        cmap = matplotlib.cm.get_cmap(colormap)
        mlab.triangular_mesh(X1, Y1, ve*Z1, triangles, color = cmap(norm(layer_n))[:3], opacity = opacity)
    if color_mode == 'property': # color based on property map
        mlab.triangular_mesh(X1, Y1, ve*Z1, triangles, scalars=scalars, colormap=str(colormap), vmin=vmin, vmax=vmax, opacity = opacity)
    if color_mode == 'facies':
        mlab.triangular_mesh(X1,Y1,ve*Z1, triangles, color=tuple(colors[int(facies[0, 0, layer_n])]), opacity = opacity)

def create_random_section_2_points(strat,facies,scale,ve,color_mode,colors,colormap,x1,x2,y1,y2,s1,dx,bottom,export,opacity):
    r, c, ts = np.shape(strat)
    dist = dx*((x2-x1)**2 + (y2-y1)**2)**0.5
    s2 = s1*dx+dist
    num = int(dist/float(dx))
    cmap = matplotlib.cm.get_cmap(colormap)
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=ts-1)
    Xrand, Yrand, Srand = np.linspace(x1,x2,num), np.linspace(y1,y2,num), np.linspace(s1*dx,s2,num)
    base = map_coordinates(strat[:,:,0], np.vstack((Yrand,Xrand)))
    vertices, triangles = create_section(base,dx,bottom) 
    gray = (0.6,0.6,0.6) # color for plotting basal part of panel
    mlab.triangular_mesh(scale*np.hstack((dx*Xrand,dx*Xrand[::-1])),scale*np.hstack((dx*Yrand,dx*Yrand[::-1])),scale*ve*vertices[:,1],triangles,color=gray)
    for layer_n in trange(0,ts-1):
        top = map_coordinates(strat[:,:,layer_n+1], np.vstack((Yrand,Xrand)))
        base = map_coordinates(strat[:,:,layer_n], np.vstack((Yrand,Xrand)))
        if np.max(top-base)>1e-6:
            Points, Inds = triangulate_layers(top,base,dx)
            for i in range(len(Points)):
                vertices = Points[i]
                inds = Inds[i]
                triangles, scalars = create_triangles(vertices)
                X1 = scale*dx*Xrand[inds]
                Y1 = scale*dx*Yrand[inds]
                Z1 = scale*vertices[:,1]
                mlab.plot3d(X1,Y1,Z1*ve,color=(0,0,0),tube_radius=0.5)
                vmin = 0; vmax = 1
                plot_layers_on_one_side(layer_n,facies,color_mode,colors,X1,Y1,Z1,ve,triangles,vertices,scalars,colormap,norm,vmin,vmax,export,opacity)
        
def create_random_section_n_points(strat,facies,topo,scale,ve,color_mode,colors,colormap,x1,x2,y1,y2,dx,bottom,export,opacity):
    r, c, ts = np.shape(strat)
    if len(x1)==1:
        create_random_section_2_points(strat,facies,scale,ve,color_mode,colors,colormap,x1,x2,y1,y2,0,dx,bottom,export,opacity)
    else:
        count = 0
        dx1,dy1,ds1,s1 = compute_derivatives(x1,y1)
        for i in range(len(x1)):
            create_random_section_2_points(strat,facies,scale,ve,color_mode,colors,colormap,x1[i],x2[i],y1[i],y2[i],s1[i],dx,bottom,export,opacity)
            count = count+1
            # print("panel "+str(count)+" done, out of "+str(len(x1))+" panels")

def create_random_cookie(strat,facies,topo,scale,ve,color_mode,colors,colormap,x1,x2,y1,y2,dx,bottom,export,opacity):
    r, c, ts = np.shape(strat)
    count = 0
    dx1,dy1,ds1,s1 = compute_derivatives(x1,y1)
    for i in range(len(x1)):
        create_random_section_2_points(strat,facies,scale,ve,color_mode,colors,colormap,x1[i],x2[i],y1[i],y2[i],s1[i],dx,bottom,export,opacity)
        count = count+1
        # print("panel "+str(count)+" done, out of "+str(len(x1)+1)+" panels")
    create_random_section_2_points(strat,facies,scale,ve,color_mode,colors,colormap,x2[-1],x1[0],y2[-1],y1[0],s1[-1]+np.sqrt((x1[0]-x2[-1])**2+(y1[0]-y2[-1])**2),dx,bottom,export,opacity)
    polygon = []
    for i in range(len(x1)):
        polygon.append((x1[i]+0.5, y1[i]+0.5))
    polygon.append((x2[-1]+0.5, y2[-1]+0.5))
    img = Image.fromarray(np.zeros(np.shape(strat[:,:,-1])))
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    img = np.array(img)
    mask = np.ones_like(strat[:,:,-1]).astype(bool)
    mask[img == 1] = False
    r,c = np.shape(strat[:,:,-1])
    Y1 = scale*(np.linspace(0,r-1,r)*dx)
    X1 = scale*(np.linspace(0,c-1,c)*dx)
    topo_min = np.min(strat[:,:,-1])
    topo_max = np.max(strat[:,:,-1])
    mlab.surf(X1, Y1, scale*strat[:,:,-1].T, mask=mask.T, warp_scale=ve, colormap='gist_earth', vmin=scale*topo_min, vmax=scale*topo_max)
        
def compute_derivatives(x,y):
    dx = np.diff(x) # first derivatives
    dy = np.diff(y)   
    ds = np.sqrt(dx**2+dy**2)
    s = np.hstack((0,np.cumsum(ds)))
    return dx, dy, ds, s

class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()

def select_random_section(strat):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax.imshow(strat[:,:,-1],cmap='viridis')
    plt.tight_layout()
    ax.set_title('click to build line segments')
    line, = ax.plot([], [])  # empty line
    linebuilder = LineBuilder(line)
    xcoords = linebuilder.xs
    ycoords = linebuilder.ys
    return xcoords, ycoords

def plot_strat_diagram(time, elevation, time_units, elev_units, end_time, max_elevation):
    fig = plt.figure(figsize=(9,6))
    ax1 = fig.add_axes([0.07, 0.08, 0.85, 0.76]) # [left, bottom, width, height]
    ax1.set_xlabel('time (' + time_units + ')', fontsize = 12)
    ax1.set_ylabel('elevation (' + elev_units + ')', fontsize = 12)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    ax2 = fig.add_axes([0.92, 0.08, 0.05, 0.76])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3 = fig.add_axes([0.07, 0.84, 0.85, 0.08])
    ax3.set_yticks([])
    ax3.set_xticks([])
    ax1.set_xlim(0, end_time)
    elev_range = max_elevation - np.min(elevation)
    ylim1 = np.min(elevation)# - 0.02 * elev_range
    ylim2 = max_elevation + 0.02 * elev_range
    ax1.set_ylim(ylim1, ylim2)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(ylim1, ylim2)
    ax3.set_ylim(0, 1)
    ax3.set_xlim(0, end_time)
    ax4 = fig.add_axes([0.07, 0.92, 0.6, 0.08])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 1)

    ax1.plot(time, elevation, 'xkcd:medium blue', linewidth = 3)

    strat = np.minimum.accumulate(elevation[::-1])[::-1] # stratigraphic 'elevation'

    unconf_inds = np.where(strat != elevation)[0] # indices where 'strat' curve is different from elevation
    inds = np.where(np.diff(unconf_inds)>1)[0] # indices where deposition starts again, after erosion
    inds = np.hstack((inds, len(unconf_inds)-1)) # add last index

    if strat[-1] - strat[-2] == 0:
        inds = np.hstack((inds, len(unconf_inds)-1))

    if len(unconf_inds) > 0:
        strat_tops = strat[unconf_inds[inds]+1] # stratigraphic tops
    else:
        strat_tops = []
    strat_top_ages = [] # ages of the stratigraphic tops
    for i in range(len(strat_tops)): # generate list of ages of stratigraphic tops
        strat_top_ages.append(np.min(time[strat >= strat_tops[i]])) 

    loc_max_elev = signal.find_peaks(elevation)[0]
    loc_min_elev = signal.find_peaks(-elevation)[0]
    if elevation[-1] < elevation[-2]:
        loc_min_elev = np.hstack((loc_min_elev, len(elevation)-1))

    if (len(loc_min_elev) > 0) & (len(loc_max_elev) > 0):
        if elevation[1] < elevation[0]: # add first point as a local maximum elevation if the series starts out erosionally
            loc_max_elev = np.hstack((0, loc_max_elev))
        if loc_min_elev[0] < loc_max_elev[0]:
            ind = np.argmax(elevation[0 : loc_min_elev[0]])
            loc_max_elev = np.sort(np.hstack((loc_max_elev, ind)))
        for i in range(len(loc_min_elev)-1):
            if len(loc_max_elev[loc_max_elev > loc_min_elev[i]]) > 0:
                if np.min(loc_max_elev[loc_max_elev > loc_min_elev[i]]) > loc_min_elev[i+1]:
                    ind = np.argmax(elevation[loc_min_elev[i] : loc_min_elev[i+1]])
                    ind = loc_min_elev[i] + ind
                    loc_max_elev = np.sort(np.hstack((loc_max_elev, ind)))
            else:
                ind = np.argmax(elevation[loc_min_elev[i] : loc_min_elev[i+1]])
                ind = loc_min_elev[i] + ind
                loc_max_elev = np.sort(np.hstack((loc_max_elev, ind)))
        for i in range(len(loc_max_elev)-1):
            if len(loc_min_elev[loc_min_elev > loc_max_elev[i]]) > 0:
                if np.min(loc_min_elev[loc_min_elev > loc_max_elev[i]]) > loc_max_elev[i+1]:
                    ind = np.argmin(elevation[loc_max_elev[i] : loc_max_elev[i+1]])
                    ind = loc_max_elev[i] + ind
                    loc_min_elev = np.sort(np.hstack((loc_min_elev, ind)))
            else:
                ind = np.argmin(elevation[loc_max_elev[i] : loc_max_elev[i+1]])
                ind = loc_max_elev[i] + ind
                loc_min_elev = np.sort(np.hstack((loc_min_elev, ind)))
    erosion_start_times = time[loc_max_elev] # times when erosion starts
    erosion_end_times = time[loc_min_elev] # times when erosion ends
    erosion_start_elevations = elevation[loc_max_elev] # elevations when erosion starts
    erosion_end_elevations = elevation[loc_min_elev] # elevations when erosion ends

    if (len(loc_min_elev) > 0) & (len(loc_max_elev) > 0):
        for i in range(len(erosion_end_times)): # plot erosional segments
            ax1.plot(time[loc_max_elev[i]:loc_min_elev[i]+1], 
                     elevation[loc_max_elev[i]:loc_min_elev[i]+1], 'xkcd:red', linewidth=3)
        if len(erosion_start_times) > len(erosion_end_times): # plot last erosional segment (if needed)
            ax1.plot(time[loc_max_elev[-1]:], 
                     elevation[loc_max_elev[-1]:], 'xkcd:red', linewidth=3)

    strat_top_labels = ['s' for strat_top in strat_tops] # labels for stratigraphic tops
    erosion_start_labels = ['es' for erosion_start_time in erosion_start_times] # labels for start of erosion
    erosion_end_labels = ['ee' for erosion_end_time in erosion_end_times] # labels for end of erosion

    time_bounds = np.hstack((strat_top_ages, erosion_start_times, erosion_end_times)) # all time boundaries
    sort_inds = np.argsort(time_bounds) # indices for sorting
    time_bounds = time_bounds[sort_inds] # sort time boundaries
    elevation_bounds = np.hstack((strat_tops, erosion_start_elevations, erosion_end_elevations)) # all elevation boundaries
    elevation_bounds = elevation_bounds[sort_inds] # sort elevation boundaries
    bound_labels = np.hstack((strat_top_labels, erosion_start_labels, erosion_end_labels)) # all boundary labels
    bound_labels = bound_labels[sort_inds] # sort boundary labels

    time_bounds = np.hstack((time[0], time_bounds, time[-1])) # add first and last time step to time boundaries
    elevation_bounds = np.hstack((elevation[0], elevation_bounds, elevation[-1])) # add first and last elevation values
    if elevation[-1] - elevation[-2] < 0: # add first and last boundary labels
        bound_labels = np.hstack(('s', bound_labels, 'ee'))
    else:
        bound_labels = np.hstack(('s', bound_labels, 's'))
        
    inds = []
    for i in range(len(bound_labels)-1):
        if (bound_labels[i] == 'es') & (bound_labels[i+1] == 's'):
            inds.append(i)
    if len(inds)>0:
        for i in range(len(inds)):
            bound_labels[inds[i]] = 's'
            bound_labels[inds[i]+1] = 'es'
        
    time_labels = []
    for i in range(len(time_bounds)-1): # plot chronostratigraphic units
        x = [time_bounds[i], time_bounds[i+1], time_bounds[i+1], time_bounds[i]]
        y = [0, 0, 1, 1]
        if (bound_labels[i] == 's') and (bound_labels[i+1] == 'es'): # vacuity
            ax3.fill(x, y, facecolor='xkcd:light grey', edgecolor='k')
            time_labels.append('v')
        elif (bound_labels[i] == 'ee') and (bound_labels[i+1] == 'es'): # vacuity 
            ax3.fill(x, y, facecolor='xkcd:light grey', edgecolor='k')
            time_labels.append('v')
        elif (bound_labels[i] == 'es') and (bound_labels[i+1] == 'ee'): # erosion 
            ax3.fill(x, y, facecolor='xkcd:red', edgecolor='k')
            time_labels.append('e')
        elif (bound_labels[i] == 's') and (bound_labels[i+1] == 'ee'): # erosion 
            time_labels.append('e')
        elif (bound_labels[i] == 'ee') and (bound_labels[i+1] == 's'): # deposition 
            ax3.fill(x, y, facecolor='xkcd:medium blue', edgecolor='k')
            time_labels.append('d')
        elif (bound_labels[i] == 's') and (bound_labels[i+1] == 's'): # deposition 
            ax3.fill(x, y, facecolor='xkcd:medium blue', edgecolor='k')
            time_labels.append('d')
        ax1.plot([time_bounds[i], time_bounds[i]], [elevation_bounds[i], max_elevation + 0.02 * elev_range], 'k--', linewidth=0.5)

    for i in range(len(strat_tops)):
        ax2.plot([0, 1], [strat_tops[i], strat_tops[i]], color = 'xkcd:red', linewidth = 3)

    if len(strat_tops) > 0:
        if elevation[0] < np.min(strat_tops):
            strat_tops = np.hstack((elevation[0], strat_tops, elevation[-1]))
            strat_top_ages = np.hstack((0, strat_top_ages, time[-1]))
        else:
            strat_tops = np.hstack((strat_tops, elevation[-1]))
            strat_top_ages = np.hstack((strat_top_ages, time[-1]))
    else:
        strat_tops = np.hstack((elevation[0], strat_tops, elevation[-1]))
        strat_top_ages = np.hstack((0, strat_top_ages, time[-1]))

    
    for i in range(len(strat_tops)-1): # plot stratigraphic units
        x = [0, 1, 1, 0]
        y = [strat_tops[i], strat_tops[i], strat_tops[i+1], strat_tops[i+1]]
        ax2.fill(x, y, facecolor='xkcd:medium blue', edgecolor='k')
        if i > 0:
            ax1.plot([strat_top_ages[i], end_time], [strat_tops[i], strat_tops[i]], 'k--', linewidth=0.5)
            
    times = np.diff(time_bounds)
    thicknesses = np.diff(elevation_bounds)
    deposition_time = np.sum([item[0] for item in zip(times, time_labels) if item[1] == 'd' ])
    vacuity_time = np.sum([item[0] for item in zip(times, time_labels) if item[1] == 'v' ])
    erosion_time = np.sum([item[0] for item in zip(times, time_labels) if item[1] == 'e' ])
    deposition_thickness = np.sum([item[0] for item in zip(thicknesses, time_labels) if item[1] == 'd' ])
    vacuity_thickness = np.sum([item[0] for item in zip(thicknesses, time_labels) if item[1] == 'v' ])
    eroded_thickness = np.sum([item[0] for item in zip(thicknesses, time_labels) if item[1] == 'e' ])
    dve_data = [deposition_time, vacuity_time, erosion_time, deposition_thickness, vacuity_thickness, eroded_thickness]
            
    y1 = 0.55
    y2 = 0.15
    y = [y1, y1, y2, y2]
    x1 = 0
    x2 = 3 * deposition_time/time[-1]
    x = [x1, x2, x2, x1]
    ax4.fill(x, y, facecolor='xkcd:medium blue', edgecolor = 'k', zorder = 1000)
    ax4.axis('off')
    ax4.text(x1, y1 + 0.07, 'deposition', fontsize = 12)
    ax4.text(x1 + 0.05, 0.27, str(np.round(deposition_time/time[-1], 3)), fontsize = 10, color = 'w',zorder=2000)
    
    x1 = 3 
    x2 = x1 + 3 * erosion_time/time[-1]
    x = [x1, x2, x2, x1]
    ax4.fill(x, y, facecolor='xkcd:red', edgecolor = 'k', zorder = 1001)
    ax4.text(x1, y1 + 0.07, 'erosion', fontsize = 12)
    ax4.text(x1 + 0.05, 0.27, str(np.round(erosion_time/time[-1], 3)), fontsize = 10, color = 'w',zorder=2000)
    
    x1 = 6 
    x2 = x1 + 3 * vacuity_time/time[-1]
    x = [x1, x2, x2, x1]
    ax4.fill(x, y, facecolor='xkcd:light grey', edgecolor = 'k', zorder = 1002)
    ax4.text(x1, y1 + 0.07, 'vacuity', fontsize = 12)
    ax4.text(x1 + 0.05, 0.27, str(np.round(vacuity_time/time[-1], 3)), fontsize = 10, color = 'w',zorder=2000)
                
    return fig

def topostrat(topo):
    # convert topography to stratigraphy
    if len(np.shape(topo)) == 2:
        strat = np.minimum.accumulate(topo[::-1, :], axis=0)[::-1, :]
    if len(np.shape(topo)) == 3:
        strat = np.minimum.accumulate(topo[:, :, ::-1], axis=2)[:, :, ::-1]
    return strat

def create_wheeler_diagram(topo):
    """create Wheeler (chronostratigraphic) diagram from a set of topographic surfaces
    """
    strat = topostrat(topo) # convert topography to stratigraphy
    wheeler = np.diff(topo, axis=2) # 'normal' Wheeler diagram
    wheeler_strat = np.diff(strat, axis=2) # array for Wheeler diagram with vacuity blanked out; this array will be a positive number if there is preserved depostion, zero otherwise
    vacuity = np.zeros(np.shape(wheeler)) # array for vacuity 
    vacuity[(wheeler>0) & (wheeler_strat==0)] = 1 # make the 'vacuity' array 1 where there was deposition (wheeler > 0) but stratigraphy is not preserved (wheeler_strat = 0)
    wheeler_strat[wheeler<0] = wheeler[wheeler<0] # add erosion to 'wheeler_strat' (otherwise it would only show deposition)
    return strat, wheeler, wheeler_strat, vacuity

def plot_model_cross_section_EW(strat, prop, facies, dx, xsec, color_mode, line_freq = 1, ve = False, map_aspect = 1, flattening_ind = False, units = 'm', list_of_colors = ['lemonchiffon', 'peru', 'sienna']):
    """Plots an E-W oriented cross section through a stratigraphic model

    :param WG: well graph
    :param strat: stratigraphic grid 
    :param prop: property array 
    :param facies: facies array
    :param dx: gridcell size in the x- and y directions
    :param xsec: index of cross section to be displayed
    :param color_mode: determines what kind of plot is created; can be 'property' or 'facies'
    :param flattening_ind: index of stratigraphic top that should be used for flattening; default is 'False' (= no flattening)
    :param ve: vertical exaggeration; default is 'False'
    :param units: units used in the model
    :param map_aspect: the aspect ratio of the inset map that shows the location of the cross section
    :param list_of_colors: list of named matplotlib colors that will be used when 'color_mode' is set to 'facies'

    :return fig: figure handle
    """
    fig = plt.figure(figsize = (10, 6))
    ax = fig.add_subplot(111)
    axin = ax.inset_axes([0.03, 0.03, 0.3, 0.3])
    r,c,ts = np.shape(strat)
    for i in trange(0, ts-1):
        if flattening_ind:
            top = (strat[xsec, :, i] - strat[xsec, :, flattening_ind]) 
            base = (strat[xsec, :, i+1] - strat[xsec, :, flattening_ind])
        else:
            top = strat[xsec, :, i] 
            base = strat[xsec, :, i+1]
        props = prop[xsec, :, i]
        faciess = facies[xsec, :, i]
        if np.max(base - top)>0:
            Points, Inds = triangulate_layers(base,top,dx)
            for j in range(len(Points)):
                vertices = Points[j]
                triangles, scalars = create_triangles(vertices)
                x = vertices[:,0]
                y = vertices[:,1]
                if color_mode == 'property':
                    colors = props[Inds[j]]
                    colors = np.mean(colors[np.array(triangles)], axis = 1)
                    ax.tripcolor(x, y, triangles=triangles, facecolors = colors, cmap = 'YlOrBr_r', 
                                  edgecolors = 'none', vmin = 0, vmax = 0.35)
                if color_mode == 'facies':
                    colors = faciess[Inds[j]]
                    colors = np.median(colors[np.array(triangles)], axis = 1)
                    cmap = ListedColormap(list_of_colors)
                    ax.tripcolor(x, y, triangles=triangles, facecolors = colors, edgecolors = 'none', cmap = cmap, vmin = 0, vmax = len(list_of_colors))
            if np.mod(i, line_freq) == 0:
                ax.plot(np.arange(0, dx*c, dx), top, 'k', linewidth = 0.25)
            if i == ts-2:
                ax.plot(np.arange(0, dx*c, dx), base, 'k', linewidth = 0.5)
    ax.set_xlim(0, dx*(c-1))
    if flattening_ind:
        ax.set_ylim(np.nanmin(strat[:,:,0] - strat[:, :, flattening_ind]), 
                    np.nanmax(strat[:,:,-1] - strat[:, :, flattening_ind]))
    else:
        ax.set_ylim(np.nanmin(strat), np.nanmax(strat))
    ax.set_xlabel('distance (' + units + ')')
    ax.set_ylabel('depth (' + units + ')')
    axin.imshow(strat[:, :, -1], cmap='viridis', aspect = map_aspect)
    axin.set_xticks([])
    axin.set_yticks([])
    axin.plot([0, c-1], [xsec, xsec], 'k')
    # axin.set_aspect('equal')
    if ve:
        ax.set_aspect(ve, adjustable='datalim')
    # plt.tight_layout()
    return fig

def plot_model_cross_section_NS(strat, prop, facies, dx, xsec, color_mode, line_freq = 1, ve = False, flattening_ind = False, units = 'm', map_aspect = 1, list_of_colors = ['lemonchiffon', 'peru', 'sienna']):
    """Plots an E-W oriented cross section through a stratigraphic model

    :param WG: well graph
    :param strat: stratigraphic grid 
    :param prop: property array 
    :param facies: facies array
    :param dx: gridcell size in the x- and y directions
    :param xsec: index of cross section to be displayed
    :param color_mode: determines what kind of plot is created; can be 'property' or 'facies'
    :param flattening_ind: index of stratigraphic top that should be used for flattening; default is 'False' (= no flattening)
    :param units: units used in the model
    :param map_aspect: the aspect ratio of the inset map that shows the location of the cross section
    :param list_of_colors: list of named matplotlib colors that will be used when 'color_mode' is set to 'facies'

    :return fig: figure handle
    """
    fig = plt.figure(figsize = (10, 6))
    ax = fig.add_subplot(111)
    axin = ax.inset_axes([0.03, 0.03, 0.3, 0.3])
    r,c,ts = np.shape(strat)
    for i in trange(0, ts-1):
        if flattening_ind:
            top = (strat[:, xsec, i] - strat[:, xsec, flattening_ind]) 
            base = (strat[:, xsec, i+1] - strat[:, xsec, flattening_ind])
        else:
            top = strat[:, xsec, i] 
            base = strat[:, xsec, i+1]
        props = prop[:, xsec, i]
        faciess = facies[:, xsec, i]
        if np.max(base - top)>0:
            Points, Inds = triangulate_layers(base,top,dx)
            for j in range(len(Points)):
                vertices = Points[j]
                triangles, scalars = create_triangles(vertices)
                x = vertices[:,0]
                y = vertices[:,1]
                if color_mode == 'property':
                    colors = props[Inds[j]]
                    colors = np.mean(colors[np.array(triangles)], axis = 1)
                    ax.tripcolor(x, y, triangles=triangles, facecolors = colors, cmap = 'YlOrBr_r', 
                                  edgecolors = 'none', vmin = 0, vmax = 0.35)
                if color_mode == 'facies':
                    colors = faciess[Inds[j]]
                    colors = np.median(colors[np.array(triangles)], axis = 1)
                    cmap = ListedColormap(list_of_colors)
                    ax.tripcolor(x, y, triangles=triangles, facecolors = colors, edgecolors = 'none', cmap = cmap, vmin = 0, vmax = len(list_of_colors))
            if np.mod(i, line_freq) == 0:
                ax.plot(np.arange(0, dx*r, dx), top, 'k', linewidth = 0.5)
            if i == ts-2:
                ax.plot(np.arange(0, dx*r, dx), base, 'k', linewidth = 0.5)
    ax.set_xlim(0, dx*(r-1))
    if flattening_ind:
        ax.set_ylim(np.nanmin(strat[:,:,0] - strat[:, :, flattening_ind]), 
                    np.nanmax(strat[:,:,-1] - strat[:, :, flattening_ind]))
    else:
        ax.set_ylim(np.nanmin(strat), np.nanmax(strat))
    ax.set_xlabel('distance (' + units + ')')
    ax.set_ylabel('depth (' + units + ')')
    axin.imshow(strat[:, :, -1], cmap='viridis', aspect = map_aspect)
    axin.set_xticks([])
    axin.set_yticks([])
    axin.plot([xsec, xsec], [0, r-1],  'k')
    # plt.tight_layout()
    if ve:
        ax.set_aspect(ve, adjustable='datalim')
    return fig

def resample_elevation_spl(time, elevation, sampling_rate):
    spl = interpolate.splrep(time, elevation, s=0.5)
    time_new = np.arange(time[0], time[-1]+1, sampling_rate)
    elevation_new = interpolate.splev(time_new, spl)
    return time_new, elevation_new
def resample_elevation_int1d(time, elevation, sampling_rate):
    f = interpolate.interp1d(time, elevation)
    time_new = np.arange(time[0], time[-1]+1, sampling_rate)
    elevation_new = f(time_new)
    return time_new, elevation_new