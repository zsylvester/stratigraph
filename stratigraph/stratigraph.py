import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
from mayavi import mlab
from scipy.ndimage import map_coordinates
from scipy import signal, interpolate
from scipy.spatial import distance
from PIL import Image, ImageDraw
from tqdm import tqdm, trange
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import split
from skimage.transform import resize
import scipy
from sklearn.cluster import KMeans
import math

def create_block_diagram(strat, dx, ve, color_mode, bottom, opacity, texture=None, sea_level=None, 
    xoffset=0, yoffset=0, scale=1, ci=None, plot_contours=None, topo_min=None, topo_max=None, plot_sides=True, 
    plot_water=False, plot_surf=True, surf_cmap='Blues', kmeans_colors=None):
    """function for creating a 3D block diagram in Mayavi
    topo - 
    strat - input array with stratigraphic surfaces
    dx - size of gridcells in the horizontal direction in 'strat'
    ve - vertical exaggeration
    color_mode
    bottom - elevation value for the bottom of the block
    opacity
    texture
    sea_level
    xoffset - offset in the x-direction relative to 0
    yoffset
    scale - scaling factor
    ci - contour interval
    plot_contours - True if you want to plot contours on the top surface
    topo_min
    topo_max
    plot_sides
    plot_water
    plot_surf
    surf_cmap
    kmeans_colors
    """

    r,c,ts = np.shape(strat)

    # if z is increasing downward:
    if np.max(strat[:, :, -1] - strat[:, :, 0]) < 0:
        strat = -1 * strat

    z = scale*strat[:,:,ts-1]

    X1 = scale*(xoffset + np.linspace(0,c-1,c)*dx) # x goes with c and y with r
    Y1 = scale*(yoffset + np.linspace(0,r-1,r)*dx)
    X1_grid , Y1_grid = np.meshgrid(X1, Y1)

    if plot_contours:
        vmin = scale * topo_min
        vmax = scale * topo_max
        contours = list(np.arange(vmin, vmax, ci*scale)) # list of contour values
        mlab.contour_surf(X1, Y1, z, contours=contours, warp_scale=ve, color=(0,0,0), line_width=1.0)
    
    if plot_surf:
        if texture is not None: 
            surf = mlab.mesh(X1_grid, Y1_grid, ve*z, scalars = texture, colormap=surf_cmap, vmin=0, vmax=255) 
            if kmeans_colors is not None:
                lut = surf.module_manager.scalar_lut_manager.lut.table.to_array()
                # lut = lut[::-1,:]
                lut[:,:3] = kmeans_colors
                lut[:, -1] = opacity * 255
                surf.module_manager.scalar_lut_manager.lut.table = lut
        else:
            surf = mlab.mesh(X1_grid, Y1_grid, ve*z, colormap=surf_cmap, vmin=topo_min, vmax=topo_max, opacity=opacity)

    if plot_sides:
        gray = (0.6,0.6,0.6) # color for plotting sides
        z1 = strat[:,:,0].copy()

        r,c,ts = np.shape(strat)
        # updip side:
        vertices, triangles = create_section(z1[:,0],dx,bottom) 
        y = scale*(yoffset + vertices[:,0])
        x = scale*(xoffset + np.zeros(np.shape(vertices[:,0])))
        z = scale*ve*vertices[:,1]
        mlab.triangular_mesh(x, y, z, triangles, color=gray, opacity = 1)

        # downdip side:
        vertices, triangles = create_section(z1[:,-1],dx,bottom) 
        y = scale*(yoffset + vertices[:,0])
        x = scale*(xoffset + (c-1)*dx*np.ones(np.shape(vertices[:,0])))
        z = scale*ve*vertices[:,1]
        mlab.triangular_mesh(x, y, z, triangles, color=gray, opacity = 1)

        # left edge (looking downdip):
        vertices, triangles = create_section(z1[0,:],dx,bottom) 
        y = scale*(yoffset + np.zeros(np.shape(vertices[:,0])))
        x = scale*(xoffset + vertices[:,0])
        z = scale*ve*vertices[:,1]
        mlab.triangular_mesh(x, y, z, triangles, color=gray, opacity = 1)

        # right edge (looking downdip):
        vertices, triangles = create_section(z1[-1,:],dx,bottom) 
        y = scale*(yoffset + (r-1)*dx*np.ones(np.shape(vertices[:,0])))
        x = scale*(xoffset + vertices[:,0])
        z = scale*ve*vertices[:,1]
        mlab.triangular_mesh(x, y, z, triangles, color=gray, opacity = 1)

        # bottom face of block:
        vertices = dx*np.array([[0,0],[r-1,0],[r-1,c-1],[0,c-1]])
        triangles = [[0,1,3],[1,3,2]]
        y = scale*(yoffset + vertices[:,0])
        x = scale*(xoffset + vertices[:,1])
        z = scale*bottom*np.ones(np.shape(vertices[:,0]))
        mlab.triangular_mesh(x, y, ve*z, triangles, color=gray, opacity = 1)


    if plot_water:
        blue = (0.255, 0.412, 0.882)
        base = strat[:, 0, -1]
        top = sea_level[ts-1] * np.ones(np.shape(base)) # updip side
        if np.max(top-base)>0:
            Points,Inds = triangulate_layers(top,base,dx)
            for i in range(len(Points)):
                vertices = Points[i]
                triangles, scalars = create_triangles(vertices)
                Y1 = scale*(yoffset + vertices[:,0])
                X1 = scale*(xoffset + dx*0*np.ones(np.shape(vertices[:,0])))
                Z1 = scale*vertices[:,1]
                mlab.triangular_mesh(X1, Y1, ve*Z1, triangles, color = blue, opacity = 0.3)

        base = strat[:, -1, -1]
        top = sea_level[ts-1] * np.ones(np.shape(base)) # downdip side
        if np.max(top-base)>0:
            Points,Inds = triangulate_layers(top,base,dx)
            for i in range(len(Points)):
                vertices = Points[i]
                triangles, scalars = create_triangles(vertices)
                Y1 = scale*(yoffset + vertices[:,0])
                X1 = scale*(xoffset + dx*(c-1)*np.ones(np.shape(vertices[:,0])))
                Z1 = scale*vertices[:,1]
                mlab.triangular_mesh(X1, Y1, ve*Z1, triangles, color = blue, opacity = 0.3)

        base = strat[-1, :, -1]
        top = sea_level[ts-1] * np.ones(np.shape(base)) # right edge (looking downdip)
        if np.max(top-base)>0:
            Points,Inds = triangulate_layers(top,base,dx)
            for i in range(len(Points)):
                vertices = Points[i]
                triangles, scalars = create_triangles(vertices)
                Y1 = scale*(yoffset + dx*(r-1)*np.ones(np.shape(vertices[:,0])))
                X1 = scale*(xoffset + vertices[:,0])
                Z1 = scale*vertices[:,1]
                mlab.triangular_mesh(X1, Y1, ve*Z1, triangles, color = blue, opacity = 0.3)

        base = strat[0, :, -1]
        top = sea_level[ts-1] * np.ones(np.shape(base)) # left edge (looking downdip)
        if np.max(top-base)>0:
            Points,Inds = triangulate_layers(top,base,dx)
            for i in range(len(Points)):
                vertices = Points[i]
                triangles, scalars = create_triangles(vertices)
                Y1 = scale*(yoffset + dx*0*np.ones(np.shape(vertices[:,0])))
                X1 = scale*(xoffset + vertices[:,0])
                Z1 = scale*vertices[:,1]
                mlab.triangular_mesh(X1, Y1, ve*Z1, triangles, color = blue, opacity = 0.3)

        X1 = scale*(xoffset + np.linspace(0,c-1,c)*dx) # x goes with c and y with r
        Y1 = scale*(yoffset + np.linspace(0,r-1,r)*dx)
        water_surf = np.ones(np.shape(strat[:, :, -1])) * sea_level[ts-1]
        water_surf[water_surf < strat[:, :, -1]] = np.nan
        water_surf = water_surf.T
        mlab.surf(X1, Y1, water_surf*scale, warp_scale=ve, color=blue, opacity=0.3)

def create_exploded_view(topo, strat, nx, ny, gap, dx, ve, color_mode, linewidth, bottom, opacity, x0=0, y0=0, water_depth=100, subsid=None, prop=None, 
    prop_cmap=None, prop_vmin=None, prop_vmax=None, facies=None, facies_colors=None, texture=None, sea_level=None, 
    xoffset=0, yoffset=0, scale=1, ci=None, plot_contours=None, topo_min=None, topo_max=None, plot_sides=True, 
    plot_water=False, plot_surf=True, surf_cmap='Blues', kmeans_colors=None, line_freq=1):
    """function for creating an exploded-view block diagram
    inputs:
    topo - stack of topographic surfaces
    strat - stack of stratigraphic surfaces
    prop
    facies - 1D array of facies codes for layers
    texture
    sea_level
    x0
    y0
    nx - number of blocks in x direction
    ny - number of blocks in y direction
    gap - gap between blocks (number of gridcells)
    dx - gridcell size
    ve - vertical exaggeration
    scale - scaling factor (for whole model)
    plot_contours - if equals 1, contours will be plotted on the top surface
    plot_sides
    plot_water
    color_mode - determines what kind of plot is created; can be 'property', 'time', or 'facies'
    colors - colors scheme for facies (list of RGB values)
    cmap
    linewidth - tube radius for plotting layers on the sides
    bottom - elevation value for the bottom of the block
    topo_min
    topo_max
    ci
    opacity
    """
    r,c,ts = np.shape(strat)
    count = 0
    x_inds = []
    y_inds = []
    for i in range(nx):
        for j in range(ny):
            x1 = i * int(c/nx)
            x2 = (i+1) * int(c/nx)
            x_inds.append(x1)
            x_inds.append(x2)
            y1 = j * int(r/ny)
            y2 = (j+1) * int(r/ny)
            y_inds.append(y1)
            y_inds.append(y2)
            xoffset = x0 + (x1+i*gap)*dx
            yoffset = y0 + (y1+j*gap)*dx
            if texture is not None:
                create_block_diagram(strat[y1:y2,x1:x2,:], dx, ve, color_mode, bottom, opacity, texture=texture[y1:y2,x1:x2], sea_level=sea_level, 
                    xoffset=xoffset, yoffset=yoffset, scale=scale, ci=ci, plot_contours=plot_contours, topo_min=topo_min, topo_max=topo_max, plot_sides=plot_sides, 
                    plot_water=plot_water, plot_surf=plot_surf, surf_cmap='Blues', kmeans_colors=kmeans_colors)
            else:
                create_block_diagram(strat[y1:y2,x1:x2,:], dx, ve, color_mode, bottom, opacity, texture=None, sea_level=sea_level, 
                    xoffset=xoffset, yoffset=yoffset, scale=scale, ci=ci, plot_contours=plot_contours, topo_min=topo_min, topo_max=topo_max, plot_sides=plot_sides, 
                    plot_water=plot_water, plot_surf=plot_surf, surf_cmap=surf_cmap, kmeans_colors=kmeans_colors)

            if color_mode == 'bathymetry' or color_mode == 'sea_level' or color_mode == 'sea_level_change':
                plot_dip_section(topo[y1:y2, x1:x2, :], strat[y1:y2, x1:x2, :], dx, 0, ve, xoffset=xoffset, yoffset=yoffset, subsid = subsid, sea_level = sea_level, linewidth=linewidth, line_freq=line_freq, 
                    color_mode=color_mode, plot_type='3D', plot_erosion=False, plot_water=False, plot_basement=False, water_depth=water_depth)
                plot_dip_section(topo[y1:y2, x1:x2, :], strat[y1:y2, x1:x2, :], dx, y2-y1-1, ve, xoffset=xoffset, yoffset=yoffset, subsid = subsid, sea_level = sea_level, linewidth=linewidth, line_freq=line_freq, 
                    color_mode=color_mode, plot_type='3D', plot_erosion=False, plot_water=False, plot_basement=False, water_depth=water_depth)
                plot_strike_section(topo[y1:y2, x1:x2, :], strat[y1:y2, x1:x2, :], dx, 0, ve, xoffset=xoffset, yoffset=yoffset, subsid = subsid, sea_level = sea_level, linewidth=linewidth, line_freq=line_freq, 
                    color_mode=color_mode, plot_type='3D', plot_erosion=False, plot_water=False, plot_basement=False, water_depth=water_depth)
                plot_strike_section(topo[y1:y2, x1:x2, :], strat[y1:y2, x1:x2, :], dx, x2-x1-1, ve, xoffset=xoffset, yoffset=yoffset, subsid = subsid, sea_level = sea_level, linewidth=linewidth, line_freq=line_freq, 
                    color_mode=color_mode, plot_type='3D', plot_erosion=False, plot_water=False, plot_basement=False, water_depth=water_depth)

            if color_mode == 'property':
                plot_dip_section(topo[y1:y2, x1:x2, :], strat[y1:y2, x1:x2, :], dx, 0, ve, xoffset=xoffset, yoffset=yoffset, subsid = subsid, prop=prop[y1:y2, x1:x2, :], 
                    prop_cmap=prop_cmap, prop_vmin=prop_vmin, prop_vmax=prop_vmax, sea_level = sea_level, linewidth=linewidth, line_freq=line_freq, 
                    color_mode='property', plot_type='3D', plot_erosion=False, plot_water=False, plot_basement=False)
                plot_dip_section(topo[y1:y2, x1:x2, :], strat[y1:y2, x1:x2, :], dx, y2-y1-1, ve, xoffset=xoffset, yoffset=yoffset, subsid = subsid, prop=prop[y1:y2, x1:x2, :], 
                    prop_cmap=prop_cmap, prop_vmin=prop_vmin, prop_vmax=prop_vmax, sea_level = sea_level, linewidth=linewidth, line_freq=line_freq, 
                    color_mode='property', plot_type='3D', plot_erosion=False, plot_water=False, plot_basement=False)
                plot_strike_section(topo[y1:y2, x1:x2, :], strat[y1:y2, x1:x2, :], dx, 0, ve, xoffset=xoffset, yoffset=yoffset, subsid = subsid, prop=prop[y1:y2, x1:x2, :], 
                    prop_cmap=prop_cmap, prop_vmin=prop_vmin, prop_vmax=prop_vmax, sea_level = sea_level, linewidth=linewidth, line_freq=line_freq, 
                    color_mode='property', plot_type='3D', plot_erosion=False, plot_water=False, plot_basement=False)
                plot_strike_section(topo[y1:y2, x1:x2, :], strat[y1:y2, x1:x2, :], dx, x2-x1-1, ve, xoffset=xoffset, yoffset=yoffset, subsid = subsid, prop=prop[y1:y2, x1:x2, :], 
                    prop_cmap=prop_cmap, prop_vmin=prop_vmin, prop_vmax=prop_vmax, sea_level = sea_level, linewidth=linewidth, line_freq=line_freq, 
                    color_mode='property', plot_type='3D', plot_erosion=False, plot_water=False, plot_basement=False) 

            if color_mode == 'facies':
                plot_dip_section(topo[y1:y2, x1:x2, :], strat[y1:y2, x1:x2, :], dx, 0, ve, xoffset=xoffset, yoffset=yoffset, subsid = subsid, facies=facies[y1:y2, x1:x2, :], 
                    facies_colors = facies_colors, sea_level = sea_level, linewidth=linewidth, line_freq=line_freq, 
                    color_mode='facies', plot_type='3D', plot_erosion=False, plot_water=False, plot_basement=False)
                plot_dip_section(topo[y1:y2, x1:x2, :], strat[y1:y2, x1:x2, :], dx, y2-y1-1, ve, xoffset=xoffset, yoffset=yoffset, subsid = subsid, facies=facies[y1:y2, x1:x2, :], 
                    facies_colors = facies_colors, sea_level = sea_level, linewidth=linewidth, line_freq=line_freq, 
                    color_mode='facies', plot_type='3D', plot_erosion=False, plot_water=False, plot_basement=False)
                plot_strike_section(topo[y1:y2, x1:x2, :], strat[y1:y2, x1:x2, :], dx, 0, ve, xoffset=xoffset, yoffset=yoffset, subsid = subsid, facies=facies[y1:y2, x1:x2, :], 
                    facies_colors = facies_colors, sea_level = sea_level, linewidth=linewidth, line_freq=line_freq, 
                    color_mode='facies', plot_type='3D', plot_erosion=False, plot_water=False, plot_basement=False)
                plot_strike_section(topo[y1:y2, x1:x2, :], strat[y1:y2, x1:x2, :], dx, x2-x1-1, ve, xoffset=xoffset, yoffset=yoffset, subsid = subsid, facies=facies[y1:y2, x1:x2, :], 
                    facies_colors = facies_colors, sea_level = sea_level, linewidth=linewidth, line_freq=line_freq, 
                    color_mode='facies', plot_type='3D', plot_erosion=False, plot_water=False, plot_basement=False)

            if color_mode == 'age':
                plot_dip_section(topo[y1:y2, x1:x2, :], strat[y1:y2, x1:x2, :], dx, 0, ve, xoffset=xoffset, yoffset=yoffset, subsid = subsid, linewidth=linewidth, line_freq=line_freq, 
                    color_mode='age', plot_type='3D', plot_erosion=False, plot_water=False, plot_basement=False)
                plot_dip_section(topo[y1:y2, x1:x2, :], strat[y1:y2, x1:x2, :], dx, y2-y1-1, ve, xoffset=xoffset, yoffset=yoffset, subsid = subsid, linewidth=linewidth, line_freq=line_freq, 
                    color_mode='age', plot_type='3D', plot_erosion=False, plot_water=False, plot_basement=False)
                plot_strike_section(topo[y1:y2, x1:x2, :], strat[y1:y2, x1:x2, :], dx, 0, ve, xoffset=xoffset, yoffset=yoffset, subsid = subsid, linewidth=linewidth, line_freq=line_freq, 
                    color_mode='age', plot_type='3D', plot_erosion=False, plot_water=False, plot_basement=False)
                plot_strike_section(topo[y1:y2, x1:x2, :], strat[y1:y2, x1:x2, :], dx, x2-x1-1, ve, xoffset=xoffset, yoffset=yoffset, subsid = subsid, linewidth=linewidth, line_freq=line_freq, 
                    color_mode='age', plot_type='3D', plot_erosion=False, plot_water=False, plot_basement=False)


            count = count+1
            # print("block "+str(count)+" done, out of "+str(nx*ny)+" blocks")
    return x_inds, y_inds

def create_fence_diagram(topo, strat, nx, ny, dx, ve, color_mode, linewidth, bottom, opacity, subsid=None, prop=None, facies=None, texture=None, sea_level=None, scale=1, plot_sides=True, plot_water=False, plot_surf=False,  
    topo_min=None, topo_max=None, facies_colors=None, surf_cmap=None, prop_cmap=None, kmeans_colors=None, prop_vmin=None, prop_vmax=None, line_freq=1):
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
    color_mode - determines what kind of plot is created; can be 'property', 'time', or 'facies'
    facies_colors - colors scheme for facies (list of RGB values)
    line_thickness - - tube radius for plotting layers on the sides
    bottom - elevation value for the bottom of the block
    opacity
    """

    r,c,ts=np.shape(strat)
    
    create_block_diagram(strat, dx, ve, color_mode, bottom, opacity=opacity, texture=texture, sea_level=sea_level, 
        xoffset=0, yoffset=0, scale=scale, ci=None, plot_contours=None, topo_min=topo_min, topo_max=topo_max, plot_sides=plot_sides, 
        plot_water=plot_water, plot_surf=plot_surf, surf_cmap=surf_cmap, kmeans_colors=kmeans_colors)

    x_inds = np.hstack((0, int(c/(nx+1)) * np.arange(1, nx+1), c-1))
    for x1 in tqdm(x_inds): # strike sections        
        if color_mode == 'bathymetry':
            plot_strike_section(topo, strat, dx, x1, ve, subsid = subsid, sea_level = sea_level, linewidth=linewidth, line_freq=line_freq, 
                color_mode='bathymetry', plot_type='3D', plot_erosion=False, plot_water=False, plot_basement=False)
        if color_mode == 'facies':
            plot_strike_section(topo, strat, dx, x1, ve,  facies=facies, facies_colors=facies_colors, 
                linewidth=linewidth, line_freq=line_freq, color_mode='facies', plot_type='3D', plot_erosion=False, plot_water=False, plot_basement=False)
        if color_mode == 'property':
            plot_strike_section(topo, strat, dx, x1, ve,  prop=prop, prop_cmap=prop_cmap, prop_vmin=prop_vmin, prop_vmax=prop_vmax, 
                linewidth=linewidth, line_freq=line_freq, color_mode='property', plot_type='3D', plot_erosion=False, plot_water=False, plot_basement=False)

    y_inds = np.hstack((0, int(r/(ny+1)) * np.arange(1, ny+1), r-1))
    for y1 in tqdm(y_inds): # dip sections
        if color_mode == 'bathymetry':
            plot_dip_section(topo, strat, dx, y1, ve, subsid = subsid, sea_level = sea_level, linewidth=linewidth, line_freq=line_freq, 
                color_mode='bathymetry', plot_type='3D', plot_erosion=False, plot_water=False, plot_basement=False)
        if color_mode == 'facies':
            plot_dip_section(topo, strat, dx, y1, ve,  facies=facies, facies_colors=facies_colors, 
                linewidth=linewidth, line_freq=line_freq, color_mode='facies', plot_type='3D', plot_erosion=False, plot_water=False, plot_basement=False)
        if color_mode == 'property':
            plot_dip_section(topo, strat, dx, y1, ve,  prop=prop, prop_cmap=prop_cmap, prop_vmin=prop_vmin, prop_vmax=prop_vmax, 
                linewidth=linewidth, line_freq=line_freq, color_mode='property', plot_type='3D', plot_erosion=False, plot_water=False, plot_basement=False)

    return x_inds, y_inds

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

def triangulate_layers2(top,base,x):
    """function for creating vertices of polygons that describe one layer"""
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

def plot_layers_on_one_side(layer_n, facies, color_mode, colors, X1, Y1, Z1, ve, triangles, vertices, scalars, colormap, norm, vmin, vmax, opacity):
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
    norm - color normalization function used in 'time' mode"""
    if color_mode == 'time':
        cmap = matplotlib.cm.get_cmap(colormap)
        mlab.triangular_mesh(X1, Y1, ve*Z1, triangles, color = cmap(norm(layer_n))[:3], opacity = opacity)
    if color_mode == 'property': # color based on property map
        mlab.triangular_mesh(X1, Y1, ve*Z1, triangles, scalars=scalars, colormap=str(colormap), vmin=vmin, vmax=vmax, opacity = opacity)
    if color_mode == 'facies': # this assumes that there is only one facies per layer!
        mlab.triangular_mesh(X1,Y1,ve*Z1, triangles, color=tuple(colors[int(facies[0, 0, layer_n])]), opacity = opacity)

def create_random_section_2_points(strat,facies,scale,ve,color_mode,colors,colormap,x1,x2,y1,y2,s1,dx,bottom,opacity):
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
                plot_layers_on_one_side(layer_n,facies,color_mode,colors,X1,Y1,Z1,ve,triangles,vertices,scalars,colormap,norm,vmin,vmax,opacity)
        
def create_random_section_n_points(strat,facies,topo,scale,ve,color_mode,colors,colormap,x1,x2,y1,y2,dx,bottom,opacity):
    r, c, ts = np.shape(strat)
    if len(x1)==1:
        create_random_section_2_points(strat,facies,scale,ve,color_mode,colors,colormap,x1,x2,y1,y2,0,dx,bottom,opacity)
    else:
        count = 0
        dx1,dy1,ds1,s1 = compute_derivatives(x1,y1)
        for i in range(len(x1)):
            create_random_section_2_points(strat,facies,scale,ve,color_mode,colors,colormap,x1[i],x2[i],y1[i],y2[i],s1[i],dx,bottom,opacity)
            count = count+1
            # print("panel "+str(count)+" done, out of "+str(len(x1))+" panels")

def create_random_cookie(strat,facies,topo,scale,ve,color_mode,colors,colormap,x1,x2,y1,y2,dx,bottom,opacity):
    r, c, ts = np.shape(strat)
    count = 0
    dx1,dy1,ds1,s1 = compute_derivatives(x1,y1)
    for i in range(len(x1)):
        create_random_section_2_points(strat,facies,scale,ve,color_mode,colors,colormap,x1[i],x2[i],y1[i],y2[i],s1[i],dx,bottom,opacity)
        count = count+1
        # print("panel "+str(count)+" done, out of "+str(len(x1)+1)+" panels")
    create_random_section_2_points(strat,facies,scale,ve,color_mode,colors,colormap,x2[-1],x1[0],y2[-1],y1[0],s1[-1]+np.sqrt((x1[0]-x2[-1])**2+(y1[0]-y2[-1])**2),dx,bottom,opacity)
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

def plot_dip_section(topo, strat, dx, loc, ve, ax=None, xoffset=0, yoffset=0, water_depth=100,
    sea_level=None, subsid=None, prop=None, prop_cmap=None, prop_vmin=None, prop_vmax=None, facies=None, facies_colors=None, 
    linewidth=1, line_freq=2, color_mode='bathymetry', plot_type='3D', plot_erosion=False, erosional_surfs_thickness=None, plot_water=False, plot_basement=False):   
    depth_range = np.nanmax(strat[loc, :, :]) - np.nanmin(strat[loc, :, :])
    if depth_range > 0.1:
        if ax is None:
            fig = plt.figure(figsize = (20, 20), frameon=False)
            ax = fig.add_subplot(111)
        else:
            fig = None

        if subsid is not None:
            topo_s = topo.copy() 
            for i in range(0, topo_s.shape[2]):
                topo_s[:,:,i] = topo_s[:,:,i]+(subsid[:,:,-1]-subsid[:,:,i])
            strat = topostrat(topo_s)

        r, c, t = strat.shape
        lines = []
        for ts in range(1, t):
            if color_mode == 'bathymetry':
                if subsid is not None:
                    subsid_corr = subsid[loc, :, -1] - subsid[loc, :, ts]
                    f = interpolate.interp1d(np.arange(0, c)*dx, subsid_corr) # subsidence correction function
                
                x = np.hstack((np.arange(0, c)*dx, np.arange(0, c)[::-1]*dx))
                if subsid is not None:
                    y = np.hstack((strat[loc, :, ts-1], strat[loc, :, ts][::-1])) - f(x) # 'unsubside' the layer
                    split_layer_by_bathymetry(x, y, sea_level, ts, c*dx, water_depth, ax, f)
                else:
                    y = np.hstack((strat[loc, :, ts-1], strat[loc, :, ts][::-1]))
                    split_layer_by_bathymetry(x, y, sea_level, ts, c*dx, water_depth, ax, f=None)
                
            else:
                x = np.hstack((np.arange(0, c)*dx, np.arange(0, c)[::-1]*dx))
                y = np.hstack((strat[loc, :, ts-1], strat[loc, :, ts][::-1]))
                coords = []
                for i in range(len(x)):
                    if (np.isnan(x[i]) == 0) & (np.isnan(y[i]) == 0):
                        coords.append((x[i],y[i]))
                sed = Polygon(LineString(coords))
                if not sed.is_valid:
                    sed = sed.buffer(0)

                if color_mode == 'age':
                    norm = matplotlib.colors.Normalize(vmin=0, vmax=t)
                    cmap = matplotlib.cm.get_cmap('viridis')
                    if type(sed) == Polygon:
                        ax.fill(sed.exterior.xy[0], sed.exterior.xy[1], color=cmap(norm(ts))[:3])
                    else:
                        for geom in sed.geoms:
                            ax.fill(geom.exterior.xy[0], geom.exterior.xy[1], color=cmap(norm(ts))[:3])

                if color_mode == 'sea_level':
                    norm = matplotlib.colors.Normalize(vmin=np.min(sea_level), vmax=np.max(sea_level))
                    cmap = matplotlib.cm.get_cmap('RdBu')
                    if type(sed) == Polygon:
                        ax.fill(sed.exterior.xy[0], sed.exterior.xy[1], color=cmap(norm(sea_level[ts]))[:3])
                    else:
                        for geom in sed.geoms:
                            ax.fill(geom.exterior.xy[0], geom.exterior.xy[1], color=cmap(norm(sea_level[ts]))[:3])

                if color_mode == 'sea_level_change':
                    sl_change = np.diff(sea_level)
                    norm = matplotlib.colors.Normalize(vmin=np.min(sl_change), vmax=np.max(sl_change))
                    cmap = matplotlib.cm.get_cmap('RdBu')
                    if type(sed) == Polygon:
                        ax.fill(sed.exterior.xy[0], sed.exterior.xy[1], color=cmap(norm(sl_change[ts-1]))[:3])
                    else:
                        for geom in sed.geoms:
                            ax.fill(geom.exterior.xy[0], geom.exterior.xy[1], color=cmap(norm(sl_change[ts-1]))[:3])

                if color_mode == 'property':
                    top = strat[loc, :, ts-1] 
                    base = strat[loc, :, ts]
                    props = prop[loc, :, ts-1]
                    if np.max(base - top)>0:
                        Points, Inds = triangulate_layers(base, top, dx)
                        for j in range(len(Points)):
                            vertices = Points[j]
                            triangles, scalars = create_triangles(vertices)
                            x = vertices[:,0]
                            y = vertices[:,1]
                            colors = props[Inds[j]]
                            colors = np.mean(colors[np.array(triangles)], axis = 1)
                            ax.tripcolor(x, y, triangles=triangles, facecolors = colors, cmap = prop_cmap, 
                                          edgecolors = 'none', vmin = prop_vmin, vmax = prop_vmax)

                if color_mode == 'facies':
                    top = strat[loc, :, ts-1] 
                    base = strat[loc, :, ts]
                    faciess = facies[loc, :, ts-1]
                    if np.max(base - top)>0:
                        Points, Inds = triangulate_layers(base, top, dx)
                        for j in range(len(Points)):
                            vertices = Points[j]
                            triangles, scalars = create_triangles(vertices)
                            x = vertices[:,0]
                            y = vertices[:,1]
                            colors = faciess[Inds[j]]
                            colors = np.median(colors[np.array(triangles)], axis = 1)
                            cmap = ListedColormap(facies_colors)
                            ax.tripcolor(x, y, triangles=triangles, facecolors = colors, edgecolors = 'none', 
                                cmap = cmap, vmin = 0, vmax = len(facies_colors))                   

            if np.mod(ts, line_freq) == 0:
                line = ax.plot(dx*np.arange(c), strat[loc, :, ts], 'k', linewidth=linewidth) # plot stratigraphic surfaces

            if plot_erosion:
                temp = strat[loc,:,ts].copy()
                temp[erosional_surfs_thickness[loc,:,ts] == -1] = np.nan
                for j in range(len(temp)-1):
                    if erosional_surfs_thickness[loc,:,ts][j] != -1:
                        if erosional_surfs_thickness[loc,:,ts][j] > 1:
                            if [dx*j, dx*(j+1), np.round(temp[j], 5), np.round(temp[j+1], 5)] not in lines:
                                ax.plot([dx*j, dx*(j+1)], [temp[j], temp[j+1]], color='firebrick', linewidth=2, zorder=2*t)
                                lines.append([dx*j, dx*(j+1), np.round(temp[j], 5), np.round(temp[j+1], 5)])

        # if color_mode == 'sea_level_change' or color_mode == 'sea_level':
            # add colorbar

        if plot_water:
            if sea_level[t-1] > np.nanmin(strat[loc, :, -1]): # plot sea level and water
                x = np.hstack((np.arange(0, c)*dx, c*dx-dx, 0))
                y = np.hstack((strat[loc, :, -1], 100, 100))
                coords = []
                for i in range(len(x)):
                    if (np.isnan(x[i]) == 0) & (np.isnan(y[i]) == 0):
                        coords.append((x[i],y[i]))
                sed = Polygon(LineString(coords))
                x = np.hstack((0, c*dx-dx, c*dx-dx, 0))
                y = np.hstack((sea_level[t-1], sea_level[t-1], 0, 0))
                coords = []
                for i in range(len(x)):
                    if (np.isnan(x[i]) == 0) & (np.isnan(y[i]) == 0):
                        coords.append((x[i],y[i]))
                sl = Polygon(LineString(coords))
                sld = sl.symmetric_difference(sed)
                if type(sld) == Polygon:
                    ax.fill(sld.exterior.xy[0], sld.exterior.xy[1], facecolor='lightblue')
                else:
                    for poly in sld.geoms:
                        if np.min(poly.exterior.xy[1]) < sea_level[t-1]:
                            ax.fill(poly.exterior.xy[0], poly.exterior.xy[1], facecolor='lightblue')

        if plot_basement: # plot basement:
            x = np.hstack((np.arange(c) * dx, (c-1)*dx, 0))
            y = np.hstack((strat[loc, :, 0], np.nanmin(strat[loc, :, :])-20, np.nanmin(strat[loc, :, :])-20))
            ax.fill(x, y, color='lightgray')

        # plot top and base surfaces:
        base = ax.plot(dx*np.arange(c), strat[loc, :, 0], 'k', linewidth=2)
        top = ax.plot(dx*np.arange(c), strat[loc, :, -1], 'k', linewidth=2)

        if fig is not None:
            ax.set_xlim(0, dx*(c-1))
            ax.set_ylim(np.nanmin(strat[loc, :, :])-20, np.nanmax(strat[loc, :, :])+20)

        if plot_type == '3D':
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            fig.savefig('temp.png', transparent=True, dpi=300)
            plt.close(fig)
            im = plt.imread('temp.png')
            # get rid of figure margins:
            col_1 = np.where(im[im.shape[0]//2,:,0] == 0)[0][0]
            row_1 = np.where(im[:,im.shape[1]//2,0] == 0)[0][0]
            col_2 = np.where(im[im.shape[0]//2,:,0] == 0)[0][-1]
            row_2 = np.where(im[:,im.shape[1]//2,0] == 0)[0][-1]
            im = im[row_1:row_2, col_1:col_2, :]
            alpha = np.ones(np.shape(im[:,:,0]))
            alpha[im[:,:,-1] == 0] = 0
            base_pix = (ylim[1] - base[0].get_ydata())*im.shape[0]/(ylim[1] - ylim[0])
            top_pix = (ylim[1] - top[0].get_ydata())*im.shape[0]/(ylim[1] - ylim[0])
            xg, yg = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
            ind = np.where(np.diff(np.where(im[10, :, 0] < 1)[0]) > 1)[0][0]
            alpha[:,:ind+1][yg[:,:ind+1] <= top_pix[0]] = 0
            alpha[:,:ind+1][yg[:,:ind+1] >= base_pix[0]] = 0
            alpha[:,-ind:][yg[:,-ind:] <= top_pix[-1]] = 0
            alpha[:,-ind:][yg[:,-ind:] >= base_pix[-1]] = 0
            alpha[:ind+1,:] = 0
            alpha[-ind:,:] = 0
            alpha = alpha*255
            im = im[:,:,:3]*255
            try:
                alpha[0]
            except:
                alpha = np.ones(im.shape[0] * im.shape[1]) * alpha
            if len(alpha.shape) != 1:
                alpha = alpha.flatten()
            myLut = np.c_[im.reshape(-1, 3), alpha] # create colormap
            myLutLookupArray = np.arange(im.shape[0] * im.shape[1]).reshape(im.shape[0], im.shape[1])
            min_y = -ylim[1]*ve
            max_y = -ylim[0]*ve
            min_x = xlim[0]
            max_x = xlim[1]
            obj = mlab.imshow(myLutLookupArray, colormap='binary', interpolate=False,
                             extent=[min_y, max_y, min_x, max_x, 0, 0], opacity=1) # display image
            obj.module_manager.scalar_lut_manager.lut.table = myLut # change colormap
            obj.actor.orientation = [0, 90, 90]  
            obj.actor.position = [0, yoffset + loc*dx, 0]     
            obj.actor.scale = [1, -1, 1]
            obj.actor.force_opaque = True
            src = obj.mlab_source
            # note that the 'x' and 'y' axes end up being switched in Mayavi relative to the original cross section
            src.y = xoffset + min_x + (max_x - min_x)*(src.y - src.y.min())/(src.y.max() - src.y.min())
            src.x = min_y + (max_y - min_y)*(src.x - src.x.min())/(src.x.max() - src.x.min()) # vertical dimensions
            mlab.draw()
    
def plot_strike_section(topo, strat, dx, loc, ve, ax=None, xoffset=0, yoffset=0, water_depth=100, 
    sea_level=None, subsid=None, prop=None, prop_cmap=None, prop_vmin=None, prop_vmax=None, facies=None, facies_colors=None, 
    linewidth=1, line_freq=2, color_mode='bathymetry', plot_type='3D', plot_erosion=False, erosional_surfs_thickness=None, plot_water=False, plot_basement=False):
    depth_range = np.nanmax(strat[:, loc, :]) - np.nanmin(strat[:, loc, :])
    if depth_range > 0.1:
        if ax is None:
            fig = plt.figure(figsize = (20, 20), frameon=False)
            ax = fig.add_subplot(111)
        else:
            fig = None

        if subsid is not None:
            topo_s = topo.copy() 
            for i in range(0, topo_s.shape[2]):
                topo_s[:,:,i] = topo_s[:,:,i]+(subsid[:,:,-1]-subsid[:,:,i])
            strat = topostrat(topo_s)

        r, c, t = strat.shape

        lines = []
        for ts in range(1, t):
            if color_mode == 'bathymetry':
                if subsid is not None:
                    subsid_corr = subsid[:, loc, -1] - subsid[:, loc, ts]
                    f = interpolate.interp1d(np.arange(0, r)*dx, subsid_corr) # subsidence correction function

                x = np.hstack((np.arange(0, r)*dx, np.arange(0, r)[::-1]*dx))
                if subsid is not None:
                    y = np.hstack((strat[:, loc, ts-1], strat[:, loc, ts][::-1])) - f(x) # 'unsubside' the layer
                    split_layer_by_bathymetry(x, y, sea_level, ts, r*dx, water_depth, ax, f)
                else:
                    y = np.hstack((strat[:, loc, ts-1], strat[:, loc, ts][::-1]))
                    split_layer_by_bathymetry(x, y, sea_level, ts, r*dx, water_depth, ax, f=None)            

            else:
                x = np.hstack((np.arange(0, r)*dx, np.arange(0, r)[::-1]*dx))
                y = np.hstack((strat[:, loc, ts-1], strat[:, loc, ts][::-1])) 
                coords = []
                for i in range(len(x)):
                    if (np.isnan(x[i]) == 0) & (np.isnan(y[i]) == 0):
                        coords.append((x[i],y[i]))
                sed = Polygon(LineString(coords))
                if not sed.is_valid:
                    sed = sed.buffer(0)

                if color_mode == 'age':
                    norm = matplotlib.colors.Normalize(vmin=0, vmax=t)
                    cmap = matplotlib.cm.get_cmap('viridis')
                    if type(sed) == Polygon:
                        ax.fill(sed.exterior.xy[0], sed.exterior.xy[1], color=cmap(norm(ts))[:3])
                    else:
                        for geom in sed.geoms:
                            ax.fill(geom.exterior.xy[0], geom.exterior.xy[1], color=cmap(norm(ts))[:3])

                if color_mode == 'sea_level':
                    norm = matplotlib.colors.Normalize(vmin=np.min(sea_level), vmax=np.max(sea_level))
                    cmap = matplotlib.cm.get_cmap('RdBu')
                    if type(sed) == Polygon:
                        ax.fill(sed.exterior.xy[0], sed.exterior.xy[1], color=cmap(norm(sea_level[ts]))[:3])
                    else:
                        for geom in sed.geoms:
                            ax.fill(geom.exterior.xy[0], geom.exterior.xy[1], color=cmap(norm(sea_level[ts]))[:3])

                if color_mode == 'sea_level_change':
                    sl_change = np.diff(sea_level)/3.0 # mm/hour
                    norm = matplotlib.colors.Normalize(vmin=np.min(sl_change), vmax=np.max(sl_change))
                    cmap = matplotlib.cm.get_cmap('RdBu')
                    if type(sed) == Polygon:
                        ax.fill(sed.exterior.xy[0], sed.exterior.xy[1], color=cmap(norm(sl_change[ts-1]))[:3])
                    else:
                        for geom in sed.geoms:
                            ax.fill(geom.exterior.xy[0], geom.exterior.xy[1], color=cmap(norm(sl_change[ts-1]))[:3])

                if color_mode == 'property':
                    top = strat[:, loc, ts-1] 
                    base = strat[:, loc, ts]
                    props = prop[:, loc, ts-1]
                    if np.max(base - top)>0:
                        Points, Inds = triangulate_layers(base, top, dx)
                        for j in range(len(Points)):
                            vertices = Points[j]
                            triangles, scalars = create_triangles(vertices)
                            x = vertices[:,0]
                            y = vertices[:,1]
                            colors = props[Inds[j]]
                            colors = np.mean(colors[np.array(triangles)], axis = 1)
                            ax.tripcolor(x, y, triangles=triangles, facecolors = colors, cmap = prop_cmap, 
                                          edgecolors = 'none', vmin = prop_vmin, vmax = prop_vmax)

                if color_mode == 'facies':
                    top = strat[:, loc, ts-1] 
                    base = strat[:, loc, ts]
                    faciess = facies[:, loc, ts-1]
                    if np.max(base - top)>0:
                        Points, Inds = triangulate_layers(base, top, dx)
                        for j in range(len(Points)):
                            vertices = Points[j]
                            triangles, scalars = create_triangles(vertices)
                            x = vertices[:,0]
                            y = vertices[:,1]
                            colors = faciess[Inds[j]]
                            colors = np.median(colors[np.array(triangles)], axis = 1)
                            cmap = ListedColormap(facies_colors)
                            ax.tripcolor(x, y, triangles=triangles, facecolors = colors, edgecolors = 'none', 
                                cmap = cmap, vmin = 0, vmax = len(facies_colors))

            if np.mod(ts, line_freq) == 0:
                line = ax.plot(dx*np.arange(r), strat[:, loc, ts], 'k', linewidth=linewidth) # plot stratigraphic surfaces

            if plot_erosion:
                temp = strat[:,loc,ts].copy()
                temp[erosional_surfs_thickness[:,loc,ts] == -1] = np.nan
                ax.plot(np.arange(len(temp))*dx, temp, color='firebrick', linewidth=1, zorder=2*t)
                # for j in range(len(temp)-1):
                #     if erosional_surfs_thickness[:,loc,ts][j] != -1:
                #         if erosional_surfs_thickness[:,loc,ts][j] > 1:
                #             if [dx*j, dx*(j+1), np.round(temp[j], 5), np.round(temp[j+1], 5)] not in lines:
                #                 ax.plot([dx*j, dx*(j+1)], [temp[j], temp[j+1]], color='firebrick', linewidth=1, zorder=2*t)
                #                 lines.append([dx*j, dx*(j+1), np.round(temp[j], 5), np.round(temp[j+1], 5)])

        if plot_water:
            if sea_level[t-1] > np.nanmin(strat[:, loc, -1]): # plot sea level and water
                # x = np.hstack((np.arange(0, c)*dx, c*dx-dx, 0))
                x = np.hstack((np.arange(0, r)*dx, r*dx-dx, 0))
                y = np.hstack((strat[:, loc, -1], 100, 100))
                coords = []
                for i in range(len(x)):
                    if (np.isnan(x[i]) == 0) & (np.isnan(y[i]) == 0):
                        coords.append((x[i],y[i]))
                sed = Polygon(LineString(coords))
                x = np.hstack((0, r*dx-dx, r*dx-dx, 0))
                y = np.hstack((sea_level[t-1], sea_level[t-1], 0, 0))
                coords = []
                for i in range(len(x)):
                    if (np.isnan(x[i]) == 0) & (np.isnan(y[i]) == 0):
                        coords.append((x[i],y[i]))
                sl = Polygon(LineString(coords))
                sld = sl.symmetric_difference(sed)
                if type(sld) == Polygon:
                    ax.fill(sld.exterior.xy[0], sld.exterior.xy[1], facecolor='lightblue')
                else:
                    for poly in sld.geoms:
                        if np.min(poly.exterior.xy[1]) < sea_level[t-1]:
                            ax.fill(poly.exterior.xy[0], poly.exterior.xy[1], facecolor='lightblue')
                
        if plot_basement: # plot basement:
            x = np.hstack((np.arange(r) * dx, (r-1)*dx, 0))
            y = np.hstack((strat[:, loc, 0], np.nanmin(strat), np.nanmin(strat)))
            ax.fill(x, y, color='lightgray')

        base = ax.plot(dx * np.arange(r), strat[:, loc, 0], 'k', linewidth=2)
        top = ax.plot(dx * np.arange(r), strat[:, loc, -1], 'k', linewidth=2)

        if fig is not None:
            ax.set_xlim(0, dx*(r-1))
            ax.set_ylim(np.nanmin(strat[:, loc, :])-20, np.nanmax(strat[:, loc, :])+20)

        if plot_type == '3D':
            # ax.set_xlim(0, dx*(r-1))
            # ax.set_ylim(np.nanmin(strat[:, loc, :])-20, np.nanmax(strat[:, loc, :])+20)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            fig.savefig('temp.png', transparent=True, dpi=300)
            plt.close(fig)
            im = plt.imread('temp.png')
            # get rid of figure margins:
            col_1 = np.where(im[im.shape[0]//2,:,0] == 0)[0][0]
            row_1 = np.where(im[:,im.shape[1]//2,0] == 0)[0][0]
            col_2 = np.where(im[im.shape[0]//2,:,0] == 0)[0][-1]
            row_2 = np.where(im[:,im.shape[1]//2,0] == 0)[0][-1]
            im = im[row_1:row_2, col_1:col_2, :]
            alpha = np.ones(np.shape(im[:,:,0]))
            alpha[im[:,:,-1] == 0] = 0
            base_pix = (ylim[1] - base[0].get_ydata())*im.shape[0]/(ylim[1] - ylim[0])
            top_pix = (ylim[1] - top[0].get_ydata())*im.shape[0]/(ylim[1] - ylim[0])
            xg, yg = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
            ind = np.where(np.diff(np.where(im[10, :, 0] < 1)[0]) > 1)[0][0]
            alpha[:,:ind+1][yg[:,:ind+1] <= top_pix[0]] = 0
            alpha[:,:ind+1][yg[:,:ind+1] >= base_pix[0]] = 0
            alpha[:,-ind:][yg[:,-ind:] <= top_pix[-1]] = 0
            alpha[:,-ind:][yg[:,-ind:] >= base_pix[-1]] = 0
            alpha[:ind+1,:] = 0
            alpha[-ind:,:] = 0
            alpha = alpha*255
            im = im[:,:,:3]*255
            try:
                alpha[0]
            except:
                alpha = np.ones(im.shape[0] * im.shape[1]) * alpha
            if len(alpha.shape) != 1:
                alpha = alpha.flatten()
            myLut = np.c_[im.reshape(-1, 3), alpha] # create colormap
            myLutLookupArray = np.arange(im.shape[0] * im.shape[1]).reshape(im.shape[0], im.shape[1])
            min_y = -ylim[1]*ve
            max_y = -ylim[0]*ve
            min_x = xlim[0]
            max_x = xlim[1]
            obj = mlab.imshow(myLutLookupArray, colormap='binary', interpolate=False,
                             extent=[min_y, max_y, min_x, max_x, 0, 0], opacity=1) # display image
            obj.module_manager.scalar_lut_manager.lut.table = myLut # change colormap
            obj.actor.orientation = [0, 90, 0]  
            obj.actor.position = [xoffset + loc*dx, 0, 0]     
            obj.actor.scale = [1, 1, 1]
            obj.actor.force_opaque = True
            src = obj.mlab_source
            # note that the 'x' and 'y' axes end up being switched in Mayavi relative to the original cross section
            src.y = yoffset + min_x + (max_x - min_x)*(src.y - src.y.min())/(src.y.max() - src.y.min())
            src.x = min_y + (max_y - min_y)*(src.x - src.x.min())/(src.x.max() - src.x.min())
            mlab.draw()

def plot_random_section_2_points(topo, strat, dx, x1, x2, y1, y2, s1, ve, bottom, xoffset=0, yoffset=0, water_depth=100, ax=None,
    sea_level=None, subsid=None, prop=None, prop_cmap=None, prop_vmin=None, prop_vmax=None, facies=None, facies_colors=None, 
    linewidth=1, line_freq=2, color_mode='bathymetry', plot_type='3D', plot_erosion=False, erosional_surfs_thickness=None, plot_water=False, plot_basement=False):

    if ax is None:
        fig = plt.figure(figsize = (20, 20), frameon=False)
        ax = fig.add_subplot(111)
    else:
        fig = None

    dist = dx*((x2-x1)**2 + (y2-y1)**2)**0.5
    s2 = s1*dx+dist
    num = int(dist/float(dx))
    Xrand, Yrand, Srand = np.linspace(x1,x2,num), np.linspace(y1,y2,num), np.linspace(s1*dx,s2,num)

    if subsid is not None:
        topo_s = topo.copy() 
        for i in range(0, topo_s.shape[2]):
            topo_s[:,:,i] = topo_s[:,:,i]+(subsid[:,:,-1]-subsid[:,:,i])
        strat = topostrat(topo_s)

    r, c, t = strat.shape
    lines = []
    for layer_n in trange(0, t-1):
        top = map_coordinates(strat[:,:,layer_n+1], np.vstack((Yrand,Xrand)))
        base = map_coordinates(strat[:,:,layer_n], np.vstack((Yrand,Xrand)))
        if color_mode == 'bathymetry':
            if subsid is not None:
                subsid_corr = map_coordinates(subsid[:,:,-1], np.vstack((Yrand,Xrand))) - \
                                map_coordinates(subsid[:,:,layer_n], np.vstack((Yrand,Xrand)))
                f = interpolate.interp1d(Srand, subsid_corr) # subsidence correction function
            s = np.hstack((Srand, Srand[::-1]))
            if subsid is not None:
                z = np.hstack((base, top[::-1])) - f(s) # 'unsubside' the layer
                split_layer_by_bathymetry(s, z, sea_level, layer_n, Srand[-1], water_depth, ax, f)
            else:
                z = np.hstack((base, top[::-1]))
                split_layer_by_bathymetry(s, z, sea_level, layer_n, Srand[-1], water_depth, ax, f=None)
        else:
            s = np.hstack((Srand, Srand[::-1])) 
            z = np.hstack((base, top[::-1]))
            coords = []
            for i in range(len(s)):
                if (np.isnan(s[i]) == 0) & (np.isnan(z[i]) == 0):
                    coords.append((s[i], z[i]))
            sed = Polygon(LineString(coords))
            if not sed.is_valid:
                sed = sed.buffer(0)

            if color_mode == 'age':
                norm = matplotlib.colors.Normalize(vmin=0, vmax=t)
                cmap = matplotlib.cm.get_cmap('viridis')
                if type(sed) == Polygon:
                    ax.fill(sed.exterior.xy[0], sed.exterior.xy[1], color=cmap(norm(layer_n))[:3])
                if type(sed) == MultiPolygon:
                    for geom in sed.geoms:
                        ax.fill(geom.exterior.xy[0], geom.exterior.xy[1], color=cmap(norm(layer_n))[:3])

#             if color_mode == 'sea_level':
#                 norm = matplotlib.colors.Normalize(vmin=np.min(sea_level), vmax=np.max(sea_level))
#                 cmap = matplotlib.cm.get_cmap('RdBu')
#                 ax.fill(sed.exterior.xy[0], sed.exterior.xy[1], color=cmap(norm(sea_level[ts]))[:3])

#             if color_mode == 'sea_level_change':
#                 sl_change = np.diff(sea_level)/3.0 # mm/hour
#                 norm = matplotlib.colors.Normalize(vmin=np.min(sl_change), vmax=np.max(sl_change))
#                 cmap = matplotlib.cm.get_cmap('RdBu')
#                 ax.fill(sed.exterior.xy[0], sed.exterior.xy[1], color=cmap(norm(sl_change[ts-1]))[:3])

#             if color_mode == 'property' or color_mode == 'facies':
#                 top = strat[:, loc, ts-1] 
#                 base = strat[:, loc, ts]
#                 props = prop[:, loc, ts-1]
#                 # faciess = facies[:, loc, ts-1]
#                 if np.max(base - top)>0:
#                     Points, Inds = triangulate_layers(base, top, dx)
#                     for j in range(len(Points)):
#                         vertices = Points[j]
#                         triangles, scalars = create_triangles(vertices)
#                         x = vertices[:,0]
#                         y = vertices[:,1]
#                         if color_mode == 'property':
#                             colors = props[Inds[j]]
#                             colors = np.mean(colors[np.array(triangles)], axis = 1)
#                             ax.tripcolor(x, y, triangles=triangles, facecolors = colors, cmap = prop_cmap, 
#                                           edgecolors = 'none', vmin = vmin, vmax = vmax)
#                         if color_mode == 'facies':
#                             colors = faciess[Inds[j]]
#                             colors = np.median(colors[np.array(triangles)], axis = 1)
#                             cmap = ListedColormap(facies_colors)
#                             ax.tripcolor(x, y, triangles=triangles, facecolors = colors, edgecolors = 'none', 
#                                 cmap = cmap, vmin = 0, vmax = len(facies_colors))

        if np.mod(layer_n, line_freq) == 0:
            line = ax.plot(Srand, base, 'k', linewidth=linewidth) # plot stratigraphic surfaces

        if plot_erosion:
            temp = base.copy()
            erosion = map_coordinates(erosional_surfs_thickness[:,:,layer_n], np.vstack((Yrand,Xrand)), order=0)
            temp[erosion == -1] = np.nan
            for j in range(len(temp)-1):
                if erosion[j] != -1:
                    if erosion[j] > 1:
                        if [Srand[j], Srand[j+1], np.round(temp[j], 5), np.round(temp[j+1], 5)] not in lines:
                            ax.plot([Srand[j], Srand[j+1]], [temp[j], temp[j+1]], color='firebrick', linewidth=2, zorder=2*t)
                            lines.append([Srand[j], Srand[j+1], np.round(temp[j], 5), np.round(temp[j+1], 5)])

    ax.set_xlim(0, Srand[-1])
    ax.set_ylim(bottom, np.nanmax(strat)+50)
                
    if plot_basement: # plot basement:
        top = map_coordinates(strat[:,:,0], np.vstack((Yrand,Xrand)))
        x = np.hstack((Srand, Srand[-1], 0))
        y = np.hstack((top, bottom, bottom)) #np.nanmin(strat)-50, np.nanmin(strat)-50))
        basement = ax.fill(x, y, facecolor='lightgray')

    base = ax.plot(Srand, map_coordinates(strat[:,:,0], np.vstack((Yrand,Xrand))), 'k', linewidth=2)
    top = ax.plot(Srand, map_coordinates(strat[:,:,-1], np.vstack((Yrand,Xrand))), 'k', linewidth=2)

    if plot_basement:
        base = ax.plot([0, Srand[-1]], [bottom, bottom], 'k', linewidth=2)        

    if plot_type == '3D':
        ax.set_xlim(0, Srand[-1])
        ax.set_ylim(bottom, np.nanmax(strat)+50)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        fig.savefig('temp.png', transparent=True, dpi=300)
        plt.close(fig)
        im = plt.imread('temp.png')
        # get rid of figure margins:
        col_1 = np.where(im[im.shape[0]//2,:,0] == 0)[0][0]
        row_1 = np.where(im[:,im.shape[1]//2,0] == 0)[0][0]
        col_2 = np.where(im[im.shape[0]//2,:,0] == 0)[0][-1]
        row_2 = np.where(im[:,im.shape[1]//2,0] == 0)[0][-1]
        im = im[row_1:row_2, col_1:col_2, :]
        alpha = np.ones(np.shape(im[:,:,0]))
        alpha[im[:,:,-1] == 0] = 0
        base_pix = (ylim[1] - base[0].get_ydata())*im.shape[0]/(ylim[1] - ylim[0])
        top_pix = (ylim[1] - top[0].get_ydata())*im.shape[0]/(ylim[1] - ylim[0])
        xg, yg = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
        ind = np.where(np.diff(np.where(im[10, :, 0] < 1)[0]) > 1)[0][0]
        alpha[:,:ind+1][yg[:,:ind+1] <= top_pix[0]] = 0
        alpha[:,:ind+1][yg[:,:ind+1] >= base_pix[0]] = 0
        alpha[:,-ind:][yg[:,-ind:] <= top_pix[-1]] = 0
        alpha[:,-ind:][yg[:,-ind:] >= base_pix[-1]] = 0
        alpha[:ind+1,:] = 0
        alpha[-ind:,:] = 0
        alpha = alpha*255
        im = im[:,:,:3]*255
        try:
            alpha[0]
        except:
            alpha = np.ones(im.shape[0] * im.shape[1]) * alpha
        if len(alpha.shape) != 1:
            alpha = alpha.flatten()
        myLut = np.c_[im.reshape(-1, 3), alpha] # create colormap
        myLutLookupArray = np.arange(im.shape[0] * im.shape[1]).reshape(im.shape[0], im.shape[1])
        min_y = -ylim[1]*ve
        max_y = -ylim[0]*ve
        min_x = xlim[0]
        max_x = xlim[1]
        obj = mlab.imshow(myLutLookupArray, colormap='binary', interpolate=False,
                         extent=[min_y, max_y, min_x, max_x, 0, 0], opacity=1) # display image
        obj.module_manager.scalar_lut_manager.lut.table = myLut # change colormap
        angle = line_orientation(x1, y1, x2, y2)
        obj.actor.orientation = [0, 90, angle-90]  
        obj.actor.position = [x1*dx, y1*dx, 0]     
        obj.actor.scale = [1, 1, 1]
        obj.actor.force_opaque = True
        src = obj.mlab_source
        # note that the 'x' and 'y' axes end up being switched in Mayavi relative to the original cross section
        src.y = min_x + (max_x - min_x)*(src.y - src.y.min())/(src.y.max() - src.y.min())
        src.x = min_y + (max_y - min_y)*(src.x - src.x.min())/(src.x.max() - src.x.min()) # vertical dimensions

        scale = 1

        if plot_water:
            blue = (0.255, 0.412, 0.882)
            base = map_coordinates(strat[:,:,-1], np.vstack((Yrand,Xrand)))
            top = sea_level[strat.shape[-1]-1] * np.ones(np.shape(base)) 
            if np.max(top-base)>0:
                Points,Inds = triangulate_layers2(top,base,Srand)
                for i in range(len(Points)):
                    vertices = Points[i]
                    triangles, scalars = create_triangles(vertices)
                    Z1 = scale*vertices[:,1]
                    mlab.triangular_mesh(Xrand[Inds[i]]*dx, Yrand[Inds[i]]*dx, ve*Z1, triangles, color = blue, opacity = 0.3)

        mlab.draw()

def split_layer_by_bathymetry(x, y, sea_level, ts, max_x, water_depth, ax, f=None):
    coords = []
    for i in range(len(x)):
        if (np.isnan(x[i]) == 0) & (np.isnan(y[i]) == 0):
            coords.append((x[i],y[i]))
    sed = Polygon(LineString(coords))
    if not sed.is_valid:
        sed = sed.buffer(0)

    paleo_sl = sea_level[ts]
    paleo_sl_deep = sea_level[ts] - water_depth
    line1 = LineString([(0, paleo_sl), (max_x, paleo_sl)]) # line used to split continental from marine
    line2 = LineString([(0, paleo_sl_deep), (max_x, paleo_sl_deep)])
    
    if line1.intersects(sed):
        polys = split(sed, line1)
        if line2.intersects(sed):
            polys = split(MultiPolygon(polys), line2)
        for poly in polys.geoms:
            if np.min(poly.exterior.xy[1]) < paleo_sl:
                polys2 = split(poly, line2)
                for poly2 in polys2.geoms:
                    x1 = poly2.exterior.xy[0]
                    if f is not None:
                        y1 = poly2.exterior.xy[1] + f(x1) # resubside the y coordinates before plotting
                    else:
                        y1 = poly2.exterior.xy[1]
                    if np.min(poly2.exterior.xy[1]) < paleo_sl_deep:
                        ax.fill(x1, y1, color='sienna') # deep marine
                    else:
                        ax.fill(x1, y1, color='peru') # shallow marine
            else:
                x1 = poly.exterior.xy[0]
                if f is not None:
                    y1 = poly.exterior.xy[1] + f(x1) # resubside the y coordinates before plotting
                else:
                    y1 = poly.exterior.xy[1]
                ax.fill(x1, y1, color='lemonchiffon') # fluvio-deltaic
                    
    elif line2.intersects(sed):
        polys = split(sed, line2)
        for poly in polys.geoms:
            if np.min(poly.exterior.xy[1]) < paleo_sl:
                polys2 = split(poly, line2)
                for poly2 in polys2.geoms:
                    x1 = poly2.exterior.xy[0]
                    if f is not None:
                        y1 = poly2.exterior.xy[1] + f(x1) # resubside the y coordinates before plotting
                    else:
                        y1 = poly2.exterior.xy[1]
                    if np.min(poly.exterior.xy[1]) < paleo_sl_deep:
                        ax.fill(x1, y1, color='sienna') # deep marine
                    else:
                        ax.fill(x1, y1, color='peru') # shallow marine
            else:
                x1 = poly.exterior.xy[0]
                if f is not None:
                    y1 = poly.exterior.xy[1] + f(x1) # resubside the y coordinates before plotting
                else:
                    y1 = poly.exterior.xy[1]
                ax.fill(x1, y1, color='lemonchiffon') # fluvio-deltaic
    else:
        if type(sed) == MultiPolygon:
            for geom in sed.geoms:
                x1 = geom.exterior.xy[0]
                if f is not None:
                    y1 = geom.exterior.xy[1] + f(x1) # resubside the y coordinates before plotting
                else:
                    y1 = geom.exterior.xy[1]
                if np.min(geom.exterior.xy[1]) < paleo_sl:
                    if np.min(geom.exterior.xy[1]) < paleo_sl_deep:
                        ax.fill(x1, y1, color='sienna') # deep marine
                    else:
                        ax.fill(x1, y1, color='peru') # shallow marine
                else:
                    ax.fill(x1, y1, color='lemonchiffon') # fluvio-deltaic
        elif type(sed) == Polygon:
            x1 = sed.exterior.xy[0]
            if f is not None:
                y1 = sed.exterior.xy[1] + f(x1) # resubside the y coordinates before plotting
            else:
                y1 = sed.exterior.xy[1]
            if len(y1) > 0:
                if np.min(sed.exterior.xy[1]) < paleo_sl:
                    if np.min(sed.exterior.xy[1]) < paleo_sl_deep:
                        ax.fill(x1, y1, color='sienna') # deep marine
                    else:
                        ax.fill(x1, y1, color='peru') # shallow marine
                else:
                    ax.fill(x1, y1, color='lemonchiffon') # fluvio-deltaic

def line_orientation(x1, y1, x2, y2):
    # Calculate the angle of the line in radians
    angle_rad = math.atan2(y2 - y1, x2 - x1)

    # Convert the angle to degrees
    angle_deg = math.degrees(angle_rad)

    return angle_deg

def plot_strat_diagram(elevation, elevation_units, time, time_units, res, max_elevation, max_time, plotting=True, plot_raw_data=True):
    if plotting:
        fig = plt.figure(figsize=(9,6))
        ax1 = fig.add_axes([0.09, 0.08, 0.85, 0.76]) # [left, bottom, width, height]
        ax1.set_xlabel('time (' + time_units + ')', fontsize = 12)
        ax1.set_ylabel('elevation (' + elevation_units + ')', fontsize = 12)
        for tick in ax1.xaxis.get_major_ticks():
            tick.label1.set_fontsize(10)
        for tick in ax1.yaxis.get_major_ticks():
            tick.label1.set_fontsize(10)
        ax2 = fig.add_axes([0.94, 0.08, 0.05, 0.76], sharey = ax1)
        ax2.set_xticks([])
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax3 = fig.add_axes([0.09, 0.84, 0.85, 0.08], sharex = ax1)
        ax3.set_yticks([])
        plt.setp(ax3.get_xticklabels(), visible=False)
        # ax1.set_xlim(0, time[-1])
        ax1.set_xlim(0, max_time)
        # elev_range = np.max(elevation) - np.min(elevation)
        elev_range = max_elevation - np.min(elevation)
        ylim1 = np.min(elevation) - 0.02 * elev_range
        # ylim2 = np.max(elevation) + 0.02 * elev_range
        ylim2 = max_elevation + 0.02 * elev_range
        ax1.set_ylim(ylim1, ylim2)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(ylim1, ylim2)
        ax3.set_ylim(0, 1)
        # ax3.set_xlim(0, time[-1])
        ax3.set_xlim(min(time), max_time)
        ax4 = fig.add_axes([0.09, 0.92, 0.6, 0.08])
        ax4.set_xlim(0, 10)
        ax4.set_ylim(0, 1)
    else:
        fig = None
    
    elevation_smooth = smooth_elevation_series(elevation, res)
    dep_inds = np.where(np.diff(elevation_smooth) >= res)[0] + 1 # deposition
    er_inds = np.where(np.diff(elevation_smooth) <= -res)[0] + 1 # erosion
    st_inds = np.where((np.diff(elevation_smooth) > -res) & (np.diff(elevation_smooth) < res))[0] + 1 # stasis

    strat = np.minimum.accumulate(elevation_smooth[::-1])[::-1] # stratigraphic 'elevation'
    unconf_inds = np.where(np.abs(strat - elevation_smooth) > 0)[0]
    inds = np.where(np.diff(unconf_inds)>1)[0] # indices where deposition starts again, after erosion
    if len(inds)>0:
        inds = np.hstack((inds, len(unconf_inds)-1)) # add last index
        unconf_inds = np.hstack((unconf_inds, unconf_inds[inds]+1))
        unconf_inds = np.sort(unconf_inds)
    if len(unconf_inds) > 0:
        if unconf_inds[0] == 0:
            unconf_inds = unconf_inds[1:]
    
    # find unconformities:
    unconf_end_inds = np.where(np.abs(strat - elevation_smooth) > 0)[0]
    inds = np.where(np.diff(unconf_end_inds)>1)[0] # indices where deposition starts again, after erosion
    inds = np.hstack((inds, len(unconf_end_inds)-1)) # add last index
    if strat[-1] - strat[-2] == 0:
        inds = np.hstack((inds, len(unconf_end_inds)-1))
    if len(unconf_end_inds) > 0:
        strat_tops = strat[unconf_end_inds[inds]+1] # stratigraphic tops
        strat_top_inds = unconf_end_inds[inds]+1
        strat_top_ages = time[unconf_end_inds[inds]+1]
    else:
        strat_tops = []
        strat_top_ages = []
        strat_top_inds = []

    # time step labels (-1, 0, 1 for now):
    ts_labels = np.zeros((len(time)-1,)) # stasis
    ts_labels[dep_inds-1] = 1 # deposition
    ts_labels[er_inds-1] = -1 # erosion

    if len(unconf_inds) > 0: # update time step labels if there is erosion
        eroded_dep_inds = np.where(ts_labels[unconf_inds-1] == 1)[0]
        ts_labels[unconf_inds[eroded_dep_inds]-1] = 2
        # ts_labels[unconf_inds[eroded_st_inds]-1] = 3 # this is needed if you want to include the eroded stasis in the vacuity category

    # this is needed so that deposition is not underestimated when deposition rates are high:
    inds = np.where(np.abs(strat - elevation_smooth) > 0)[0]
    if len(inds)>0:
        unconf_start_inds = np.where(np.diff(inds) > 1)[0] + 1
        unconf_start_inds = inds[np.hstack((0, unconf_start_inds))]
        if len(unconf_start_inds) > 0:
            if unconf_start_inds[0] == 0:
                unconf_start_inds = unconf_start_inds[1:]
        # Define the endpoints of the two lines
        x1, y1 = time[unconf_start_inds-1], elevation_smooth[unconf_start_inds-1]
        x2, y2 = time[unconf_start_inds], elevation_smooth[unconf_start_inds]
        x3, y3 = time[unconf_start_inds], strat[unconf_start_inds]
        x4, y4 = time[unconf_start_inds+1], strat[unconf_start_inds+1]
        # Find the intersection point
        intersection_x, intersection_y = find_intersection_point(x1, y1, x2, y2, x3, y3, x4, y4)
        new_times = np.hstack((time, intersection_x))
        new_elevation_smooth = np.hstack((elevation_smooth, intersection_y))
        new_elevation = np.hstack((elevation, intersection_y))
        sort_inds = np.argsort(new_times)
        new_times = new_times[sort_inds]
        new_elevation_smooth = new_elevation_smooth[sort_inds]
        new_elevation = new_elevation[sort_inds]
        new_ts_labels = ts_labels.copy()
        for i in range(len(unconf_start_inds)):
            new_ts_labels = np.insert(new_ts_labels, unconf_start_inds[i]-1+i, 1)
    else:
        new_times = time
        new_elevation_smooth = elevation_smooth
        new_elevation = elevation
        new_ts_labels = ts_labels

    # applying some median filtering to the stasis-dominated sections:
    temp = new_ts_labels.copy()
    temp[temp != 0] = 1
    temp = signal.medfilt(temp, 3)
    temp[temp != 0] = new_ts_labels[temp != 0]
    # make sure that single-step changes in elevation that are significant are not filtered out:
    temp[np.abs(np.diff(new_elevation)) > res] = new_ts_labels[np.abs(np.diff(new_elevation)) > res]
    new_ts_labels = temp

    stasis_inds = np.where(new_ts_labels == 0)[0]
    # Calculate the differences between consecutive target_indices
    differences = np.diff(stasis_inds)
    # Find the indices where differences are not equal to 1
    split_indices = np.where(differences != 1)[0]
    # Split the target_indices into contiguous blocks using split_indices
    blocks = np.split(stasis_inds, split_indices + 1)
    # Get the beginning and end indices of each block
    if len(blocks[0]) > 0:
        stasis_start_inds = [block[0] for block in blocks]
        stasis_end_inds = [block[-1]+1 for block in blocks]
        stasis_tops = new_elevation[stasis_end_inds]
        stasis_ages = new_times[stasis_end_inds]
    else:
        stasis_tops = []

    if plotting:
        ax1.plot(time, strat, 'k--')
        # if plot_raw_data:
        #     for i in range(len(new_ts_labels)):
        #         if new_ts_labels[i] == 0:
        #             ax1.plot([new_times[i], new_times[i+1]], [new_elevation[i], new_elevation[i+1]], 'orange', linewidth=3)
        #         if new_ts_labels[i] == -1:
        #             ax1.plot([new_times[i], new_times[i+1]], [new_elevation[i], new_elevation[i+1]], 'r', linewidth=3)
        #         if new_ts_labels[i] == 1:
        #             ax1.plot([new_times[i], new_times[i+1]], [new_elevation[i], new_elevation[i+1]], 'xkcd:medium blue', linewidth=3)
        #         if new_ts_labels[i] == 2:
        #             ax1.plot([new_times[i], new_times[i+1]], [new_elevation[i], new_elevation[i+1]], 'gray', linewidth=3)
        #         if new_ts_labels[i] == 3:
        #             ax1.plot([new_times[i], new_times[i+1]], [new_elevation[i], new_elevation[i+1]], 'y', linewidth=3)
        # else:
        #     for i in range(len(new_ts_labels)):
        #         if new_ts_labels[i] == 0:
        #             ax1.plot([new_times[i], new_times[i+1]], [new_elevation_smooth[i], new_elevation_smooth[i+1]], 'orange', linewidth=3)
        #         if new_ts_labels[i] == -1:
        #             ax1.plot([new_times[i], new_times[i+1]], [new_elevation_smooth[i], new_elevation_smooth[i+1]], 'r', linewidth=3)
        #         if new_ts_labels[i] == 1:
        #             ax1.plot([new_times[i], new_times[i+1]], [new_elevation_smooth[i], new_elevation_smooth[i+1]], 'xkcd:medium blue', linewidth=3)
        #         if new_ts_labels[i] == 2:
        #             ax1.plot([new_times[i], new_times[i+1]], [new_elevation_smooth[i], new_elevation_smooth[i+1]], 'gray', linewidth=3)
        #         if new_ts_labels[i] == 3:
        #             ax1.plot([new_times[i], new_times[i+1]], [new_elevation_smooth[i], new_elevation_smooth[i+1]], 'y', linewidth=3)
    
        # plot stratigraphic column:
        for i in range(len(strat_tops)):
            ax2.plot([0, 1], [strat_tops[i], strat_tops[i]], color = 'xkcd:red', linewidth = 2)
            ax1.plot([strat_top_ages[i], max_time], [strat_tops[i], strat_tops[i]], 'k--', linewidth=0.5)
        if len(strat_tops) > 0:
            if elevation[0] < np.min(strat_tops):
                strat_tops = np.hstack((elevation[0], strat_tops, elevation[-1]))
            else:
                strat_tops = np.hstack((strat_tops, elevation[-1]))
        else:
            strat_tops = np.hstack((elevation[0], strat_tops, elevation[-1]))
        for i in range(len(strat_tops)-1): # plot stratigraphic units
            x = [0, 1, 1, 0]
            y = [strat_tops[i], strat_tops[i], strat_tops[i+1], strat_tops[i+1]]
            ax2.fill(x, y, facecolor='xkcd:medium blue', edgecolor='k')


        ax1.plot([time[-1], max_time], [strat_tops[-1], strat_tops[-1]], 'k--', linewidth=0.5)
        ax1.plot([time[-1], time[-1]], [strat_tops[-1], ylim2], 'k--', linewidth=0.5)
            
        # # plot chronostratigraphic units:      
        # for i in range(len(new_ts_labels)): 
        #     x = [new_times[i], new_times[i+1], new_times[i+1], new_times[i]]
        #     y = [0, 0, 1, 1]
        #     if (new_ts_labels[i] == 2 or new_ts_labels[i] == 3): # vacuity
        #         ax3.fill(x, y, facecolor='xkcd:light grey')
        #     elif (new_ts_labels[i] == -1): # erosion 
        #         ax3.fill(x, y, facecolor='xkcd:red')
        #     elif (new_ts_labels[i] == 1): # preserved deposition 
        #         ax3.fill(x, y, facecolor='xkcd:medium blue')
        #     elif (new_ts_labels[i] == 0): # preserved stasis 
        #         ax3.fill(x, y, facecolor='orange')
        #     if i>0:
        #         if new_ts_labels[i] != new_ts_labels[i-1]:
        #             ax1.plot([new_times[i], new_times[i]], 
        #                      [new_elevation[i], max_elevation + 0.02 * elev_range], 'k--', linewidth=0.5)

    inds = np.where(new_ts_labels == 1)[0]
    deposition_time = np.sum(np.diff(new_times)[inds])
    deposition_thickness = np.sum(np.diff(new_elevation_smooth)[inds])
    inds = np.where(new_ts_labels == -1)[0]
    erosion_time = np.sum(np.diff(new_times)[inds])
    erosion_thickness = np.sum(np.diff(new_elevation_smooth)[inds])
    inds = np.where(new_ts_labels == 0)[0]
    stasis_time = np.sum(np.diff(new_times)[inds])
    stasis_thickness = np.sum(np.diff(new_elevation_smooth)[inds])
    inds = np.where((new_ts_labels == 2) | (new_ts_labels == 3))[0]
    vacuity_time = np.sum(np.diff(new_times)[inds])
    dve_data = [deposition_time, erosion_time, stasis_time, vacuity_time, deposition_thickness, 
                erosion_thickness, stasis_thickness]

    temp = ts_labels.copy()
    temp[temp == 2] = 1
    bound_inds = np.where(np.diff(temp) != 0)[0] + 1
    bound_inds = np.hstack((0, bound_inds, len(temp)))
    time_bounds = time[bound_inds]
    elevation_bounds = elevation[bound_inds]
    interval_labels = temp[bound_inds[:-1]]
    stasis_durations = np.diff(time_bounds)[interval_labels == 0]
    deposition_durations = np.diff(time_bounds)[interval_labels == 1]
    erosion_durations = np.diff(time_bounds)[interval_labels == -1]
    deposition_thicknesses = np.diff(elevation_bounds)[interval_labels == 1]
    erosion_thicknesses = np.diff(elevation_bounds)[interval_labels == -1]
    duration_thickness_data = [deposition_durations, stasis_durations, erosion_durations, deposition_thicknesses, erosion_thicknesses]

    temp = new_ts_labels.copy()
    bound_inds = np.where(np.diff(temp) != 0)[0] + 1
    bound_inds = np.hstack((0, bound_inds, len(temp)))
    interval_labels = temp[bound_inds[:-1]]

    if plotting:
        for i in range(len(interval_labels)):
            if interval_labels[i] == 0:
                ax1.plot(new_times[bound_inds[i]:bound_inds[i+1]+1], new_elevation[bound_inds[i]:bound_inds[i+1]+1], 'orange', linewidth=3)
            if interval_labels[i] == -1:
                ax1.plot(new_times[bound_inds[i]:bound_inds[i+1]+1], new_elevation[bound_inds[i]:bound_inds[i+1]+1], 'r', linewidth=3)
            if interval_labels[i] == 1:
                ax1.plot(new_times[bound_inds[i]:bound_inds[i+1]+1], new_elevation[bound_inds[i]:bound_inds[i+1]+1], 'xkcd:medium blue', linewidth=3)
            if interval_labels[i] == 2:
                ax1.plot(new_times[bound_inds[i]:bound_inds[i+1]+1], new_elevation[bound_inds[i]:bound_inds[i+1]+1], 'gray', linewidth=3)

        # plot stasis surfaces:
        if len(stasis_tops) > 0:
            for i in range(len(stasis_end_inds)):
                if new_ts_labels[stasis_start_inds[i]-1] != 2.0 and new_ts_labels[stasis_start_inds[i]-1] != -1.0:
                    ax1.plot([stasis_ages[i], max_time], [stasis_tops[i], stasis_tops[i]], 'k--', linewidth=0.5)
                    if stasis_tops[i] not in strat_tops:
                        ax2.plot([0, 1], [stasis_tops[i], stasis_tops[i]], color = 'orange', linewidth = 1)

        # plot chronostratigraphic units:      
        for i in range(len(interval_labels)): 
            x = [new_times[bound_inds[i]], new_times[bound_inds[i+1]], new_times[bound_inds[i+1]], new_times[bound_inds[i]]]
            y = [0, 0, 1, 1]
            if (interval_labels[i] == 2 or interval_labels[i] == 3): # vacuity
                ax3.fill(x, y, facecolor='xkcd:light grey')
            elif (interval_labels[i] == -1): # erosion 
                ax3.fill(x, y, facecolor='xkcd:red')
            elif (interval_labels[i] == 1): # preserved deposition 
                ax3.fill(x, y, facecolor='xkcd:medium blue')
            elif (interval_labels[i] == 0): # preserved stasis 
                ax3.fill(x, y, facecolor='orange')
            ax1.plot([new_times[bound_inds[i]], new_times[bound_inds[i]]], 
                [new_elevation[bound_inds[i]], max_elevation + 0.02 * elev_range], 'k--', linewidth=0.5)

        y1 = 0.55
        y2 = 0.15
        y = [y1, y1, y2, y2]
        x1 = 0
        x2 = 2 * deposition_time/(time[-1]-time[0])
        x = [x1, x2, x2, x1]
        ax4.fill(x, y, facecolor='xkcd:medium blue', edgecolor = 'k', zorder = 1000)
        ax4.axis('off')
        ax4.text(x1, y1 + 0.07, 'deposition', fontsize = 12)
        ax4.text(x2 + 0.05, 0.27, str(np.round(deposition_time/(time[-1]-time[0]), 3)), fontsize = 10, color = 'k',zorder=2000)
        
        x1 = 2 
        x2 = x1 + 2 * erosion_time/(time[-1]-time[0])
        x = [x1, x2, x2, x1]
        ax4.fill(x, y, facecolor='xkcd:red', edgecolor = 'k', zorder = 1001)
        ax4.text(x1, y1 + 0.07, 'erosion', fontsize = 12)
        ax4.text(x2 + 0.05, 0.27, str(np.round(erosion_time/(time[-1]-time[0]), 3)), fontsize = 10, color = 'k',zorder=2000)

        x1 = 4
        x2 = x1 + 2 * stasis_time/(time[-1]-time[0])
        x = [x1, x2, x2, x1]
        ax4.fill(x, y, facecolor='xkcd:marigold', edgecolor = 'k', zorder = 1002)
        ax4.text(x1, y1 + 0.07, 'stasis', fontsize = 12)
        ax4.text(x2 + 0.05, 0.27, str(np.round(stasis_time/(time[-1]-time[0]), 3)), fontsize = 10, color = 'k',zorder=2000)
        
        x1 = 6
        x2 = x1 + 2 * vacuity_time/(time[-1]-time[0])
        x = [x1, x2, x2, x1]
        ax4.fill(x, y, facecolor='xkcd:light grey', edgecolor = 'k', zorder = 1002)
        ax4.text(x1, y1 + 0.07, 'vacuity', fontsize = 12)
        ax4.text(x2 + 0.05, 0.27, str(np.round(vacuity_time/(time[-1]-time[0]), 3)), fontsize = 10, color = 'k',zorder=2000)
    return fig, dve_data, duration_thickness_data, new_ts_labels, strat_tops, strat_top_inds, bound_inds, interval_labels

def find_intersection_point(x1, y1, x2, y2, x3, y3, x4, y4):
    # Calculate slopes and y-intercepts of the two lines
    m1 = (y2 - y1) / (x2 - x1)
    b1 = y1 - m1 * x1
    m2 = (y4 - y3) / (x4 - x3)
    b2 = y3 - m2 * x3
    # Calculate the x-coordinate of the intersection point
    x_intersect = (b2 - b1) / (m1 - m2)
    # Calculate the y-coordinate using either of the line equations
    y_intersect = m1 * x_intersect + b1
    return x_intersect, y_intersect

def smooth_elevation_series(etas, res):
    etas_new = np.zeros(np.shape(etas))
    eta_old = etas[0]
    for i in range(len(etas)):
        eta_test = etas[i]
        if np.abs(eta_test - eta_old) >= res:
            eta_old = eta_test
            etas_new[i] = eta_test
        if np.abs(eta_test - eta_old) < res:
            etas_new[i] = eta_old
    return etas_new

def smooth_elevation_series_2D(topo, res):
    topo_new = np.zeros(np.shape(topo))
    topo_old = topo[:,0].copy()
    for i in range(1,topo.shape[1]):
        topo_test = topo[:,i].copy()
        topo_old[np.abs(topo_test - topo_old) >= res] = topo_test[np.abs(topo_test - topo_old) >= res]
        topo_new[:,i][np.abs(topo_test - topo_old) >= res] = topo_test[np.abs(topo_test - topo_old) >= res]
        topo_new[:,i][np.abs(topo_test - topo_old) < res] = topo_old[np.abs(topo_test - topo_old) < res]
    return topo_new

def smooth_elevation_series_3D(topo, res):
    topo_new = np.zeros(np.shape(topo))
    topo_old = topo[:,:,0].copy()
    for i in range(1,topo.shape[2]):
        topo_test = topo[:,:,i].copy()
        topo_old[np.abs(topo_test - topo_old) >= res] = topo_test[np.abs(topo_test - topo_old) >= res]
        topo_new[:,:,i][np.abs(topo_test - topo_old) >= res] = topo_test[np.abs(topo_test - topo_old) >= res]
        topo_new[:,:,i][np.abs(topo_test - topo_old) < res] = topo_old[np.abs(topo_test - topo_old) < res]
    return topo_new

def smooth_strat_attribute(surf, smoothing_window):
    smooth_surf = surf.copy()
    smooth_surf[np.isnan(smooth_surf) == 1] = 0
    smooth_surf=sgolay2d(smooth_surf, smoothing_window, 3)
    smooth_surf[np.isnan(surf)==1] = np.nan
    return smooth_surf

def find_stasis(elevation):
    s = np.minimum.accumulate(elevation[::-1])[::-1] # stratigraphic 'elevation'
    inds = np.where(np.diff(s) == 0)[0]
    if len(inds) > 0:
        inds1 = np.where(np.diff(inds) > 1)[0] + 1
        inds1 = np.hstack((0, inds1))
        inds1 = inds[inds1]
        inds2 = []
        for i in range(1, len(s)-1):
            if s[i] - s[i-1] == 0 and s[i+1] - s[i] > 0:
                inds2.append(i)
    else:
        inds1 = []
        inds2 = []
    return inds1, inds2

def topostrat(topo):
    # convert topography to stratigraphy
    if len(np.shape(topo)) == 2:
        # strat = np.minimum.accumulate(topo[::-1, :], axis=0)[::-1, :]
        strat = np.minimum.accumulate(topo[:, ::-1], axis=1)[:, ::-1]
    if len(np.shape(topo)) == 3:
        strat = np.minimum.accumulate(topo[:, :, ::-1], axis=2)[:, :, ::-1]
    return strat

# def create_wheeler_diagram_old(topo):
#     """create Wheeler (chronostratigraphic) diagram from a set of topographic surfaces
#     """
#     strat = topostrat(topo) # convert topography to stratigraphy
#     wheeler = np.diff(topo, axis=2) # 'normal' Wheeler diagram
#     wheeler_strat = np.diff(strat, axis=2) # array for Wheeler diagram with vacuity blanked out; this array will be a positive number if there is preserved depostion, zero otherwise
#     vacuity = np.zeros(np.shape(wheeler)) # array for vacuity 
#     vacuity[(wheeler>0) & (wheeler_strat==0)] = 1 # make the 'vacuity' array 1 where there was deposition (wheeler > 0) but stratigraphy is not preserved (wheeler_strat = 0)
#     wheeler_strat[wheeler<0] = wheeler[wheeler<0] # add erosion to 'wheeler_strat' (otherwise it would only show deposition)
#     return strat, wheeler, wheeler_strat, vacuity

def create_wheeler_diagram(topo, res):
    # rewritten so that it is consistent with the way stasis is computed with the 'plot_strat_diagram' function
    topo_new = smooth_elevation_series_3D(topo, res)
    wheeler = np.zeros((topo.shape[0], topo.shape[1], topo.shape[2]-1))
    wheeler[np.diff(topo_new, axis=-1) > 0] = np.diff(topo, axis=-1)[np.diff(topo_new, axis=-1) > 0]
    wheeler[np.diff(topo_new, axis=-1) < 0] = np.diff(topo, axis=-1)[np.diff(topo_new, axis=-1) < 0]
    strat = topostrat(topo)
    wheeler_strat = np.diff(strat, axis=2)
    wheeler_strat[wheeler == 0] = 0
    vacuity = np.zeros(np.shape(wheeler)) # array for vacuity 
    vacuity[(wheeler>0) & (wheeler_strat==0)] = 1 # make the 'vacuity' array 1 where there was deposition (wheeler > 0) but stratigraphy is not preserved (wheeler_strat = 0)
    wheeler_strat[wheeler<0] = wheeler[wheeler<0] # add erosion to 'wheeler_strat' (otherwise it would only show deposition)
    return strat, wheeler, wheeler_strat, vacuity

def create_wheeler_diagram_2D(topo, res):
    # rewritten so that it is consistent with the way stasis is computed with the 'plot_strat_diagram' function
    topo_new = smooth_elevation_series_2D(topo, res)
    wheeler = np.zeros((topo.shape[0], topo.shape[1]-1))
    wheeler[np.diff(topo_new, axis=-1) > 0] = np.diff(topo, axis=-1)[np.diff(topo_new, axis=-1) > 0]
    wheeler[np.diff(topo_new, axis=-1) < 0] = np.diff(topo, axis=-1)[np.diff(topo_new, axis=-1) < 0]
    strat = topostrat(topo)
    wheeler_strat = np.diff(strat, axis=-1)
    wheeler_strat[wheeler == 0] = 0
    vacuity = np.zeros(np.shape(wheeler)) # array for vacuity 
    vacuity[(wheeler>0) & (wheeler_strat==0)] = 1 # make the 'vacuity' array 1 where there was deposition (wheeler > 0) but stratigraphy is not preserved (wheeler_strat = 0)
    wheeler_strat[wheeler<0] = wheeler[wheeler<0] # add erosion to 'wheeler_strat' (otherwise it would only show deposition)
    stasis = np.zeros(np.shape(wheeler))
    stasis[wheeler == 0] = 1
    return strat, wheeler, wheeler_strat, vacuity, stasis

def compute_strat_maps(strat, wheeler, wheeler_strat, vacuity):
    temp = wheeler_strat.copy()
    temp[wheeler_strat<=0] = 0
    temp[wheeler_strat>0] = 1
    deposition_time = np.sum(temp, axis=-1)/wheeler.shape[-1]
    temp = wheeler.copy()
    temp[wheeler>=0] = 0
    temp[wheeler<0] = 1
    erosion_time = np.sum(temp, axis=-1)/wheeler.shape[-1]
    temp = np.zeros(np.shape(wheeler))
    temp[wheeler==0] = 1
    stasis_time = np.sum(temp, axis=-1)/wheeler.shape[-1]
    vacuity_time = np.sum(vacuity, axis=-1)/wheeler.shape[-1]
    deposition_thickness = strat[:,:,-1] - strat[:,:,0]
    temp = wheeler.copy()
    temp[wheeler>=0] = 0
    erosion_thickness = np.sum(temp, axis=-1)
    return deposition_time, erosion_time, stasis_time, vacuity_time, deposition_thickness, erosion_thickness

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

def resample_and_smooth(x, y, delta_s, smoothing_factor):
    dx = np.diff(x); dy = np.diff(y)      
    ds = np.sqrt(dx**2 + dy**2)
    tck, u = scipy.interpolate.splprep([x,y],s=smoothing_factor) # parametric spline representation of curve
    unew = np.linspace(0, 1, int(1+sum(ds)/delta_s)) # vector for resampling
    out = scipy.interpolate.splev(unew, tck) # resampling
    xs = out[0]
    ys = out[1]
    return xs, ys

def resample_elevation_spl(time, elevation, sampling_rate):
    spl = interpolate.splrep(time, elevation, s=5)
    time_new = np.arange(time[0], time[-1]+1, sampling_rate)
    elevation_new = interpolate.splev(time_new, spl)
    return time_new, elevation_new

def resample_elevation_int1d(time, elevation, sampling_rate):
    f = interpolate.interp1d(time, elevation)
    time_new = np.arange(time[0], time[-1]+0.001, sampling_rate)
    elevation_new = f(time_new)
    return time_new, elevation_new

def sgolay2d(z, window_size, order, derivative=None): # this is from: https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
    """
    """
    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0

    if  window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial. 
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ... 
    # this line gives a list of two item tuple. Each tuple contains 
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]

    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat( ind, window_size )
    dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty( (window_size**2, len(exps)) )
    for i, exp in enumerate( exps ):
        A[:,i] = (dx**exp[0]) * (dy**exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros( (new_shape) )
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( z[1:half_size+1, :] ) - band )
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( z[-half_size-1:-1, :] )  -band )
    # left band
    band = np.tile( z[:,0].reshape(-1,1), [1,half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band )
    # right band
    band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )
    Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band )
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0,0]
    Z[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - band )
    # bottom right corner
    band = z[-1,-1]
    Z[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - band )

    # top right corner
    band = Z[half_size,-half_size:]
    Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band )
    # bottom left corner
    band = Z[-half_size:,half_size].reshape(-1,1)
    Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band )

    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid'), scipy.signal.fftconvolve(Z, -c, mode='valid')

def plot_surf_w_texture(strat, scale, dx, ve, texture, opacity, cmap, mask, kmeans_colors):
    """
    Plot surface in 3D and drape it with an RGB image, using k-means clustering 
    (because other approaches don't seem to work)
    """
    r,c,ts = np.shape(strat)
    z = scale*strat[:,:,ts-1].copy()
    z[mask==1] = np.nan
    X1 = scale*(np.linspace(0,c-1,c)*dx) # x goes with c and y with r
    Y1 = scale*(np.linspace(0,r-1,r)*dx)
    X1_grid , Y1_grid = np.meshgrid(X1, Y1)
    surf = mlab.mesh(X1_grid, Y1_grid, z*ve, scalars = texture, mask = mask, colormap=cmap, vmin=0, vmax=255)
    lut = surf.module_manager.scalar_lut_manager.lut.table.to_array()
    lut[:,:3] = kmeans_colors
    surf.module_manager.scalar_lut_manager.lut.table = lut

def line_coefficients(x1, y1, x2, y2):
    # Calculate the slope of the line
    slope = (y2 - y1) / (x2 - x1)
    # Calculate the y-intercept
    y_intercept = y1 - slope * x1
    return slope, y_intercept

# this function is from: https://github.com/NapsterInBlue/Movie-Posters-2017/blob/master/imagetools.py
def pick_colors(vec, numColors):
    '''
    Do k-means clustering over ``vec`` to return ``numColors``
    '''
    vec = vec.reshape(-1, 3)
    model = KMeans(n_clusters=numColors, n_init='auto').fit(vec)
    return model, model.cluster_centers_

def create_animation_frames_from_views(mlab_view_params1, mlab_view_params2, n_steps, fname, start_no=0):
    """
    Save Mayavi figures as animation frames after picking the 'mlab.view' parameters from a spectrum 
    between two end member views.
    """
    for i in range(n_steps):
        az = np.linspace(mlab_view_params1[0], mlab_view_params2[0], n_steps)[i]
        el = np.linspace(mlab_view_params1[1], mlab_view_params2[1], n_steps)[i]
        dist = np.linspace(mlab_view_params1[2], mlab_view_params2[2], n_steps)[i]
        fp1 = np.linspace(mlab_view_params1[3], mlab_view_params2[3], n_steps)[i]
        fp2 = np.linspace(mlab_view_params1[4], mlab_view_params2[4], n_steps)[i]
        fp3 = np.linspace(mlab_view_params1[5], mlab_view_params2[5], n_steps)[i]
        mlab.view(azimuth=az, elevation=el, distance=dist, focalpoint=np.array([fp1, fp2, fp3]))
        mlab.savefig(fname+'%03d.png'%(start_no+i))

def read_and_track_line(filename):
    t = plt.imread(filename).astype(float)
    t[t==0] = 1 # make sure that t only has zeros and ones
    t[t==255] = 0
    y_pix,x_pix = np.where(t==1)
    # find starting pixel on left side of image
    ind = np.where(x_pix==np.min(x_pix))[0][0]
    x0 = x_pix[ind]
    y0 = y_pix[ind]
    start_ind = np.where((x_pix==x0) & (y_pix==y0))[0][0]
    dist = distance.cdist(np.array([x_pix,y_pix]).T,np.array([x_pix,y_pix]).T)
    dist[np.diag_indices_from(dist)]=100.0
    ind = start_ind
    clinds = [ind]
    count = 0
    while count<len(x_pix):
        d = dist[ind,:].copy()
        if len(clinds)>2:
            d[clinds[-2]]=d[clinds[-2]]+100.0
            d[clinds[-3]]=d[clinds[-3]]+100.0
        if len(clinds)==2:
            d[clinds[-2]]=d[clinds[-2]]+100.0
        ind = np.argmin(d)
        clinds.append(ind)
        count=count+1
    x_pix = x_pix[clinds]
    y_pix = y_pix[clinds]
    return x_pix, y_pix

def compute_erosional_surf_attributes(strat, time, topo_s, erosion_threshold = 0.1):
    erosional_surfs_age_below = -1*np.ones(np.shape(strat))
    erosional_surfs_age_above = -1*np.ones(np.shape(strat))
    erosional_surfs_time = -1*np.ones(np.shape(strat))
    erosional_surfs_thickness = -1*np.ones(np.shape(strat))
    if strat.ndim == 3:
        for r in trange(strat.shape[0]):
            for c in range(strat.shape[1]):
                elevation = topo_s[r, c, :].copy()
                fig, dve_data, duration_thickness_data, ts_labels, strat_tops, strat_top_inds, bound_inds, interval_labels\
                      = plot_strat_diagram(elevation, 'mm', time, 'seconds', 0.5, 
                            np.max(elevation), np.max(time), plotting=False, plot_raw_data=True)
                for i in range(0,len(strat_tops)):
                    inds = np.where(np.abs(strat[r, c, :] - strat[r, c, strat_top_inds[i]])<erosion_threshold)[0] # 0.0001
                    if len(inds) > 0:
                        for ind in inds:
                            erosional_surfs_age_below[r, c, ind] = inds[0]
                            erosional_surfs_age_above[r, c, ind] = inds[-1]
                            erosional_surfs_time[r, c, ind] = len(inds)
                        eroded_thickness = np.max(elevation[inds[0]:inds[-1]+1]) - elevation[inds[-1]]
                        erosional_surfs_thickness[r, c, inds[0]:inds[-1]+1] = eroded_thickness
    if strat.ndim == 2:
        for r in trange(strat.shape[0]):
            elevation = topo_s[r, :].copy()
            fig, dve_data, duration_thickness_data, ts_labels, strat_tops, strat_top_inds, bound_inds, interval_labels \
                  = plot_strat_diagram(elevation, 'mm', time, 'minutes', 0.5, 
                        np.max(elevation), np.max(time), plotting=False, plot_raw_data=True)
            for i in range(0,len(strat_top_inds)):
                inds = np.where(np.abs(strat[r, :] - strat[r, strat_top_inds[i]])<erosion_threshold)[0]
                if len(inds) > 0:
                    for ind in inds:
                        erosional_surfs_age_below[r, ind] = inds[0]
                        erosional_surfs_age_above[r, ind] = inds[-1]
                        erosional_surfs_time[r, ind] = len(inds)
                    eroded_thickness = np.max(elevation[inds[0]:inds[-1]+1]) - elevation[inds[-1]]
                    erosional_surfs_thickness[r, inds[0]:inds[-1]+1] = eroded_thickness
    return erosional_surfs_age_below, erosional_surfs_age_above, erosional_surfs_time, erosional_surfs_thickness