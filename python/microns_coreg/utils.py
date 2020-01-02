import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.transforms import Bbox
from IPython.display import HTML
from matplotlib import animation
from mpl_toolkits.mplot3d import axes3d, Axes3D
import torch
from torch.nn import functional as F
import re
from PIL import Image
import json
import urllib
from scipy import ndimage

def clock2math(t): 
    """from Paul Fahey""" 
    return ((np.pi / 2) - t) % (2 * np.pi)

def create_grid(um_sizes, desired_res=1):
    """ Create a grid corresponding to the sample position of each pixel/voxel in a FOV of
     um_sizes at resolution desired_res. The center of the FOV is (0, 0, 0).
    In our convention, samples are taken in the center of each pixel/voxel, i.e., a volume
    centered at zero of size 4 will have samples at -1.5, -0.5, 0.5 and 1.5; thus edges
    are NOT at -2 and 2 which is the assumption in some libraries.
    :param tuple um_sizes: Size in microns of the FOV, .e.g., (d1, d2, d3) for a stack.
    :param float or tuple desired_res: Desired resolution (um/px) for the grid.
    :return: A (d1 x d2 x ... x dn x n) array of coordinates. For a stack, the points at
    each grid position are (x, y, z) points; (x, y) for fields. Remember that in our stack
    coordinate system the first axis represents z, the second, y and the third, x so, e.g.,
    p[10, 20, 30, 0] represents the value in x at grid position 10, 20, 30.
    """
    # Make sure desired_res is a tuple with the same size as um_sizes
    if np.isscalar(desired_res):
        desired_res = (desired_res,) * len(um_sizes)

    # Create grid
    out_sizes = [int(round(um_s / res)) for um_s, res in zip(um_sizes, desired_res)]
    um_grids = [np.linspace(-(s - 1) * res / 2, (s - 1) * res / 2, s, dtype=np.float32)
                for s, res in zip(out_sizes, desired_res)] # *
    full_grid = np.stack(np.meshgrid(*um_grids, indexing='ij')[::-1], axis=-1)
    # * this preserves the desired resolution by slightly changing the size of the FOV to
    # out_sizes rather than um_sizes / desired_res.

    return full_grid


def resize(original, um_sizes, desired_res):
    """ Resize array originally of um_sizes size to have desired_res resolution.
    We preserve the center of original and resized arrays exactly in the middle. We also
    make sure resolution is exactly the desired resolution. Given these two constraints,
    we cannot hold FOV of original and resized arrays to be exactly the same.
    :param np.array original: Array to resize.
    :param tuple um_sizes: Size in microns of the array (one per axis).
    :param int or tuple desired_res: Desired resolution (um/px) for the output array.
    :return: Output array (np.float32) resampled to the desired resolution. Size in pixels
        is round(um_sizes / desired_res).
    """

    # Create grid to sample in microns
    grid = create_grid(um_sizes, desired_res) # d x h x w x 3

    # Re-express as a torch grid [-1, 1]
    um_per_px = np.array([um / px for um, px in zip(um_sizes, original.shape)])
    torch_ones = np.array(um_sizes) / 2 - um_per_px / 2  # sample position of last pixel in original
    grid = grid / torch_ones[::-1].astype(np.float32)

    # Resample
    input_tensor = torch.from_numpy(original.reshape(1, 1, *original.shape).astype(
        np.float32))
    grid_tensor = torch.from_numpy(grid.reshape(1, *grid.shape))
    resized_tensor = F.grid_sample(input_tensor, grid_tensor, padding_mode='border')
    resized = resized_tensor.numpy().squeeze()

    return resized


def affine_product(X, A, b):
    """ Special case of affine transformation that receives coordinates X in 2-d (x, y)
    and affine matrix A and translation vector b in 3-d (x, y, z). Y = AX + b
    :param torch.Tensor X: A matrix of 2-d coordinates (d1 x d2 x 2).
    :param torch.Tensor A: The first two columns of the affine matrix (3 x 2).
    :param torch.Tensor b: A 3-d translation vector.
    :return: A (d1 x d2 x 3) torch.Tensor corresponding to the transformed coordinates.
    """
    return torch.einsum('ij,klj->kli', (A, X)) + b

def sample_grid(volume, grid):
    """ 
    Sample grid in volume.

    Assumes center of volume is at (0, 0, 0) and grid and volume have the same resolution.

    :param torch.Tensor volume: A d x h x w tensor. The stack.
    :param torch.Tensor grid: A d1 x d2 x 3 (x, y, z) tensor. The coordinates to sample.

    :return: A d1 x d2 tensor. The grid sampled in the stack.
    """
    # Make sure input is tensor
    volume = torch.as_tensor(volume, dtype=torch.float32)
    grid = torch.as_tensor(grid, dtype=torch.float32)

    # Rescale grid so it ranges from -1 to 1 (as expected by F.grid_sample)
    norm_factor = torch.as_tensor([s / 2 - 0.5 for s in volume.shape[::-1]])
    norm_grid = grid / norm_factor

    # Resample
    resampled = F.grid_sample(volume[None, None, ...], norm_grid[None, None, ...], padding_mode='zeros')
    resampled = resampled.squeeze() # drop batch and channel dimension

    return resampled

def html_to_json(url_string, return_parsed_url=False, fragment_prefix='!'):
    # Parse neuromancer url to logically separate the json state dict from the rest of it.
    full_url_parsed = urllib.parse.urlparse(url_string)
    # Decode percent-encoding in url, and skip "!" from beginning of string.
    decoded_fragment = urllib.parse.unquote(full_url_parsed.fragment)
    if decoded_fragment.startswith(fragment_prefix):
        decoded_fragment = decoded_fragment[1:]
    # Load the json state dict string into a python dictionary.
    json_state_dict = json.loads(decoded_fragment)

    if return_parsed_url:
        return json_state_dict, full_url_parsed
    else:
        return json_state_dict

def add_point_annotations(provided_link, ano_name, ano_list, voxelsize, color='#f1ff00', overwrite=True):
    # format annotation list
    ano_list_dict = []
    if ano_list.ndim<2:
        ano_list = np.expand_dims(ano_list,0)
    if ano_list.ndim>2:
        return print('The annotation list must be 1D or 2D')
    for i, centroid in enumerate(ano_list.tolist()):
        ano_list_dict.append({'point':centroid, 'type':'point', 'id':str(i+1), "tagIds":[]})

    json_data, parsed_url = html_to_json(provided_link, return_parsed_url=True)
    # if annotation layer doesn't exist, create it
    if re.search(ano_name,json.dumps(json_data)) is None:
        json_data['layers'].append({'tool': 'annotatePoint',
                               'type': 'annotation',
                               'annotations': [],
                               'annotationColor': color,
                               'annotationTags': [],
                               'voxelSize': voxelsize,
                               'name': ano_name})
        print('annotation layer does not exist... creating it')
    annotation_dict = list(filter(lambda d: d['name'] == ano_name, json_data['layers']))
    annotation_ind = np.where(np.array(json_data['layers']) == annotation_dict)[0][0].squeeze()
    # test if voxel size of annotation matches provided voxel size
    if json_data['layers'][annotation_ind]['voxelSize']!=voxelsize:
        return print('The annotation layer already exists but does not match your provided voxelsize')
    # add annotations
    if overwrite:
        json_data['layers'][annotation_ind]['annotations'] = ano_list_dict
    else:
        json_data['layers'][annotation_ind]['annotations'].extend(ano_list_dict)

    return urllib.parse.urlunparse([parsed_url.scheme, parsed_url.netloc, parsed_url.path, parsed_url.params, parsed_url.query, '!'+ urllib.parse.quote(json.dumps(json_data))])

def add_ellipsoid_annotations(provided_link, ano_name, ano_list, radii, voxelsize, color='#00ff2c', overwrite=True):
    # format annotation list
    ano_list_dict = []
    if ano_list.ndim<2:
        ano_list = np.expand_dims(ano_list,0)
    if ano_list.ndim>2:
        return print('The annotation list must be 1D or 2D')
    for i, centroid in enumerate(ano_list.tolist()):
        ano_list_dict.append({'center':centroid, 'radii':radii, 'type':'ellipsoid', 'id':str(i+1), "tagIds":[]})

    json_data, parsed_url = html_to_json(provided_link, return_parsed_url=True)
    # if annotation layer doesn't exist, create it
    if re.search(ano_name,json.dumps(json_data)) is None:
        json_data['layers'].append({'tool': 'annotateSphere',
                               'type': 'annotation',
                               'annotations': [],
                               'annotationColor': color,
                               'annotationTags': [],
                               'voxelSize': voxelsize,
                               'name': ano_name})
        print('annotation layer does not exist... creating it')
    annotation_dict = list(filter(lambda d: d['name'] == ano_name, json_data['layers']))
    annotation_ind = np.where(np.array(json_data['layers']) == annotation_dict)[0][0].squeeze()
    # test if voxel size of annotation matches provided voxel size
    if json_data['layers'][annotation_ind]['voxelSize']!=voxelsize:
        return print('The annotation layer already exists but does not match your provided voxelsize')
    # add annotations
    if overwrite:
        json_data['layers'][annotation_ind]['annotations'] = ano_list_dict
    else:
        json_data['layers'][annotation_ind]['annotations'].extend(ano_list_dict)

    return urllib.parse.urlunparse([parsed_url.scheme, parsed_url.netloc, parsed_url.path, parsed_url.params, parsed_url.query, '!'+ urllib.parse.quote(json.dumps(json_data))])



def add_segments(provided_link, segments, overwrite=True, color=None):
    json_data, parsed_url = html_to_json(provided_link, return_parsed_url=True)
    seg_strings = []
    for seg in segments:
        seg_strings.append(seg.astype(np.str))
    segmentation_layer = list(filter(lambda d: d['type'] == 'segmentation', json_data['layers']))
    if re.search('segments',json.dumps(json_data)) is None:
        segmentation_layer[0].update({'segments':[]})
    if overwrite:
        segmentation_layer[0]['segments'] = seg_strings
    else:
        segmentation_layer[0]['segments'].extend(seg_strings)
    if color is not None:
        if re.search('segmentColors',json.dumps(json_data)) is None:
            segmentation_layer[0].update({'segmentColors':{}})
        color_dict = {}
        for seg in segments:
            color_dict.update({str(seg):color})
        segmentation_layer[0]['segmentColors'] = color_dict
            
    return urllib.parse.urlunparse([parsed_url.scheme, parsed_url.netloc, parsed_url.path, parsed_url.params, parsed_url.query, '!'+ urllib.parse.quote(json.dumps(json_data))])

def transfer_annotations(source_link, target_link, ano_name, voxelsize):
    src_json_data, src_parsed_url = html_to_json(source_link, return_parsed_url=True)
    trg_json_data, trg_parsed_url = html_to_json(target_link, return_parsed_url=True)
    
    # check if annotation layer exists in source link
    if re.search(ano_name,json.dumps(src_json_data)) is None:
        return print('annotation layer does not exist')
    # get annotation to transfer
    src_annotation_dict = list(filter(lambda d: d['name'] == ano_name, src_json_data['layers']))
    src_annotation_ind = np.where(np.array(src_json_data['layers']) == src_annotation_dict)[0][0].squeeze()
    # test if voxel size of annotation matches provided voxel size
    if src_json_data['layers'][src_annotation_ind]['voxelSize']!=voxelsize:
        return print('The annotation layer already exists but does not match your provided voxelsize')
    # check if annotation exists in target, if it doesn't then create it
    if re.search(ano_name,json.dumps(trg_json_data)) is None:
        trg_json_data['layers'].append({'name': ano_name})
    
    # find annotation in target
    trg_annotation_dict = list(filter(lambda d: d['name'] == ano_name, trg_json_data['layers']))
    trg_annotation_ind = np.where(np.array(trg_json_data['layers']) == trg_annotation_dict)[0][0].squeeze()
    
    # transfer annotations
    trg_json_data['layers'][trg_annotation_ind] = src_json_data['layers'][src_annotation_ind]

    return urllib.parse.urlunparse([trg_parsed_url.scheme, trg_parsed_url.netloc, trg_parsed_url.path, trg_parsed_url.params, trg_parsed_url.query, '!'+ urllib.parse.quote(json.dumps(trg_json_data))])

def coordinate(grid_to_transform):
    x = grid_to_transform.shape[0]
    y = grid_to_transform.shape[1]
    return grid_to_transform.reshape(x*y,-1)

def uncoordinate(transformed_coordinates,x,y):
    return transformed_coordinates.reshape(x,y,-1)

def lcn(image, sigmas=(12, 12)):
    """ Local contrast normalization.
    Normalize each pixel using mean and stddev computed on a local neighborhood.
    We use gaussian filters rather than uniform filters to compute the local mean and std
    to soften the effect of edges. Essentially we are using a fuzzy local neighborhood.
    Equivalent using a hard defintion of neighborhood will be:
        local_mean = ndimage.uniform_filter(image, size=(32, 32))
    :param np.array image: Array with raw two-photon images.
    :param tuple sigmas: List with sigmas (one per axis) to use for the gaussian filter.
        Smaller values result in more local neighborhoods. 15-30 microns should work fine
    """
    local_mean = ndimage.gaussian_filter(image, sigmas)
    local_var = ndimage.gaussian_filter(image ** 2, sigmas) - local_mean ** 2
    local_std = np.sqrt(np.clip(local_var, a_min=0, a_max=None))
    norm = (image - local_mean) / (local_std + 1e-7)

    return norm


def sharpen_2pimage(image, laplace_sigma=0.7, low_percentile=3, high_percentile=99.9):
    """ Apply a laplacian filter, clip pixel range and normalize.
    :param np.array image: Array with raw two-photon images.
    :param float laplace_sigma: Sigma of the gaussian used in the laplace filter.
    :param float low_percentile, high_percentile: Percentiles at which to clip.
    :returns: Array of same shape as input. Sharpened image.
    """
    sharpened = image - ndimage.gaussian_laplace(image, laplace_sigma)
    clipped = np.clip(sharpened, *np.percentile(sharpened, [low_percentile, high_percentile]))
    norm = (clipped - clipped.mean()) / (clipped.max() - clipped.min() + 1e-7)
    return norm

def normalize(image, clip_bounds=None, newrange=[0,255], astype=np.uint8):
    assert type(image)==type(np.array(1))
    if clip_bounds is not None:
        image = np.clip(image,clip_bounds[0], clip_bounds[1]) 
    return (((image - image.min())*(newrange[1]-newrange[0])/(image.max() - image.min())) + newrange[0]).astype(astype)

def fix_boundaries(image):
    image = np.maximum(0,image - np.quantile(image,0.05))
    image = np.minimum(1, image / np.quantile(image,0.995))
    return image

def fetch_as_list_dict(dj_relation, attrs, keys_to_append=None):
    out = {}
    for i, entry in enumerate(np.stack(dj_relation.fetch(*attrs)).T):
        if len(attrs)>1:
            out.update({i:{attr:val for attr, val in zip(attrs, entry)}})
        else:
            out.update({i:{attrs[0]:entry}})
    if keys_to_append is not None:
        for key in keys_to_append:
            for i in range(len(out)):
                out[i].update(key)
    list_dict = []
    for item in out.items():
        list_dict.append(item[1])
    return list_dict

def animate_frames(frames, display_inches=4.0, vmin=0.0, vmax=1200.0, fps=10.0):
    """
    creator: Eric Wang
    Create an HTML5 video from raw video frames
    Args:
        frames: np.array of shape [frames, height, width]
        fps: int
        display_inches: float
    Return:
        HTML5 video
    """
    _, height, width = frames.shape
    interval = 1 / fps * 1000
    fig, ax = plt.subplots(1, 1, figsize=(display_inches * width / height, display_inches))
    ims = [[ax.imshow(im, animated=True, vmin=vmin, vmax=vmax)] for im in frames]
    ani = animation.ArtistAnimation(fig, ims, interval=interval)
    plt.tight_layout()
    plt.close(fig)
    return HTML(ani.to_html5_video())