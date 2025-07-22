import meshio as io
import numpy as np
from skimage.measure import label, regionprops
import nibabel as nib
from scipy import ndimage as ndi
from os import listdir
import os
import copy

def affineMatrix(control_p_div,affine):
    """
    Point projection using affine matrix. Scale is multiplied by -1 at x and y axis
    to orient the image Cranial-Caudal direction
    @param control_p_div: ndarray with spatial points
    @param affine: affine matrix
    @return: projected points
    """
    scale = np.asarray([-affine[0,0], -affine[1,1],affine[2,2]])
    pos   = np.asarray([affine[0,3], affine[1,3],affine[2,3]])

    control_p_aff = np.zeros((np.shape(control_p_div)[0],3))
    control_p_aff[:,0] = control_p_div[:,0]*scale[0]+pos[0]
    control_p_aff[:,1] = control_p_div[:,1]*scale[1]+pos[1]
    control_p_aff[:,2] = control_p_div[:,2]*scale[2]+pos[2]

    return control_p_aff

def get_last_fslash(path):
    """
    Finad last backslash
    Args:
        path: string with paths

    Returns:
        pos: int with last backslash position
    """
    word = copy.copy(path)
    pos = []
    for p,ch in enumerate(word):
        if '/' == ch:
            pos.append(p)
    pos = np.max(pos)
    return pos

def pipeLongitude(branch_father, segm = False):
    x = branch_father[:,0]
    x_dif = (x[:-1] - x[1:]) ** 2
    y = branch_father[:,1]
    y_dif = (y[:-1] - y[1:]) ** 2
    z = branch_father[:,2]
    z_dif = (z[:-1] - z[1:]) ** 2

    if not segm:
        dist = np.sum(np.sqrt(x_dif+y_dif+z_dif))
    else:
        dist = np.sqrt(x_dif + y_dif + z_dif)

    return dist

def get_paths(path,list_dir,ext=".gz"):
    """
    Recursively find files with extension 'ext' inside folder path 'path'.

    @param path: folder path input
    @param list_dir: as input, auxiliar function for recursion
    @param ext: file extension to search
    @return: list of paths to files with input extension
    """
    for f in listdir(path):
        path1 = f'{path}/{f}'
        if path1.endswith(ext):
            print(f'File found: {path1}')
            list_dir.append(path1)
        elif os.path.isdir(path1):
            list_dir = get_paths(path1, list_dir, ext)

    return list_dir

def off2vtu(fnamei,fnameo):
    mesh = io.read(fnamei)
    xyz_1ord,ien_1ord = mesh.points, mesh.cells[0].data
    cells={'triangle':ien_1ord}
    io.write_points_cells(fnameo,xyz_1ord,cells)


def geometry(path_nifti):
    """
    Load nifti image and properties.
    @param path_nifti: path to nifti image '.nii'
    @return:
        affine: affine matrix
        airway: ndarray containing the airway mask. Preprocess is made
                to select the object with the biggest size, avoiding
                disconnected airways to been considered.
        perim: airways perimeter from morphological operations.
    """
    img, airway, hdr, affine = load_nifty_image(path_nifti)

    airway = np.round(airway)
    airway = selMax_vol(airway)
    perim = np.nonzero(BWperim(airway).ravel() == 1)[0]
    perim = np.column_stack(np.unravel_index(perim, airway.shape))

    return affine, airway, perim

def load_nifty_image(img_dir):
    img = nib.load(img_dir)
    # data = np.array(img.get_data())      # <--- LÍNEA VIEJA, NO USAR
    data = img.get_fdata()                 # <--- LÍNEA NUEVA (array tipo float)
    hdr = img.header
    affine = img.affine
    return img, data, hdr, affine

def selMax_vol(BW):
    """
    Max volume airway selection to avoid disconnected airways.

    @param BW: ndarray containing the binary mask of the airways
    @return: selected biggest airway in ndarray
    """
    m, n, d = BW.shape
    L = label(BW)
    BW_stats = regionprops(L)

    volumen = [BW_stats[i].area for i in range(len(BW_stats))]
    volumen = np.asarray(volumen)

    ar = np.arange(0, len(volumen), 1.)

    volumen = ar[volumen == np.max(volumen)]

    BW = np.zeros((m, n, d))
    BW[L == volumen + 1] = 1

    return BW


def BWperim(image,neighbourhood=1):
    """
    Calculate binary mask image perimeter using morphological operations.
    @param image: input mask image.
    @param neighbourhood:
    @return: image containing binary mask perimeter.
    """
    if neighbourhood == 1:
        strel = ndi.generate_binary_structure(3, 1)
    else:
        strel = ndi.generate_binary_structure(3, 2)

    image = image.astype(np.uint8)
    eroded_image = ndi.binary_erosion(image, strel, border_value=0)
    border_image = image - eroded_image

    return border_image