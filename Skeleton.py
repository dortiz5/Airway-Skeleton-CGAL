import helpers as fn

import nibabel as nib
import numpy as np
import meshio as io
from skimage.measure import label, regionprops
import copy
from scipy import interpolate, ndimage


def niiCut(pathr, paths):
    affine, airway, _ = fn.geometry(pathr)

    strel = ndimage.generate_binary_structure(3, 1)
    airway = ndimage.binary_closing(airway, strel, border_value=0)

    xp, yp, zp = np.where(airway == 1)
    x_min, x_max = np.min(xp), np.max(xp)
    y_min, y_max = np.min(yp), np.max(yp)
    z_min, z_max = np.min(zp), np.max(zp)
    xs, ys, zs = x_max - x_min, y_max - y_min, z_max - z_min

    im_cut = np.zeros((xs + 20, ys + 20, zs + 20))
    im_cut[10:xs + 10, 10:ys + 10, 10:zs + 10] = \
        airway[x_min:x_max, y_min:y_max, z_min:z_max]

    im_cut = ndimage.binary_dilation(im_cut, strel, border_value=0, iterations=1)

    affine[0:3, -1] = 0  # +=np.asarray([x_min-10,y_min-10,z_min-10])
    save_nifty_image(paths + '.nii.gz', im_cut.astype(int), affine)


def save_nifty_image(img_dir, data, affine):
    # Fuerza el tipo de datos adecuado antes de guardar (por ejemplo int16)
    if data.dtype == 'int64':
        data = data.astype('int32')  # o int16 si tus datos caben en 16 bits
    new_image = nib.Nifti1Image(data, affine)
    nib.save(new_image, img_dir)


def load_nifty_image(img_dir):
    img = nib.load(img_dir)
    data = np.array(img.get_data())
    hdr = img.get_header()
    affine = img.get_affine()
    return img, data, hdr, affine


def DB_nii2inr(path):
    path_nifti = fn.get_paths(path, [], ".gz")
    path_nifti.sort()

    for nif_path in path_nifti:
        pos = nif_path.find('.', 2, len(nif_path))
        nif_path_w = nif_path[:pos] + '_cut'
        niiCut(nif_path, nif_path_w)
        nii2inr(nif_path, nif_path_w)

    print('Data base .inr save')


def selMax_vol(BW):
    m, n, d = BW.shape

    BW = ndimage.binary_fill_holes(BW)
    L = label(BW)
    BW_stats = regionprops(L)

    volumen = [BW_stats[i].area for i in range(len(BW_stats))]
    volumen = np.asarray(volumen)

    ar = np.arange(0, len(volumen), 1.)

    volumen = ar[volumen == np.max(volumen)]

    BW = np.zeros((m, n, d))
    BW[L == volumen + 1] = 1

    return BW


def write_inr(inr_name, data, affine, Type="float32", CPU="decm"):
    """ Function to write .inr File

    Args:
        inr_name (str): File name to output
        data (np.array): Array with data values at each voxel
        affine (dx, dy, dz): Affine of the image (note no translation is made)

    Kwargs:
        Type (str): data format type
        CPU (str): This is for the header

    """

    dx, dy, dz = affine
    s = int(len(data.shape) / 4)  # 1 vfield 0 sfield
    vdim = 3 if s == 1 else 1
    header = "#INRIMAGE-4#{\n"
    header += "XDIM=" + str(data.shape[0 + s]) + "\n"  # x dimension
    header += "YDIM=" + str(data.shape[1 + s]) + "\n"  # y dimension
    header += "ZDIM=" + str(data.shape[2 + s]) + "\n"  # z dimension
    header += "VDIM=" + str(vdim) + "\n"
    header += "VX=" + str(dx) + "\n"  # voxel size in x
    header += "VY=" + str(dy) + "\n"  # voxel size in y
    header += "VZ=" + str(dz) + "\n"  # voxel size in z

    Scale = -1
    Type_np = Type.lower()
    if (Type_np in ['float32', 'single', 'float', 'float_vec']):
        Type = "float"
        Pixsize = "32 bits"
    elif (Type_np in ['float64', 'double', 'double_vec']):
        Type = "float"
        Pixsize = "64 bits"
    elif (Type_np in ['uint8']):
        Type = "unsigned fixed"
        Pixsize = "8 bits"
        Scale = "2**0"
    elif (Type_np in ['int16']):
        Type = "signed fixed"
        Pixsize = "16 bits"
        Scale = "2**0"
    elif (Type_np in ['uint16']):
        Type = "unsigned fixed"
        Pixsize = "16 bits"
        Scale = "2**0"
    else:
        print("Incorrect Data Type.")

    header += "TYPE=" + Type + "\n"  # float, signed fixed, or unsigned fixed
    header += "PIXSIZE=" + Pixsize + "\n"  # 8, 16, 32, or 64
    if Scale != -1: header += "SCALE=" + Scale + "\n"  # not used in my program
    header += "CPU=" + CPU + "\n"
    # decm, alpha, pc, sun, sgi ; ittle endianness : decm, alpha,
    # pc; big endianness :sun, sgi
    for i in range(256 - (len(header) + 4)):
        header += "\n"
    header += "##}\n"

    file = open(inr_name, 'w+')
    file.write(header)
    file.close()
    file = open(inr_name, 'ab')
    file.write(data.transpose().astype(Type_np).tostring())
    file.close()

    print('domain values: ', np.unique(data)[1::])


def read_inr(inrfile):
    """
    For Gray Level Images Only (Scalar fields)
    """

    # Read header
    file = open(inrfile, 'r')
    n = 0
    nline = 0

    while n < 256:
        line = file.readline()
        n += len(line)
        nline += 1
        # Get dimensions
        if line.find("XDIM") != -1: xdim = int(line.split("=")[1])
        if line.find("YDIM") != -1: ydim = int(line.split("=")[1])
        if line.find("ZDIM") != -1: zdim = int(line.split("=")[1])

        # Get affine
        if line.find("VX") != -1: vx = float(line.split("=")[1])
        if line.find("VY") != -1: vy = float(line.split("=")[1])
        if line.find("VZ") != -1: vz = float(line.split("=")[1])

        # Get type of data
        if line.find("TYPE") != -1: typ = line.split("=")[1].rstrip()
        if line.find("PIXSIZE") != -1: bits = int(line.split("=")[1].split(' ')[0])

    if typ == "unsigned fixed":
        Type = "uint"
    elif typ == "signed fixed":
        Type = "int"
    elif typ == "float":
        Type = "float"

    Type += str(bits)

    # Read data
    file = open(inrfile, 'rb')
    n = 0

    raw_data = file.readlines()[nline]

    data = np.frombuffer(raw_data, dtype=Type)
    #    data = data/np.max(data)*domains

    # size=XDIM*YDIM*ZDIM*VDIM*PIXSIZE/8
    if len(data) == xdim * ydim * zdim:
        # read Transposed file          ##IMPORTANT##
        return data.reshape((zdim, ydim, xdim)).transpose(), [vx, vy, vz], Type
    else:
        print("Error Reading the file make sure the data type its correct.")
        print("[data saved in \"data\"]")
        return data


def nii2inr(ifile, ofile):
    """ Reads a .nii file and returns a .inr File

    Args:
        ifile (str): Input file name
        ofile (str): output file name
    """

    # Load NII
    nii_mask = nib.load(ifile)
    data = np.asanyarray(nii_mask.dataobj)  # CORREGIDO
    affine = nii_mask.affine

    dl = affine.diagonal().__abs__()[0:3]  # affine in inr format

    strel = ndimage.generate_binary_structure(3, 1)
    data = ndimage.binary_erosion(data, strel, border_value=0, iterations=1)
    # data = ndimage.binary_closing(data, strel, border_value=0)

    # data with domains to inr format
    data = selMax_vol(data)

    inr_data = data * 255 / np.max(data)

    Type = 'uint8'
    inr_data = inr_data.astype('uint8')

    write_inr(ofile + '.inr', inr_data, dl, Type=Type)



def HC_Laplacian_Smoothing(fname, parameters):
    """
        The idea of the HC-algorithm (HC stands for Humphrey s
    Classes and has no deeper meaning) is to push the modified
    points p i (produced by the Laplacian algorithm e.g.) back
    towards the previous points q i and (or) the original points o i
    by the average of the differences. (Improved Laplacian
        Smoothing of Noisy Surface Meshes, 1999)

    fname (str): Meshio object | Ex: "Data/Aorta/Aorta_refined.mesh"

    parameter (alpha, beta, iterations):
        alpha : Preserves original mesh if alpha=1,and smooths the
         mesh in certain quantity depending his value. | Ex:{Float:0-1} 0.1,0.2,..0.8,etc.

        beta : Pushes back the original points to preserve volume
         like a correction factor. | Ex:{Float:0-1} 0.1,0.2,..0.8,etc.

        iterations : number of iterations of the method.
    """

    # alpha, beta, iterations = parameters
    # x,IEN=fname
    # x1,_ = HCL(x, np.vstack(IEN).astype(int), alpha, beta, iterations)  # Cython algorithm ##mesh.cells['triangle'],
    mesh = io.read(fname)
    # mesh.points, adj = laplacianSmooth(mesh.points, mesh.cells[0].data, alpha, beta,
    #                        iterations)  # Cython algorithm ##mesh.cells['triangle'],
    mesh.points, adj = laplacianSmooth(mesh.points, mesh.cells[0].data, parameters)

    io.write(fname, mesh)


def laplacianSmooth(x, IEN, params=(0.1, 1, 1)):  # 0.1,1,1  #0.1,1,2 #0.1, 0, 3
    #     """
    #         The idea of the HC-algorithm (HC stands for Humphrey s
    #     Classes and has no deeper meaning) is to push the modified
    #     points p i (produced by the Laplacian algorithm e.g.) back
    #     towards the previous points q i and (or) the original points o i
    #     by the average of the differences. (Improved Laplacian
    #         Smoothing of Noisy Surface Meshes, 1999)
    #
    #     fname (str): Meshio object | Ex: "Data/Aorta/Aorta_refined.mesh"
    #
    #     parameter (alpha, beta, iterations):
    #         alpha : Preserves original mesh if alpha=1,and smooths the
    #          mesh in certain quantity depending his value. | Ex:{Float:0-1} 0.1,0.2,..0.8,etc.
    #
    #         beta : Pushes back the original points to preserve volume
    #          like a correction factor. | Ex:{Float:0-1} 0.1,0.2,..0.8,etc.
    #
    #         iterations : number of iterations of the method.
    #     """

    # all nodes laplacian smoothing
    alpha, beta, it = params

    IEN1 = IEN.astype(int)
    p = copy.deepcopy(x)
    l = np.shape(x)[0]
    b = np.zeros((l, 3), dtype=float)
    elem_size = np.shape(IEN1[0, :])[0]

    # Founding Neighbors
    adj = [set([]) for i in range(l)]
    for tr in IEN1:
        for i in tr:
            for j in tr:
                if j != i:
                    adj[i].add(j)

    # HC Smoothing Algorithm
    for t in range(it):
        q = p
        for i in range(l):
            n = len(adj[i])
            if n != 0:
                p[i] = sum([q[j] for j in adj[i]]) / n
                b[i] = p[i] - (alpha * x[i] + (1 - alpha) * q[i])

        for i in range(l):
            n = len(adj[i])
            if n != 0:
                p[i] = p[i] - (beta * b[i] + (1 - beta) * sum([b[j] for j in adj[i]]) / n)

    return p, adj


def skel_interpol(coords, distp = 5., rate = 2.):
    ncdr = np.shape(coords)[0]

    ps = np.arange(ncdr)
    nps = np.linspace(0, ncdr - 1, np.ceil(ncdr * rate).astype(int), endpoint=True)

    dist = fn.pipeLongitude(coords)
    if ncdr <= 5 or dist <= distp:
        if ncdr<=3:
            inter = 'linear'
        else:
            inter = 'quadratic'
        f = interpolate.interp1d(ps, coords.T, kind=inter)
        coords = f(nps).T
    # if dist <= distp:
    #     inter = 'quadratic'
    #     f = interpolate.interp1d(ps, coords.T, kind=inter)
    #     coords = f(nps)
    npoints = np.shape(coords)[0]

    return coords, npoints


def skeleton_xyz_ien(data, affine):
    """
    Function to obtain the coordinates and connectivity given by the CGAL output.
    It also returns the bifurcations and the start and end point of each segment

    Args:
        data (np.array): data on skel-poly.cgal

    Returns:
        xyz (np.array): nodal coordinates
        ien (np.array): skeleton connectivity
        bif (np.array): bifurcations node number
        seg_start (np.array): segment start node number
        seg_end (np.array): segment end node number
        segdata (np.array): segment id number
    """

    xyz = []
    ien = []
    npts = []
    seg_start = []
    seg_end = []
    bif = []
    st = 0
    cont = 0
    segdata = np.array([], dtype=int)

    for line in data:
        vals = line.strip().split(' ')
        # npoints = int(vals[0])
        coords = np.array(vals[1::], dtype=float)
        coords = coords.reshape([-1, 3])
        coords, npoints = skel_interpol(coords, 5, 2)

        # coords = fns.affineMatrix(coords, affine)
        coords[:, 0] = coords[:, 0] * affine[0]
        coords[:, 1] = coords[:, 1] * affine[1]
        coords[:, 2] = coords[:, 2] * affine[2]

        ien_loc = np.vstack([np.arange(npoints - 1), np.arange(1, npoints)]).T + st
        xyz.append(coords)
        ien.append(ien_loc)
        segdata = np.append(segdata, np.ones(npoints - 1) * cont)

        bif += [ien_loc[0, 0], ien_loc[-1, 1]]
        seg_start += [ien_loc[0, 0]] * (npoints - 1)
        seg_end += [ien_loc[-1, 1]] * (npoints - 1)

        npts.append(npoints)
        cont += 1
        st += npoints

    # xyz = np.vstack(xyz)
    # ien = np.vstack(ien)
    bif = np.array(bif)
    seg_start = np.array(seg_start)
    seg_end = np.array(seg_end)

    return xyz, ien, bif, seg_start, seg_end, segdata


def remove_duplicate_bifs(xyz, ien, bif, seg_start, seg_end):
    """
    Args:
        xyz (np.array): nodal coordinates
        ien (np.array): skeleton connectivity
        bif (np.array): bifurcations node number
        seg_start (np.array): segment start node number
        seg_end (np.array): segment end node number

    Returns:
        xyz (np.array): corrected nodal coordinates
        ien (np.array): corrected skeleton connectivity
        bif (np.array): corrected bifurcations node number
        start (np.array): corrected segment start node number
        end (np.array): corrected segment end node number
    """

    change = []
    to_del = []
    to_del_bif = []
    for i, node in enumerate(bif[1::]):
        dist = np.linalg.norm(xyz[node] - xyz[bif[0:i + 1]], axis=1)
        tol = 1e-3
        arr = np.where(dist < tol)[0]

        if len(arr) != 0:
            change.append([node, bif[np.min(arr)]])
            to_del.append(node)
            to_del_bif.append(np.where(bif == node)[0][0])
            ien[ien == node] = bif[np.min(arr)]

    for i in range(len(change)):
        seg_start[seg_start == change[i][0]] = change[i][1]
        seg_end[seg_end == change[i][0]] = change[i][1]

    sk_ien = np.copy(ien)
    sk_bif = np.copy(bif)
    start = np.copy(seg_start)
    end = np.copy(seg_end)
    for i in to_del:  # FIX ME (too slow)
        sk_ien[ien >= i] -= 1
        sk_bif[bif >= i] -= 1
        start[seg_start >= i] -= 1
        end[seg_end >= i] -= 1

    xyz = np.delete(xyz, to_del, axis=0)
    ien = sk_ien
    bif = np.delete(sk_bif, to_del_bif)

    return xyz, ien, bif, start, end


def mark_branches(tree):
    """
    Mark branches by generation number
    Args:
        xyz (np.array): nodal coordinates
        ien (np.array): skeleton connectivity
        start (np.array): segment start node number
        end (np.array): segment end node number

    Returns:
        branch (np.array): branch segment number
        flipped (np.array): flipped segment number
    """
    start = []
    end = []

    for i in range(len(tree)):
        start.append(tree[i].father)
        end.append(tree[i].son)

    start = np.asarray(start)
    end = np.asarray(end)

    start_point = start[0]  # Point to start analysis
    branch = np.ones(len(tree)) * -1
    nbranch = 0

    starts = np.array([start_point], dtype=int)
    while np.min(branch) < 0:
        ends = np.array([], dtype=int)
        for s in starts:
            arr1 = start == s
            endings = np.unique(end[arr1])

            arr2 = end == s
            endings = np.append(endings, np.unique(start[arr2]))

            arr = arr1 + arr2
            arr[branch != -1] = False
            branch[arr] = nbranch

            ends = np.append(ends, endings)

        starts = ends
        nbranch += 1

    branch = branch.astype(int)
    for i in range(len(tree)):
        tree[i].generation = branch[i]

    return branch, tree


def find_terminals(bif, ien):
    """
    Args:
        bif (np.array): bifurcations node number
        ien (np.array): skeleton connectivity

    Returns:
        ter (np.array): terminals node number
        bif (np.array): corrected bifurcations node number
    """

    ter = []
    to_del = []
    for i, n in enumerate(bif):
        arr = np.where(ien == n)[0]
        if len(arr) < 3:
            ter.append(n)
            to_del.append(i)

    bif = np.delete(bif, to_del)
    ter = np.asarray(ter)

    return ter, bif


def skeleton_axis(xyz, ien):
    """
    Get axis vector at each segment
    Args:
        xyz (np.array): nodal coordinates
        ien (np.array): skeleton connectivity

    Returns:
        sk_normals (np.array): axis vector at each segment
    """
    sk_normals = np.zeros((ien.shape[0], 3))
    for e in range(ien.shape[0]):
        nodes = ien[e]
        vector = xyz[nodes[1]] - xyz[nodes[0]]
        vector = vector / np.linalg.norm(vector)
        sk_normals[e] = vector

    return sk_normals

class Branch:
    def __init__(self, data, ien, father, son):
        self.data_skel = data
        self.ien_skel = ien
        self.father = father
        self.son = son
        self.radius = [np.nan, np.nan, np.nan]
        self.generation = None
        self.surf = None

        self.CP = None
        self.IEN = None
        self.affine = None
        self.patches = []
        self.measures = None
        
        

def BranchesTree(xyz, ien, surf_xyz):
    tree = []
    branches = copy.deepcopy(xyz)
    ien_copy = copy.deepcopy(ien)
    xyz_all = np.vstack(branches)
    start_node = np.argmax(xyz_all[:, 2])  # nodo más alto en Z

    pending_nodes = [start_node]
    
    while pending_nodes:
        node_father = pending_nodes.pop(0)
        xyz_father = xyz_all[node_father]

        new_branches = []
        new_ien = []

        for i in range(len(branches)):
            branch = branches[i]
            pos = np.where((branch == xyz_father).all(axis=1))[0]

            if len(pos) == 0:
                new_branches.append(branch)
                new_ien.append(ien_copy[i])
                continue

            idx = pos[0]

            if idx != 0 and idx != len(branch) - 1:
                print(f'Nodo {node_father} está en el medio de la rama {i}, se ignora')
                new_branches.append(branch)
                new_ien.append(ien_copy[i])
                continue

            # Definir dirección
            indices = np.unique(ien_copy[i])
            if idx == len(branch) - 1:
                indices = np.flip(indices)
                ien_copy[i] = np.flip(ien_copy[i], axis=0)

            segment_xyz = xyz_all[indices]
            node_son = indices[-1]

            br = Branch(segment_xyz, ien_copy[i], node_father, node_son)
            tree.append(br)
            pending_nodes.append(node_son)

        branches = new_branches
        ien_copy = new_ien

    tree = bif2trif(sk_surf_corr(surf_xyz, tree))

    xyz, ien = copy.copy(xyz_all), []
    for t in tree:
        ien.append(t.ien_skel)
        indexs = np.hstack([t.ien_skel[:, 0], t.ien_skel[-1, -1]])
        xyz[indexs] = t.data_skel

    return tree, ien, xyz




def bif2trif(tree, tresh=0.3):
    cont = 0
    lbranch = len(tree)

    while cont < lbranch:
        # for each branch get children branches
        t = tree[cont]
        n_son = t.son
        b_childs, i_childs = [], []

        for i in range(cont + 1, lbranch):
            tt = tree[i]
            if tt.father == n_son:
                b_childs.append(tt)
                i_childs.append(i)

        # check a bifurcation
        if len(b_childs) == 2:
            lb0 = fn.pipeLongitude(b_childs[0].data_skel)
            lb1 = fn.pipeLongitude(b_childs[1].data_skel)
            ldata = np.asarray([lb0, lb1])
            short_b_i = np.argmin(ldata)

            rad0 = b_childs[0].radius[2]
            rad1 = b_childs[1].radius[2]
            rad = np.asarray([rad0, rad1])
            longs_rad = rad[np.abs(short_b_i - 1)]

            lpc = ldata[short_b_i] - longs_rad
            lpca = lpc / rad[short_b_i]

            # decide whether is a bifurcation or a trifurcacion
            if lpca < tresh:
                short_i = i_childs[short_b_i]  # global index on tree
                branch = b_childs[short_b_i]
                n_son_son = branch.son

                for b in range(short_i, lbranch):
                    if tree[b].father == n_son_son:
                        tree[b].data_skel[0, :] = t.data_skel[-1, :]
                        for i in range(len(tree[b].data_skel)-4):
                            tree[b].data_skel[i+1, :] = np.mean(tree[b].data_skel[i:i+4, :], axis=0)  # np.mean([tree[b].data[0, :], tree[b].data[1, :]], axis=0)
                        # tree[b].data_skel[2, :] = np.mean(tree[b].data_skel[1:4, :], axis=0)
                        tree[b].father = n_son
                        tree[b].ien_skel[0, 0] = n_son

                tree.pop(short_i)
                lbranch = len(tree)

        cont += 1

    return tree


def patient_skeleton(path, affine, **kwargs):
    tree, xyz, ien, surf_xyz, branch = \
        skeleton_reader(path, True, affine, **kwargs)

    return  tree, xyz, ien, surf_xyz, branch


def skeleton_reader(fname, savevtu, affine, **kwargs):
    # Load file
    sname = f'{fname}_skel_poly.cgal'
    f = open(sname)
    data = f.readlines()
    f.close()

    # Get connectivity and coordinates
    scale = 1. / np.asarray([-affine[0, 0], -affine[1, 1], affine[2, 2]])
    xyz, ien, bif, start, end, segdata = skeleton_xyz_ien(data, scale)
    sk_seg = segdata.astype(int)  # segment id number

    # get tree subdivisions
    if 'surface' in kwargs:
        surf_xyz = kwargs.get('surface')
    else:
        surf_mesh = io.read(f'{fname}_out.off')
        surf_xyz = scale * surf_mesh.points
    tree, ien, xyz = BranchesTree(xyz, ien, surf_xyz)

    # Remove duplicate bifurcations
    xyz = np.vstack(xyz)
    ien = np.vstack(ien)
    # xyz, ien, bif, start, end = remove_duplicate_bifs(xyz, ien, bif, start, end)

    # Mark branches with generation number
    branch, tree = mark_branches(tree)

    # Find terminals
    ter, bif = find_terminals(bif, ien)

    # Get axis vector
    vector = skeleton_axis(xyz, ien)

    # Save .vtu
    if savevtu:
        export_branches_to_vtk(tree, affine, fname)
        print('Skeleton save')
    # skel_vars = [ter, bif, start, end, sk_seg, branch, vector]
    
    return tree, xyz, ien, surf_xyz, branch


def export_branches_to_vtk(tree, affine, filename) -> None:
    points = []
    lines = []
    generations = []

    point_offset = 0

    for branch in tree:
        # Aplicar transformación afín a los nodos
        coords_affine = fn.affineMatrix(branch.data_skel, affine)
        n_points = coords_affine.shape[0]

        points.append(coords_affine)
        lines.extend([[i + point_offset, i + 1 + point_offset] for i in range(n_points - 1)])
        generations.extend([branch.generation if branch.generation is not None else -1] * (n_points - 1))
        point_offset += n_points

    points = np.vstack(points)
    lines = np.array(lines, dtype=int)
    generations = np.array(generations, dtype=int)

    mesh = io.Mesh(
        points=points,
        cells=[("line", lines)],
        cell_data={"generation": [generations]}
    )

    mesh.write(f'{filename}_skeleton_generations.vtk')


def sk_surf_corr(surf_xyz, tree):
    new_segm = segmentDivision(tree)
    nb = np.shape(new_segm)[0]
    nel_perim = np.shape(surf_xyz)[0]

    branch_f = new_segm[:, 0:3]
    branch_s = new_segm[:, 3:6]

    radius_max = np.zeros((nb))
    radius_min = np.ones((nb)) * np.inf
    radius_avg = np.zeros((nel_perim))
    p_branch_dum = np.zeros((nel_perim))
    new_branch_num = new_segm[:, 6]

    q1, q2 = branch_f, branch_s
    invlen2 = 1. / np.sum((q2 - q1) * (q2 - q1), axis=1)
    rads = []

    for i in range(nel_perim):
        p = surf_xyz[i, :]

        var_min = np.min([np.ones((np.shape(q1)[0])), -np.sum((q1 - p) * (q2 - q1), axis=1) * invlen2], axis=0)
        var_min[np.isnan(var_min)] = -np.inf
        t = np.max([np.zeros((np.shape(q1)[0])), var_min], axis=0).reshape(-1, 1)
        v = q1 + t * (q2 - q1)

        closest = np.linalg.norm(v - p, axis=1)

        rad, I = np.min(closest), np.argmin(closest)
        if t[I] != 0 and t[I] !=1:
            radius_max[I] = np.max([radius_max[I], rad])
            radius_min[I] = np.min([radius_min[I], rad])
            radius_avg[i] = rad
        else:
            pass
        p_branch_dum[i] = new_branch_num[I]
        rads.append(rad)

    nb = len(tree)
    surf_mean = np.mean(rads)
    for i in range(nb):
        dum_max = radius_max[new_branch_num == i]
        dum_max = np.delete(dum_max, np.where(dum_max == 0))
        if len(dum_max) == 0: dum_max = surf_mean
        tree[i].radius[0] = np.mean(dum_max)

        dum_min = radius_min[new_branch_num == i]
        dum_min = np.delete(dum_min, np.where(dum_min == np.inf))
        if len(dum_min) == 0: dum_min = surf_mean
        tree[i].radius[1] = np.mean(dum_min)

        dum_avg = radius_avg[p_branch_dum == i]
        dum_avg = np.delete(dum_avg, np.where(dum_avg == 0))
        if len(dum_avg) == 0: dum_avg = surf_mean
        tree[i].radius[2] = np.mean(dum_avg)

        tree[i].surf = surf_xyz[p_branch_dum == i, :]
        # print(f'Branch: {i}, rad: {tree[i].radius}')
    return tree


def segmentDivision(tree):
    nb = len(tree)
    new_segm = []

    for i in range(nb):
        branch = tree[i].data_skel
        nodes_f = branch[:-1,:]
        nodes_s = branch[1:,:]

        index = np.ones((np.shape(nodes_f)[0], 1)) * i
        new_nodes = np.hstack([nodes_f, nodes_s, index])
        new_segm.append(new_nodes)
    return np.vstack(new_segm)


# def segmentDivision(tree, max_length=2, dmax=0):
#     nb = len(tree)
#     new_segm = []
#
#     for i in range(nb):
#         branch = tree[i].data_skel
#         new_nodes = []
#         l_branch = np.shape(branch)[0]
#         ndiv = np.floor(l_branch / max_length).astype(int)
#         t = np.floor(np.linspace(1, l_branch, ndiv + 1)).astype(int)
#
#         if l_branch > max_length :
#             ntry, ind1, ind2 = 0, 0, 1
#             for j in range(ndiv):
#                 q1 = branch[t[ind1] - 1, :]
#                 q2 = branch[t[ind2] - 1, :]
#                 p = branch[t[j]:t[j + 1] - 1, :]
#
#                 invlen2 = 1. / np.dot(q2 - q1, q2 - q1)
#                 min_var = np.min([np.ones((np.shape(p)[0])),
#                                   np.sum(-(q1 - p) * np.tile(q2 - q1, [np.shape(p)[0], 1]), axis=1).T * invlen2],
#                                  axis=0)
#                 tt = np.max([np.zeros((np.shape(p)[0])), min_var], axis=0).reshape(-1, 1)
#                 v = q1 + tt * (q2 - q1)
#
#                 th = np.max(np.linalg.norm(v - p))
#
#                 if th >= dmax or ntry == 3 and j <= ndiv:
#                     new_nodes.append(np.array([ind1, ind2 - 1]))
#                     ind1, ind2 = ind2, ind2 + 1
#                     ntry = 0
#                 else:
#                     ind2, ntry = ind2 + 1, ntry + 1
#         if not new_nodes:
#             new_nodes = np.array([0, 0])
#
#         new_nodes = np.hstack(new_nodes)
#         dum1 = np.vstack([branch[t[new_nodes[1:]]], branch[-1]])
#         dum2 = np.ones((np.shape(new_nodes)[0], 1)) * i
#         new_segm.append(np.hstack([branch[t[new_nodes]], dum1, dum2]))
#
#     return np.vstack(new_segm)


# def sk_surf_correspondance(fname, sk_xyz, surf_xyz, tree):
#
#     from scipy.spatial import KDTree
#
#     corr = np.loadtxt(fname)
#
#     # Find node number in skeleton correspondance
#     sk_points = corr[:,1:4]
#     Tree1 = KDTree(sk_points)
#     Tree2 = KDTree(sk_xyz)
#     sk_corr = np.asarray(Tree1.query_ball_tree(Tree2, 0.01)).flatten()
#
#     # Find node number in airway correspondance
#     surf_points = corr[:,4::]
#     Tree1 = KDTree(surf_points)
#     Tree2 = KDTree(surf_xyz)
#     surf_corr = np.asarray(Tree1.query_ball_tree(Tree2, 0.01)).flatten()
#
#     ien_sksurf = sk_corr[np.argsort(surf_corr)]
#
#     return ien_sksurf



# __________________________________________________________________
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib
# matplotlib.use('TkAgg')
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# ax.scatter(surf_xyz[:,0],surf_xyz[:,1],surf_xyz[:,2], c='g', marker='o')
# plt.pause(0.1)