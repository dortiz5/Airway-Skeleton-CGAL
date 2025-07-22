import os
import subprocess
import Skeleton as ch
import helpers as fn
from collections import Counter

# Paths base
build_dir = os.path.join(os.path.dirname(__file__), "build")

# Lista de casos a procesar
paths = [
    './Data/Airways/ATM_001_0000',
    './Data/Airways/ATM_002_0000',
    './Data/Airways/ATM_003_0000',
    './Data/Airways/ATM_004_0000',
    './Data/Airways/ATM_005_0000',
    # ... agrega todos los paths que quieras procesar
]

for path in paths:
    print(f'Procesando {path}...')

    # Paso 1: NIfTI → INR
    ch.niiCut(f'{path}.nii.gz', f'{path}_cut')
    ch.nii2inr(f'{path}_cut.nii.gz', f'{path}_cut')
    print(f'{path}.inr save')

    # Paso 2: Mesh (INR → OFF)
    input_inr = f'{path}_cut.inr'
    output_off = f'{path}_out.off'
    mesh_exe = os.path.join(build_dir, "mesh_a_3d_gray_image")
    subprocess.run([mesh_exe, input_inr, output_off], check=True)

    # Paso 3: Suavizado
    ch.HC_Laplacian_Smoothing(output_off, [0.6, 1, 3])
    print('Laplacian smoothing: Done')

    # Paso 4: Skeleton (OFF → skeleton)
    skel_exe = os.path.join(build_dir, "simple_mcfskel_example")
    output_base = f'{path}'
    input_inr = f'{path}_cut.inr'
    subprocess.run([skel_exe, output_off, output_base], check=True)
    # Salidas: {path}_skel-sm.polylines.txt, {path}_correspondance-sm.polylines.txt

    # Paso 5: Exportar VTU u otros procesos
    fn.off2vtu(output_off, f'{path}_mesht.vtu')

    # Paso 6: Post-procesamiento Python (opcional)
    path_nifti = f'{path}_cut.nii.gz'
    affine, airway, perim = fn.geometry(path_nifti)
    branches = ch.patient_skeleton(path, affine)[-1]
    conteo = Counter(branches)
    skeleton_clean = [(int(k), int(v)) for (k, v) in sorted(conteo.items())]
    print(f'Branches: {skeleton_clean}')

    print(f'Finalizado {path}\n')
