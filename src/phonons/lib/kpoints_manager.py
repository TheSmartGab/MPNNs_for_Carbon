import numpy as np

SIMMETRY_POINTS = {
    "G" : [0, 0, 0],
    "M" : [1/2, 0, 0],
    "K" : [1/3, 1/3, 0],
    # 3D diamond / fcc
    "k" : [0.375, 0.375, 0.75],
    "x": np.array([0.5, 0.0, 0.5]),
    "l": np.array([0.5, 0.5, 0.5]),
    "w": np.array([0.5, 0.25, 0.75]),
    "u": np.array([0.625, 0.25, 0.625]),
}

def string_to_vector(s_kpoints):
    """From a stirng of high simmetry points, return vector of points"""
    v_kpoints = []
    for point in s_kpoints:
        if point in  SIMMETRY_POINTS.keys():
            v_kpoints.append(SIMMETRY_POINTS[point])
        else:
            print("[ERROR] in file", __file__, "point", point, "not in the keys of SIMMETRY_KPOINTS. add it! exit")
            exit(-1)

    return np.array(v_kpoints)


def assign_positions(kpoints : np.ndarray ):
    shape = kpoints.shape
    assert shape[1] == 3

    distances = np.array([kpoints[i]-kpoints[i-1] for i in range(1,shape[0])])
    distances = np.linalg.norm(distances, axis=1)

    tot_dist = np.sum(distances)
    scaled_distances = distances / tot_dist

    cum_distances = np.cumsum(scaled_distances)

    return np.concatenate(([0], cum_distances))

def kpoint_equal(kp, sp, tol=1e-8):

    if np.allclose(kp, sp, atol=tol):
        return True

    return False



def slice_paths(kpoints: np.ndarray, frequencies: np.ndarray, path: str, tol=1e-8):
    """
    Slice kpoints and frequencies to follow the given high-symmetry path.
    Robust against repeated symmetry points and reversed segments.

    Parameters
    ----------
    kpoints : (nkpoints, 3) array
    frequencies : (1, nkpoints, nbands) array
    path : str
        High-symmetry path (e.g. "GKMG")
    """

    print("[INFO] slice_paths called")
    print("[INFO] kpoints shape:", kpoints.shape)
    print("[INFO] frequencies shape:", frequencies.shape)

    sp = string_to_vector(path)
    labels = list(path)

    # Map label index -> all matching kpoint indices
    label_indices = {i: [] for i in range(len(sp))}

    for idx, kp in enumerate(kpoints):
        for i, s in enumerate(sp):
            if kpoint_equal(kp, s, tol):
                label_indices[i].append(idx)

    kp_segments = []
    freq_segments = []

    for i in range(len(labels) - 1):
        starts = label_indices[i]
        ends = label_indices[i + 1]

        if not starts or not ends:
            print(f"[WARNING] Missing symmetry point {labels[i]} or {labels[i+1]}")
            continue

        # Find closest pair (minimal index distance)
        best_pair = None
        best_dist = np.inf

        for i1 in starts:
            for i2 in ends:
                d = abs(i2 - i1)
                if d > 0 and d < best_dist:
                    best_dist = d
                    best_pair = (i1, i2)

        if best_pair is None:
            print(f"[WARNING] Path {labels[i]} → {labels[i+1]} not found")
            continue

        i1, i2 = best_pair

        if i2 > i1:
            kp_seg = kpoints[i1:i2 + 1]
            fr_seg = frequencies[:, i1:i2 + 1, :]
        else:
            kp_seg = kpoints[i2:i1 + 1][::-1]
            fr_seg = frequencies[:, i2:i1 + 1, :][:, ::-1, :]

        # Avoid duplicating junction points
        if kp_segments:
            kp_seg = kp_seg[1:]
            fr_seg = fr_seg[:, 1:, :]

        kp_segments.append(kp_seg)
        freq_segments.append(fr_seg)

    if not kp_segments:
        return None, None

    kpoints_out = np.vstack(kp_segments)
    frequencies_out = np.concatenate(freq_segments, axis=1)

    return kpoints_out, frequencies_out


