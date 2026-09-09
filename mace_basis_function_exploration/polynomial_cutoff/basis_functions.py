import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, sph_harm_y, jn_zeros

CUTOFF=5.0
XS = np.linspace(0, CUTOFF, 100)
THETAS = np.linspace(0, np.pi, 50)
PHIS = np.linspace(0, 2*np.pi, 100)

def eRBF(x, n, c):
    x = np.asarray(x)
    return np.sqrt(2.0 / c) * np.sin(n * np.pi * x / c) / np.where(x == 0, 1.0, x)


def pol(x, p, c):
    x = np.asarray(x)
    y = x / c
    return (
        1
        - ((p + 1) * (p + 2)) * y**p / 2
        + p * (p + 2) * y**(p + 1)
        - (p * (p + 1)) * y**(p + 2) / 2
    )

def spherical_bessel_zeros(l, n_max):
    """
    Zeros z_{l n} of the spherical Bessel function j_l.
    """
    return jn_zeros(l, n_max)


def aSBF(d, theta, phi, l, m, n, c, z_ln=None):
    """
    Angular spherical Bessel function (cutoffed).

    Parameters
    ----------
    d : array_like
        Distances, shape (...,)
    theta : array_like
        Azimutal angles, shape (...,)
    phi :   array_like
        Polar angles, shape (...,)
    l : int
        Angular momentum index
    n : int
        Radial index (1-based)
    c : float
        Cutoff radius
    z_ln : array_like, optional
        Precomputed zeros of j_l

    Returns
    -------
    sbf : ndarray
        Shape broadcasted from d, theta, phi
    """

    d = np.asarray(d)
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    if z_ln is None:
        z_ln = spherical_bessel_zeros(l, n)

    z = z_ln[n - 1]

    # Radial part
    radial = spherical_jn(l, z * d / c)

    # Normalization
    norm = np.sqrt(2.0 / c**3) / spherical_jn(l + 1, z)

    # Angular part (real Y_l^m)
    Y_l0 = sph_harm_y(l, m, theta, phi).real

    return norm * radial * Y_l0


def base_function(x, theta, phi, p, l, m, n, c):
    return (
        eRBF(x, n, c)
        * pol(x, p, c)
        * aSBF(x, theta, phi, l, m, n, c)
    )

def process_p(p, l_max, n_max):

    for l in range(0, l_max+1):
        process_l(p, l, n_max)

def process_l(p, l, n_max):
    for n in range(1, n_max+1):
        values = base_function(XS, 0, 0, p, l, 0, n, CUTOFF)
        plt.plot(XS, values, label=f"n={n}")

    plt.legend()
    plt.title(f"Base function p={p}, l={l}")
    plt.grid()
    plt.xlabel("d")
    plt.tight_layout()
    print(f"[INFO] saving fig p{p}_l{l}.pdf")
    plt.savefig(f"p{p}_l{l}.pdf")
    plt.close()



def main():
    
    for p in range(0, 11):
        process_p(p, 5, 16)


if __name__ == "__main__":
    main()

