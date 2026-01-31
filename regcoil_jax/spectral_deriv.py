from __future__ import annotations
import jax
import jax.numpy as jnp
from .constants import twopi

def _fft_freqs(n: int):
    # returns integer frequencies [0,1,...,n/2, -(n/2-1),..., -1] for even n
    # and [0,1,...,(n-1)/2, -((n-1)/2),..., -1] for odd n
    return jnp.fft.fftfreq(n) * n

@jax.jit
def deriv_theta(f_tz3):
    """Spectral derivative wrt theta on a uniform grid theta in [0,2pi).
    f_tz3: (3,T,Z) or (T,Z) etc; derivative along axis=1 (T).
    """
    f = f_tz3
    T = f.shape[1]
    k = _fft_freqs(T)  # integer
    ik = (1j * k).astype(jnp.complex128)
    F = jnp.fft.fft(f, axis=1)
    dF = F * ik[None,:,None]
    df = jnp.fft.ifft(dF, axis=1).real
    return df

@jax.jit
def deriv_zeta(f_tz3, nfp: int):
    """Spectral derivative wrt zeta on a uniform grid zeta in [0,2pi/nfp).
    derivative along axis=2 (Z).
    """
    f = f_tz3
    Z = f.shape[2]
    k = _fft_freqs(Z)
    # physical wavenumber is nfp * k because Lz=2pi/nfp
    ik = (1j * (nfp * k)).astype(jnp.complex128)
    F = jnp.fft.fft(f, axis=2)
    dF = F * ik[None,None,:]
    df = jnp.fft.ifft(dF, axis=2).real
    return df
