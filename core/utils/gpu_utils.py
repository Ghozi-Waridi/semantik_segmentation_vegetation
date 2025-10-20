"""
GPU Utility Module
Menyediakan fungsi untuk deteksi dan inisialisasi GPU (NVIDIA atau Intel Iris)
dengan fallback otomatis ke CPU jika GPU tidak tersedia.
"""

import sys
import warnings

_gpu_available = False
_backend = "cpu"
_device_info = {}

# Try NVIDIA CUDA via CuPy first
_cp_mod = None
try:
    import cupy as _cupy
    _cp_mod = _cupy
    _gpu_available = True
    _backend = "cupy"
    try:
        device = _cupy.cuda.Device()
        _device_info = {
            "name": device.name.decode() if isinstance(device.name, bytes) else device.name,
            "compute_capability": getattr(device, 'compute_capability', (0, 0)),
            "total_memory": _cupy.cuda.runtime.memGetInfo()[1] / (1024**3),
            "free_memory": _cupy.cuda.runtime.memGetInfo()[0] / (1024**3),
            "driver_version": _cupy.cuda.runtime.driverGetVersion(),
        }
    except Exception as e:
        warnings.warn(f"Could not get CUDA device info: {e}")
        _device_info = {"name": "CUDA GPU"}
except Exception:
    _cp_mod = None

# If no CuPy, try Intel GPU via dpnp (oneAPI)
_dpnp_mod = None
if _cp_mod is None:
    try:
        import dpnp as _dpnp
        _dpnp_mod = _dpnp
        _gpu_available = True  # assume GPU device available via default queue
        _backend = "dpnp"
        try:
            # Try to get more detailed device info via dpctl if available
            try:
                import dpctl
                with dpctl.device_context(dpctl.select_default_device()):
                    dev = dpctl.get_current_device()
                    _device_info = {
                        "name": str(dev)
                    }
            except Exception:
                _device_info = {"name": "Intel GPU (dpnp)"}
        except Exception as e:
            warnings.warn(f"Could not get dpnp device info: {e}")
            _device_info = {"name": "Intel GPU (dpnp)"}
    except Exception:
        _dpnp_mod = None

# Fallback to NumPy CPU
if not _gpu_available:
    warnings.warn(
        "GPU backend not available. Falling back to NumPy (CPU).\n"
        "For NVIDIA GPUs: pip install 'cupy-cuda12x' (match your CUDA).\n"
        "For Intel GPUs: pip install 'dpnp' and Intel oneAPI runtime."
    )
    _backend = "cpu"
    _device_info = {"name": "CPU (NumPy)"}
    import numpy as _np
else:
    import numpy as _np

# Import numpy untuk operasi CPU
import numpy as np


def get_array_module(x=None):
    """
    Mendapatkan module array yang sesuai (cupy atau numpy)
    
    Args:
        x: Optional array untuk deteksi module
        
    Returns:
        cupy atau numpy module
    """
    if x is None:
        if _backend == 'cupy':
            return _cp_mod
        if _backend == 'dpnp':
            return _dpnp_mod
        return _np
    # When using CuPy, use its array module detection
    if _backend == 'cupy' and _cp_mod is not None:
        try:
            return _cp_mod.get_array_module(x)
        except Exception:
            return _cp_mod
    # dpnp does not provide get_array_module - return dpnp module
    if _backend == 'dpnp' and _dpnp_mod is not None:
        return _dpnp_mod
    return _np


def is_gpu_available():
    """
    Check apakah GPU tersedia
    
    Returns:
        True jika GPU tersedia, False jika tidak
    """
    return _gpu_available


def get_backend():
    """
    Mendapatkan nama backend yang digunakan
    
    Returns:
        String 'cupy' atau 'cpu'
    """
    return _backend


def get_device_info():
    """
    Mendapatkan informasi device yang digunakan
    
    Returns:
        Dictionary berisi informasi device
    """
    return _device_info


def to_gpu(x):
    """
    Transfer array ke GPU
    
    Args:
        x: NumPy array atau CuPy array
        
    Returns:
        CuPy array jika GPU tersedia, NumPy array jika tidak
    """
    if not _gpu_available:
        return x
    if _backend == 'cupy' and _cp_mod is not None:
        if isinstance(x, _np.ndarray):
            return _cp_mod.asarray(x)
        return x
    if _backend == 'dpnp' and _dpnp_mod is not None:
        if isinstance(x, _np.ndarray):
            return _dpnp_mod.asarray(x)
        return x
    return x


def to_cpu(x):
    """
    Transfer array ke CPU
    
    Args:
        x: NumPy array atau CuPy array
        
    Returns:
        NumPy array
    """
    try:
        if _backend == 'cupy' and _cp_mod is not None and isinstance(x, _cp_mod.ndarray):
            return _cp_mod.asnumpy(x)
        if _backend == 'dpnp' and _dpnp_mod is not None and isinstance(x, _dpnp_mod.ndarray):
            # dpnp exposes asnumpy
            return _dpnp_mod.asnumpy(x)
    except Exception:
        pass
    return x


def empty_cache():
    """
    Kosongkan cache GPU memory
    """
    if _backend == 'cupy' and _cp_mod is not None:
        try:
            _cp_mod.get_default_memory_pool().free_all_blocks()
            _cp_mod.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass


def print_gpu_info():
    """
    Print informasi GPU yang digunakan
    """
    print("=" * 60)
    print("GPU/CPU INFORMATION")
    print("=" * 60)
    
    if _gpu_available:
        print(f"✓ GPU Mode: ENABLED")
        print(f"  Backend: {_backend.upper()}")
        print(f"  Device: {_device_info.get('name', 'Unknown')}")
        if _backend == 'cupy':
            if "compute_capability" in _device_info:
                cc = _device_info["compute_capability"]
                if isinstance(cc, tuple) and len(cc) == 2:
                    print(f"  Compute Capability: {cc[0]}.{cc[1]}")
            if "total_memory" in _device_info:
                print(f"  Total Memory: {_device_info['total_memory']:.2f} GB")
                print(f"  Free Memory: {_device_info['free_memory']:.2f} GB")
            if "driver_version" in _device_info:
                driver = _device_info['driver_version']
                try:
                    print(f"  CUDA Driver Version: {driver // 1000}.{(driver % 1000) // 10}")
                except Exception:
                    print(f"  CUDA Driver Version: {driver}")
    else:
        print(f"⚠ GPU Mode: DISABLED")
        print(f"  Backend: CPU (NumPy)")
        print(f"  Reason: CuPy not installed or GPU not available")
        print(f"  To enable GPU: pip install cupy-cuda11x (or cupy-cuda12x)")
    
    print("=" * 60)


def synchronize():
    """
    Sinkronisasi GPU (menunggu semua operasi GPU selesai)
    """
    if _backend == 'cupy' and _cp_mod is not None:
        try:
            _cp_mod.cuda.Stream.null.synchronize()
        except Exception:
            pass


# Alias untuk kompatibilitas
if _backend == 'cupy' and _cp_mod is not None:
    xp = _cp_mod
elif _backend == 'dpnp' and _dpnp_mod is not None:
    xp = _dpnp_mod
else:
    xp = _np


# Print info saat module di-import
if __name__ != "__main__":
    # Only print when imported, not when run directly
    import logging
    logger = logging.getLogger(__name__)
    if _gpu_available:
        logger.info(f"GPU mode enabled: {_device_info.get('name', 'Unknown GPU')}")
    else:
        logger.info("Running in CPU mode (NumPy)")
