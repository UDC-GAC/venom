import hashlib
import textwrap
from pathlib import Path
import subprocess
import ctypes
import numpy as np
import timeit
import filelock


def stringify(params):
    if isinstance(params, dict):
        return "_".join([stringify(k) + "_" + stringify(v) for k, v in params.items()])
    elif isinstance(params, str):
        return params
    elif isinstance(params, list):
        return "_".join([stringify(p) for p in params])
    else:
        res = str(params)
        if " " in res:
            raise ValueError(f"Stringified <{res}> contains not allowed symbols")
        return res


def compile(
    src,
    inc_path=None,
    cache_path=".compile_cache",
    lang="cpp",
    opts=["-g", "-O3"],
    recompile=False,
):
    cache_path = Path(cache_path)
    uid_str = stringify([src, inc_path, cache_path, lang, opts])
    uid = hashlib.sha256(uid_str.encode()).hexdigest()
    cache_path.mkdir(exist_ok=True)
    lock = filelock.FileLock(cache_path / f"{uid}.lock", timeout=10)
    with lock:
        src_path = cache_path / f"{uid}.{lang}"
        with src_path.open(mode="w") as f:
            f.write(src)
        lib_path = cache_path / f"{uid}.so"
        if recompile or not lib_path.exists():
            compiler = {
                'cpp': 'g++',
                'c': 'gcc',
                'cu': 'nvcc',
            }[lang]
            flags = {
                'cpp': "-march=native -fopenmp -lpthread -shared -fPIC".split(),
                'cu': "-Xcompiler -march=native,-fopenmp,-lpthread,-fPIC -shared".split(),
            }[lang]
            compile_cmd = [compiler] + flags
            compile_cmd += opts
            compile_cmd.append(src_path.resolve())
            if inc_path is not None:
                compile_cmd += ["-I", Path(inc_path).resolve()]
            compile_cmd += ["-o", lib_path.resolve()]
            subprocess.run(compile_cmd, check=True)
    lib = ctypes.CDLL(lib_path)
    return lib


def mul2(A, B):
    if not hasattr(mul2, "loaded_lib"):
        src = textwrap.dedent(
            """
            #include <cstdlib>
            extern "C"
            void foo(float* A, float* B, size_t N) {
                for (size_t i = 0; i < N; i++) {
                    B[i] = A[i] * 2;
                }
            }
            """
        )
        mul2.loaded_lib = compile(src, recompile=False)
        mul2.loaded_lib.foo.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
    mul2.loaded_lib.foo(A.ctypes.data, B.ctypes.data, A.size)


if __name__ == "__main__":
    A = np.arange(100, dtype=np.float32)
    B = np.zeros_like(A)
    print(timeit.repeat("mul2(A, B)", globals={**globals(), **locals()}, number=1))
    assert np.allclose(A * 2, B)
