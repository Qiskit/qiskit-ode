from multiprocessing import Pool
import numpy as np


def scan_parameters(func,*args, method="parallel",**kwargs):
    s = [len(arg) for arg in args]
    arr = [arg.reshape(*((1,)*i+(s[i],)+(1,)*(len(args)-i))) for i, arg in enumerate(args)]
    arr = np.broadcast_arrays(*arr)
    arr=np.concatenate(arr,axis=-1).reshape(np.product(s),len(args))

    if str(method)=="parallel":
        p = Pool()
        res = p.starmap(func,arr)

    elif method == "serial":
        res = []
        for arg in arr:
            res.append(func(*arg))
    else:
        raise ValueError(f"Method {method} is not supported.")

    return res