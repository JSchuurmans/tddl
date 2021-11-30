import os

def select_hardware(
    cuda: str = None,
    cpu: str = None,
) -> None:
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

    if cuda is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda

    if cpu is not None:
        os.environ["MKL_NUM_THREADS"] = cpu
        os.environ["NUMEXPR_NUM_THREADS"] = cpu
        os.environ["OMP_NUM_THREADS"] = cpu
