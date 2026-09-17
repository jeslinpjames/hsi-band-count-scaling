import h5py
import numpy as np

def load_houston(scene):
    """scene: '13' or '18'"""
    DATA = '../data/Houston'
    with h5py.File(f'{DATA}/Houston{scene}.mat', 'r') as f:
        cube = np.array(f['ori_data'])          # (48, 954, 210)
    with h5py.File(f'{DATA}/Houston{scene}_7gt.mat', 'r') as f:
        gt = np.array(f['map'])                  # (954, 210)

    cube = cube.transpose(1, 2, 0).astype(np.float32)  # -> (954, 210, 48)
    gt = gt.astype(np.float32)
    return cube, gt

if __name__ == "__main__":
    cube, gt = load_houston('13')
    print("Houston13 cube:", cube.shape, "gt:", gt.shape, "classes:", int(gt.max()))
    cube18, gt18 = load_houston('18')
    print("Houston18 cube:", cube18.shape, "gt:", gt18.shape, "classes:", int(gt18.max()))
