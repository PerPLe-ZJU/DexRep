import json
from scipy.spatial.transform import Rotation as R
import numpy as np
class DotDict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value


def pretty(d):
    return json.dumps(d, indent=4, ensure_ascii=False)

def get_T_matrix(Roation, xyz):
    if Roation.size == 3:
        matrix = R.from_euler('xyz', Roation.flatten(), degrees=False).as_matrix()
    elif Roation.size == 9:
        matrix = Roation
    else:
        raise AttributeError
    T = np.eye(4)
    T[:3, :3] = matrix
    T[:3, -1] = xyz

    return T
def get_quat(matrix):
    quat = R.from_matrix(matrix).as_quat()
    return quat

import torch
import matplotlib.pyplot as plt

def plot_tensor_data(y: list, title:str, x_name: str = "step", y_name: str="value"):
    # Extract x and y coordinates
    x = np.arange(len(y))

    fig, ax = plt.subplots()
    # Create a scatter plot
    ax.scatter(x, y, label='Original Data')

    # Set labels and title
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    ax.set_ylim(0,15)

    # Display the legend
    plt.legend()

    # Show the plot
    plt.show()

def mean_and_std(data_list):
    import math

    mean = sum(data_list) / len(data_list)

    sqrt_sum = 0
    for d in data_list:
        sqrt_sum += (d - mean) ** 2
    std = math.sqrt(sqrt_sum / (len(data_list) - 1))
    print(f"mean: {mean}, std: {std}")

def split_batch_process(split_batch, input_tensor_list, process_chain):
    batch_num = input_tensor_list[0].shape[0]
    batch_split_num = int((batch_num + split_batch - 1) / split_batch)
    # split input
    split_input_list = []
    for it in input_tensor_list:
        assert batch_num == it.shape[0]
        it_slices = [
            it[i*split_batch:(i+1)*split_batch, ...] \
            for i in range(batch_split_num)
        ]
        split_input_list.append(it_slices)
    # transpose input list
    split_input_list = [list(item) for item in zip(*split_input_list)]
    # process
    result_list = []
    for iter in range(batch_split_num):
        chain_idx = 0
        result = None
        for my_func in process_chain:
            if chain_idx == 0: # input
                result = my_func(*split_input_list[iter])
            else:
                result = my_func(result)
            chain_idx += 1
        result_list.append(result)

def occumpy_mem(factor=0.96): # try to occupy 96% GPU mem
    # 指定 GPU 设备
    # cuda_dev = os.getenv('CUDA_VISIBLE_DEVICES')
    device = torch.device(f'cuda:0')  # 例如使用第一个 GPU
    # torch.cuda.set_device(device)
    # 清空之前的 CUDA 缓存，以便从干净的状态开始
    torch.cuda.empty_cache()

    # 获取当前可用的显存量
    total_memory = torch.cuda.get_device_properties(device).total_memory
    free_memory = total_memory - torch.cuda.memory_allocated(device)

    # 尝试分配尽可能多的显存
    allocated = False
    while not allocated and factor > 0.1:
        try:
            # 计算当前尝试分配的显存大小
            num_elements = int(free_memory * factor / 4)  # float32 类型，每个元素 4 字节
            large_tensor = torch.empty(num_elements, dtype=torch.float32, device=device)
            allocated = True
            del large_tensor
            print(f"Allocated a tensor with {num_elements} elements, approximately {num_elements * 4 / 1e6} MB.")
        except RuntimeError as e:
            print(f"Failed at {factor * 100}% of free memory, reducing factor and trying again.")
            factor -= 0.02  # 减小尝试分配的百分比

    # 注意：在实际应用中，占用过多显存可能会影响系统稳定性和其他程序的运行。


def split_torch_dist(finger_points, obj_pcb, split_batch):
    assert finger_points.shape[0] == obj_pcb.shape[0]
    batch_num = finger_points.shape[0]
    batch_split_num = int((batch_num + split_batch - 1) / split_batch)
    # split input
    finger_points_split = [
        finger_points[i*split_batch:(i+1)*split_batch, ...] \
            for i in range(batch_split_num)
    ]
    obj_pcb_split = [
        obj_pcb[i * split_batch:(i + 1) * split_batch, ...] \
            for i in range(batch_split_num)
    ]
    # process
    dis_min_slices, dis_min_idx_slices = [], []
    for iter in range(batch_split_num):
        dis_split = torch.cdist(finger_points_split[iter], obj_pcb_split[iter])
        dis_min_split, dis_min_idx_split = torch.min(dis_split, dim=-1)
        dis_min_slices.append(dis_min_split)
        dis_min_idx_slices.append(dis_min_idx_split)
    # cat
    dis_min = torch.cat(dis_min_slices, dim=0)
    dis_min_idx = torch.cat(dis_min_idx_slices, dim=0)

    return dis_min, dis_min_idx


if __name__ == "__main__":
    mean_and_std([
        38.75,
        32.5,
        33.75

    ])