import os
import numpy as np
import scipy.io as io
import imageio.v2 as iio
# from pdb import set_trace as stx  # 注释掉调试

import pydicom

category = 'teeth5_50'
dcm_folder = '/root/autodl-fs/SAX-NeRF/dataGenerator/CT_512'  # 修改为你的 DICOM 文件夹路径

# 获取所有的 .dcm 文件路径
dcm_files = [os.path.join(dcm_folder, f) for f in sorted(os.listdir(dcm_folder)) if f.endswith('.dcm')]

# 检查是否有273个文件
assert len(dcm_files) == 273, "文件数量不对，应该是273个"

# 读取所有的 DICOM 文件并堆叠成一个三维数组
slices = []
for dcm_file in dcm_files:
    ds = pydicom.dcmread(dcm_file)
    slices.append(ds.pixel_array)

# 将列表转换为 numpy 数组
raw_data_vis = np.stack(slices, axis=-1)  # 堆叠成 (512, 512, 273)

# 由于数据类型是 uint16，因此不需要进一步转换
# 直接进行数据归一化
data_float = np.float32(raw_data_vis) / raw_data_vis.max()
data_uint8 = np.uint8(data_float * 255)

# 选择切片展示
show_slice = 50
show_step = data_uint8.shape[-1] // show_slice
show_image = data_uint8[..., ::show_step]

# 创建可视化目录并保存切片图像
vis_dir = f'CT_vis/{category}/'
os.makedirs(vis_dir, exist_ok=True)

for i in range(show_slice + 1):
    iio.imwrite(f'{vis_dir}CT_dcm_{i}.png', show_image[..., i])

# 设置断点调试
# stx()

# 将处理后的数据保存为 .mat 文件
img_data = data_float
dict_data = {'img': img_data}
save_dir = f'raw_data/{category}/'
os.makedirs(save_dir, exist_ok=True)
io.savemat(f'{save_dir}img.mat', dict_data)