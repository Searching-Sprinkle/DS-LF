# 可设置 验证集 固定采样策略
import os
import os.path as osp
import tigre
from tigre.utilities.geometry import Geometry
from tigre.utilities import gpu
import numpy as np
import yaml

import pickle
import scipy.io
import scipy.ndimage.interpolation
from tigre.utilities import CTnoise

import cv2
import matplotlib.pyplot as plt

import argparse
import imageio.v2 as iio

from pdb import set_trace as stx

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctName", default="P191", type=str,
                        help="Name of CT")
    parser.add_argument("--outputName", default="P191_50_49", type=str,
                        help="Name of output data")
    parser.add_argument("--dataFolder", default="raw_data", type=str,
                        help="folder of raw data")
    parser.add_argument("--outputFolder", default="./data", type=str,
                        help="folder of output data")
    return parser

def main():
    parser = config_parser()
    args = parser.parse_args()
    dataType = args.ctName
    dataFolder = args.dataFolder
    outputName = args.outputName
    outputFolder = args.outputFolder
    matPath = f"./dataGenerator/{dataFolder}/{dataType}/img.mat"
    configPath = f"./dataGenerator/{dataFolder}/{dataType}/config.yml"
    outputPath = osp.join(outputFolder, f"{outputName}.pickle")
    generator(matPath, configPath, outputPath, dataFolder, dataType, True)

# %% Geometry
class ConeGeometry_special(Geometry):
    def __init__(self, data):
        Geometry.__init__(self)
        self.DSD = data["DSD"] / 1000  # Distance Source Detector (m)
        self.DSO = data["DSO"] / 1000  # Distance Source Origin (m)
        self.nDetector = np.array(data["nDetector"])  # number of pixels (px)
        self.dDetector = np.array(data["dDetector"]) / 1000  # size of each pixel (m)
        self.sDetector = self.nDetector * self.dDetector  # total size of the detector (m)
        self.nVoxel = np.array(data["nVoxel"][::-1])  # number of voxels (vx)
        self.dVoxel = np.array(data["dVoxel"][::-1]) / 1000  # size of each voxel (m)
        self.sVoxel = self.nVoxel * self.dVoxel  # total size of the image (m)
        self.offOrigin = np.array(data["offOrigin"][::-1]) / 1000  # Offset of image from origin (m)
        self.offDetector = np.array([data["offDetector"][1], data["offDetector"][0], 0]) / 1000  # Offset of Detector (m)
        self.accuracy = data["accuracy"]  # Accuracy of FWD proj (vx/sample)
        self.mode = data["mode"]  # parallel, cone ...
        self.filter = data["filter"]

def convert_to_attenuation(data: np.array, rescale_slope: float, rescale_intercept: float):
    HU = data * rescale_slope + rescale_intercept
    mu_water = 0.206
    mu_air = 0.0004
    mu = mu_water + (mu_water - mu_air) / 1000 * HU
    return mu

def loadImage(dirname, nVoxels, convert, rescale_slope, rescale_intercept, normalize=True):
    if nVoxels is None:
        nVoxels = np.array((256, 256, 256))
    test_data = scipy.io.loadmat(dirname)
    image_ori = test_data["img"].astype(np.float32)
    if convert:
        print("Convert from HU to attenuation")
        image = convert_to_attenuation(image_ori, rescale_slope, rescale_intercept)
    else:
        image = image_ori
    imageDim = image.shape
    zoom_x = nVoxels[0] / imageDim[0]
    zoom_y = nVoxels[1] / imageDim[1]
    zoom_z = nVoxels[2] / imageDim[2]
    if zoom_x != 1.0 or zoom_y != 1.0 or zoom_z != 1.0:
        print(f"Resize ct image from {imageDim[0]}x{imageDim[1]}x{imageDim[2]} to "
              f"{nVoxels[0]}x{nVoxels[1]}x{nVoxels[2]}")
        image = scipy.ndimage.interpolation.zoom(
            image, (zoom_x, zoom_y, zoom_z), order=3, prefilter=False
        )
    image_max = np.max(image)
    image_min = np.min(image)
    image_mean = np.mean(image)
    print("Range of CT image is [%f, %f], mean: %f" % (image_min, image_max, image_mean))
    if normalize and image_min != 0 and image_max != 1:
        print("Normalize range to [0, 1]")
        image = (image - image_min) / (image_max - image_min)
    return image

def generator(matPath, configPath, outputPath, dataFolder, dataType, show=False):
    with open(configPath, "r") as handle:
        data = yaml.safe_load(handle)
    geo = ConeGeometry_special(data)
    img = loadImage(matPath, data["nVoxel"], data["convert"],
                    data["rescale_slope"], data["rescale_intercept"], data["normalize"])
    data["image"] = img.copy()

    # Generate training images
    if data["randomAngle"] is False:
        data["train"] = {"angles": np.linspace(0, data["totalAngle"] / 180 * np.pi, data["numTrain"]+1)[:-1] + data["startAngle"]/ 180 * np.pi}
    else:
        data["train"] = {"angles": np.sort(np.random.rand(data["numTrain"]) * data["totalAngle"] / 180 * np.pi) + data["startAngle"]/ 180 * np.pi}
    projections = tigre.Ax(np.transpose(img, (2, 1, 0)).copy(), geo, data["train"]["angles"])[:, ::-1, :]
    if data["noise"] != 0 and data["normalize"]:
        noise_projections = CTnoise.add(projections, Poisson=1e5, Gaussian=np.array([0, data["noise"]]))
        data["train"]["projections"] = noise_projections
    else:
        data["train"]["projections"] = projections

    # Generate validation images
    if data.get("randomAngleVal", False) is False:  # 如果没有randomAngleVal，默认为False
        data["val"] = {"angles": np.linspace(0, data["totalAngle"] / 180 * np.pi, data["numVal"]+1)[:-1] + data["startAngle"]/ 180 * np.pi}
    else:
        data["val"] = {"angles": np.sort(np.random.rand(data["numVal"]) * data["totalAngle"] / 180 * np.pi) + data["startAngle"]/ 180 * np.pi}
    projections = tigre.Ax(np.transpose(img, (2, 1, 0)).copy(), geo, data["val"]["angles"])[:, ::-1, :]
    if data["noise"] != 0 and data["normalize"]:
        noise_projections = CTnoise.add(projections, Poisson=1e5, Gaussian=np.array([0, data["noise"]]))
        data["val"]["projections"] = noise_projections
    else:
        data["val"]["projections"] = projections

    if show:
        save_dir_train_ct = osp.join('dataGenerator/', dataFolder, dataType, "show_vis_train_ct/")
        save_dir_train_proj = osp.join('dataGenerator/', dataFolder, dataType, "show_vis_train_proj/")
        save_dir_vali_proj = osp.join('dataGenerator/', dataFolder, dataType, "show_vis_vali_proj/")

        os.makedirs(save_dir_train_ct, exist_ok=True)
        os.makedirs(save_dir_train_proj, exist_ok=True)
        os.makedirs(save_dir_vali_proj, exist_ok=True)

        print("Shape of data['val']['projections']: ", data["val"]["projections"].shape)
        print("Shape of data['train']['projections']: ", data["train"]["projections"].shape)

        show_step = 5
        show_num_train = data["train"]["projections"].shape[0] // show_step
        show_num_val = data["val"]["projections"].shape[0] // show_step

        show_image_train_ct = img[...,::show_step]
        show_dir_train_proj = data["train"]["projections"][::show_step,...]
        show_dir_vali_proj  = data["val"]["projections"][::show_step,...]

        for i in range(show_num_val):  # 使用 show_num_val 而不是 show_num
            print(f"Processing index {i}...")
            try:
                iio.imwrite(save_dir_train_ct+'CT_'+str(i)+'.png', (show_image_train_ct[...,i]*255).astype(np.uint8))
                iio.imwrite(save_dir_train_proj+'projs_'+str(i)+'.png', (show_dir_train_proj[i,...]*255).astype(np.uint8))
                iio.imwrite(save_dir_vali_proj+'projs_'+str(i)+'.png', (show_dir_vali_proj[i,...]*255).astype(np.uint8))
            except IndexError as e:
                print(f"Error writing image at index {i}: {e}")

    # Save data
    os.makedirs(osp.dirname(outputPath), exist_ok=True)
    with open(outputPath, "wb") as handle:
        pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)

    print(f"Save files in {outputPath}")

if __name__ == "__main__":
    main()