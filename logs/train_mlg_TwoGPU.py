# import os
# import os.path as osp
# import torch
# import imageio.v2 as iio
# from tqdm import tqdm
# import numpy as np
# import argparse

# # 参数部分，定义各项参数和默认值
# def config_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", default="./config/Lineformer/chest_50.yaml", help="configs file path")
#     parser.add_argument("--gpu_ids", default="0,1", help="gpus to use, separated by comma")
#     return parser

# parser = config_parser()
# args = parser.parse_args()

# # 设置环境变量，确保使用指定的GPU来运行代码。
# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

# # 导入项目中的自定义模块
# from src.config.configloading import load_config
# from src.render import render, run_network
# from src.trainer_mlg import Trainer
# from src.loss import calc_mse_loss
# from src.utils import get_psnr, get_ssim, get_psnr_3d, get_ssim_3d, get_lpips, cast_to_image

# # 加载指定的配置文件
# cfg = load_config(args.config)

# # 指定设备，选择GPU进行训练
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class BasicTrainer(Trainer):
#     def __init__(self):
#         super().__init__(cfg, device)
#         print(f"[Start] exp: {cfg['exp']['expname']}, net: Basic network")

#         # 将模型移动到多个GPU上
#         if torch.cuda.device_count() > 1:
#             print(f"Using {torch.cuda.device_count()} GPUs!")
#             self.net = torch.nn.DataParallel(self.net)
#             if self.net_fine is not None:
#                 self.net_fine = torch.nn.DataParallel(self.net_fine)

#     def compute_loss(self, data, global_step, idx_epoch):
#         rays = data["rays"].reshape(-1, 8)             # [1, 1024, 8] -> [1024, 8]
#         projs = data["projs"].reshape(-1)            # projection 的 ground truth [1, 1024] -> [1024]
        
#         ret = render(rays, self.net, self.net_fine, **self.conf["render"])
#         projs_pred = ret["acc"]
        
#         loss = {"loss": 0.}
#         calc_mse_loss(loss, projs_pred, projs)

#         for ls in loss.keys():
#             self.writer.add_scalar(f"train/{ls}", loss[ls].item(), global_step)

#         return loss["loss"]

#     def eval_step(self, global_step, idx_epoch):
#         projs = self.eval_dset.projs                 # [256, 256] -> [50, 256, 256]
#         rays = self.eval_dset.rays.reshape(-1, 8)    # [65536,8]  -> [3276800, 8]
#         N, H, W = projs.shape
#         projs_pred = []
#         for i in tqdm(range(0, rays.shape[0], self.n_rays)):
#             projs_pred.append(render(rays[i:i+self.n_rays], self.net, self.net_fine, **self.conf["render"])["acc"])
#         projs_pred = torch.cat(projs_pred, 0).reshape(N, H, W)

#         image = self.eval_dset.image
#         image_pred = run_network(self.eval_dset.voxels, self.net_fine if self.net_fine is not None else self.net, self.netchunk)
#         image_pred = image_pred.squeeze()

#         loss = {
#             "proj_psnr": get_psnr(projs_pred, projs),
#             "proj_ssim": get_ssim(projs_pred, projs),
#             "psnr_3d": get_psnr_3d(image_pred, image),
#             "ssim_3d": get_ssim_3d(image_pred, image),
#             "lpips": get_lpips(projs_pred, projs)*1000
#         }

#         if loss["psnr_3d"] > self.best_psnr_3d:
#             torch.save({
#                 "epoch": idx_epoch,
#                 "network": self.net.module.state_dict(),
#                 "network_fine": self.net_fine.module.state_dict() if self.n_fine > 0 else None,
#                 "optimizer": self.optimizer.state_dict(),
#             }, self.ckpt_best_dir)
#             self.best_psnr_3d = loss["psnr_3d"]
#             self.logger.info(f"best model update, epoch:{idx_epoch}, best 3d psnr:{self.best_psnr_3d:.4g}")

#         show_slice = 5
#         show_step = image.shape[-1]//show_slice
#         show_image = image[...,::show_step]
#         show_image_pred = image_pred[...,::show_step]
#         show = []
#         for i_show in range(show_slice):
#             show.append(torch.concat([show_image[..., i_show], show_image_pred[..., i_show]], dim=0))
#         show_density = torch.concat(show, dim=1)

#         self.writer.add_image("eval/density (row1: gt, row2: pred)", cast_to_image(show_density), global_step, dataformats="HWC")

#         proj_pred_origin_dir = osp.join(self.expdir, "proj_pred_origin")
#         proj_gt_origin_dir = osp.join(self.expdir, "proj_gt_origin")
#         proj_pred_dir = osp.join(self.expdir, "proj_pred")
#         proj_gt_dir = osp.join(self.expdir, "proj_gt")
#         os.makedirs(proj_pred_origin_dir, exist_ok=True)
#         os.makedirs(proj_gt_origin_dir, exist_ok=True)
#         os.makedirs(proj_pred_dir, exist_ok=True)
#         os.makedirs(proj_gt_dir, exist_ok=True)

#         for i in tqdm(range(N)):
#             iio.imwrite(osp.join(proj_pred_origin_dir, f"proj_pred_{str(i)}.png"), (cast_to_image(projs_pred[i])*255).astype(np.uint8))
#             iio.imwrite(osp.join(proj_gt_origin_dir, f"proj_gt_{str(i)}.png"), (cast_to_image(projs[i])*255).astype(np.uint8))
#             iio.imwrite(osp.join(proj_pred_dir, f"proj_pred_{str(i)}.png"), ((1-cast_to_image(projs_pred[i]))*255).astype(np.uint8))
#             iio.imwrite(osp.join(proj_gt_dir, f"proj_gt_{str(i)}.png"), ((1-cast_to_image(1-projs[i]))*255).astype(np.uint8))

#         for ls in loss.keys():
#             self.writer.add_scalar(f"eval/{ls}", loss[ls], global_step)

#         eval_save_dir = osp.join(self.evaldir, f"epoch_{idx_epoch:05d}")
#         os.makedirs(eval_save_dir, exist_ok=True)
#         np.save(osp.join(eval_save_dir, "image_pred.npy"), image_pred.cpu().detach().numpy())
#         np.save(osp.join(eval_save_dir, "image_gt.npy"), image.cpu().detach().numpy())
#         iio.imwrite(osp.join(eval_save_dir, "slice_show_row1_gt_row2_pred.png"), (cast_to_image(show_density)*255).astype(np.uint8))
#         with open(osp.join(eval_save_dir, "stats.txt"), "w") as f: 
#             for key, value in loss.items(): 
#                 f.write("%s: %f\n" % (key, value.item()))

#         return loss
    
# trainer = BasicTrainer()
# trainer.start()



import os
import os.path as osp
import torch
import imageio.v2 as iio
from tqdm import tqdm
import numpy as np
import argparse

# 参数部分，定义各项参数和默认值
def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/Lineformer/chest_50.yaml", help="configs file path")
    parser.add_argument("--gpu_ids", default="0,1", help="gpus to use, separated by comma")
    return parser

parser = config_parser()
args = parser.parse_args()

# 设置环境变量，确保使用指定的GPU来运行代码。
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

# 导入项目中的自定义模块
from src.config.configloading import load_config
from src.render import render, run_network
from src.trainer_mlg import Trainer
from src.loss import calc_mse_loss
from src.utils import get_psnr, get_ssim, get_psnr_3d, get_ssim_3d, get_lpips, cast_to_image

# 加载指定的配置文件
cfg = load_config(args.config)

# 指定设备，选择GPU进行训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BasicTrainer(Trainer):
    def __init__(self):
        super().__init__(cfg, device)
        print(f"[Start] exp: {cfg['exp']['expname']}, net: Basic network")

        # 将模型移动到多个GPU上
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.net = torch.nn.DataParallel(self.net)
            if self.net_fine is not None:
                self.net_fine = torch.nn.DataParallel(self.net_fine)

    def compute_loss(self, data, global_step, idx_epoch):
        rays = data["rays"].reshape(-1, 8)             # [1, 1024, 8] -> [1024, 8]
        projs = data["projs"].reshape(-1)            # projection 的 ground truth [1, 1024] -> [1024]
        
        ret = render(rays, self.net, self.net_fine, **self.conf["render"])
        projs_pred = ret["acc"]
        
        loss = {"loss": 0.}
        calc_mse_loss(loss, projs_pred, projs)

        for ls in loss.keys():
            self.writer.add_scalar(f"train/{ls}", loss[ls].item(), global_step)

        # 清理缓存
        torch.cuda.empty_cache()

        return loss["loss"]

    def eval_step(self, global_step, idx_epoch):
        projs = self.eval_dset.projs                 # [256, 256] -> [50, 256, 256]
        rays = self.eval_dset.rays.reshape(-1, 8)    # [65536,8]  -> [3276800, 8]
        N, H, W = projs.shape
        projs_pred = []
        
        # 负载均衡：将rays分块处理
        chunk_size = 1024  # 根据实际情况调整这个值
        for i in tqdm(range(0, rays.shape[0], chunk_size)):
            torch.cuda.empty_cache()  # 在每个chunk开始前清理缓存
            chunk_rays = rays[i:i+chunk_size]
            projs_pred.append(render(chunk_rays, self.net, self.net_fine, **self.conf["render"])["acc"])
        
        projs_pred = torch.cat(projs_pred, 0).reshape(N, H, W)

        image = self.eval_dset.image
        image_pred = run_network(self.eval_dset.voxels, self.net_fine if self.net_fine is not None else self.net, self.netchunk)
        image_pred = image_pred.squeeze()

        loss = {
            "proj_psnr": get_psnr(projs_pred, projs),
            "proj_ssim": get_ssim(projs_pred, projs),
            "psnr_3d": get_psnr_3d(image_pred, image),
            "ssim_3d": get_ssim_3d(image_pred, image),
            "lpips": get_lpips(projs_pred, projs)*1000
        }

        if loss["psnr_3d"] > self.best_psnr_3d:
            torch.save({
                "epoch": idx_epoch,
                "network": self.net.module.state_dict(),
                "network_fine": self.net_fine.module.state_dict() if self.n_fine > 0 else None,
                "optimizer": self.optimizer.state_dict(),
            }, self.ckpt_best_dir)
            self.best_psnr_3d = loss["psnr_3d"]
            self.logger.info(f"best model update, epoch:{idx_epoch}, best 3d psnr:{self.best_psnr_3d:.4g}")

        # show_slice = 5
        show_slice = 10
        show_step = image.shape[-1]//show_slice
        show_image = image[...,::show_step]
        show_image_pred = image_pred[...,::show_step]
        show = []
        for i_show in range(show_slice):
            show.append(torch.concat([show_image[..., i_show], show_image_pred[..., i_show]], dim=0))
        show_density = torch.concat(show, dim=1)

        self.writer.add_image("eval/density (row1: gt, row2: pred)", cast_to_image(show_density), global_step, dataformats="HWC")

        proj_pred_origin_dir = osp.join(self.expdir, "proj_pred_origin")
        proj_gt_origin_dir = osp.join(self.expdir, "proj_gt_origin")
        proj_pred_dir = osp.join(self.expdir, "proj_pred")
        proj_gt_dir = osp.join(self.expdir, "proj_gt")
        os.makedirs(proj_pred_origin_dir, exist_ok=True)
        os.makedirs(proj_gt_origin_dir, exist_ok=True)
        os.makedirs(proj_pred_dir, exist_ok=True)
        os.makedirs(proj_gt_dir, exist_ok=True)

        for i in tqdm(range(N)):
            iio.imwrite(osp.join(proj_pred_origin_dir, f"proj_pred_{str(i)}.png"), (cast_to_image(projs_pred[i])*255).astype(np.uint8))
            iio.imwrite(osp.join(proj_gt_origin_dir, f"proj_gt_{str(i)}.png"), (cast_to_image(projs[i])*255).astype(np.uint8))
            iio.imwrite(osp.join(proj_pred_dir, f"proj_pred_{str(i)}.png"), ((1-cast_to_image(projs_pred[i]))*255).astype(np.uint8))
            iio.imwrite(osp.join(proj_gt_dir, f"proj_gt_{str(i)}.png"), ((1-cast_to_image(1-projs[i]))*255).astype(np.uint8))

        for ls in loss.keys():
            self.writer.add_scalar(f"eval/{ls}", loss[ls], global_step)

        eval_save_dir = osp.join(self.evaldir, f"epoch_{idx_epoch:05d}")
        os.makedirs(eval_save_dir, exist_ok=True)
        np.save(osp.join(eval_save_dir, "image_pred.npy"), image_pred.cpu().detach().numpy())
        np.save(osp.join(eval_save_dir, "image_gt.npy"), image.cpu().detach().numpy())
        iio.imwrite(osp.join(eval_save_dir, "slice_show_row1_gt_row2_pred.png"), (cast_to_image(show_density)*255).astype(np.uint8))
        with open(osp.join(eval_save_dir, "stats.txt"), "w") as f: 
            for key, value in loss.items(): 
                f.write("%s: %f\n" % (key, value.item()))

        # 最后再清理一次缓存
        torch.cuda.empty_cache()

        return loss
    
trainer = BasicTrainer()
trainer.start()