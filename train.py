# import os
# import os.path as osp
# import torch
# import imageio.v2 as iio
# import numpy as np
# from tqdm import tqdm
# import argparse

# def config_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", default=f"./config/tensorf/chest_50.yaml", help="configs file path")
#     parser.add_argument("--gpu_id", default="0", help="gpu to use")
#     return parser

# parser = config_parser()
# args = parser.parse_args()

# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# from src.config.configloading import load_config
# from src.render import render, run_network
# from src.trainer import Trainer
# from src.loss import calc_mse_loss
# from src.utils import get_psnr, get_ssim, get_lpips, cast_to_image

# cfg = load_config(args.config)

# device = torch.device("cuda")

# class BasicTrainer(Trainer):
#     def __init__(self):
#         super().__init__(cfg, device)
#         print(f"[Start] exp: {cfg['exp']['expname']}, net: Basic network")

#     def compute_loss(self, data, global_step, idx_epoch):
#         rays = data["rays"].reshape(-1, 8)          
#         projs = data["projs"].reshape(-1)           
#         ret = render(rays, self.net, self.net_fine, **self.conf["render"])
#         projs_pred = ret["acc"]

#         loss = {"loss": 0.}
#         calc_mse_loss(loss, projs, projs_pred)

#         for ls in loss.keys():
#             self.writer.add_scalar(f"train/{ls}", loss[ls].item(), global_step)

#         return loss["loss"]

#     def eval_step(self, global_step, idx_epoch):
#         projs = self.eval_dset.projs                 
#         rays = self.eval_dset.rays.reshape(-1, 8)    
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
#             "lpips": get_lpips(projs_pred, projs)
#         }

#         if loss["proj_psnr"] > self.best_proj_psnr:
#             torch.save(
#                 {
#                     "epoch": idx_epoch,
#                     "network": self.net.state_dict(),
#                     "network_fine": self.net_fine.state_dict() if self.n_fine > 0 else None,
#                     "optimizer": self.optimizer.state_dict(),
#                 },
#                 self.ckpt_best_dir,
#             )
#             self.best_proj_psnr = loss["proj_psnr"]
#             self.logger.info(f"best model update, epoch:{idx_epoch}, best proj psnr:{self.best_proj_psnr:.4g}")

#         show_slice = 5
#         show_step = image.shape[-1] // show_slice
#         show_image = image[..., ::show_step]
#         show_image_pred = image_pred[..., ::show_step]
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
#             iio.imwrite(osp.join(proj_pred_origin_dir, f"proj_pred_{str(i)}.png"), (cast_to_image(projs_pred[i]) * 255).astype(np.uint8))
#             iio.imwrite(osp.join(proj_gt_origin_dir, f"proj_gt_{str(i)}.png"), (cast_to_image(projs[i]) * 255).astype(np.uint8))
#             iio.imwrite(osp.join(proj_pred_dir, f"proj_pred_{str(i)}.png"), ((1 - cast_to_image(projs_pred[i])) * 255).astype(np.uint8))
#             iio.imwrite(osp.join(proj_gt_dir, f"proj_gt_{str(i)}.png"), ((1 - cast_to_image(1 - projs[i])) * 255).astype(np.uint8))

#         for ls in loss.keys():
#             self.writer.add_scalar(f"eval/{ls}", loss[ls], global_step)

#         eval_save_dir = osp.join(self.evaldir, f"epoch_{idx_epoch:05d}")
#         os.makedirs(eval_save_dir, exist_ok=True)
#         np.save(osp.join(eval_save_dir, "image_pred.npy"), image_pred.cpu().detach().numpy())
#         np.save(osp.join(eval_save_dir, "image_gt.npy"), image.cpu().detach().numpy())
#         iio.imwrite(osp.join(eval_save_dir, "slice_show_row1_gt_row2_pred.png"), (cast_to_image(show_density) * 255).astype(np.uint8))
#         with open(osp.join(eval_save_dir, "stats.txt"), "w") as f: 
#             for key, value in loss.items(): 
#                 f.write("%s: %f\n" % (key, value.item()))

#         return loss

# trainer = BasicTrainer()
# trainer.start()
#单精度双GPU
import os
import os.path as osp
import torch
import imageio.v2 as iio
import numpy as np
from tqdm import tqdm
import argparse

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=f"./config/tensorf/chest_50.yaml", help="configs file path")
    parser.add_argument("--gpu_id", default="0", help="gpus to use, separated by comma")
    return parser

parser = config_parser()
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

from src.config.configloading import load_config
from src.render import render, run_network
from src.trainer import Trainer
from src.loss import calc_mse_loss
from src.utils import get_psnr, get_ssim, get_lpips, get_psnr_3d, get_ssim_3d, cast_to_image

cfg = load_config(args.config)

device_ids = list(map(int, args.gpu_id.split(',')))
device = torch.device(f"cuda:{device_ids[0]}")

class BasicTrainer(Trainer):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.cfg = cfg  # Ensure cfg is stored in the instance
        print(f"[Start] exp: {self.cfg['exp']['expname']}, net: Basic network")
        self.best_proj_psnr = -np.inf  # Initialize best_proj_psnr as a scalar

    def compute_loss(self, data, global_step, idx_epoch):
        rays = data["rays"].reshape(-1, 8).to(device)          
        projs = data["projs"].reshape(-1).to(device)           
        ret = render(rays, self.net, self.net_fine, **self.conf["render"])
        projs_pred = ret["acc"]

        loss = {"loss": 0.}
        calc_mse_loss(loss, projs, projs_pred)

        for ls in loss.keys():
            self.writer.add_scalar(f"train/{ls}", loss[ls].item(), global_step)

        return loss["loss"]

    def eval_step(self, global_step, idx_epoch):
        projs = self.eval_dset.projs.to(device)                 
        rays = self.eval_dset.rays.reshape(-1, 8).to(device)    
        N, H, W = projs.shape
        projs_pred = []
        for i in tqdm(range(0, rays.shape[0], self.n_rays)):     
            projs_pred.append(render(rays[i:i+self.n_rays], self.net, self.net_fine, **self.conf["render"])["acc"])
        projs_pred = torch.cat(projs_pred, 0).reshape(N, H, W)

        image = self.eval_dset.image.to(device)
        image_pred = run_network(self.eval_dset.voxels.to(device), self.net_fine if self.net_fine is not None else self.net, self.netchunk)
        image_pred = image_pred.squeeze()

        loss = {
            "proj_psnr": get_psnr(projs_pred, projs),
            "proj_ssim": get_ssim(projs_pred, projs),
            # "psnr_3d": get_psnr_3d(image_pred, image),
            # "ssim_3d": get_ssim_3d(image_pred, image),
            "lpips": get_lpips(projs_pred, projs) * 1000
        }

        if loss["proj_psnr"] > self.best_proj_psnr:
            torch.save(
                {
                    "epoch": idx_epoch,
                    "network": self.net.module.state_dict() if isinstance(self.net, torch.nn.DataParallel) else self.net.state_dict(),
                    "network_fine": (self.net_fine.module.state_dict() if isinstance(self.net_fine, torch.nn.DataParallel) else self.net_fine.state_dict()) if self.n_fine > 0 else None,
                    "optimizer": self.optimizer.state_dict(),
                },
                self.ckpt_best_dir,
            )
            self.best_proj_psnr = loss["proj_psnr"].item()  # Convert to scalar before assignment
            self.logger.info(f"best model update, epoch:{idx_epoch}, best proj psnr:{self.best_proj_psnr:.4g}")

        show_slice = 10
        show_step = image.shape[-1] // show_slice
        show_image = image[..., ::show_step]
        show_image_pred = image_pred[..., ::show_step]
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
            iio.imwrite(osp.join(proj_pred_origin_dir, f"proj_pred_{str(i)}.png"), (cast_to_image(projs_pred[i]) * 255).astype(np.uint8))
            iio.imwrite(osp.join(proj_gt_origin_dir, f"proj_gt_{str(i)}.png"), (cast_to_image(projs[i]) * 255).astype(np.uint8))
            iio.imwrite(osp.join(proj_pred_dir, f"proj_pred_{str(i)}.png"), ((1 - cast_to_image(projs_pred[i])) * 255).astype(np.uint8))
            iio.imwrite(osp.join(proj_gt_dir, f"proj_gt_{str(i)}.png"), ((1 - cast_to_image(1 - projs[i])) * 255).astype(np.uint8))

        for ls in loss.keys():
            self.writer.add_scalar(f"eval/{ls}", loss[ls], global_step)

        eval_save_dir = osp.join(self.evaldir, f"epoch_{idx_epoch:05d}")
        os.makedirs(eval_save_dir, exist_ok=True)
        np.save(osp.join(eval_save_dir, "image_pred.npy"), image_pred.cpu().detach().numpy())
        np.save(osp.join(eval_save_dir, "image_gt.npy"), image.cpu().detach().numpy())
        iio.imwrite(osp.join(eval_save_dir, "slice_show_row1_gt_row2_pred.png"), (cast_to_image(show_density) * 255).astype(np.uint8))
        with open(osp.join(eval_save_dir, "stats.txt"), "w") as f: 
            for key, value in loss.items(): 
                f.write("%s: %f\n" % (key, value.item()))

        return loss

trainer = BasicTrainer(cfg, device)
if len(device_ids) > 1:
    trainer.net = torch.nn.DataParallel(trainer.net, device_ids=device_ids)
    if trainer.net_fine is not None:
        trainer.net_fine = torch.nn.DataParallel(trainer.net_fine, device_ids=device_ids)

# Manually set optimizer with unique parameters
unique_params = set()
for param_group in trainer.optimizer.param_groups:
    for p in param_group['params']:
        unique_params.add(p)
trainer.optimizer = torch.optim.Adam(unique_params, lr=trainer.cfg['train']['lrate'])

trainer.start()


# import os
# import os.path as osp
# import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# import imageio.v2 as iio
# import numpy as np
# from tqdm import tqdm
# import argparse
# from src.config.configloading import load_config
# from src.render import render, run_network
# from src.trainer import Trainer
# from src.loss import calc_mse_loss
# from src.utils import get_psnr, get_ssim, get_lpips, get_psnr_3d, get_ssim_3d, cast_to_image

# # 配置解析
# def config_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", default="./config/tensorf/chest_50.yaml", help="configs file path")
#     parser.add_argument("--gpu_id", default="0,1", help="gpus to use, separated by comma")
#     return parser

# parser = config_parser()
# args = parser.parse_args()

# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# device_ids = list(map(int, args.gpu_id.split(',')))

# # 初始化分布式环境
# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)

# # 清理分布式环境
# def cleanup():
#     dist.destroy_process_group()

# # BasicTrainer 类
# class BasicTrainer(Trainer):
#     def __init__(self, cfg, device, rank, world_size):
#         super().__init__(cfg, device)
#         self.cfg = cfg
#         self.rank = rank
#         self.world_size = world_size
#         self.device = device  # 确保 device 属性被正确初始化
#         print(f"[Start] exp: {self.cfg['exp']['expname']}, net: Basic network")
#         self.best_proj_psnr = -np.inf

#         # 加载评估数据集
#         self.eval_dset = load_eval_dataset(cfg)  # 确保 load_eval_dataset 函数已实现

#         # 初始化 DistributedDataParallel
#         self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[device], output_device=device)
#         if self.net_fine is not None:
#             self.net_fine = torch.nn.parallel.DistributedDataParallel(self.net_fine, device_ids=[device], output_device=device)

#         # 确保 optimizer 中没有重复参数
#         params = list(dict.fromkeys(list(self.net.parameters()) + (list(self.net_fine.parameters()) if self.net_fine is not None else [])))
#         self.optimizer = torch.optim.Adam(params, lr=self.cfg['train']['lrate'])

#     def compute_loss(self, data, global_step, idx_epoch):
#         rays = data["rays"].reshape(-1, 8).to(self.device)
#         projs = data["projs"].reshape(-1).to(self.device)
        
#         ret = render(rays, self.net, self.net_fine, **self.conf["render"])
#         projs_pred = ret["acc"]

#         loss = calc_mse_loss(projs, projs_pred)

#         self.writer.add_scalar("train/loss", loss.item(), global_step)

#         return loss

#     def eval_step(self, global_step, idx_epoch):
#         projs = self.eval_dset.projs.to(self.device)
#         rays = self.eval_dset.rays.reshape(-1, 8).to(self.device)
#         N, H, W = projs.shape
#         projs_pred = []
        
#         with torch.no_grad():
#             for i in tqdm(range(0, rays.shape[0], self.n_rays)):
#                 projs_pred.append(render(rays[i:i+self.n_rays], self.net, self.net_fine, **self.conf["render"])["acc"])
        
#         projs_pred = torch.cat(projs_pred, 0).reshape(N, H, W)

#         image = self.eval_dset.image.to(self.device)
#         image_pred = run_network(self.eval_dset.voxels.to(self.device), self.net_fine if self.net_fine is not None else self.net, self.netchunk)
#         image_pred = image_pred.squeeze()

#         loss = {
#             "proj_psnr": get_psnr(projs_pred, projs),
#             "proj_ssim": get_ssim(projs_pred, projs),
#             "psnr_3d": get_psnr_3d(image_pred, image),
#             "ssim_3d": get_ssim_3d(image_pred, image),
#             "lpips": get_lpips(projs_pred, projs) * 1000
#         }

#         # 使用 all_reduce 来聚合不同进程的结果
#         for key in loss.keys():
#             dist.all_reduce(loss[key])
#             loss[key] /= self.world_size

#         if loss["proj_psnr"] > self.best_proj_psnr and self.rank == 0:
#             torch.save(
#                 {
#                     "epoch": idx_epoch,
#                     "network": self.net.module.state_dict(),
#                     "network_fine": (self.net_fine.module.state_dict() if self.net_fine is not None else None),
#                     "optimizer": self.optimizer.state_dict(),
#                 },
#                 self.ckpt_best_dir,
#             )
#             self.best_proj_psnr = loss["proj_psnr"].item()
#             self.logger.info(f"best model update, epoch:{idx_epoch}, best proj psnr:{self.best_proj_psnr:.4g}")

#         show_slice = 5
#         show_step = image.shape[-1] // show_slice
#         show_image = image[..., ::show_step]
#         show_image_pred = image_pred[..., ::show_step]
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

#         if self.rank == 0:
#             for i in tqdm(range(N)):
#                 iio.imwrite(osp.join(proj_pred_origin_dir, f"proj_pred_{str(i)}.png"), (cast_to_image(projs_pred[i]) * 255).astype(np.uint8))
#                 iio.imwrite(osp.join(proj_gt_origin_dir, f"proj_gt_{str(i)}.png"), (cast_to_image(projs[i]) * 255).astype(np.uint8))
#                 iio.imwrite(osp.join(proj_pred_dir, f"proj_pred_{str(i)}.png"), ((1 - cast_to_image(projs_pred[i])) * 255).astype(np.uint8))
#                 iio.imwrite(osp.join(proj_gt_dir, f"proj_gt_{str(i)}.png"), ((1 - cast_to_image(1 - projs[i])) * 255).astype(np.uint8))

#             for ls in loss.keys():
#                 self.writer.add_scalar(f"eval/{ls}", loss[ls], global_step)

#             eval_save_dir = osp.join(self.evaldir, f"epoch_{idx_epoch:05d}")
#             os.makedirs(eval_save_dir, exist_ok=True)
#             np.save(osp.join(eval_save_dir, "image_pred.npy"), image_pred.cpu().detach().numpy())
#             np.save(osp.join(eval_save_dir, "image_gt.npy"), image.cpu().detach().numpy())
#             iio.imwrite(osp.join(eval_save_dir, "slice_show_row1_gt_row2_pred.png"), (cast_to_image(show_density) * 255).astype(np.uint8))
#             with open(osp.join(eval_save_dir, "stats.txt"), "w") as f:
#                 for key, value in loss.items():
#                     f.write("%s: %f\n" % (key, value.item()))

#         return loss

# # 主函数
# def main(rank, world_size):
#     setup(rank, world_size)
#     device = torch.device(f"cuda:{rank}")
    
#     # 加载配置
#     cfg = load_config(args.config)
    
#     # 初始化训练器
#     trainer = BasicTrainer(cfg, device, rank, world_size)
    
#     # 启动训练
#     trainer.start()
    
#     # 清理分布式环境
#     cleanup()

# if __name__ == "__main__":
#     world_size = len(device_ids)
#     mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)






 