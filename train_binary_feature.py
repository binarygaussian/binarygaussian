# Borrowed from SAGA (https://github.com/Jumpat/SegAnyGAussians)
# We provide partial training code here, and will release the complete code as open source later.

import os
import torch
from random import randint
from gaussian_renderer import render_binary_feature
import sys
from scene import Scene, GaussianModel, FeatureGaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
import numpy as np
import torch
from torch import nn

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, iteration, saving_iterations, checkpoint_iterations, debug_from):
    print("RFN weight:", opt.rfn)
    print("Smooth K:", opt.smooth_K)
    print("Scale aware dim:", opt.scale_aware_dim)
    print("opacity_lr:", opt.opacity_lr)
    assert opt.ray_sample_rate > 0 or opt.num_sampled_rays > 0

    dataset.need_features = False
    dataset.need_masks = True
    gaussians = GaussianModel(dataset.sh_degree)

    feature_gaussians = FeatureGaussianModel(dataset.feature_dim)
    scene = Scene(dataset, gaussians, feature_gaussians, load_iteration=iteration, shuffle=False, target='binary_feature', mode='train', sample_rate=sample_rate)
    feature_gaussians.change_to_segmentation_mode(opt, "binary_feature", fixed_feature=False)

    smooth_weights = None

    del gaussians
    torch.cuda.empty_cache()

    background = torch.ones([dataset.feature_dim], dtype=torch.float32, device="cuda") if dataset.white_background else torch.zeros([dataset.feature_dim], dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    first_iter = 0
    viewpoint_stack = None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    gates = torch.zeros(3,32).cuda()
    number_dim = [0, 12, 22, 32]
    for i in range(3):
        gates[i, number_dim[i]:number_dim[i+1]] = 1

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        
        if iteration < -1:
            viewpoint_cam = viewpoint_stack[0]
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        with torch.no_grad():

            sam_masks = torch.load(viewpoint_cam.original_masks).cuda().float()
            # sam_masks = viewpoint_cam.original_masks.cuda().float()
            viewpoint_cam.feature_height, viewpoint_cam.feature_width = viewpoint_cam.image_height, viewpoint_cam.image_width

            # cal size
            scale_2d = sam_masks.sum(dim=(1,2))
            sort_indices = torch.argsort(scale_2d, descending=True)
            sam_masks = sam_masks[sort_indices, :, :]

            ray_sample_rate = opt.ray_sample_rate if opt.ray_sample_rate > 0 else opt.num_sampled_rays / (sam_masks.shape[-1] * sam_masks.shape[-2])
            # ray_sample_rate = 1000
            sampled_ray = torch.rand(sam_masks.shape[-2], sam_masks.shape[-1]).cuda() < ray_sample_rate * 2
            non_mask_region = sam_masks.sum(dim = 0) == 0

            sampled_ray = torch.logical_and(sampled_ray, ~non_mask_region)
            mask_add = torch.cumsum(sam_masks, dim=0)
            # cal mask level
            masks_level = mask_add * sam_masks # N_mask H W

            # Mask Balanced Sampling
            sampled_for_pos = mask_balanced_sampling(masks_level, num_samples=ray_sample_rate * 8, max_level=3)
            sampled_ray = torch.logical_or(sampled_ray, sampled_for_pos)

            sam_masks_sampled_ray = sam_masks[:, sampled_ray]   # N_mask N_pixel

            gt_corrs = []
            zero_masks = []   # no finer mask region
            sampled_level = [1,2,3]
            sam_masks_level = torch.cumsum(sam_masks_sampled_ray, dim=0) * sam_masks_sampled_ray # N_mask N_pixel

            for l in sampled_level:
                gt_vec = torch.where(sam_masks_level == l, sam_masks_sampled_ray, torch.zeros_like(sam_masks_sampled_ray))
                if l > 1:
                    # no mask for two pixel -> back to before mask
                    gt_vec_sum = gt_vec.sum(dim=0)  # N_pixel
                    zero_mask = gt_vec_sum == 0   # N_pixel
                    gt_vec[:, zero_mask] = gt_vec_ori[:, zero_mask]
                    zero_masks.append(zero_mask)

                gt_vec_ori = gt_vec.clone()
                # N_mask N_pixel , N_mask N_pixel -> N_pixel N_pixel
                gt_corr = torch.einsum('nh,nj->hj', gt_vec, gt_vec)

                gt_corr[gt_corr != 0] = 1
                gt_corrs.append(gt_corr)
                # sam_masks_level
            # N_scale N_pixel N_pixel
            gt_corrs = torch.stack(gt_corrs, dim = 0)
            zero_masks = torch.stack(zero_masks, dim = 0)

        render_pkg_feat = render_binary_feature(viewpoint_cam, feature_gaussians, pipe, background, norm_point_features=False, smooth_type = 'traditional', smooth_weights=torch.softmax(smooth_weights, dim = -1) if smooth_weights is not None else None, smooth_K = opt.smooth_K)
        rendered_features_ori = render_pkg_feat["render"]
        rendered_features_ori = rendered_features_ori[:,sampled_ray]  #32 N

        rendered_features, loss_norm = binarize_with_ste(rendered_features_ori)  #32 N
        rendered_feature_norm = torch.abs(rendered_features_ori - 0.5).mean()

        rendered_features_granularity = rendered_features.unsqueeze(0).repeat([len(sampled_level), 1, 1]) # 1 32 N
        rendered_features_granularity = rendered_features_granularity * gates.unsqueeze(-1)  # num_level 32 N
        rendered_features_granularity = rendered_features_granularity.permute([0,2,1]) # num_level N 32
        corr_t0 = (rendered_features_granularity ** 2).sum(dim=-1) # num_level N
        corr = corr_t0.unsqueeze(2) + corr_t0.unsqueeze(1) - 2 * torch.bmm(rendered_features_granularity, rendered_features_granularity.transpose(1, 2)) # num_level N N

        loss = loss_norm * 10
        diag_mask = torch.eye(corr.shape[1], dtype=bool, device=corr.device)
        max_dis = [12,10,10]
        gt_corrs_pre = torch.cat((torch.ones(gt_corrs.shape[1:]).unsqueeze(0).cuda(), gt_corrs[:-1]), dim=0) # for Gradient independence

        for i in sampled_level:
            j = i - 1

            sum_0 = gt_corrs[j]
            consistent_negative = sum_0 == 0
            consistent_positive = sum_0 == 1

            sampled_mask_positive = torch.logical_and(consistent_positive, ~diag_mask)
            sampled_mask_positive = torch.logical_and(sampled_mask_positive, gt_corrs_pre[j])
            sampled_mask_positive = torch.triu(sampled_mask_positive, diagonal=0)
            sampled_mask_positive = sampled_mask_positive.bool()
            
            sampled_mask_negative = torch.logical_and(consistent_negative, ~diag_mask)
            sampled_mask_negative = torch.logical_and(sampled_mask_negative, gt_corrs_pre[j])
            sampled_mask_negative = torch.triu(sampled_mask_negative, diagonal=0)
            sampled_mask_negative = sampled_mask_negative.bool()

            loss_corr = corr[j, sampled_mask_positive].mean() + torch.relu(max_dis[j] - corr[j, sampled_mask_negative]).mean()
            loss = loss + loss_corr

            # loss for Virtual Negative
            if j > 0: # skip level 1
                t0 = zero_masks[j-1] # no finer mask region
                loss_vn = torch.relu(max_dis[j] - corr_t0[j, t0]).mean()

                loss = loss + loss_vn


        with torch.no_grad():
            corr = torch.stack(corr, dim = 0)
            cosine_pos = corr[gt_corrs == 1].mean()
            cosine_neg = corr[gt_corrs == 0].mean()

        loss.backward()

        feature_gaussians.optimizer.step()
        feature_gaussians.optimizer.zero_grad(set_to_none = True)

        iter_end.record()

        if iteration % 10 == 0:
            progress_bar.set_postfix({
                "RFN": f"{rendered_feature_norm.item():.{3}f}",
                "Pos cos": f"{cosine_pos.item():.{3}f}",
                "Neg cos": f"{cosine_neg.item():.{3}f}",
                "Loss": f"{loss.item():.{3}f}",
            })
            progress_bar.update(10)

    
    scene.save_feature(iteration, target = 'binary_feature', smooth_weights = torch.softmax(smooth_weights, dim = -1) if smooth_weights is not None else None, smooth_type = 'traditional', smooth_K = smooth_K)


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output_v2/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=np.random.randint(10000, 20000))
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--target', default='binary_feature', const='binary_feature', nargs='?', choices=['scene', 'seg', 'feature', 'coarse_seg_everything', 'binary_feature'])
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--which_scene", default='figurines', type=str)
    
    # args = parser.parse_args(sys.argv[1:])
    args = get_combined_args(parser, target_cfg_file = 'cfg_args')
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.iteration, args.save_iterations, args.checkpoint_iterations, args.debug_from)

    # All done
    print("\nTraining complete.")
