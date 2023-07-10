import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    对 input 进行处理，应用 神经网络 fn
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]]) # [1024*64, 3]
    embedded = embed_fn(inputs_flat)
    # 如果视图方向不为 None，即输入了视图方向，
    # 那么我们就应该考虑对视图方向作出处理，用以生成颜色
    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        # 对输入方向进行编码
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        # 将编码后的位置和方向拼接到一起
        embedded = torch.cat([embedded, embedded_dirs], -1)
    # 将编码过的点以批处理的形式输入到网络模型中得到输出(RGB,A)
    # batchify() 函数会把 embedded 数组分批输入到网络 fn 中，前向传播得到对应的 (RGB，A)
    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

# rays_flat [1024, 8]
def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM(Out-Of-Memory).
       小批量渲染光线以避免显存爆炸
    """
    all_ret = {}
    # shape: rays_flat[1024,8]
    for i in range(0, rays_flat.shape[0], chunk):
        # 渲染光线
        # ret是一个字典,shape:rgb_map[4096,3] disp_map[4096] acc_map[4096] raw[4096,64,4]
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        # 每一个key对应一个list，list包含了所有的ret对应key的value
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      focal: 针孔相机的焦距
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      chunk: 并行处理的光线的数量,即需要同时处理的最大射线数。用来控制最大内存使用。不影响最终结果。
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      rays: 批次中每个示例的射线原点和方向。即经由上一步骤我们挑选出的光线。
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      c2w: 相机坐标到世界坐标的变换矩阵
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      ndc: 如果为True，则以NDC坐标表示光线原点和方向。NDC:标准化/归一化的设备坐标(Normalized Device Coordinates)。
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      use_viewdirs: 如果为True，请使用模型中空间点的观测方向。
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
      c2w_staticcam: 如果不是None，则将此变换矩阵用于相机，同时将其他c2w参数用于观测方向。
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays. 光线的预测RGB值。
      disp_map: [batch_size]. Disparity map. Inverse of depth. 视差图。深度的倒数。
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray. 沿光线累积的不透明度alpha。
      extras: dict with everything returned by render_rays(). 字典，使用render_rays()返回的所有内容。

    render函数返回的是光束对应的rgb图、视差图、不透明度，以及raw
    """
    if c2w is not None:
        # special case to render full image 渲染完整图像的特殊情况
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch 使用提供的射线批次,默认是这种情况
        rays_o, rays_d = rays  # rays_o [1024, 3]  rays_d [1024, 3] rays [2, 1024, 3] 
    # 如果使用视图方向，根据光线的 ray_d 计算单位方向作为 view_dirs, 默认没有使用这种情况
    if use_viewdirs:
        # provide ray directions as input 提供光线方向作为输入
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs 可视化空间点的观测方向效果的特例
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [1024, 3]
    if ndc:
        # for forward facing scenes 对于前向场景效果好，适用于前向场景构建
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float() # [1024, 3]
    rays_d = torch.reshape(rays_d, [-1,3]).float() # [1024, 3]
    # 生成光线的远近端，用于确定边界框，并将其拼接到 rays 中
    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1]) # near、far [1024, 1]
    rays = torch.cat([rays_o, rays_d, near, far], -1) # [1024, 8]
    # 视图方向拼接到光线中
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape 
    # 开始并行计算光线属性
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


# 根据pose等信息获得颜色和视差
def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


# 初始化nerf的MLP模型
def create_nerf(args):
    """Instantiate NeRF's MLP model.
        先调用get_embedder获得一个对应的embedding函数，
        然后构建NeRF模型
    """
    # 获得一个编码器 embed_fn 以及一个编码后的维度 input_ch
    # 给定 embed_fn 一个输入，就可以获得输入的编码后的数据
    # i_embed 设置是否使用位置编码。0表示使用，-1不使用
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed) # embed_fn:一个实例化的关于位置点x的embedder函数, input_ch = 63

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed) # embeddirs_fn:一个实例化的关于方向view的embedder函数, input_ch_views = 27 
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4] # 在第i=4层有一个skip连接
    # 初始化MLP模型参数
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    # 模型中的梯度变量
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        # 精细网络
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())
    # 定义一个查询点的颜色和密度的匿名函数，
    # 实际上给定点坐标，方向，以及查询网络，
    # 我们就可以得到该点在该网络下的输出([rgb,alpha])
    # network_query_fn 是一个匿名函数，真正起作用的函数是 run_network()
    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer 创建优化器
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints 加载模型
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model 加载已有模型
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################
    # 整体初始化已完成，对返回值进行一些处理
    '''
    1.network_query_fn  这是一个匿名函数，给这个函数输入位置坐标，方向坐标，以及神经网络，
                        就可以利用神经网络返回该点对应的 颜色和密度
    2.perturb           扰动，对整体算法理解没有影响
    3.N_importance      args.N_importance,每条光线上细采样点的数量
    4.network_fine      model_fine,论文中的 精细网络
    5.N_samples         args.N_samples,每条光线上粗采样点的数量
    6.network_fn        model,论文中的 粗网络
    7.use_viewdirs      args.use_viewdirs,是否使用视点方向，影响到神经网络是否输出颜色
    8.white_bkgd        args.white_bkgd,如果为 True 将输入的 png 图像的透明部分转换成白色
    9.raw_noise_std     args.raw_noise_std,归一化密度
    '''
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }
    # NDC 空间，只对前向场景有效，具体解释可以看论文
    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


# 把模型的预测转化为有实际意义的表达，
# 输入预测、时间和光束方向，将离散的点进行积分，得到对应的像素颜色，(体渲染公式的离散形式的具体代码)
# 输出光束颜色、视差、密度、每个采样点的权重和深度
def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    # 定义一个匿名函数 raw2alpha
    # raw2alpha 代表体渲染公式中的 1−exp(−σ∗δ) 计算每个点的透明度
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    # 计算两点Z轴之间的距离
    dists = z_vals[...,1:] - z_vals[...,:-1]  # coarse [1024, 63] fine [1024, 191]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples] coarse [1024, 64] fine [1024, 192]
    # 将 Z 轴之间的距离转换为实际距离
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)  # coarse [1024, 64] fine [1024, 192]

    # 获取模型预测的每个点的 RGB 值
    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3] coarse [1024, 64, 3] fine [1024, 192, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    # 给透明度加噪音
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples] coarse [1024, 64] fine [1024, 192]
    # alpha 代表体渲染公式中的 1−exp(−σ∗δ)
    # torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1) 即代表公式中的 Ti, torch.cumprod累乘
    # weights 实际上代表的是渲染公式中的Ti(1−exp(−σi∗δi)) 
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1] # [1024, 64] [1024, 192]
    # rgb_map 代表了最终的渲染颜色
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3] [1024, 3]

    depth_map = torch.sum(weights * z_vals, -1) # [N_rays] [1024]
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)) # [N_rays] [1024]
    acc_map = torch.sum(weights, -1) # [N_rays] [1024]

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None]) # [1024, 3]

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,          # 1024
                network_fn,         # NeRF class
                network_query_fn,   # 
                N_samples,          # 64
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,     # 128
                network_fine=None,  # NeRF class
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    # 1.从ray_batch提取需要的数据，即从 ray 中分离出 rays_o, rays_d, viewdirs, near, far
    # 光束数量默认4096或1024
    N_rays = ray_batch.shape[0] # 1024
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each, 即都是[1024, 3]
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None # None
    # shape: bounds[4096,1,2] near[4096,1] far[4096,1]
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])  # [1024, 1, 2]
    near, far = bounds[...,0], bounds[...,1] # 都是[1024,1]
    # 2.确定空间中一个坐标的 Z 轴位置
    # 每个光束上取N_samples个点,即在 0-1 内生成 N_samples 个等差点，默认64个。
    t_vals = torch.linspace(0., 1., steps=N_samples) # [64]
    # 根据参数确定不同的采样方式,从而确定 Z 轴在边界框内的的具体位置
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples]) # [1024, 64]

    # 如果有增加扰动
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand # [1024, 64]

    # 生成光线上每个采样点的位置
    # 空间中的点的位置 = 原点位置 + 方向和距离的乘积
    # 光束打到的位置(采样点)，可用来输入网络查询颜色和密度 
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3] [1024, 64, 3]

    # 将光线上的每个点投入到 MLP 网络 network_fn 中前向传播得到每个点对应的(RGB，A)
    # 根据pts,viewdirs进行前向计算。raw[4096,64,4]，最后一个维是RGB+density。
    raw = network_query_fn(pts, viewdirs, network_fn)
    # 这一步相当于是在做volume render，将光束颜色合成图像上的点
    # 对这些离散点进行体积渲染，即进行积分操作(累加操作)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
    # 分层采样的细采样阶段(精细网络)，会再算一遍上述步骤，然后也封装到ret
    if N_importance > 0:
        # 保存前面的值
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
        # 重新采样光束上的点
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1]) # [1024, 63]
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest) # [1024, 128]
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1) # 拼接后进行排序, shape=[1024, 192]
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3] [1024, 192, 3]

        run_fn = network_fn if network_fine is None else network_fine
        # 生成新采样点的颜色密度
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    # 不管有无精细网络都要
    # shape: rgb_map[4096,3] disp_map[4096] acc_map[4096]
    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map} # [1024, 3] [1024] [1024]
    if retraw:
        ret['raw'] = raw # [1024, 192, 5]
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0 # [1024, 3]
        ret['disp0'] = disp_map_0 # [1024]
        ret['acc0'] = acc_map_0 # [1024]
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays] [1024] torch.std返回尺寸为 dim 的 input 张量的每一行的标准差

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    # config文件，放在./configs/下
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    # 指定实验名称
    parser.add_argument("--expname", type=str, default='blender_paper_lego', 
                        help='experiment name')
    # 指定输出目录
    parser.add_argument("--basedir", type=str, default='./logs', 
                        help='where to store ckpts and logs')
    # 指定输入数据的目录                    
    parser.add_argument("--datadir", type=str, default='./data/nerf_synthetic/lego',
                        help='input data directory')

    # training options 训练选项
    # 设置（粗）网络深度/层数
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    # 设置I（粗）网络宽度，即每层神经元个数
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    # 设置细网络深度
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    # 设置细网络宽度
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    # 设置每个梯度步长的随机射线的数量
    parser.add_argument("--N_rand", type=int, default=1024, 
                        help='batch size (number of random rays per gradient step)')
    # 设置学习率
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    # 设置指数学习率衰减（in 1000 steps）
    parser.add_argument("--lrate_decay", type=int, default=500, 
                        help='exponential learning rate decay (in 1000 steps)')
    # 设置并行处理的光线数量，如果溢出则减少
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    # 设置并行发送的点数，如果溢出则减少
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    # 设置是否（默认：不）一次只能从一张图片中获取随机光线
    parser.add_argument("--no_batching", type=bool, default=False, 
                        help='only take random rays from 1 image at a time')
    # 设置是否（默认：不）要从保存的模型中加载权重
    parser.add_argument("--no_reload", type=bool, default=True, 
                        help='do not reload weights from saved ckpt')
    # 设置是否为粗网络重新加载特定权重
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options 渲染选项
    # 每条射线的粗样本数
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    # 设置每条射线附加的细样本数
    parser.add_argument("--N_importance", type=int, default=128,
                        help='number of additional fine samples per ray')
    # 设置是否考虑抖动，0表示无抖动，1表示有抖动
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    # 设置使用5D输入
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    # 设置是否使用位置编码。0表示使用，-1不使用
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    # 设置多分辨率 log2的位置编码的最大频率（3D位置）
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    # 设置多分辨率 log2的位置编码的最大频率（2D方向）
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    # 设置噪音方差
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    # 不要优化、重新加载权重和渲染render_poses路径
    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    # 渲染测试集而不是render_poses路径
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    # 下采样因子以加快渲染速度，设置为 4 或 8 用于快速预览
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options 训练选项
    # 设置中心裁剪区的训练步数
    parser.add_argument("--precrop_iters", type=int, default=500,
                        help='number of steps to train on central crops')
    # 设置图片用于中心裁剪区的那部分的大小
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options 数据集选项
    # 设置数据集类型：llff / blender / deepvoxels
    parser.add_argument("--dataset_type", type=str, default='blender', 
                        help='options: llff / blender / deepvoxels')
    # 设置从测试/验证集中加载 1/N 图像，这对于像 deepvoxels 这样的大型数据集很有用
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags 专门用于deepvoxels的设置
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags 专门用于blender的设置
    parser.add_argument("--white_bkgd", type=bool, default=True, 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", type=bool, default=True,
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags 专门用于llff的设置
    # llff下采样因子
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    # 设置不使用标准化的设备坐标（为非正面场景设置）
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    # 设置在视差而不是深度中线性采样
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    # 设置为针对球形360的场景
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    # 设置将每1/N张图像作为LLFF测试集，论文使用8张
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options 日志/保存选项
    # 控制台打印输出和度量日志记录的频率 默认每隔 100 epoch
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric logging')
    # tensorboard记录图像的频率
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    # ckpt权重文件的保存频率
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    # 测试集保存频率
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    # render_pose视频保存的频率
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    return parser


# 训练函数
def train():

    parser = config_parser()
    args = parser.parse_args()

    ##########################################################################################
    # 1.加载数据
    # Load data 
    K = None
    if args.dataset_type == 'llff':
        # shape: images[20,378,504,3] poses[20,3,5] render_poses[120,3,5]
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        # hwf=[378,504,focal] poses每个batch的每一行最后一个元素拿出来
        hwf = poses[0,:3,-1]
        # shape: poses [20,3,4] hwf给出去之后把每一行的第5个元素删掉
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        # 确定景深范围
        near = 2.
        far = 6.

        # 将 RGBA 转换成 RGB 图像
        if args.white_bkgd:
            # 如果使用白色背景
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types 将内部转换为正确的类型
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # 获取相机内参K [3, 3]
    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    # 把所有args参数写入到logs/fern_test/args.txt
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    # 把args.config参数写入到logs/fern_test/config.txt
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    ##########################################################################################
    # 2.创建 NeRF 网络
    # Create nerf model 初始化 NeRF 网络模型
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    # 更新字典，加入三维空间的边界框 bounding box
    # 本来都是dict类型，都有9个元素，加了bds(near+far)之后就是11个元素了
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model 如果仅从训练模型中渲染就短路
    # 在测试渲染时只需要将 render_only 参数置 True，就不再对网络进行训练，直接得到渲染结果
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    ################################################################################################
    # 3. 如果以批处理的形式对进行训练
    # 3.1 首先生成所有图片的光线
    
    # Prepare raybatch tensor if batching random rays 
    # 开始读取光线以及光线对应的像素值,默认N_rand=1024
    N_rand = args.N_rand

    # use_batching 参数决定了是否从多个角度进行光线投射。
    # 源代码中对 lego 小车重建时参数为 False
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        # 获取光束，rays shape:[138,2,400,400,3] poses [138, 4, 4]
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3] [138, 2, 400, 400, 3]
        print('done, concats')
        # 沿axis=1拼接,rays_rgb shape:[138,3,400,400,3] images shape:[138,400,400,3]
        # 这里把光线的原点、方向、以及光线对应的像素颜色结合到一起，便于后面的 shuffle 操作
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3] [138, 3, 400, 400, 3]
        # 改变shape,rays_rgb shape:[138,400,400,3,3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3] [138, 400, 400, 3, 3]
        # rays_rgb shape:[训练样本数目=100,400,400,3,3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only [100, 400, 400, 3, 3]
        # 得到了(训练样本数目)*H*W个光束,rays_rgb shape:[(N_train)*H*W=16000000,3,3]
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3] [16000000, 3, 3]
        # 生成了所有图片的像素点对应的光线原点和方向，
        # 并将光线对应的像素颜色与光线拼接到了一起构成 rays_rgb
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        # 打乱光束的顺序 np.random.shuffle()在多维矩阵中，只对第一维（行）做打乱顺序操作
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    N_iters = 200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    ##########################################################################################
    # 4.开始训练
    # 默认训练200000次
    '''
    注意到不论是 use_batching 的情况还是单张图像的情况，
    每个 epoch 选择的光线的数量是恒定的，即 N_rand 。
    这么做实际上是为了减少计算的工程量。
    虽然每次都只随机挑选了一部分像素对应的光线，
    但是经过多达 200 000 次的训练实际上已经足以把所有的像素对应的光线都挑选一遍了。
    '''
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # 4.1 如果以批处理的形式进行训练
        # Sample random ray batch
        if use_batching:
            # Random over all images
            # 取一个batch, 
            # 分批加载光线，大小为 N_rand = 1024
            # batch shape [1024,3,3]
            batch = rays_rgb[i_batch:i_batch+N_rand] # [1024, 3, 3]
            # 转换0维和1维的位置， batch shape [ro+rd+rgb=3,1024,3]
            batch = torch.transpose(batch, 0, 1) # [3, 1024, 3]
            # shape: batch_rays shape[ro+rd=2,1024,3] target_s[1024,3]对应的是rgb
            # 将光线batch_rays和对应的像素点颜色target_s 分离
            batch_rays, target_s = batch[:2], batch[2] # [3, 1024, ro+rd], [3, 1024, rgb]
            
            i_batch += N_rand
            # 经过一定批次的处理后，所有的图片都经过了一次。这时候要对数据打乱，重新再挑选。
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                # torch.randperm(n)：将0~n-1（包括0和n-1）随机打乱后获得的数字序列
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0
        # 4.2 如果不以批处理的形式进行训练，即对单张图片进行训练
        else:
            # Random from one image 从所有的图像中随机选择一张图像用于训练
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                # 生成这张图像中每个像素点对应的光线的原点和方向
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
                # 生成每个像素点的笛卡尔坐标，前 precrop_iters 生成图像中心的像素坐标坐标
                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    # 生成图像中每个像素的坐标
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                # 注意，在训练的时候并不是给图像中每个像素都打光线，而是加载一批光线，批大小为 N_rand
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                # 选择像素坐标
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                # 选择对应的光线
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        '''
        我们得到生成的光线以后就可以对光线进行渲染操作了。
        chunk=4096, batch_rays[2,4096,3]
        返回渲染出的一个batch的rgb，disp(视差图)，acc(不透明度)和extras(其他信息)
        rgb shape [4096, 3]刚好可以和target_s 对应上
        disp shape 4096，对应4096个光束
        acc shape 4096， 对应4096个光束
        extras 是一个dict，含有5个元素 shape:[4096,64,4]
        '''
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        optimizer.zero_grad()
        # 计算MSE损失，求RGB的MSE img_loss shape:[20,378,504,3]
        img_loss = img2mse(rgb, target_s)
        # trans shape:[4096,64]
        trans = extras['raw'][...,-1]
        loss = img_loss
        # 将损失转换为 PSNR 指标，计算 PSNR shape:[1]
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        # 反向传播
        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate 动态更新学习率   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        # 保存ckpt
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        # 输出mp4视频
        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        # 保存测试数据集
        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        # 默认每隔 100 epoch 打印一次
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
