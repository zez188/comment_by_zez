import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        # 获取字典kwargs
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        # 如果包含原始位置
        if self.kwargs['include_input']:
            # 把一个不对数据做出改变的匿名函数添加到列表中
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            # 得到 [2^0, 2^1, ... ,2^(L-1)] 参考论文 5.1 中的公式
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            # 得到 [2^0, 2^(L-1)] 的等差数列，列表中有 L 个元素
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                # sin(x * 2^n)  参考位置编码公式
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                # 每使用子编码公式一次就要把输出维度加 3，因为每个待编码的位置维度是 3
                out_dim += d
        # 相当于是一个编码公式列表
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        # 对各个输入进行编码，给定一个输入，使用编码列表中的公式分别对他编码
        # 输出(3x2xN)x1的张量矩阵
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


# 获取位置编码信息
def get_embedder(multires, i=0):
    if i == -1:
        # nn.Identity()表示输入是啥，输出就是啥，不做任何的改变
        # 这个网络层的设计是用于占位的，即不干活，只是有这么一个层，
        # 放到残差网络里就是在跳过连接的地方用这个层
        return nn.Identity(), 3
    '''
    1.include_input     如果为真，最终的编码结果包含原始坐标
    2.input_dims        输入给编码器的数据的维度
    3.max_freq_log2     multires-1
    4.num_freqs         multires,即论文中 5.1 节位置编码公式中的 L 
    5.log_sampling      if True:采用log2形式进行采样, else:采用等差数列形式进行采样
    6.periodic_fns      [torch.sin, torch.cos]
    '''
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    # embed 现在相当于一个编码器，具体的编码公式与论文中的一致。
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D  # D = 8
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        '''
        # 第0层: nn.Linear(input_ch, W)
        # 第1-4层(i=0-3): nn.Linear(W, W)
        # 第5层(i=4): nn.Linear(W + input_ch, W)
        # 第6-7层(i=5-6): nn.Linear(W, W)
        '''
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            # 在第4层时将输入input_pts和h拼接，使第5层的nn.Linear(W + input_ch, W)能运行
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, K, c2w):
    '''
    torch.linspace(start, end, steps=100, out=None) → Tensor 返回一个1维张量，包含在区间start和end上均匀间隔的step个点。
    torch.meshgrid()的功能是生成网格，可以用于生成坐标。
    函数输入两个数据类型相同的一维张量，
    两个输出张量的行数为第一个输入张量的元素个数，列数为第二个输入张量的元素个数，当两个输入张量数据类型不同或维度不是一维时会报错。
    其中第一个输出张量填充第一个输入张量中的元素，各行元素相同；第二个输出张量填充第二个输入张量中的元素,各列元素相同。
    '''
    # torch.meshgrid(a, b) 返回的是 a.shape() 行 ，b.shape() 列的二维数组。
    # 需要一个转置操作 i.t()，得到一张图片的每个像素点的笛卡尔坐标
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()   # 转置
    j = j.t()
    # 利用相机内参 K 计算每个像素坐标相对于光心的单位方向
    # OpenCV/Colmap的相机坐标系里相机的Up/Y朝下, 相机光心朝向+Z轴，而NeRF/OpenGL相机坐标系里相机的Up/朝上，相机光心朝向-Z轴，
    # 所以这里代码在方向向量dir的第二和第三项乘了个负号。
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # dirs保存了每个方向下的像素点到光心的单位方向(Z轴归一化), shape [H, W, 3]
    # 我们有了这个单位方向就可以通过调整 Z 轴坐标生成空间中每一个点坐标，以此模拟一条光线。
    # 利用相机外参转置矩阵将光线方向相机坐标转换为世界坐标
    # 因为是方向向量，所以转换为世界坐标时不需要平移部分，只需要旋转矩阵c2w[:3,:3]
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    # 转换为世界坐标时取平移部分，将相机帧的原点平移到世界帧。它是所有射线的起源。
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    
    # 返回所有像素坐标相对于光心的射线的原点组成的矩阵和方向向量组成的矩阵
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w): # get_rays_np(H, W, K, p)
    # np.meshgrid(a, b，indexing = "xy") 函数会返回 b.shape() 行 ，a.shape() 列的二维数组。
    # 因此 i, j 都是 [H, W] 的二维数组。i 的每一行都一样，j 的每一列都一样
    # 得到一张图片的每个像素点的笛卡尔坐标
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy') # i:[400,400], j:[400,400]
    # 利用相机内参 K 计算每个像素坐标相对于光心的单位方向
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1) # dirs:[400, 400, 3]
    # Rotate ray directions from camera frame to the world frame 把光线方向从相机坐标系转移到世界坐标系
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs] [400, 400, 3]
    # Translate camera frame's origin to the world frame. It is the origin of all rays. 
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))  # [400, 400, 3]
    # 生成每个方向下的像素点到光心的单位方向（Z 轴为 1）。
    # 我们有了这个单位方向就可以通过调整 Z 轴坐标生成空间中每一个点坐标，借此模拟一条光线。
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans [1024, 62]
    pdf = weights / torch.sum(weights, -1, keepdim=True) # [1024, 62]
    cdf = torch.cumsum(pdf, -1)  # [1024, 62]
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins)) [1024, 63]

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples]) # [1024, 128]

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF  contiguous来返回一个深拷贝
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True) # [1024, 128]
    below = torch.max(torch.zeros_like(inds-1), inds-1) # [1024, 128]
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds) # [1028, 128]
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2) [1024, 128, 2]

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]] # matched_shape=[1024, 128, 63], shape=[3]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g) # [1024, 128, 2]
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g) # [1024, 128, 2]

    denom = (cdf_g[...,1]-cdf_g[...,0]) # [1024, 128]
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom) # [1024, 128]
    t = (u-cdf_g[...,0])/denom # [1024, 128]
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0]) # [1024, 128]

    return samples
