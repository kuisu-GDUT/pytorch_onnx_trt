from build_utils.layers import *
from build_utils.parse_config import *

ONNX_EXPORT = False


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class MaxPoolDark(nn.Module):
    def __init__(self, size=2, stride=1):
        super(MaxPoolDark, self).__init__()
        self.size = size
        self.stride = stride

    def forward(self, x):
        '''
        darknet output_size = (input_size + p - k) / s +1
        p : padding = k - 1
        k : size
        s : stride
        torch output_size = (input_size + 2*p -k) / s +1
        p : padding = k//2
        '''
        p = self.size // 2
        if ((x.shape[2] - 1) // self.stride) != ((x.shape[2] + 2 * p - self.size) // self.stride):
            padding1 = (self.size - 1) // 2
            padding2 = padding1 + 1
        else:
            padding1 = (self.size - 1) // 2
            padding2 = padding1
        if ((x.shape[3] - 1) // self.stride) != ((x.shape[3] + 2 * p - self.size) // self.stride):
            padding3 = (self.size - 1) // 2
            padding4 = padding3 + 1
        else:
            padding3 = (self.size - 1) // 2
            padding4 = padding3
        x = F.max_pool2d(F.pad(x, (padding3, padding4, padding1, padding2), mode='replicate'),
                         self.size, stride=self.stride)
        return x

class Upsample_expand(nn.Module):
    def __init__(self, stride=2):
        super(Upsample_expand, self).__init__()
        self.stride = stride

    def forward(self, x):
        assert (x.data.dim() == 4)

        x = x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1). \
            expand(x.size(0), x.size(1), x.size(2), self.stride, x.size(3), self.stride).contiguous(). \
            view(x.size(0), x.size(1), x.size(2) * self.stride, x.size(3) * self.stride)

        return x

class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert (x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert (H % stride == 0)
        assert (W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H / hs, hs, W / ws, ws).transpose(3, 4).contiguous()
        x = x.view(B, C, H / hs * W / ws, hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, H / hs, W / ws).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, H / hs, W / ws)
        return x
class Upsample_interpolate(nn.Module):
    def __init__(self, stride):
        super(Upsample_interpolate, self).__init__()
        self.stride = stride

    def forward(self, x):
        assert (x.data.dim() == 4)

        out = F.interpolate(x, size=(x.size(2) * self.stride, x.size(3) * self.stride), mode='nearest')
        return out

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x


# for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x

def create_modules(modules_defs: list, img_size):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    :param modules_defs: 通过.cfg文件解析得到的每个层结构的列表
    :param img_size:
    :return:
    """

    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size
    # 删除解析cfg列表中的第一个配置(对应[net]的配置)
    modules_defs.pop(0)  # cfg training hyperparams (unused)
    output_filters = [3]  # input channels
    module_list = nn.ModuleList()
    # 统计哪些特征层的输出会被后续的层使用到(可能是特征融合，也可能是拼接)
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1

    # 遍历搭建每个层结构
    for i, mdef in enumerate(modules_defs):
        modules = nn.Sequential()

        if mdef["type"] == "convolutional":
            bn = mdef["batch_normalize"]  # 1 or 0 / use or not
            filters = mdef["filters"]
            k = mdef["size"]  # kernel size
            stride = mdef["stride"] if "stride" in mdef else (mdef['stride_y'], mdef["stride_x"])
            if isinstance(k, int):
                modules.add_module("Conv2d", nn.Conv2d(in_channels=output_filters[-1],
                                                       out_channels=filters,
                                                       kernel_size=k,
                                                       stride=stride,
                                                       padding=k // 2 if mdef["pad"] else 0,
                                                       bias=not bn))
            else:
                raise TypeError("conv2d filter size must be int type.")

            if bn:
                modules.add_module("BatchNorm2d", nn.BatchNorm2d(filters))
            else:
                # 如果该卷积操作没有bn层，意味着该层为yolo的predictor
                routs.append(i)  # detection output (goes into yolo layer)

            if mdef["activation"] == "leaky":
                modules.add_module("activation", nn.LeakyReLU(0.1, inplace=True))
            else:
                pass

        elif mdef["type"] == "BatchNorm2d":
            modules.add_module(f"batch_norm_{i}",
                               nn.BatchNorm2d(filters, momentum=0.1, eps=1e-5))

        elif mdef["type"] == "maxpool":
            #reference: https://github.com/eriklindernoren/PyTorch-YOLOv3/tree/e54d52e6500b7bbd687769f7abfd8581e9f59fd1
            # k = mdef["size"]  # kernel size
            # stride = mdef["stride"]
            # modules = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)
            kernel_size = int(mdef["size"])
            stride = int(mdef["stride"])
            # if kernel_size == 2 and stride == 1:
            #     modules.add_module(f"_debug_padding_{i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
                                   padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{i}", maxpool)

        elif mdef["type"] == "upsample":
            if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
                g = (yolo_index + 1) * 2 / 32  # gain
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))
            else:
                modules = nn.Upsample(scale_factor=mdef["stride"])

        elif mdef["type"] == "route":  # [-2],  [-1,-3,-5,-6], [-1, 61]
            layers = mdef["layers"]
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat(layers=layers)

        elif mdef["type"] == "shortcut":
            layers = mdef["from"]
            filters = output_filters[-1]
            # routs.extend([i + l if l < 0 else l for l in layers])
            routs.append(i + layers[0])
            modules = WeightedFeatureFusion(layers=layers, weight="weights_type" in mdef)

        elif mdef["type"] == "yolo":
            yolo_index += 1  # 记录是第几个yolo_layer [0, 1, 2]
            stride = [32, 16, 8]  # 预测特征层对应原图的缩放比例
            #todo YOLO-v3的头构建
            modules = YOLOLayer(anchors=mdef["anchors"][mdef["mask"]],  # anchor list; mask:#对应的anchors的索引
                                nc=mdef["classes"],  # number of classes
                                img_size=img_size,
                                stride=stride[yolo_index])

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try:
                j = -1
                # bias: shape(255,) 索引0对应Sequential中的Conv2d
                # view: shape(3, 85)
                b = module_list[j][0].bias.view(modules.na, -1)
                b.data[:, 4] += -4.5  # obj
                b.data[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            except Exception as e:
                print('WARNING: smart bias initialization failure.', e)
        else:
            print("Warning: Unrecognized Layer Type: " + mdef["type"])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    routs_binary = [False] * len(modules_defs)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary


class YOLOLayer(nn.Module):
    """
    对YOLO的输出进行处理
    """
    def __init__(self, anchors, nc, img_size, stride):#anchor, 类别数, 图像大小, 缩小比率
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.stride = stride  # layer stride 特征图上一步对应原图上的步距 [32, 16, 8]
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85: x, y, w, h, obj, cls1, ...)
        self.nx, self.ny, self.ng = 0, 0, (0, 0)  # initialize number of x, y and gridpoints
        # 将anchors大小缩放到grid尺度
        self.anchor_vec = self.anchors / self.stride
        # batch_size, na, grid_h, grid_w, wh,
        # 值为1的维度对应的值不是固定值，后续操作可根据broadcast广播机制自动扩充
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        self.grid = None

        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

    def create_grids(self, ng=(13, 13), device="cpu"):
        """
        更新grids信息并生成新的grids参数
        :param ng: 特征图大小
        :param device:
        :return:
        """
        self.nx, self.ny = ng
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets 构建每个cell处的anchor的xy偏移量(在feature map上的)
        if not self.training:  # 训练模式不需要回归到最终预测boxes
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device),
                                     torch.arange(self.nx, device=device)])
            # batch_size, na, grid_h, grid_w, wh
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p):
        if ONNX_EXPORT:
            bs = p.shape[0]  # batch size
        else:
            bs, _, ny, nx = p.shape  # batch_size, predict_param(255), grid(13), grid(13)
            if (self.nx, self.ny) != (nx, ny) or self.grid is None:  # fix no grid bug
                self.create_grids((nx, ny), p.device)

        # view: (batch_size, 255, 13, 13) -> (batch_size, 3, 85, 13, 13)
        # permute: (batch_size, 3, 85, 13, 13) -> (batch_size, 3, 13, 13, 85)
        # [bs, anchor, grid, grid, xywh + obj + classes]
        # p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p
        elif ONNX_EXPORT:
            # Avoid broadcasting for ANE operations
            # m = self.na * self.nx * self.ny  #numbers of anchor 3*
            # ng = 1. / self.ng.repeat(m, 1)
            # grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            # anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng
            #
            # p = p.view(bs,m, self.no)
            # # xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
            # # wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            # # p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
            # #     torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # conf
            # p[:,:, :2] = (torch.sigmoid(p[:,:, 0:2]) + grid) * ng  # x, y
            # p[:,:, 2:4] = torch.exp(p[:,:, 2:4]) * anchor_wh  # width, height
            # p[:,:, 4:] = torch.sigmoid(p[:,:, 4:])
            # p[:,:, 5:] = p[:,:, 5:self.no] * p[:,:, 4:5]
            return p
        else:  # inference
            # [bs, anchor, grid, grid, xywh + obj + classes]
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy 计算在feature map上的xy坐标
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method 计算在feature map上的wh
            io[..., :4] *= self.stride  # 换算映射回原图尺度
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]


class Darknet(nn.Module):
    """
    YOLOv3 spp object detection model
    """
    def __init__(self, cfg, img_size=(608, 608), verbose=False):
        super(Darknet, self).__init__()
        # 这里传入的img_size只在导出ONNX模型时起作用
        self.input_size = [img_size] * 2 if isinstance(img_size, int) else img_size
        # 解析网络对应的.cfg文件
        self.module_defs = parse_model_cfg(cfg)
        # 根据解析的网络结构一层一层去搭建
        self.module_list, self.routs = create_modules(self.module_defs, img_size)
        # 获取所有YOLOLayer层的索引
        self.yolo_layers = get_yolo_layers(self)

        # 打印下模型的信息，如果verbose为True则打印详细信息
        self.info(verbose) if not ONNX_EXPORT else None  # print model description[layers, parameters, gradients]

    def forward(self, x, verbose=False):
        return self.forward_once(x, verbose=verbose)

    def forward_once(self, x, verbose=False):
        # yolo_out收集每个yolo_layer层的输出
        # out收集每个模块的输出
        yolo_out, out = [], []
        if verbose:
            print('0', x.shape)
            str = ""

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in ["WeightedFeatureFusion", "FeatureConcat"]:  # sum, concat
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
            elif name == "YOLOLayer":
                yolo_out.append(module(x))
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                x = module(x)

            out.append(x if self.routs[i] else [])#保证下采样的层输出, 用于特征融合或者拼接
            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
                str = ''

        if self.training:  # train
            return yolo_out
        elif ONNX_EXPORT:  # export
            # x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            # return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
            # p = torch.cat(yolo_out, dim=1)

            # # 根据objectness虑除低概率目标
            # mask = torch.nonzero(torch.gt(p[:, 4], 0.1), as_tuple=False).squeeze(1)
            # # onnx不支持超过一维的索引（pytorch太灵活了）
            # # p = p[mask]
            # p = torch.index_select(p, dim=0, index=mask)
            #
            # # 虑除小面积目标，w > 2 and h > 2 pixel
            # # ONNX暂不支持bitwise_and和all操作
            # mask_s = torch.gt(p[:, 2], 2./self.input_size[0]) & torch.gt(p[:, 3], 2./self.input_size[1])
            # mask_s = torch.nonzero(mask_s, as_tuple=False).squeeze(1)
            # p = torch.index_select(p, dim=0, index=mask_s)  # width-height 虑除小目标
            #
            # if mask_s.numel() == 0:
            #     return torch.empty([0, 85])

            return yolo_out
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs

            return x, p

    def info(self, verbose=False):
        """
        打印模型的信息
        :param verbose:
        :return:
        """
        torch_utils.model_info(self, verbose)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            # First five are header values
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        # If the weights file has a cutoff, we can find out about it by looking at the filename
        # examples: darknet53.conv.74 -> cutoff is 74
        filename = os.path.basename(weights_path)
        if ".conv." in filename:
            try:
                cutoff = int(filename.split(".")[-1])  # use last part of filename
            except ValueError:
                pass

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(
                    weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w


def get_yolo_layers(self):
    """
    获取网络中三个"YOLOLayer"模块对应的索引
    :param self:
    :return:
    """
    return [i for i, m in enumerate(self.module_list) if m.__class__.__name__ == 'YOLOLayer']  # [89, 101, 113]



