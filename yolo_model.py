#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：yolo_onnx2trt 
@File    ：yolo_model.py
@Author  ：kuisu
@Email     ：kuisu_dgut@163.com
@Date    ：2022/1/14 10:48 
'''
from models import *
import torch

# Initialize model
if __name__ == '__main__':
    cfg="cfg/yolov3-spp-dlh.cfg"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(cfg).to(device)
    weights = "weights/yolov3_spp_dlh101_DatasetV7-99.pt"
    if weights.endswith(".pt") or weights.endswith(".pth"):
        ckpt = torch.load(weights, map_location=device)

        # load model
        try:
            ckpt["model"] = {k: v for k, v in ckpt["model"].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt["model"], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e
    # saveModel = torch.load(weights, map_location=device)["model"]
    # model.load_state_dict(torch.load(weights, map_location=device)["model"])
    save_files = {'model':model.state_dict()}
    torch.save(save_files,"weights/yolov3_spp_0118.pt")