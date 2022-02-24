import os
import torch
import cv2
import torch.onnx
import onnx
import onnxruntime
import numpy as np
import models
from build_utils import img_utils

from data_processing import PostProcessYOLO,PreprocessYOLO

device = torch.device("cpu")
models.ONNX_EXPORT = True


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def main():
    img_size = 608  # 必须是32的整数倍 [416, 512, 608]
    cfg = "cfg/yolov3-spp-dlh.cfg"
    # weights = "weights/yolov3_tiny_dlh101_v3-99.pt"
    weights = "weights/yolov3_spp_0118.pt"
    assert os.path.exists(cfg), "cfg file does not exist..."
    assert os.path.exists(weights), "weights file does not exist..."

    input_size = (img_size, img_size)  # [h, w]

    # create model
    model = models.Darknet(cfg, input_size)
    # load model weights
    model.load_state_dict(torch.load(weights, map_location=device)["model"])
    model.to(device)
    model.eval()
    # input to the model
    # [batch, channel, height, width]
    # x = torch.rand(1, 3, *input_size, requires_grad=True)
    img_path = "test.jpg"
    img_o = cv2.imread(img_path)  # BGR
    shape_orig_WH = img_o.shape[:2]
    assert img_o is not None, "Image Not Found " + img_path

    # preprocessing img
    img = img_utils.letterbox(img_o, new_shape=input_size, auto=False, color=(0, 0, 0))[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img).astype(np.float32)

    img /= 255.0  # scale (0, 255) to (0, 1)
    img = np.expand_dims(img, axis=0) .repeat(1,axis=0) # add batch dimension

    x = torch.tensor(img)
    torch_out = model(x)

    # save_path = "weights/onnx_weights/yolov3_spp_0118.onnx"
    save_path = "weights/onnx_weights/yolov3_spp_0224.onnx"
    # export the model
    torch.onnx.export(model,                       # model being run
                      x,                           # model input (or a tuple for multiple inputs)
                      save_path,                   # where to save the model (can be a file or file-like object)
                      export_params=True,          # store the trained parameter weights inside the model file
                      opset_version=12,            # the ONNX version to export the model to
                      do_constant_folding=False,    # whether to execute constant folding for optimization
                      input_names=["images"],       # the model's input names
                      # output_names=["classes", "boxes"],     # the model's output names
                      output_names=["prediction1","prediction2"],
                      dynamic_axes={"images": {0: "batch_size"},  # variable length axes
                                    "prediction": {0: "batch_size"}})
                                    # "classes": {0: "batch_size"},
                                    # "confidence": {0: "batch_size"},
                                    # "boxes": {0: "batch_size"}})

    # check onnx model
    onnx_model = onnx.load(save_path)

    #optimizer from onnx
    import onnxoptimizer
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(save_path)

    # compute ONNX Runtime output prediction
    output_shapes = [(1, 13 * 3, 19, 19), (1, 13 * 3, 38, 38), (1, 13 * 3, 76, 76)]
    input_resolution_yolov3_HW=(608,608)
    ort_inputs = {"images": to_numpy(x)}
    onnx_outs = ort_session.run(None, ort_inputs)

    trt_outputs = [output.reshape(shape) for output, shape in zip(onnx_outs, output_shapes)]
    # anchors-tiny: (10,14),  (23,27),  (37,58),  (81,82),  (135,169),  (344,319)
    # anchors-tiny: (10,13),  (16,30),  (33,23),  (30,61),  (62,45),  (59,119),  (116,90),  (156,198),  (373,326)
    postprocessor_args = {"yolo_masks": [(3, 4, 5), (0, 1, 2)],
                          # A list of 3 three-dimensional tuples for the yolo masks
                          "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90),
                                           (156, 198), (373, 326)],
                          "obj_threshold": 0.6,  # Threshold for object coverage, float value between 0 and 1
                          'nms_threshold': 0.5,
                          # Threshold for non-max suppression algorithm, float value between 0 and 1
                          "yolo_input_resolution": input_resolution_yolov3_HW}

    postprocessor = PostProcessYOLO(**postprocessor_args)

    # Run the post-processing algorithms on the TensorRT outputs and get the bouding box details of detected objects
    boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_WH))

    # compare ONNX Runtime and Pytorch results
    # assert_allclose: Raises an AssertionError if two objects are not equal up to desired tolerance.
    # np.testing.assert_allclose(to_numpy(torch_out), onnx_outs[0], rtol=1e-03, atol=1e-05)
    # np.testing.assert_allclose(to_numpy(torch_out[1]), onnx_outs[1], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(to_numpy(torch_out[2]), onnx_outs[2], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == '__main__':
    main()
