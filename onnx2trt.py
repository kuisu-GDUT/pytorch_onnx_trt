#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：yolo_onnx2trt 
@File    ：onnx2trt.py
@Author  ：kuisu
@Email     ：kuisu_dgut@163.com
@Date    ：2022/1/14 10:56 
'''
import numpy as np
import tensorrt as trt
import os
import common
from PIL import ImageDraw

from data_processing import PreprocessYOLO,PostProcessYOLO,ALL_CATEGORIES

TRT_LOGGER = trt.Logger()
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def draw_bboxes(image_raw,bboxes,confidences,categories,all_categories,bbox_color='blue'):
    """
    Draw the bouding boxes on the original input image and return it
    :param image_raw:a raw PIL Image
    :param bboxes: NumPy array containing the bouding box coordinates of N objects, with shape(N,4)
    :param confidences: NumPy array containing the corresponding category for each object, with spape (N,)
    :param categories: NumPy array containing the corresponding confidence for each object, with shape (N,)
    :param all_categories: a list of all categories in the correct ordered (required for looking up the category name)
    :param bbox_color: an optional string specifying the color of the bouding boxes (defalut: "blue")
    :return:
    """
    draw = ImageDraw.Draw(image_raw)
    print(bboxes,confidences,categories)
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text((left, top - 12), '{0} {1:.2f}'.format(all_categories[category], score), fill=bbox_color)

    return image_raw


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
            config.max_workspace_size = 1 << 28 # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 3, 608, 608]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            #     pass
            return engine
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

if __name__ == '__main__':
    onnx_file_path="weights/onnx_weights/yolov3_spp_0118.onnx"
    engine_file_path="weights/onnx_weights/yolov3_spp_0118.trt"
    #Download a dog image and save it to the following file path
    input_image_path = "test.jpg"
    #Two-dimensionnal tuple with the target network's input resolution in HW ordered
    input_resolution_yolov3_HW = (608,608)
    import time

    #Create a pre-processor object by specifying the required input resolution for YOLOv3
    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
    #Load an image from the specified input path, and return it together with a pre-processed version
    image_raw, image = preprocessor.process(input_image_path)
    #Store the shape of the original input image in WH format, we will need it for later
    shape_orig_WH = image_raw.size

    # Output shapes expected by the post-processor
    # output_shapes = [(1, 13*3, 19, 19), (1, 13*3, 38, 38)]
    output_shapes = [(1, 13*3, 19, 19), (1, 13*3, 38, 38), (1, 13*3, 76, 76)]
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        print('Running inference on image {}...'.format("random.uninform(1,3,608,608)"))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = image
        time_start = time.time()
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
    trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
    #anchors-tiny: (10,14),  (23,27),  (37,58),  (81,82),  (135,169),  (344,319)
    #anchors-tiny: (10,13),  (16,30),  (33,23),  (30,61),  (62,45),  (59,119),  (116,90),  (156,198),  (373,326)
    postprocessor_args = {"yolo_masks":[(6,7,8),(3,4,5),(0,1,2)],               # A list of 3 three-dimensional tuples for the yolo masks
                          "yolo_anchors":[(10,13),  (16,30),  (33,23),  (30,61),  (62,45),  (59,119),  (116,90),  (156,198),  (373,326)],
                          "obj_threshold":0.4,                          # Threshold for object coverage, float value between 0 and 1
                          'nms_threshold':0.5,                          # Threshold for non-max suppression algorithm, float value between 0 and 1
                          "yolo_input_resolution":input_resolution_yolov3_HW}

    postprocessor = PostProcessYOLO(**postprocessor_args)

    #Run the post-processing algorithms on the TensorRT outputs and get the bouding box details of detected objects
    boxes, classes, scores = postprocessor.process(trt_outputs,(shape_orig_WH))
    #Draw the bouding boxes onto the original input image and save it as a PNG file
    obj_detected_img = draw_bboxes(image_raw,boxes,scores,classes,ALL_CATEGORIES)
    output_image_path = 'test_bboxes.png'
    obj_detected_img.save(output_image_path,'PNG')
    print("Saved image with bouding boxes of detected objects to {}.".format(output_image_path))