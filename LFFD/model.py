import cv2
import mxnet as mx
from LFFD import predict

use_gpu = True
version = 'v2'

if use_gpu:
    ctx = mx.gpu(0)
else:
    ctx = mx.cpu()

if version == 'v1':
    from LFFD.config_farm import configuration_10_560_25L_8scales_v1 as cfg

    symbol_file_path = \
        './LFFD/symbol_farm/symbol_10_560_25L_8scales_v1_deploy.json'
    model_file_path = \
        './LFFD/saved_model/configuration_10_560_25L_8scales_v1/train_10_560_25L_8scales_v1_iter_1400000.params'
elif version == 'v2':
    from LFFD.config_farm import configuration_10_320_20L_5scales_v2 as cfg

    symbol_file_path = \
        './LFFD/symbol_farm/symbol_10_320_20L_5scales_v2_deploy.json'
    model_file_path = \
        './LFFD/saved_model/configuration_10_320_20L_5scales_v2/train_10_320_20L_5scales_v2_iter_1000000.params'
else:
    raise TypeError('Unsupported LFFD Version.')

face_predictor = predict.Predict(mxnet=mx,
                                 symbol_file_path=symbol_file_path,
                                 model_file_path=model_file_path,
                                 ctx=ctx,
                                 receptive_field_list=cfg.param_receptive_field_list,
                                 receptive_field_stride=cfg.param_receptive_field_stride,
                                 bbox_small_list=cfg.param_bbox_small_list,
                                 bbox_large_list=cfg.param_bbox_large_list,
                                 receptive_field_center_start=cfg.param_receptive_field_center_start,
                                 num_output_scales=cfg.param_num_output_scales)


def get_faces_lffd(img_lffd):
    faces_lffd = face_predictor.predict(img_lffd, resize_scale=1, score_threshold=0.6,
                                        top_k=10000, NMS_threshold=0.4, NMS_flag=True, skip_scale_branch_list=[])
    return faces_lffd


def draw_box_lffd(img_lffd):
    faces, _ = get_faces_lffd(img_lffd)
    if not faces or len(faces) < 1:
        print("No faces detection")
    else:
        for face in faces:
            cv2.rectangle(img_lffd, (face[0], face[1]), (face[2], face[3]), (0, 0, 255), 2)
