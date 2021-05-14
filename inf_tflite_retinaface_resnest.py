from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import numpy as np
import cv2
import os
import time

from tensorflow.lite.python.interpreter import Interpreter

from modules.utils import (set_memory_growth, load_yaml, get_bbox_landm, draw_bbox_landm,
                           pad_input_image, recover_pad_output)
from modules.utils import preprocess_input, prior_box, decode, non_max_suppression, postprocess

# メインの実行
def inference():
    target_dir = 'samples'
    det_model_file = os.path.join("Models", "converted_savedmodel.tflite")
    land_model_file = os.path.join("Models", "./wflw_moruhard_grow_68_Res50.tflite")
    cfg_path = './configs/retinaface_res50.yaml'
    iou_th = 0.4
    score_th = 0.5
    cfg = load_yaml(cfg_path)

    ####### retinaface interpreter
    # インタプリタの生成
    det_interpreter = Interpreter(model_path=det_model_file)
    det_interpreter.allocate_tensors()
    # 入力情報と出力情報の取得
    det_input_details = det_interpreter.get_input_details()
    det_output_details = det_interpreter.get_output_details()
    # 幅と高さの取得
    # import pdb; pdb.set_trace()
    det_width  = det_input_details[0]['shape'][1]
    det_height = det_input_details[0]['shape'][2]

    ######### resnest interpreter
    # インタプリタの生成
    land_interpreter = Interpreter(model_path=land_model_file)
    land_interpreter.allocate_tensors()
    # 入力情報と出力情報の取得
    land_input_details = land_interpreter.get_input_details()
    land_output_details = land_interpreter.get_output_details()
    # 幅と高さの取得
    image_size  = land_input_details[0]['shape'][1]

    ######### process img
    print("[*] Processing on images {}".format(target_dir))
    img_paths = glob.glob(target_dir + '/*')
    for img_path in img_paths:
        target = cv2.imread(img_path)
        # 入力画像前処理
        img_height_raw, img_width_raw, img, pad_params = preprocess_input(target, cfg)

        # 入力データの生成
        det_input_data = np.expand_dims(img, axis=0)
        det_interpreter.set_tensor(det_input_details[0]['index'], det_input_data)

        # 推論の実行
        st = time.time()
        det_interpreter.invoke()
        print("bb el: ", time.time() - st)

        det_output_data = [
            det_interpreter.get_tensor(det_output_details[1]['index']),
            det_interpreter.get_tensor(det_output_details[2]['index']),
            det_interpreter.get_tensor(det_output_details[0]['index'])
        ]

        outputs = postprocess(det_output_data, cfg, iou_th, score_th)
        # print("outputs: ", outputs.shape)

        # recover padding effect
        outputs = recover_pad_output(outputs, pad_params)

        for prior_index in range(len(outputs)):
            box, _ = get_bbox_landm(
                outputs[prior_index], img_height_raw, img_width_raw)
            x1, y1, x2, y2 = (np.array(box[:4])+0.5).astype(np.int32)
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            size = int(max([w, h])*1.1)
            cx = x1 + w//2
            cy = y1 + h//2
            x1 = cx - size//2
            x2 = x1 + size
            y1 = cy - size//2
            y2 = y1 + size

            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - img_width_raw)
            edy = max(0, y2 - img_height_raw)
            x2 = min(img_width_raw, x2)
            y2 = min(img_height_raw, y2)
            cropped = target[y1:y2, x1:x2]

            if dx > 0 or dy > 0 or edx > 0 or edy > 0:
                cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
            # cv2.imwrite('croped.jpg', cropped)
            # 輪郭点モデルの入力データの生成
            input = cv2.resize(cropped, (image_size, image_size))
            input_data = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
            input_data = input_data.astype(np.float32) / 256
            input_data = np.expand_dims(input_data, 0)
            print(input_data.shape)
            land_interpreter.set_tensor(land_input_details[0]['index'], input_data)

            # 輪郭点推論の実行
            st = time.time()
            land_interpreter.invoke()
            print("land el: ", time.time() - st)

            output_data = land_interpreter.get_tensor(land_output_details[-1]['index'])
            pre_landmark = output_data.reshape(-1, 2) * [image_size, image_size]

            # visualization
            cv2.rectangle(target, (x1, y1), (x2, y2), (0, 255, 0))
            pre_landmark = pre_landmark * [size/image_size, size/image_size] - [dx, dy]
            for (x, y) in pre_landmark.astype(np.int32):
                cv2.circle(target, (x1 + x, y1 + y), 2, (0, 0, 255), 2)
        save_img_path = os.path.join('results', 'out_' + os.path.basename(img_path))
        cv2.imwrite(save_img_path, target)
        print(f"[*] save result at {save_img_path}")


if __name__ == '__main__':
    inference()