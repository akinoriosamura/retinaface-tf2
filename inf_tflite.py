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

from modules.utils import (load_yaml, draw_bbox_landm, recover_pad_output, preprocess_input, postprocess)


def inference():
    target_dir = 'samples'
    model_file = os.path.join("Models", "converted_kerasmodel.tflite")
    cfg_path = './configs/retinaface_res50.yaml'
    iou_th = 0.4
    score_th = 0.5
    cfg = load_yaml(cfg_path)

    # インタプリタの生成
    interpreter = Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    # 入力情報と出力情報の取得
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 幅と高さの取得
    width  = input_details[0]['shape'][1]
    height = input_details[0]['shape'][2]

    print("[*] Processing on images {}".format(target_dir))
    img_paths = glob.glob(target_dir + '/*')
    for img_path in img_paths:
        target = cv2.imread(img_path)
        # 入力画像前処理
        img_height_raw, img_width_raw, img, pad_params = preprocess_input(target, cfg)

        # 入力データの生成
        input_data = np.expand_dims(img, axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # 推論の実行
        st = time.time()
        interpreter.invoke()
        print("el: ", time.time() - st)

        # output_data: [bb, land, cls]
        output_data = [
            interpreter.get_tensor(output_details[1]['index']),
            interpreter.get_tensor(output_details[2]['index']),
            interpreter.get_tensor(output_details[0]['index'])
        ]

        outputs = postprocess(output_data, cfg, iou_th, score_th)
        print("outputs: ", outputs.shape)

        # recover padding effect
        outputs = recover_pad_output(outputs, pad_params)

        # draw and save results
        save_img_path = os.path.join('results', 'out_' + os.path.basename(img_path))
        # import pdb;pdb.set_trace()
        for prior_index in range(len(outputs)):
            draw_bbox_landm(target, outputs[prior_index], img_height_raw,
                            img_width_raw)
            cv2.imwrite(save_img_path, target)
        print(f"[*] save result at {save_img_path}")


if __name__ == '__main__':
    inference()