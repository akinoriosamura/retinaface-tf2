from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import glob
import os
import numpy as np
import tensorflow as tf
import time

from modules.models import RetinaFaceModel
from modules.utils import (set_memory_growth, load_yaml, draw_bbox_landm,
                           pad_input_image, recover_pad_output, preprocess_input, postprocess)


flags.DEFINE_string('cfg_path', './configs/retinaface_res50.yaml',
                    'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('img_path', '', 'path to input image')
flags.DEFINE_string('img_dir', '', 'path to input image dir')
flags.DEFINE_boolean('webcam', False, 'get image source from webcam or not')
flags.DEFINE_float('iou_th', 0.4, 'iou threshold for nms')
flags.DEFINE_float('score_th', 0.5, 'score threshold for nms')
flags.DEFINE_float('down_scale_factor', 0.1, 'down-scale factor for inputs')


def save_model(model):
    model.save(os.path.join("Models", "model.h5"))
    model.save(os.path.join("Models", "saved_model"))
    # import pdb;pdb.set_trace()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                        tf.lite.OpsSet.SELECT_TF_OPS]
    converter.experimental_new_converter = True
    tflite_model = converter.convert()
    open(os.path.join("Models", "converted_kerasmodel.tflite"), "wb").write(tflite_model)

    converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                        tf.lite.OpsSet.SELECT_TF_OPS]
    converter.experimental_new_converter = True
    tflite_model = converter.convert()
    open(os.path.join("Models", "converted_savedmodel.tflite"), "wb").write(tflite_model)


def main(_argv):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    # define network
    model = RetinaFaceModel(cfg, training=False, iou_th=FLAGS.iou_th,
                            score_th=FLAGS.score_th)

    # load checkpoint
    checkpoint_dir = './checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("[*] load ckpt from {}.".format(
            tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()

    if not FLAGS.webcam:
        if os.path.exists(FLAGS.img_dir):
            print("[*] Processing on images {}".format(FLAGS.img_dir))
            img_paths = glob.glob(FLAGS.img_dir + '/*')
            for img_path in img_paths:
                import pdb;pdb.set_trace()
                img_raw = cv2.imread(img_path)                    
                # img_height_raw, img_width_raw, _ = img_raw.shape
                # img = np.float32(img_raw.copy())

                # if FLAGS.down_scale_factor < 1.0:
                #     # import pdb;pdb.set_trace()
                #     img = cv2.resize(img, (0, 0), fx=FLAGS.down_scale_factor,
                #                     fy=FLAGS.down_scale_factor,
                #                     interpolation=cv2.INTER_LINEAR)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # # pad input image to avoid unmatched shape problem
                # img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))
                img_height_raw, img_width_raw, img, pad_params = preprocess_input(img_raw, cfg)

                # run model
                outputs = model(img[np.newaxis, ...])# .numpy()
                # import pdb;pdb.set_trace()
                outputs = postprocess(outputs, cfg, FLAGS.iou_th, FLAGS.score_th)

                # recover padding effect
                outputs = recover_pad_output(outputs, pad_params)
                # import pdb;pdb.set_trace()
                save_model(model)


                # draw and save results
                save_img_path = os.path.join('results', 'out_' + os.path.basename(img_path))
                # import pdb;pdb.set_trace()
                for prior_index in range(len(outputs)):
                    draw_bbox_landm(img_raw, outputs[prior_index], img_height_raw,
                                    img_width_raw)
                    cv2.imwrite(save_img_path, img_raw)
                print(f"[*] save result at {save_img_path}")
        elif os.path.exists(FLAGS.img_path):
            print("[*] Processing on single image {}".format(FLAGS.img_path))

            img_raw = cv2.imread(FLAGS.img_path)
            img_height_raw, img_width_raw, _ = img_raw.shape
            img = np.float32(img_raw.copy())

            if FLAGS.down_scale_factor < 1.0:
                # import pdb;pdb.set_trace()
                img = cv2.resize(img, (0, 0), fx=FLAGS.down_scale_factor,
                                fy=FLAGS.down_scale_factor,
                                interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # pad input image to avoid unmatched shape problem
            img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))

            # run model
            outputs = model(img[np.newaxis, ...]).numpy()

            # recover padding effect
            outputs = recover_pad_output(outputs, pad_params)

            # draw and save results
            save_img_path = os.path.join('results', 'out_' + os.path.basename(FLAGS.img_path))
            for prior_index in range(len(outputs)):
                draw_bbox_landm(img_raw, outputs[prior_index], img_height_raw,
                                img_width_raw)
                cv2.imwrite(save_img_path, img_raw)
            print(f"[*] save result at {save_img_path}")
        else:
            print(f"cannot find image path from {FLAGS.img_path}")
            exit()

    else:
        cam = cv2.VideoCapture(0)

        start_time = time.time()
        while True:
            _, frame = cam.read()
            if frame is None:
                print("no cam input")

            frame_height, frame_width, _ = frame.shape
            img = np.float32(frame.copy())
            if FLAGS.down_scale_factor < 1.0:
                img = cv2.resize(img, (0, 0), fx=FLAGS.down_scale_factor,
                                 fy=FLAGS.down_scale_factor,
                                 interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # pad input image to avoid unmatched shape problem
            img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))

            # run model
            outputs = model(img[np.newaxis, ...]).numpy()

            # recover padding effect
            outputs = recover_pad_output(outputs, pad_params)

            # draw results
            for prior_index in range(len(outputs)):
                draw_bbox_landm(frame, outputs[prior_index], frame_height,
                                frame_width)

            # calculate fps
            fps_str = "FPS: %.2f" % (1 / (time.time() - start_time))
            start_time = time.time()
            cv2.putText(frame, fps_str, (25, 25),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)

            # show frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                exit()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
