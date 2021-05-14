import cv2
import yaml
import sys
import time
import numpy as np
import tensorflow as tf
from absl import logging

from modules.anchor import decode, prior_box
from modules.dataset import load_tfrecord_dataset
from modules.nms import non_max_suppression


def load_yaml(load_path):
    """load yaml file"""
    with open(load_path, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.Loader)

    return loaded


def set_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices(
                    'GPU')
                logging.info(
                    "Detect {} Physical GPUs, {} Logical GPUs.".format(
                        len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logging.info(e)


def load_dataset(cfg, priors, shuffle=True, buffer_size=10240):
    """load dataset"""
    logging.info("load dataset from {}".format(cfg['dataset_path']))
    dataset = load_tfrecord_dataset(
        tfrecord_name=cfg['dataset_path'],
        batch_size=cfg['batch_size'],
        img_dim=cfg['input_size'],
        using_bin=cfg['using_bin'],
        using_flip=cfg['using_flip'],
        using_distort=cfg['using_distort'],
        using_encoding=True,
        priors=priors,
        match_thresh=cfg['match_thresh'],
        ignore_thresh=cfg['ignore_thresh'],
        variances=cfg['variances'],
        shuffle=shuffle,
        buffer_size=buffer_size)
    return dataset


class ProgressBar(object):
    """A progress bar which can print the progress modified from
       https://github.com/hellock/cvbase/blob/master/cvbase/progress.py"""
    def __init__(self, task_num=0, completed=0, bar_width=25):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width
                          if bar_width <= max_bar_width else max_bar_width)
        self.completed = completed
        self.first_step = completed
        self.warm_up = False

    def _get_max_bar_width(self):
        if sys.version_info > (3, 3):
            from shutil import get_terminal_size
        else:
            from backports.shutil_get_terminal_size import get_terminal_size
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            logging.info('terminal width is too small ({}), please consider '
                         'widen the terminal for better progressbar '
                         'visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def reset(self):
        """reset"""
        self.completed = 0
        self.fps = 0

    def update(self, inf_str=''):
        """update"""
        self.completed += 1

        if not self.warm_up:
            self.start_time = time.time() - 1e-1
            self.warm_up = True

        if self.completed > self.task_num:
            self.completed = self.completed % self.task_num
            self.start_time = time.time() - 1 / self.fps
            self.first_step = self.completed - 1
            sys.stdout.write('\n')

        elapsed = time.time() - self.start_time
        self.fps = (self.completed - self.first_step) / elapsed
        percentage = self.completed / float(self.task_num)
        mark_width = int(self.bar_width * percentage)
        bar_chars = '>' * mark_width + ' ' * (self.bar_width - mark_width)
        stdout_str = '\rTraining [{}] {}/{}, {}  {:.1f} step/sec'
        sys.stdout.write(stdout_str.format(
            bar_chars, self.completed, self.task_num, inf_str, self.fps))

        sys.stdout.flush()


def preprocess_input(img_raw, cfg):
    # import pdb;pdb.set_trace()
    height, width, _ = img_raw.shape
    img = np.float32(img_raw.copy())

    image_size = img.size
    input_size = cfg['input_size']
    output_size = (int(input_size), int(input_size))
    down_scale_factor = output_size[0] / max(width, height)
    resized_img = cv2.resize(img, (0, 0), fx=down_scale_factor,
                        fy=down_scale_factor,
                        interpolation=cv2.INTER_LINEAR)
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    re_height, re_width, _ = resized_img.shape

    pad_x, pad_y = 0., 0.
    # perform letterboxing if required
    out_aspect = output_size[1] / output_size[0]    # type: ignore[index]
    roi_aspect = re_height / re_width
    new_width, new_height = int(re_width), int(re_height)
    new_height = output_size[1]
    new_width = output_size[0]
    if new_width != int(re_width) or new_height != int(re_height):
        img_pad_w, img_pad_h = int(new_width - re_width), int(new_height - re_height)
        padd_val = np.mean(img, axis=(0, 1)).astype(np.uint8)
        padded_img = cv2.copyMakeBorder(resized_img, 0, img_pad_h, 0, img_pad_w,
                                cv2.BORDER_CONSTANT, value=padd_val.tolist())
        pad_params = (re_height, re_width, img_pad_h, img_pad_w)

    return height, width, padded_img, pad_params


def postprocess(outputs, cfg, iou_th=0.4, score_th=0.02):
    # select high cls value
    # nms
    # only for batch size 1
    (bbox_regressions, landm_regressions, classifications) = outputs
    preds = np.concatenate(  # [bboxes, landms, landms_valid, conf]
        [bbox_regressions[0], landm_regressions[0],
            np.ones_like(classifications[0, :, 0][..., np.newaxis]),
            classifications[0, :, 1][..., np.newaxis]], 1)
    priors = prior_box([cfg['input_size'], cfg['input_size']],
                            cfg['min_sizes'],  cfg['steps'], cfg['clip'])
    decode_preds = decode(preds, priors, cfg['variances'])

    selected_indices = non_max_suppression(
        boxes=decode_preds[:, :4],
        scores=decode_preds[:, -1],
        max_output_size=decode_preds.shape[0],
        iou_threshold=iou_th,
        score_threshold=score_th)
    out = np.take(decode_preds, selected_indices, 0)
    print("out: ", out.shape)

    return out


###############################################################################
#   Testing                                                                   #
###############################################################################
def pad_input_image(img, max_steps):
    """pad image to suitable shape"""
    img_h, img_w, _ = img.shape

    img_pad_h = 0
    if img_h % max_steps > 0:
        img_pad_h = max_steps - img_h % max_steps

    img_pad_w = 0
    if img_w % max_steps > 0:
        img_pad_w = max_steps - img_w % max_steps

    padd_val = np.mean(img, axis=(0, 1)).astype(np.uint8)
    img = cv2.copyMakeBorder(img, 0, img_pad_h, 0, img_pad_w,
                             cv2.BORDER_CONSTANT, value=padd_val.tolist())
    pad_params = (img_h, img_w, img_pad_h, img_pad_w)

    return img, pad_params


def recover_pad_output(outputs, pad_params):
    """recover the padded output effect"""
    img_h, img_w, img_pad_h, img_pad_w = pad_params
    recover_xy = np.reshape(outputs[:, :14], [-1, 7, 2]) * \
        [(img_pad_w + img_w) / img_w, (img_pad_h + img_h) / img_h]
    outputs[:, :14] = np.reshape(recover_xy, [-1, 14])

    return outputs


def get_bbox_landm(ann, img_height, img_width):
    """get bboxes and landmarks"""
    bb = [
        int(ann[0] * img_width), int(ann[1] * img_height), \
        int(ann[2] * img_width), int(ann[3] * img_height)
    ]
    land = [
        [int(ann[4] * img_width), int(ann[5] * img_width)],
        [int(ann[6] * img_width), int(ann[7] * img_width)],
        [int(ann[8] * img_width), int(ann[9] * img_width)],
        [int(ann[10] * img_width), int(ann[11] * img_width)],
        [int(ann[12] * img_width), int(ann[13] * img_width)]
    ]

    return bb, land


###############################################################################
#   Visulization                                                              #
###############################################################################
def draw_bbox_landm(img, ann, img_height, img_width):
    """draw bboxes and landmarks"""
    # bbox
    x1, y1, x2, y2 = int(ann[0] * img_width), int(ann[1] * img_height), \
                     int(ann[2] * img_width), int(ann[3] * img_height)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # confidence
    text = "{:.4f}".format(ann[15])
    cv2.putText(img, text, (int(ann[0] * img_width), int(ann[1] * img_height)),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    # landmark
    if ann[14] > 0:
        cv2.circle(img, (int(ann[4] * img_width),
                         int(ann[5] * img_height)), 1, (255, 255, 0), 2)
        cv2.circle(img, (int(ann[6] * img_width),
                         int(ann[7] * img_height)), 1, (0, 255, 255), 2)
        cv2.circle(img, (int(ann[8] * img_width),
                         int(ann[9] * img_height)), 1, (255, 0, 0), 2)
        cv2.circle(img, (int(ann[10] * img_width),
                         int(ann[11] * img_height)), 1, (0, 100, 255), 2)
        cv2.circle(img, (int(ann[12] * img_width),
                         int(ann[13] * img_height)), 1, (255, 0, 100), 2)


def draw_anchor(img, prior, img_height, img_width):
    """draw anchors"""
    x1 = int(prior[0] * img_width - prior[2] * img_width / 2)
    y1 = int(prior[1] * img_height - prior[3] * img_height / 2)
    x2 = int(prior[0] * img_width + prior[2] * img_width / 2)
    y2 = int(prior[1] * img_height + prior[3] * img_height / 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 1)
