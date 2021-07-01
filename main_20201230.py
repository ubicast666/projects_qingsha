# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import argparse
# create by zhangmingchao 2020.12.30
# change cout number problem

import configparser
import pymysql
import numpy as np
import logging
import time
import cv2
import numpy
from tkinter import *
import sys
import os

from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DecodeImage(object):
    def __init__(self, to_rgb=True):
        self.to_rgb = to_rgb

    def __call__(self, img):
        data = np.frombuffer(img, dtype='uint8')
        img = cv2.imdecode(data, 1)
        if self.to_rgb:
            assert img.shape[2] == 3, 'invalid shape of image[%s]' % (
                img.shape)
            img = img[:, :, ::-1]

        return img


class ResizeImage(object):
    def __init__(self, resize_short=None):
        self.resize_short = resize_short

    def __call__(self, img):
        img_h, img_w = img.shape[:2]
        percent = float(self.resize_short) / min(img_w, img_h)
        w = int(round(img_w * percent))
        h = int(round(img_h * percent))
        return cv2.resize(img, (w, h))


class CropImage(object):
    def __init__(self, size):
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img):
        w, h = self.size
        img_h, img_w = img.shape[:2]
        w_start = (img_w - w) // 2
        h_start = (img_h - h) // 2

        w_end = w_start + w
        h_end = h_start + h
        return img[h_start:h_end, w_start:w_end, :]


class NormalizeImage(object):
    def __init__(self, scale=None, mean=None, std=None):
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, img):
        return (img.astype('float32') * self.scale - self.mean) / self.std


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = img.transpose((2, 0, 1))
        return img


# def parse_args():
#     def str2bool(v):
#         return v.lower() in ("true", "t", "1")
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-i", "--image_file", type=str)
#     parser.add_argument("-m", "--model_file", type=str)
#     parser.add_argument("-p", "--params_file", type=str)
#     parser.add_argument("-b", "--batch_size", type=int, default=1)
#     parser.add_argument("--use_fp16", type=str2bool, default=False)
#     parser.add_argument("--use_gpu", type=str2bool, default=True)
#     parser.add_argument("--ir_optim", type=str2bool, default=True)
#     parser.add_argument("--use_tensorrt", type=str2bool, default=False)
#     parser.add_argument("--gpu_mem", type=int, default=8000)
#     parser.add_argument("--enable_benchmark", type=str2bool, default=False)
#     parser.add_argument("--model_name", type=str)
#
#     return parser.parse_args()


def create_predictor(model_file, params_file, use_gpu, gpu_mem, ir_optim, use_tensorrt, use_fp16, batch_size):
    config = AnalysisConfig(model_file, params_file)

    if use_gpu:
        config.enable_use_gpu(gpu_mem, 0)
    else:
        config.disable_gpu()

    config.disable_glog_info()
    config.switch_ir_optim(ir_optim)  # default true
    if use_tensorrt:
        config.enable_tensorrt_engine(
            precision_mode=AnalysisConfig.Precision.Half
            if use_fp16 else AnalysisConfig.Precision.Float32,
            max_batch_size=batch_size)

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)
    predictor = create_paddle_predictor(config)

    return predictor


def create_operators():
    size = 224
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    img_scale = 1.0 / 255.0

    decode_op = DecodeImage()
    resize_op = ResizeImage(resize_short=256)
    crop_op = CropImage(size=(size, size))
    normalize_op = NormalizeImage(
        scale=img_scale, mean=img_mean, std=img_std)
    totensor_op = ToTensor()

    return [decode_op, resize_op, crop_op, normalize_op, totensor_op]


def preprocess(fname, ops):
    data = open(fname, 'rb').read()
    for op in ops:
        data = op(data)

    return data


def main(res):
    # args = parse_args()
    conf = configparser.ConfigParser()
    conf.read('config.conf')
    image_file = conf.get("info", "image_file")
    image_file = image_file + res + ".jpg"
    print(image_file)
    model_file = conf.get("info", "model_file")
    params_file = conf.get("info", "params_file")
    batch_size = conf.getint("info", "batch_size")
    use_fp16 = conf.getboolean("info", "use_fp16")
    use_gpu = conf.getboolean("info", "use_gpu")
    ir_optim = conf.getboolean("info", "ir_optim")
    use_tensorrt = conf.getboolean("info", "use_tensorrt")
    gpu_mem = conf.getint("info", "gpu_mem")
    enable_benchmark = conf.getboolean("info", "enable_benchmark")
    model_name = conf.get("info", "model_name")
    print(image_file, model_file, params_file, batch_size, use_fp16, use_gpu, ir_optim, use_tensorrt, gpu_mem,
          enable_benchmark, model_name)

    if not enable_benchmark:
        assert batch_size == 1
        assert use_fp16 is False
    else:
        assert use_gpu is True
        assert model_name is not None
        assert use_tensorrt is True
    # HALF precission predict only work when using tensorrt
    if use_fp16 is True:
        assert use_tensorrt is True

    operators = create_operators()
    predictor = create_predictor(model_file, params_file, use_gpu, gpu_mem, ir_optim, use_tensorrt, use_fp16,
                                 batch_size)

    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_tensor(input_names[0])

    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_tensor(output_names[0])

    test_num = 500
    test_time = 0.0
    if not enable_benchmark:
        inputs = preprocess(image_file, operators)
        inputs = np.expand_dims(
            inputs, axis=0).repeat(
            batch_size, axis=0).copy()
        input_tensor.copy_from_cpu(inputs)

        predictor.zero_copy_run()

        output = output_tensor.copy_to_cpu()
        output = output.flatten()
        cls = np.argmax(output)
        score = output[cls]
        logger.info("class: {0}".format(cls))
        logger.info("score: {0}".format(score))
        return cls, score
    else:
        for i in range(0, test_num + 10):
            inputs = np.random.rand(batch_size, 3, 224,
                                    224).astype(np.float32)
            start_time = time.time()
            input_tensor.copy_from_cpu(inputs)

            predictor.zero_copy_run()

            output = output_tensor.copy_to_cpu()
            output = output.flatten()
            if i >= 10:
                test_time += time.time() - start_time

        fp_message = "FP16" if use_fp16 else "FP32"
        logger.info("{0}\t{1}\tbatch size: {2}\ttime(ms): {3}".format(
            model_name, fp_message, batch_size, 1000 * test_time /
                                                test_num))


def process():
    # 连接数据库 读取状态位的值，根据值来判定是否需要打开摄像头，进行图片的获取，保存，以及结果位的修改，以及预测行为
    if running:
        db = pymysql.connect(host="127.0.0.1",
                             port=3306,
                             user="root",
                             password="123456",
                             database="detection",
                             charset="utf8")
        # 创建游标
        cur = db.cursor()
        sql = "select test_num,test_batch,name,status,status2,data_time from info order by id DESC limit 1;"
        cur.execute(sql)
        res = cur.fetchone()
        # print(res)
        # print(res[0])
        datatime = str(res[5]).replace("-", "_").replace(":", "_").replace(" ", "_")
        pic_name = res[0] + "_" + res[1] + "_" + res[2] + "_" + datatime
        print(pic_name)

        # if not res[0]:
        #     pass
        if res[3] == 1 and res[4] == 0:
            # 打开摄像头，采集信息，保存图片到指定的路径
            cap = cv2.VideoCapture(0)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("can't receive the frame (stream end) Exiting...")
                    break
                # 窗口显示
                # cv2.imshow("Capture_Test", frame)  # 窗口显示，显示名为 Capture_Test
                # cv2.imwrite("D:/work/projects_qingsha/save/" + pic_name + ".jpg", frame)
                if not os.path.exists("c:/temp"):
                    os.makedirs("c:/temp")
                cv2.imwrite("c:/temp/" + pic_name + ".jpg", frame)
                print("success to save jpg")
                print('-----------------------------------')
                time.sleep(2)
                break
            try:
                cls, score = main(pic_name)
                print(cls, score)
                global count
                global totle_count
                if cls == 0 and score > 0.5:
                    sql2 = "select count from info order by id DESC limit 1;"
                    cur.execute(sql2)
                    res1 = cur.fetchone()
                    print("res1:", res1)
                    count = res1[0] + 1
                    show(count)
                    sql3 = "update info set result=1,count={},status2=1,new_obj=1 order by id desc limit 1;".format(count)
                    cur.execute(sql3)
                    db.commit()
                    totle_count += 1
                    show2(totle_count)
                    print("吹砂干净")
                else:
                    sql2 = "select count from info order by id DESC limit 1;"
                    cur.execute(sql2)
                    res1 = cur.fetchone()
                    print("res1:", res1)
                    count = res1[0] + 1
                    show(count)
                    sql = "update info set result=0,count={},status2=1,new_obj=1 order by id desc limit 1;".format(count)
                    cur.execute(sql)
                    db.commit()
                    # time.sleep(2)
                    # if count > 5:
                    #     return
            except FileNotFoundError:
                print("图片没有保存")
        else:
            print("等待...进行下一次的检测")
            # time.sleep(30)
        cur.close()
        db.close()
    # window.after(1000, process)
    window.after(1, process)


def stop():
    global running
    running = False


def start():
    global running
    running = True


if __name__ == "__main__":
    window = Tk()
    window.title("First Window")
    window.geometry("350x200")
    # 给之前的例子添加一个标签组件，我们可以使用Label类
    app = Frame(window)
    app.grid()

    running = True


    def show(count):
        lbl = Label(window,
                    text="目前工件一共拍照了%d次" % count,
                    font=("Arial Bold", 10),
                    # bg='red',
                    width=40,
                    height=3,
                    wraplength=80,
                    # anchor='w',
                    justify='left',
                    )
        lbl.grid(column=0, row=1)


    def show2(totle_count):
        lbl = Label(window, text="今天一共吹净了%d个" % totle_count,
                    # bg='red',
                    width=40,
                    height=3,
                    wraplength=80,
                    justify='left',
                    font=("Arial Bold", 10))
        lbl.grid(column=0, row=2)


    # 添加按钮组件
    # 我们可以用fg参数设置按钮或其他组件的前景色
    # 我们可以用bg参数设置按钮或其他组件的背景色
    # btn = Button(window,text="Click me",bg="orange")
    btn = Button(app, text="开始", fg="red", command=start)
    btn.grid(column=0, row=0)
    # btn.pack(side='left', anchor='nw')

    btn = Button(app, text="运行", fg="red", command=process)
    btn.grid(column=3, row=0)
    # btn.pack(side='top', anchor='n')
    # btn.pack(side='top')

    btn = Button(app, text="结束", fg='red', command=stop)
    btn.grid(column=5, row=0)
    # btn.pack(side='right',expand='yes',anchor='ne')

    count = 0
    totle_count = 0
    show(count)
    show2(totle_count)

    # window.after(1000,process)
    window.mainloop()
