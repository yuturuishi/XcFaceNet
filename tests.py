from PIL import Image
import logging
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from nets.facenet import Facenet
from utils.logger import setup_logger, fmt_duration
from utils.utils import preprocess_input, resize_image, show_config

# 项目根目录：权重 / 样例图等路径以项目根为基准，任意目录执行都不会跑偏
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

class Test(object):

    # ---------------------------------------------------#
    #   初始化Facenet
    # ---------------------------------------------------#
    def __init__(self,model_path,input_shape,backbone,letterbox_image,cuda):
        self.model_path = model_path
        self.input_shape = input_shape
        self.backbone = backbone
        self.letterbox_image = letterbox_image
        self.cuda=cuda
        # ---------------------------------------------------#
        #   载入模型与权值
        # ---------------------------------------------------#
        logging.info("正在载入模型权重...")
        # 修复：按用户配置决定权重落点，避免 cuda=False 时权重先落 GPU 再拷回 CPU
        device = torch.device('cuda' if (self.cuda and torch.cuda.is_available()) else 'cpu')

        self.net = Facenet(backbone=self.backbone, mode="predict").eval()
        self.net.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        logging.info("模型载入完成: %s", self.model_path)

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image_1, image_2):
        # ---------------------------------------------------#
        #   图片预处理，归一化
        # ---------------------------------------------------#
        with torch.no_grad():
            image_1 = resize_image(image_1, [self.input_shape[1], self.input_shape[0]],
                                   letterbox_image=self.letterbox_image)
            image_2 = resize_image(image_2, [self.input_shape[1], self.input_shape[0]],
                                   letterbox_image=self.letterbox_image)

            image_1_tp = np.transpose(preprocess_input(np.array(image_1, np.float32)), (2, 0, 1))


            photo_1 = torch.from_numpy(
                np.expand_dims(image_1_tp, 0))
            photo_2 = torch.from_numpy(
                np.expand_dims(np.transpose(preprocess_input(np.array(image_2, np.float32)), (2, 0, 1)), 0))



            if self.cuda:
                photo_1 = photo_1.cuda()
                photo_2 = photo_2.cuda()

            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            output1 = self.net(photo_1).cpu().numpy()
            output2 = self.net(photo_2).cpu().numpy()

            # ---------------------------------------------------#
            #   计算二者之间的距离
            # ---------------------------------------------------#
            l1 = np.linalg.norm(output1 - output2, axis=1)

        """
        plt.subplot(1, 2, 1)
        plt.imshow(np.array(image_1))

        plt.subplot(1, 2, 2)
        plt.imshow(np.array(image_2))
        plt.text(-12, -12, 'Distance:%.3f' % l1, ha='center', va= 'bottom',fontsize=11)
        plt.show()


        """

        return l1

if __name__ == "__main__":
    #---------------------------------------------------#
    #   初始化日志系统：log/test-年月日-时分秒.log
    #---------------------------------------------------#
    _, log_file = setup_logger("test_pytorch")

    logging.info("=" * 78)
    logging.info("XcFaceNet 人脸相似度测试")
    logging.info("=" * 78)
    logging.info("[启动] 测试时间: %s | 日志文件: %s", time.strftime("%Y-%m-%d %H:%M:%S"), log_file)

    __params = {
        "model_path" : os.path.join(PROJECT_ROOT, "checkpoints", "model_best_lfw0.9898_ep067.pth"),
        "input_shape": [112, 112, 3],
        "backbone": "mobilenet",
        "letterbox_image": False,  # 与训练/部署一致：直接 resize 112，不做 letterbox
        "cuda": False,
    }

    test = Test(**__params)
    url1 = os.path.join(PROJECT_ROOT, "data", "Adrien_Brody_0001.jpg")
    url2 = os.path.join(PROJECT_ROOT, "data", "Adrien_Brody_0012.jpg")

    logging.info("[输入] 图片1: %s", url1)
    logging.info("[输入] 图片2: %s", url2)
    show_config(**__params)

    image_1 = Image.open(url1)
    image_2 = Image.open(url2)

    t1 = time.time()
    probability = test.detect_image(image_1, image_2)
    t2 = time.time()

    # 特征已 L2 归一化：余弦 = 1 - d²/2（与线上部署同口径）
    distance = float(np.asarray(probability).reshape(-1)[0])
    cosine = 1.0 - distance ** 2 / 2.0
    logging.info("[结果] 推理耗时: %s | 空间距离: %.5f | 余弦相似度: %.4f（越大越相似）",
                 fmt_duration(t2 - t1), distance, cosine)
