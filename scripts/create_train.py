"""
训练数据集生成train.txt文件的脚本

"""
import os

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if __name__ == "__main__":
    train_path  = r"D:\datasets\face\ms1mv2\train"  # 训练数据集图片根目录
    train_desc_path = r"D:\datasets\face\ms1mv2\train.txt" # 训练数据集描述文件

    if os.path.exists(train_desc_path):
        os.remove(train_desc_path)

    types_name  = os.listdir(train_path)
    types_name  = sorted(types_name)
    train_desc_f = open(train_desc_path, 'w')

    cls_id = 0
    for type_name in types_name:
        photos_path = os.path.join(train_path, type_name)
        if not os.path.isdir(photos_path):
            continue
        photos_name = os.listdir(photos_path)
        image_names = [
            photo_name for photo_name in photos_name
            if os.path.isfile(os.path.join(photos_path, photo_name))
            and os.path.splitext(photo_name)[1].lower() in IMAGE_EXTS
        ]
        if len(image_names) == 0:
            continue
        for photo_name in image_names:
            train_desc_f.write(str(cls_id) + ";" + '%s'%(os.path.join(os.path.abspath(train_path), type_name, photo_name)))
            train_desc_f.write('\n')
        cls_id += 1
    train_desc_f.close()
