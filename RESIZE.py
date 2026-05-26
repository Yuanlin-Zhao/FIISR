import os
import cv2

# 输入文件夹
input_folder = r"D:\\zyl\\sr\\dataset\\FLIR-align-3class\\val\GT"

# 输出文件夹
output_folder = r"D:\\zyl\\sr\\dataset\\FLIR-align-3class\\val\LR"

os.makedirs(output_folder, exist_ok=True)

# 支持的图片格式
img_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

for filename in os.listdir(input_folder):
    if filename.lower().endswith(img_ext):

        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            continue

        h, w = img.shape[:2]

        # 宽高缩小为1/2（面积1/4）
        new_size = (w // 4, h // 4)

        img_small = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, img_small)

print("全部图片压缩完成！")