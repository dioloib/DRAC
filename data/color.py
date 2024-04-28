import cv2
import os
import numpy as np

SAVE_FOLDER = r"../data/DRAC2022_Seg_Color"

def read_list(Listtxt):
    with open(Listtxt) as f:
        Path = []
        for line in f:
            try:
                value = line[:-1]
            except ValueError:
                value = line.strip('\n')
            Path.append(value)
    return Path

def enhance_contrast(img):
    # 使用直方图均衡化增强对比度
    return cv2.equalizeHist(img)

def gamma_correction(img, gamma):
    # Gamma校正
    return np.power(img / 255.0, gamma) * 255.0

def white_balance(img):
    # 计算每个通道的平均值
    avg_r = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_b = np.mean(img[:, :, 2])

    # 计算缩放因子，使每个通道的平均值相等
    scale_r = avg_g / avg_r
    scale_b = avg_g / avg_b

    # 对每个通道进行缩放
    img[:, :, 0] = np.clip(img[:, :, 0] * scale_r, 0, 255).astype(np.uint8)
    img[:, :, 2] = np.clip(img[:, :, 2] * scale_b, 0, 255).astype(np.uint8)

    return img

def apply_filters(img):
    # 使用高斯滤波器平滑图像
    return cv2.GaussianBlur(img, (5, 5), 0)

def alpha_rooting(img):
    # α-rooting增强
    alpha = 0.7
    return np.power(img, alpha)

def augment_data(img):
    # 随机旋转和翻转
    angle = np.random.randint(-30, 30)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)

    # 随机裁剪和缩放
    h, w, _ = img.shape
    crop_size = np.random.randint(h // 2, h)
    x, y = np.random.randint(0, h - crop_size), np.random.randint(0, w - crop_size)
    img = img[x:x+crop_size, y:y+crop_size]

    # 添加高斯噪声
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = np.clip(img + noise, 0, 255)

    return img

def coloring(imgpaths):
    thres = 120
    for imgpath in imgpaths:
        img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 应用增强技术
        gray_img = enhance_contrast(gray_img)
        gray_img = gamma_correction(gray_img, gamma=1.2)
        gray_img = apply_filters(gray_img)
        
        
        ch0 = gray_img
        ch1 = np.zeros_like(gray_img)
        ch2 = np.zeros_like(gray_img)
        ch1[gray_img > thres] = gray_img[gray_img > thres]
        ch2[gray_img <= thres] = gray_img[gray_img <= thres]
        result_img = cv2.merge((ch0, ch1, ch2))
        
        result_img = white_balance(result_img)
        # 数据增强
        #result_img = alpha_rooting(result_img)
        result_img = augment_data(result_img)
        
        result_path = imgpath.replace('Seg', 'Seg_Color')
        cv2.imwrite(result_path, result_img)

if __name__ == '__main__':
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    imglist = 'drac2022seg_all.txt'
    imgpaths = read_list(imglist)
    coloring(imgpaths)
