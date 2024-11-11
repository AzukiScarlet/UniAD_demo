import cv2
import os

def jpg2avi(img_path: str, video_path: str, fps: int):

    print("Converting JPG images to video...")
    
    # 设置输入图像路径和输出视频文件名
    image_folder = img_path
    video_filename = video_path
    fps = fps  # 每秒帧数

    # 获取所有 JPG 图像文件
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort()  # 可选，按文件名排序

    # 读取第一张图像以获取尺寸
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_image.shape

    # 创建视频编写对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 可以选择不同的编码方式
    video = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    # 将每张图像写入视频
    for image in images:
        img = cv2.imread(os.path.join(image_folder, image))
        video.write(img)  # 写入视频

    # 释放视频对象
    video.release()
    cv2.destroyAllWindows()

    print(f'Video {video_filename} created successfully!')

def main():

    # 设置输入图像路径和输出视频文件名
    image_folder = '/home2/lixiang/UniAD_demo/experiments/origin/stage2/origin/output/figures'  # 替换为你的图像文件夹路径
    video_filename = '/home2/lixiang/UniAD_demo/experiments/origin/stage2/origin/output/output_video.avi'  # 输出视频文件名
    fps = 2  # 每秒帧数
    jpg2avi(image_folder, video_filename, fps)

if __name__ == "__main__":
    main()
