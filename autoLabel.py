import os
from ultralytics import YOLO
import cv2
import numpy as np
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 加载训练好的YOLO模型
model = YOLO('best.pt')  # 替换为你的模型权重文件路径

# 设置输入和输出目录
input_folder = './'  # 当前目录
output_image_folder = './yolo_output/images/'  # 输出带标注图像的目录
output_label_folder = './yolo_output/labels/'  # 输出标注文件的目录

# 创建输出目录（如果不存在）
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)

# 定义每个类别的颜色（根据需要修改）
class_colors = {
    0: (0, 255, 0),    # 绿色表示healthy
    1: (255, 0, 0),    # 蓝色表示rubbish
    2: (0, 0, 255),    # 红色表示unhealthy
    3: (255, 255, 0)   # 黄色表示bothcells
}

# 支持的图像格式
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
image_files = [file for file in os.listdir(input_folder) if os.path.splitext(file)[1].lower() in image_extensions]

# 处理每一张图片
for image_file in image_files:
    try:
        # 输入图片的完整路径
        input_image_path = os.path.join(input_folder, image_file)
        
        # 读取图片
        image = cv2.imread(input_image_path)
        if image is None:
            logging.warning(f"无法读取图片 {image_file}。跳过。")
            continue

        height, width = image.shape[:2]

        # 使用YOLO模型进行推理
        results = model.predict(source=input_image_path, save=False)  # 禁用自动保存

        # 创建带标注的图片副本
        annotated_image = image.copy()

        # 准备写入标签文件
        label_lines = []

        # 遍历每个结果
        for result in results:
            if result.boxes is None:
                continue  # 该结果中没有检测到任何框

            # 遍历每个边界框
            for i, box in enumerate(result.boxes.xyxy):
                # 将边界框张量移到CPU并转换为NumPy数组
                box_np = box.cpu().numpy()
                
                # 检查是否有NaN值
                if np.isnan(box_np).any():
                    logging.warning(f"在图片 {image_file} 的边界框中检测到NaN。跳过此框。")
                    continue  # 跳过无效的框

                # 获取边界框坐标
                x1, y1, x2, y2 = box_np
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # 获取类别ID和置信度
                cls = int(result.boxes.cls[i].cpu().numpy()) if len(result.boxes.cls) > 0 else -1
                conf = float(result.boxes.conf[i].cpu().numpy()) if len(result.boxes.conf) > 0 else 0.0

                if cls not in model.names:
                    logging.warning(f"检测到未知类别ID {cls} 在图片 {image_file}。跳过此框。")
                    continue  # 跳过未知类别

                # 计算YOLO格式的标注值（相对于图片尺寸的比例）
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                bbox_width = (x2 - x1) / width
                bbox_height = (y2 - y1) / height

                # 确保所有值在0到1之间
                x_center = min(max(x_center, 0), 1)
                y_center = min(max(y_center, 0), 1)
                bbox_width = min(max(bbox_width, 0), 1)
                bbox_height = min(max(bbox_height, 0), 1)

                # 创建标签行
                label_line = f"{cls} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
                label_lines.append(label_line)

                # 获取类别名称和颜色
                label = f"{model.names[cls]} {conf:.2f}"
                color = class_colors.get(cls, (255, 255, 255))  # 如果类别ID不在class_colors中，则默认白色

                # 在图片上绘制边界框
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

                # 在图片上绘制标签
                cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, color, 2)

        # 保存带标注的图片
        output_image_path = os.path.join(output_image_folder, image_file)
        cv2.imwrite(output_image_path, annotated_image)

        # 保存标签文件
        label_file_name = os.path.splitext(image_file)[0] + '.txt'
        label_file_path = os.path.join(output_label_folder, label_file_name)
        with open(label_file_path, 'w') as f:
            f.write('\n'.join(label_lines))

        logging.info(f'已处理 {image_file}，保存标注图片到 {output_image_path}，标签文件到 {label_file_path}')

    except Exception as e:
        logging.error(f"处理 {image_file} 时出错: {e}")
        continue  # 继续处理下一个图片

logging.info('处理完成。所有结果已保存到: {}'.format(output_image_folder))
