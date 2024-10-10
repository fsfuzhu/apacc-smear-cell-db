from ultralytics import YOLO

def main():
    # 定义图片和标注数据的路径
    image_path = r'C:\Users\Administrator\Desktop\Pap Semar\apacc-smear-cell-db\data\input\images'

    # 定义训练配置
    data_config = """
    # YOLO数据配置
    train: {image_path}
    val: {image_path}

    # 类别名称列表
    names:
      0: healthy
      1: rubbish
      2: unhealthy
      3: bothcells
    """.format(image_path=image_path)

    # 创建一个新的数据配置文件
    data_file = 'smear_data.yaml'
    with open(data_file, 'w') as file:
        file.write(data_config)

    # 初始化YOLO模型并加载更大的预训练权重（精度优先）
    model = YOLO('yolov8n.pt')  # 使用更大的模型 'yolov8x.pt' 提升精度

    # 设置自定义的超参数
    model.train(
        data=data_file,
        epochs=300,  # 增加 epoch 数，进行更长时间的训练
        batch=16,  # 如果显存允许，使用更大的 batch size，建议从 16 或 32 开始
        imgsz=640,  # 保持图像大小为 640x640
        save_period=10,  # 每隔 10 个 epoch 保存一次模型
        name='smear_cell_detection',
        lr0=1e-4,  # 初始学习率
        optimizer='AdamW',  # 使用 AdamW 优化器，提升训练稳定性
        amp=False,  # 启用混合精度训练
        augment=True,  # 启用数据增强
        patience=50  # 使用 early stopping，等待 50 个 epoch 后验证不再提升才停止
    )

if __name__ == '__main__':
    main()
