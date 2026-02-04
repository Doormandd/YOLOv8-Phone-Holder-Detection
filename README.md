# Phone Holder Detection with YOLOv8 / 手机支架检测（YOLOv8）

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-v8-orange)](https://github.com/ultralytics/ultralytics)

---

## 项目简介 / Project Introduction

This project implements a **phone holder detection** system using **YOLOv8** (You Only Look Once version 8), one of the most advanced object detection architectures.

本项目使用 **YOLOv8**（You Only Look Once 第8版），最先进的目标检测架构之一，实现了一个**手机支架检测**系统。

---

## 项目特点 / Features

- ✅ **High Accuracy**: Achieved 99.975% precision and 100% recall / **高精度**：达到了99.975%的精确度和100%的召回率
- ✅ **Lightweight Model**: Best.pt file size only 6.0MB / **轻量级模型**：best.pt文件仅为6.0MB
- ✅ **Easy to Use**: Simple inference interface / **易于使用**：简单的推理接口
- ✅ **Complete Training Results**: Includes training curves and visualizations / **完整的训练结果**：包含训练曲线和可视化
- ✅ **Quality Dataset**: 58 annotated images with Label Studio / **高质量数据集**：58张使用Label Studio标注的图片

---

## 数据集 / Dataset

### 数据集统计 / Dataset Statistics

| Split | Count / 数量 | Percentage / 占比 |
|--------|---------------|------------------|
| Training / 训练集 | 44 images / 张 | 75.86% |
| Validation / 验证集 | 7 images / 张 | 12.07% |
| Test / 测试集 | 7 images / 张 | 12.07% |
| **Total / 总计** | **58 images / 张** | **100%** |

### 标注工具 / Annotation Tool
- **Label Studio**: Professional labeling platform / 专业标注平台
- **YOLO Format**: Standard bounding box format / 标准边界框格式

---

## 训练结果 / Training Results

### 性能指标 / Performance Metrics

| Metric / 指标 | Value / 数值 | Description / 说明 |
|----------------|---------------|-------------------|
| **Precision** / **精确度** | 99.975% | 正确检测的正样本占检测出的正样本的比例 |
| **Recall** / **召回率** | 100% | 正确检测出的正样本占所有正样本的比例 |
| **mAP@0.5** | 83.348% | IoU阈值0.5时的平均精度 |
| **mAP@0.5:0.95** | 78.834% | IoU阈值0.5到0.95的平均精度 |

### 训练配置 / Training Configuration
- **Epochs**: 32
- **Classes**: 1 (phone holder)
- **Image Size**: Custom (as per dataset)
- **Model**: YOLOv8

---

## 项目结构 / Project Structure

```
YOLO_v8/
├── best.pt                      # 最佳模型文件 (6.0MB)
├── best.pt.zip                 # 压缩的模型文件
├── data/                      # 数据集
│   ├── classes.txt              # 类别定义
│   ├── phoneholder.yaml         # 数据集配置
│   ├── notes.json             # 数据集元数据
│   ├── images/               # 图片文件
│   │   ├── train/           # 训练集 (44张)
│   │   ├── val/             # 验证集 (7张)
│   │   └── test/           # 测试集 (7张)
│   └── labels/               # 标注文件 (YOLO格式)
│       ├── train/
│       ├── val/
│       └── test/
├── phoneholder_zsxq/          # 训练结果
│   └── runs/detect/train/
│       ├── best.pt           # 训练生成的最佳模型
│       ├── last.pt           # 最后一个epoch的模型
│       ├── results.csv       # 训练指标（32个epoch）
│       ├── results.png       # 训练曲线图
│       ├── confusion_matrix.png
│       ├── F1_curve.png
│       ├── PR_curve.png
│       ├── P_curve.png
│       ├── R_curve.png
│       ├── train_batch*.jpg  # 训练批次可视化
│       └── args.yaml        # 训练参数
├── data.zip                   # 压缩数据集 (已忽略)
├── datasets.zip               # 压缩数据集备份 (已忽略)
└── phoneholder_zsxq.zip       # 压缩训练结果 (已忽略)
```

---

## 使用方法 / Usage

### 环境要求 / Requirements

```bash
# Python环境 / Python Environment
python >= 3.8

# 依赖包 / Dependencies
pip install ultralytics
```

### 推理示例 / Inference Example

#### Python代码 / Python Code

```python
from ultralytics import YOLO

# 加载模型 / Load model
model = YOLO('best.pt')

# 进行推理 / Perform inference
results = model('image.jpg')

# 显示结果 / Display results
for r in results:
    boxes = r.boxes
    for box in boxes:
        # 获取边界框坐标 / Get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0]
        # 获取类别 / Get class
        cls = int(box.cls[0])
        # 获取置信度 / Get confidence
        conf = float(box.conf[0])
        
        print(f"Class: {cls}, Conf: {conf:.2f}, Box: [{x1}, {y1}, {x2}, {y2}]")

# 保存结果 / Save results
results[0].save('result.jpg')
```

#### 命令行使用 / Command Line Usage

```bash
# 推理单张图片 / Inference on single image
yolo predict model=best.pt source=image.jpg

# 推理整个文件夹 / Inference on entire folder
yolo predict model=best.pt source=data/images/test/

# 使用GPU加速 / Use GPU acceleration
yolo predict model=best.pt source=image.jpg device=0
```

---

## 训练 / Training

### 准备数据集 / Prepare Dataset

确保数据集按照以下结构组织：

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### 运行训练 / Run Training

```bash
# 使用自定义数据集训练 / Train with custom dataset
yolo detect train data=data/phoneholder.yaml model=yolov8n.pt epochs=32 imgsz=640

# 使用预训练模型 / Use pretrained model
yolo detect train data=data/phoneholder.yaml model=yolov8s.pt epochs=32 imgsz=640

# 使用GPU / Use GPU
yolo detect train data=data/phoneholder.yaml model=yolov8n.pt epochs=32 device=0
```

### 训练参数说明 / Training Parameters

- `data`: 数据集配置文件路径
- `model`: 模型架构（yolov8n/s/m/l/x）
- `epochs`: 训练轮数
- `imgsz`: 输入图像大小
- `device`: 设备选择（0为第一个GPU）

---

## 训练可视化 / Training Visualization

### 训练曲线 / Training Curves

训练过程中的性能变化可以在以下文件中查看：

- `results.png`: 所有指标的综合曲线
- `F1_curve.png`: F1分数曲线
- `PR_curve.png`: 精确率-召回率曲线
- `P_curve.png`: 精确率曲线
- `R_curve.png`: 召回率曲线

### 混淆矩阵 / Confusion Matrix

- `confusion_matrix.png`: 标准混淆矩阵
- `confusion_matrix_normalized.png`: 归一化混淆矩阵

### 训练批次 / Training Batches

- `train_batch*.jpg`: 训练过程中的批次可视化

---

## 为什么选择YOLOv8？ / Why YOLOv8?

### 优势 / Advantages

1. **速度与精度平衡** / **Balance Between Speed and Accuracy**
   - YOLOv8在保持高精度的同时提供实时推理速度
   - Maintains high accuracy while providing real-time inference speed

2. **易于使用** / **Easy to Use**
   - 统一的命令行接口
   - Clean and unified CLI interface
   - 丰富的文档和社区支持
   - Rich documentation and community support

3. **模型多样性** / **Model Diversity**
   - 提供nano、small、medium、large、extra-large等多种尺寸
   - Offers multiple sizes: nano, small, medium, large, extra-large
   - 适应不同的部署需求
   - Adapts to different deployment requirements

4. **先进架构** / **Advanced Architecture**
   - 改进的骨干网络和检测头
   - Improved backbone and detection head
   - 更好的特征提取和目标检测能力
   - Better feature extraction and object detection capabilities

---

## 项目亮点 / Project Highlights

### 1. 高质量数据集 / High-Quality Dataset

- 使用 **Label Studio** 专业标注工具
- 58张精心标注的图片
- 标注准确，质量可靠

### 2. 优秀的训练结果 / Excellent Training Results

- **99.975%精确度**：几乎零误检
- **100%召回率**：无漏检
- **83.348% mAP@0.5**：高平均精度

### 3. 轻量级模型 / Lightweight Model

- 最佳模型仅 **6.0MB**
- 适合边缘设备部署
- 推理速度快

### 4. 完整的实验记录 / Complete Experiment Records

- 详细的训练曲线
- 混淆矩阵分析
- PR曲线、F1曲线等

---

## 未来改进方向 / Future Improvements

- [ ] 扩展数据集，增加更多场景 / Expand dataset with more scenarios
- [ ] 测试不同YOLOv8模型尺寸（n/s/m/l/x）/ Test different YOLOv8 model sizes
- [ ] 优化模型量化和压缩 / Optimize model quantization and compression
- [ ] 添加数据增强策略 / Add data augmentation strategies
- [ ] 实现实时检测应用 / Implement real-time detection application

---

## 贡献指南 / Contributing

欢迎贡献！如果你想改进这个项目，请：

1. Fork本仓库 / Fork this repository
2. 创建特性分支 / Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 提交更改 / Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 / Push to the branch (`git push origin feature/AmazingFeature`)
5. 开启Pull Request / Open a Pull Request

---

## 许可证 / License

本项目采用 **MIT License** 开源。

This project is open source under the **MIT License**.

---

## 联系方式 / Contact

- **作者 / Author**: DR
- **邮箱 / Email**: dr1012324010@qq.com
- **GitHub**: https://github.com/Doormandd

---

## 致谢 / Acknowledgments

- **Ultralytics**: YOLOv8框架 - https://github.com/ultralytics/ultralytics
- **Label Studio**: 标注工具 - https://labelstud.io/
- **PyTorch**: 深度学习框架 - https://pytorch.org/

---

## 参考链接 / References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Label Studio Documentation](https://labelstud.io/guide/)
- [YOLOv8 Paper](https://arxiv.org/abs/2305.09972)

---

**注意 / Note**: 如果你在实际应用中使用此模型，请确保测试其在你的具体场景中的表现。

**Note**: If you use this model in production, make sure to test its performance in your specific scenarios.
