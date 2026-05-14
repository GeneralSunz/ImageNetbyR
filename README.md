# ImageNet 图像分类的R语言实现(极简版) 

使用预训练 MobileNetV2 模型对本地图片进行 ImageNet 分类，基于 R 语言和 Keras3。

## 文件结构

```
├── image_classfication.r   # 主脚本：加载图片 → 预处理 → 预测 → 解码
├── book_images/             # 存放待分类图片（jpg/.jpeg）
└── README.md
```

## 环境要求

- R >= 4.4
- Python 3.13（系统 Python，非 conda 虚拟环境）
- Python 包：`keras>=3.13`
- R 包：`keras3`、`magick`

