# 身份证提取

身份证人像面提取&透视变换。

用于解决复杂背景中，身份证难以提取问题。

可能会对特定类型背景的图片识别率比较低，可根据具体情况，重新训练生成权重文件。


## 应用场景

由于需要调用模型，预测图片中身份证人像面可能存在的位置，
所以速度慢，因此以下情况不建议使用：

1 纯色背景不建议使用，可以有更快速的方法实现

2 特别大的图片，会消耗更多的CPU和内存资源，同时处理时间过长

## 处理流程

**一 预测身份证位置**

![avatar](http://www.zhangkang.fun/assets/idcard/12.jpg)

**二 根据预测，生成蒙版**

![avatar](http://www.zhangkang.fun/assets/idcard/12_mask_50.jpeg)

**三 提取&透视变换**

![avatar](http://www.zhangkang.fun/assets/idcard/12_res.jpg)

## 训练自有数据

具体参照：https://github.com/divamgupta/image-segmentation-keras

也可参照以下步骤进行：

第一步：使用labelme标注位置，生成json文件

第二步：根据json文件生成标签label，可使用dataset/json_to_dataset2.py
批量生成图片标签

第三步：执行train.py，进行数据训练

## 错误

### 批量处理

调用idcard_multiple方法，会随着处理图片的数量和大小增加，而占用更多内存，不建议一次性输入大量图片，可以多次小批量的进行图片处理。

