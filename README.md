# Moe Music Transcription Dev branch


## 关于

本分支为开发分支，其中包含模型训练，自定义数据处理，部署导出等代码，以及一些开发文档和其他分享（例如踩坑，第三方库特性与BUG等）

## 文档

### 开发部署

**该部署文档写于项目首个预览版本完成后，可能存在遗漏，有问题可发issue或者先尝试自行解决，后续会跑一遍测试并完善**

项目使用python 3.10，nodejs 18.16.1（你也可以先尝试你现有的版本）

1. python环境：

到项目下运行以下命令：

```shell
pip install -r requirements.txt
```

pytorch请按照官网教程手动安装

2. node环境：

到`react_app/`目录下执行以下命令：

```
npm install
```

3. 运行：

​	webui:

```
npm start
```

​	python backend:

```
python backend.py
```

### ONNX导出

#### 导出方法：

完成部署后，修改`hppnet_dev/onnx_export.py`中的`model_path`,在此填入你的模型路径，然后运行此脚本。

导出仅支持hppnet官方发布的代码训练得到的模型（Dec 1, 2022 commit 85cfe09），以及本项目提供的修改版代码训练的模型。不支持在上述commit下原作者仓库内提供的模型。

**关于导出方案讨论：**

此导出方案直接导出了最小时间序列（512）但并未从最小模型单元导出，另外丢弃了时间上的隐藏状态（这个行为似乎对此模型影响不大，模型训练时也是随机切片音频，虽然有概率夹掉offset）。但最终效果来看几乎没丢失转录表现，就没有再去折腾了。

### 构建与分发

轻量化分发方案讨论：待完成

待完成

### 模型训练

#### Hppnet:

**1.自定义数据集制作：**

你需要准备16000采样率单声道wav与对应的mid文件，将它们放在同一文件夹下

打开`hppnet_dev\data_preprocess.ipynb`修改path为你的数据所在的文件夹，然后依次运行

完成后仅需要生成的h5文件即可训练。

**2.训练：**

编辑`hppnet_dev\train.py`在data_config中填写你的数据集（即上方生成的h5文件所在文件夹）路径，val与test数据可以仅提供少量。

训练配置默认为hppnet sp，可根据你的设备适当调整batch size与sequence_length

直接运行此py文件即可训练，如要训练其他规格模型，请修改装饰器（`@ex.config`为使用的配置，`@ex.named_config`为未使用配置，更多参考sacred文档）。

**继续上次进度：**

修改`resume_iteration`为上次保存的模型步数

注：sacred提供了命令行参数，也可不编辑此文件直接使用命令行传递参数。

## 更新日志

1. dev-2023-10-11
   1. [hppnet代码修改]：修改dataset为文件夹h5数据集，用于支持自定义数据集。
   2. [hppnet代码修改]：优化训练代码，重新排列了训练用参数，移除了未使用参数与代码，适配自定义数据集训练
   3. [hppnet代码修改]：移除每次运行默认创建随机输出目录。
   4. [hppnet代码修改]：修改导入方式，修复Windows下由于torch Dataloader多线程加载导致CQT被重复创建拖慢训练速度的问题。（为兼容模型保存加载等，此修复直接将workers改为0，另一个方案为：将CQT放入模型中，使其只会随着模型初始化时被创建一次，但此方案需要修改模型保存相关）
   5. [hppnet代码修改]：将模型padding参数从`‘same’`修改为等价的具体值（因为ONNX导出暂不支持same padding，已尝试拉取Pytorch PR，patch后发现又不支持在same padding下的dilation（麻了））
   6. [hppnet训练相关]：添加自定义数据集打包到h5工具
   7. [hppnet部署相关]：添加onnx导出代码
   8. [hppnet推理相关]：添加pytorch推理代码
   9. [MoeTrans后端]：修改后端代码适配开发环境
   10. [MoeTrans文档]：添加本地部署方法，添加ONNX导出方法，添加自定义数据集制作与训练方法

## TODO

1. 完善readme
2. 尝试节拍检测与钢琴左右手分离，note量化等更多后处理
3. 人声转录与多乐器转录
4. 更大的钢琴转录数据集
5. 尝试优化hppnet或尝试其他转录方案
6. 优化UI
7. 添加转录外的音乐相关功能，例如和声检测，乐谱草稿生成，乐谱OCR
8. 实时转录（感觉用处不大，先放这）

## 参考与引用

1. 转录算法：[HPPNet(GitHub)](https://github.com/WX-Wei/HPPNet)   [Arxiv](https://arxiv.org/abs/2208.14339v2)
2. 图标：《恋×シンアイ彼女》中的角色：姬野星奏

