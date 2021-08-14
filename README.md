# [AI训练营]PaddleX实现目标检测baseline

手把手教你基于PaddleX实现目标检测。你需要实现以下任务：

> 1. 配置数据集（数据集选择、数据处理）
> 2. 配置模型并训练
> 3. 项目跑通即可达到结业要求

# 一、数据集说明

本项目使用的数据集是：[[AI训练营]目标检测数据集合集](https://aistudio.baidu.com/aistudio/datasetdetail/103743)，包含口罩识别 、交通标志识别、火焰检测、锥桶识别以及中秋元素识别。

该数据集已加载至本环境中，位于：**data/data103743/objDataset.zip**


```python
# 解压数据集（解压一次即可，请勿重复解压）
!unzip -oq /home/aistudio/data/data103743/objDataset.zip
```

解压完成后，左侧文件夹处会多一个名为**objDataset**的文件夹，该文件夹下有5个子文件夹：
- **barricade**——Gazebo锥桶检测
- **facemask**——口罩检测
- **fire**——火焰检测
- **MidAutumn**——中秋元素检测
- **roadsign_voc**——交通路标检测

每个子文件夹下有2个文件夹，分别存放着图像（**JPEGImages**）和标注文件（**Annotations**），如下所示：


```python
# 查看数据集文件结构
!tree objDataset -L 2
```

    objDataset
    ├── barricade
    │   ├── Annotations
    │   └── JPEGImages
    ├── facemask
    │   ├── Annotations
    │   └── JPEGImages
    ├── fire
    │   ├── Annotations
    │   └── JPEGImages
    ├── MidAutumn
    │   ├── Annotations
    │   └── JPEGImages
    └── roadsign_voc
        ├── Annotations
        └── JPEGImages
    
    15 directories, 0 files


# 二、数据准备

本基线系统使用的数据格式是PascalVOC格式，开发者基于PaddleX开发目标检测模型时，无需对数据格式进行转换，开箱即用。

但为了进行训练，还需要将数据划分为训练集、验证集和测试集。划分之前首先需要**安装PaddleX**。


```python
# 安装PaddleX
!pip install paddlex
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Collecting paddlex
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/d6/a2/07435f4aa1e51fe22bdf06c95d03bf1b78b7bc6625adbb51e35dc0804cc7/paddlex-1.3.11-py3-none-any.whl (516kB)
    [K     |████████████████████████████████| 522kB 14.8MB/s eta 0:00:01
    [?25hRequirement already satisfied: sklearn in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (0.0)
    Requirement already satisfied: opencv-python in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (4.1.1.26)
    Requirement already satisfied: visualdl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (2.2.0)
    Collecting paddlehub==2.1.0 (from paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/7a/29/3bd0ca43c787181e9c22fe44b944b64d7fcb14ce66d3bf4602d9ad2ac76c/paddlehub-2.1.0-py3-none-any.whl (211kB)
    [K     |████████████████████████████████| 215kB 22.6MB/s eta 0:00:01
    [?25hRequirement already satisfied: tqdm in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (4.36.1)
    Requirement already satisfied: flask-cors in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (3.0.8)
    Collecting pycocotools; platform_system != "Windows" (from paddlex)
      Downloading https://mirror.baidu.com/pypi/packages/de/df/056875d697c45182ed6d2ae21f62015896fdb841906fe48e7268e791c467/pycocotools-2.0.2.tar.gz
    Collecting shapely>=1.7.0 (from paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/98/f8/db4d3426a1aba9d5dfcc83ed5a3e2935d2b1deb73d350642931791a61c37/Shapely-1.7.1-cp37-cp37m-manylinux1_x86_64.whl (1.0MB)
    [K     |████████████████████████████████| 1.0MB 11.8MB/s eta 0:00:01
    [?25hRequirement already satisfied: colorama in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (0.4.4)
    Requirement already satisfied: psutil in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (5.7.2)
    Collecting paddleslim==1.1.1 (from paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/d1/77/e257227bed9a70ff0d35a4a3c4e70ac2d2362c803834c4c52018f7c4b762/paddleslim-1.1.1-py2.py3-none-any.whl (145kB)
    [K     |████████████████████████████████| 153kB 60.5MB/s eta 0:00:01
    [?25hCollecting xlwt (from paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/44/48/def306413b25c3d01753603b1a222a011b8621aed27cd7f89cbc27e6b0f4/xlwt-1.3.0-py2.py3-none-any.whl (99kB)
    [K     |████████████████████████████████| 102kB 25.6MB/s ta 0:00:01
    [?25hRequirement already satisfied: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (5.1.2)
    Requirement already satisfied: scikit-learn in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from sklearn->paddlex) (0.24.2)
    Requirement already satisfied: numpy>=1.14.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from opencv-python->paddlex) (1.20.3)
    Requirement already satisfied: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.0.0)
    Requirement already satisfied: pandas in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.1.5)
    Requirement already satisfied: six>=1.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.15.0)
    Requirement already satisfied: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (3.8.2)
    Requirement already satisfied: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (0.8.53)
    Requirement already satisfied: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (2.22.0)
    Requirement already satisfied: flask>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.1.1)
    Requirement already satisfied: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.21.0)
    Requirement already satisfied: matplotlib in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (2.2.3)
    Requirement already satisfied: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (0.7.1.1)
    Requirement already satisfied: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (3.14.0)
    Requirement already satisfied: Pillow>=7.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (7.1.2)
    Requirement already satisfied: gitpython in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (3.1.14)
    Requirement already satisfied: paddlenlp>=2.0.0rc5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (2.0.1)
    Requirement already satisfied: packaging in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (20.9)
    Requirement already satisfied: gunicorn>=19.10.0; sys_platform != "win32" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (20.0.4)
    Requirement already satisfied: rarfile in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (3.1)
    Requirement already satisfied: filelock in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (3.0.12)
    Collecting paddle2onnx>=0.5.1 (from paddlehub==2.1.0->paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/37/80/aa6134b5f36aea45dc1b363e7af941dccabe4d7e167ac391ff046f34baf1/paddle2onnx-0.7-py3-none-any.whl (94kB)
    [K     |████████████████████████████████| 102kB 31.9MB/s ta 0:00:01
    [?25hRequirement already satisfied: colorlog in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (4.1.0)
    Requirement already satisfied: pyzmq in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (18.1.1)
    Requirement already satisfied: easydict in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (1.9)
    Requirement already satisfied: setuptools>=18.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools; platform_system != "Windows"->paddlex) (56.2.0)
    Requirement already satisfied: cython>=0.27.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools; platform_system != "Windows"->paddlex) (0.29)
    Requirement already satisfied: scipy>=0.19.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn->sklearn->paddlex) (1.6.3)
    Requirement already satisfied: joblib>=0.11 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn->sklearn->paddlex) (0.14.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn->sklearn->paddlex) (2.1.0)
    Requirement already satisfied: Jinja2>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0->paddlex) (2.10.1)
    Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0->paddlex) (2019.3)
    Requirement already satisfied: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0->paddlex) (2.8.0)
    Requirement already satisfied: python-dateutil>=2.7.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pandas->visualdl>=2.0.0->paddlex) (2.8.0)
    Requirement already satisfied: importlib-metadata; python_version < "3.8" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (0.23)
    Requirement already satisfied: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (0.6.1)
    Requirement already satisfied: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (2.2.0)
    Requirement already satisfied: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (2.6.0)
    Requirement already satisfied: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0->paddlex) (0.18.0)
    Requirement already satisfied: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0->paddlex) (3.9.9)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (2019.9.11)
    Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (2.8)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (1.25.6)
    Requirement already satisfied: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->paddlex) (7.0)
    Requirement already satisfied: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->paddlex) (1.1.0)
    Requirement already satisfied: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->paddlex) (0.16.0)
    Requirement already satisfied: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (1.4.10)
    Requirement already satisfied: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (0.10.0)
    Requirement already satisfied: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (1.3.4)
    Requirement already satisfied: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (16.7.9)
    Requirement already satisfied: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (2.0.1)
    Requirement already satisfied: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (1.3.0)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0->paddlex) (0.10.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0->paddlex) (2.4.2)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0->paddlex) (1.1.0)
    Requirement already satisfied: gitdb<5,>=4.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gitpython->paddlehub==2.1.0->paddlex) (4.0.5)
    Requirement already satisfied: seqeval in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (1.2.2)
    Requirement already satisfied: jieba in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (0.42.1)
    Requirement already satisfied: multiprocess in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (0.70.11.1)
    Requirement already satisfied: h5py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (2.9.0)
    Requirement already satisfied: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.5->Flask-Babel>=1.0.0->visualdl>=2.0.0->paddlex) (1.1.1)
    Requirement already satisfied: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->flake8>=3.7.9->visualdl>=2.0.0->paddlex) (0.6.0)
    Requirement already satisfied: smmap<4,>=3.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gitdb<5,>=4.0.1->gitpython->paddlehub==2.1.0->paddlex) (3.0.5)
    Requirement already satisfied: dill>=0.3.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from multiprocess->paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (0.3.3)
    Requirement already satisfied: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->flake8>=3.7.9->visualdl>=2.0.0->paddlex) (7.2.0)
    Building wheels for collected packages: pycocotools
      Building wheel for pycocotools (setup.py) ... [?25ldone
    [?25h  Created wheel for pycocotools: filename=pycocotools-2.0.2-cp37-cp37m-linux_x86_64.whl size=278365 sha256=98736b864c36fc5d3ac3e6a0988a1bcb0dd139ee60d2f63bcf35bb783cb1e78f
      Stored in directory: /home/aistudio/.cache/pip/wheels/fb/44/67/8baa69040569b1edbd7776ec6f82c387663e724908aaa60963
    Successfully built pycocotools
    Installing collected packages: paddle2onnx, paddlehub, pycocotools, shapely, paddleslim, xlwt, paddlex
      Found existing installation: paddlehub 2.0.4
        Uninstalling paddlehub-2.0.4:
          Successfully uninstalled paddlehub-2.0.4
    Successfully installed paddle2onnx-0.7 paddlehub-2.1.0 paddleslim-1.1.1 paddlex-1.3.11 pycocotools-2.0.2 shapely-1.7.1 xlwt-1.3.0


使用如下命令即可将数据划分为70%训练集，20%验证集和10%的测试集。


```python
# 划分数据集，使用中秋元素数据集，验证集放小一点
!paddlex --split_dataset --format VOC --dataset_dir objDataset/MidAutumn --val_value 0.1 --test_value 0.1
```

    Dataset Split Done.[0m
    [0mTrain samples: 262[0m
    [0mEval samples: 32[0m
    [0mTest samples: 32[0m
    [0mSplit files saved in objDataset/MidAutumn[0m
    [0m[0m

划分完成后，该数据集下会生成**labels.txt**, **train_list.txt**, **val_list.txt**和**test_list.txt**，分别存储类别信息，训练样本列表，验证样本列表，测试样本列表。如下图所示：

![](https://ai-studio-static-online.cdn.bcebos.com/d29a92b4cfc34b0097ef46dbbc8562af824387889f224948ae49283e0adee19d)

在这里，**你需要将path to dataset部分替换成你选择的数据集路径**。在左侧文件夹处，将鼠标放到你想选择的数据集文件夹上，会出现三个图标，第一个图标表示复制该文件夹路径，点击即可获得该文件夹路径，用这个路径替换path to dataset即可。

![](https://ai-studio-static-online.cdn.bcebos.com/c28ed88586644f64b34709a592fea0b97ec80470c0e041fd9aa6b8da21c8e283)


# 三、数据预处理

在训练模型之前，对目标检测任务的数据进行操作，从而提升模型效果。可用于数据处理的API有：
- **Normalize**：对图像进行归一化
- **ResizeByShort**：根据图像的短边调整图像大小
- **RandomHorizontalFlip**：以一定的概率对图像进行随机水平翻转
- **RandomDistort**：以一定的概率对图像进行随机像素内容变换

更多关于数据处理的API及使用说明可查看文档：
[https://paddlex.readthedocs.io/zh_CN/release-1.3/apis/transforms/det_transforms.html](https://paddlex.readthedocs.io/zh_CN/release-1.3/apis/transforms/det_transforms.html)


```python
from paddlex.det import transforms

# 定义训练和验证时的transforms
# API说明 https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html
train_transforms = transforms.Compose([
    # 此处需要补充图像预处理代码
    transforms.RandomHorizontalFlip(),
    transforms.RandomExpand(),
    transforms.RandomCrop(), # 新加入
    transforms.RandomDistort(brightness_range=1.2, brightness_prob=0.2, hue_range=0.8, hue_prob=0.2),
    transforms.Resize(target_size=608, interp='RANDOM'),
    transforms.Normalize()
])

eval_transforms = transforms.Compose([
    # 此处需要补充图像预处理代码
    transforms.Resize(target_size=608, interp='CUBIC'),
    transforms.Normalize()
])
```

读取PascalVOC格式的检测数据集，并对样本进行相应的处理。


```python
import paddlex as pdx

# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-vocdetection
# 不得不说CV dataset封装比nlp方便
train_dataset = pdx.datasets.VOCDetection(
    data_dir='objDataset/MidAutumn',
    file_list='objDataset/MidAutumn/train_list.txt',
    label_list='objDataset/MidAutumn/labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.VOCDetection(
    data_dir='objDataset/MidAutumn',
    file_list='objDataset/MidAutumn/val_list.txt',
    label_list='objDataset/MidAutumn/labels.txt',
    transforms=eval_transforms)
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/__init__.py:107: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import MutableMapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/rcsetup.py:20: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Iterable, Mapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/colors.py:53: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Sized


    2021-08-11 13:46:06 [INFO]	Starting to read file list from dataset...
    2021-08-11 13:46:07 [INFO]	262 samples in file objDataset/MidAutumn/train_list.txt
    creating index...
    index created!
    2021-08-11 13:46:07 [INFO]	Starting to read file list from dataset...
    2021-08-11 13:46:07 [INFO]	32 samples in file objDataset/MidAutumn/val_list.txt
    creating index...
    index created!


需要注意的是：
- **data_dir** (str): 数据集所在的目录路径。
- **file_list** (str): 描述数据集图片文件和对应标注文件的文件路径（文本内每行路径为相对data_dir的相对路径）。
- **label_list** (str): 描述数据集包含的类别信息文件路径。

需要将第二步数据准备时生成的labels.txt, train_list.txt, val_list.txt和test_list.txt配置到以上变量中，如下图所示：

![](https://ai-studio-static-online.cdn.bcebos.com/6462f7811da3436290948e5dde0c497d6ae51bcfe1904e0a863ff032363a4448)


# 四、模型训练

PaddleX目前提供了FasterRCNN和YOLOv3两种检测结构，多种backbone模型。本基线系统以骨干网络为MobileNetV1的YOLOv3算法为例。


```python
# 初始化模型
# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-yolov3

# 此处需要补充目标检测模型代码
model = pdx.det.PPYOLO(num_classes=len(train_dataset.labels), backbone='ResNet50_vd_ssld')
```


```python
# 模型训练
# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#id1
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html

# 此处需要补充模型训练参数
model.train(
    num_epochs=100,
    train_dataset=train_dataset,
    train_batch_size=24, # 这个数据集应该最多能加到24
    eval_dataset=eval_dataset,
    learning_rate=0.000375,
    warmup_steps=200,
    warmup_start_lr=0.0,
    # lr_decay_epochs=[210, 240],
    pretrain_weights='output/ppyolo/best_model', # 加入RandomCrop再跑一会儿
    save_dir='output/ppyolo')
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/framework.py:689: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      elif dtype == np.bool:
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/framework.py:2341: UserWarning: The Attr(force_cpu) of Op(fill_constant) will be deprecated in the future, please use 'device_guard' instead. 'device_guard' has higher priority when they are used at the same time.
      "used at the same time." % type)
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/ops.py:131
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:155
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:172
    The behavior of expression A - B has been unified with elementwise_sub(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_sub(X, Y, axis=0) instead of A - B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:172
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:174
    The behavior of expression A - B has been unified with elementwise_sub(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_sub(X, Y, axis=0) instead of A - B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:174
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:178
    The behavior of expression A - B has been unified with elementwise_sub(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_sub(X, Y, axis=0) instead of A - B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:178
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:180
    The behavior of expression A - B has been unified with elementwise_sub(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_sub(X, Y, axis=0) instead of A - B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:180
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/iou_loss.py:216
    The behavior of expression A - B has been unified with elementwise_sub(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_sub(X, Y, axis=0) instead of A - B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/iou_loss.py:217
    The behavior of expression A - B has been unified with elementwise_sub(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_sub(X, Y, axis=0) instead of A - B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/iou_loss.py:218
    The behavior of expression A + B has been unified with elementwise_add(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_add(X, Y, axis=0) instead of A + B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/iou_loss.py:219
    The behavior of expression A + B has been unified with elementwise_add(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_add(X, Y, axis=0) instead of A + B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/iou_loss.py:97
    The behavior of expression A - B has been unified with elementwise_sub(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_sub(X, Y, axis=0) instead of A - B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/iou_loss.py:97
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/iou_loss.py:99
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/iou_loss.py:101
    The behavior of expression A - B has been unified with elementwise_sub(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_sub(X, Y, axis=0) instead of A - B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/iou_loss.py:101
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/iou_loss.py:101
    The behavior of expression A + B has been unified with elementwise_add(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_add(X, Y, axis=0) instead of A + B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/iou_loss.py:102
    The behavior of expression A / B has been unified with elementwise_div(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_div(X, Y, axis=0) instead of A / B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/iou_loss.py:79
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:186
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:194
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:349
    The behavior of expression A - B has been unified with elementwise_sub(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_sub(X, Y, axis=0) instead of A - B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:350
    The behavior of expression A - B has been unified with elementwise_sub(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_sub(X, Y, axis=0) instead of A - B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:351
    The behavior of expression A + B has been unified with elementwise_add(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_add(X, Y, axis=0) instead of A + B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:352
    The behavior of expression A + B has been unified with elementwise_add(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_add(X, Y, axis=0) instead of A + B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:383
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:385
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:209
    The behavior of expression A + B has been unified with elementwise_add(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_add(X, Y, axis=0) instead of A + B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:210
    The behavior of expression A + B has been unified with elementwise_add(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_add(X, Y, axis=0) instead of A + B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:212
    The behavior of expression A + B has been unified with elementwise_add(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_add(X, Y, axis=0) instead of A + B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/iou_aware.py:64
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/iou_aware.py:40
    The behavior of expression A / B has been unified with elementwise_div(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_div(X, Y, axis=0) instead of A / B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))


    2021-08-11 13:50:32 [INFO]	Load pretrain weights from output/ppyolo/best_model.
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_0.w_0 doesn't match.(Pretrained: (16, 3, 19, 19), Actual: (24, 3, 19, 19))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_1.w_0 doesn't match.(Pretrained: (16, 3, 19, 19), Actual: (24, 3, 19, 19))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_2.w_0 doesn't match.(Pretrained: (16, 3, 19, 19), Actual: (24, 3, 19, 19))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_3.w_0 doesn't match.(Pretrained: (16, 3, 19, 19), Actual: (24, 3, 19, 19))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_4.w_0 doesn't match.(Pretrained: (16, 3, 19, 19), Actual: (24, 3, 19, 19))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_5.w_0 doesn't match.(Pretrained: (16, 3, 19, 19), Actual: (24, 3, 19, 19))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_6.w_0 doesn't match.(Pretrained: (16, 3, 19, 19), Actual: (24, 3, 19, 19))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_7.w_0 doesn't match.(Pretrained: (16, 3, 19, 19), Actual: (24, 3, 19, 19))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_8.w_0 doesn't match.(Pretrained: (16, 3, 19, 19), Actual: (24, 3, 19, 19))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_9.w_0 doesn't match.(Pretrained: (16, 3, 19, 19), Actual: (24, 3, 19, 19))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_10.w_0 doesn't match.(Pretrained: (16, 3, 19, 19), Actual: (24, 3, 19, 19))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_11.w_0 doesn't match.(Pretrained: (16, 3, 19, 19), Actual: (24, 3, 19, 19))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_12.w_0 doesn't match.(Pretrained: (16, 3, 19, 19), Actual: (24, 3, 19, 19))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_13.w_0 doesn't match.(Pretrained: (16, 3, 19, 19), Actual: (24, 3, 19, 19))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_14.w_0 doesn't match.(Pretrained: (16, 3, 19, 19), Actual: (24, 3, 19, 19))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_15.w_0 doesn't match.(Pretrained: (16, 3, 19, 19), Actual: (24, 3, 19, 19))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_16.w_0 doesn't match.(Pretrained: (16, 3, 38, 38), Actual: (24, 3, 38, 38))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_17.w_0 doesn't match.(Pretrained: (16, 3, 38, 38), Actual: (24, 3, 38, 38))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_18.w_0 doesn't match.(Pretrained: (16, 3, 38, 38), Actual: (24, 3, 38, 38))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_19.w_0 doesn't match.(Pretrained: (16, 3, 38, 38), Actual: (24, 3, 38, 38))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_20.w_0 doesn't match.(Pretrained: (16, 3, 38, 38), Actual: (24, 3, 38, 38))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_21.w_0 doesn't match.(Pretrained: (16, 3, 38, 38), Actual: (24, 3, 38, 38))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_22.w_0 doesn't match.(Pretrained: (16, 3, 38, 38), Actual: (24, 3, 38, 38))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_23.w_0 doesn't match.(Pretrained: (16, 3, 38, 38), Actual: (24, 3, 38, 38))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_24.w_0 doesn't match.(Pretrained: (16, 3, 38, 38), Actual: (24, 3, 38, 38))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_25.w_0 doesn't match.(Pretrained: (16, 3, 38, 38), Actual: (24, 3, 38, 38))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_26.w_0 doesn't match.(Pretrained: (16, 3, 38, 38), Actual: (24, 3, 38, 38))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_27.w_0 doesn't match.(Pretrained: (16, 3, 38, 38), Actual: (24, 3, 38, 38))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_28.w_0 doesn't match.(Pretrained: (16, 3, 38, 38), Actual: (24, 3, 38, 38))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_29.w_0 doesn't match.(Pretrained: (16, 3, 38, 38), Actual: (24, 3, 38, 38))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_30.w_0 doesn't match.(Pretrained: (16, 3, 38, 38), Actual: (24, 3, 38, 38))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_31.w_0 doesn't match.(Pretrained: (16, 3, 38, 38), Actual: (24, 3, 38, 38))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_32.w_0 doesn't match.(Pretrained: (16, 3, 76, 76), Actual: (24, 3, 76, 76))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_33.w_0 doesn't match.(Pretrained: (16, 3, 76, 76), Actual: (24, 3, 76, 76))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_34.w_0 doesn't match.(Pretrained: (16, 3, 76, 76), Actual: (24, 3, 76, 76))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_35.w_0 doesn't match.(Pretrained: (16, 3, 76, 76), Actual: (24, 3, 76, 76))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_36.w_0 doesn't match.(Pretrained: (16, 3, 76, 76), Actual: (24, 3, 76, 76))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_37.w_0 doesn't match.(Pretrained: (16, 3, 76, 76), Actual: (24, 3, 76, 76))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_38.w_0 doesn't match.(Pretrained: (16, 3, 76, 76), Actual: (24, 3, 76, 76))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_39.w_0 doesn't match.(Pretrained: (16, 3, 76, 76), Actual: (24, 3, 76, 76))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_40.w_0 doesn't match.(Pretrained: (16, 3, 76, 76), Actual: (24, 3, 76, 76))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_41.w_0 doesn't match.(Pretrained: (16, 3, 76, 76), Actual: (24, 3, 76, 76))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_42.w_0 doesn't match.(Pretrained: (16, 3, 76, 76), Actual: (24, 3, 76, 76))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_43.w_0 doesn't match.(Pretrained: (16, 3, 76, 76), Actual: (24, 3, 76, 76))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_44.w_0 doesn't match.(Pretrained: (16, 3, 76, 76), Actual: (24, 3, 76, 76))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_45.w_0 doesn't match.(Pretrained: (16, 3, 76, 76), Actual: (24, 3, 76, 76))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_46.w_0 doesn't match.(Pretrained: (16, 3, 76, 76), Actual: (24, 3, 76, 76))
    2021-08-11 13:50:34 [WARNING]	[SKIP] Shape of pretrained weight create_parameter_47.w_0 doesn't match.(Pretrained: (16, 3, 76, 76), Actual: (24, 3, 76, 76))
    2021-08-11 13:50:34 [INFO]	There are 392 varaibles in output/ppyolo/best_model are loaded.
    2021-08-11 13:50:52 [INFO]	[TRAIN] Epoch=1/100, Step=2/10, loss=6.965278, lr=2e-06, time_each_step=8.77s, eta=2:27:23
    2021-08-11 13:50:54 [INFO]	[TRAIN] Epoch=1/100, Step=4/10, loss=6.48839, lr=6e-06, time_each_step=4.87s, eta=1:21:42
    2021-08-11 13:50:56 [INFO]	[TRAIN] Epoch=1/100, Step=6/10, loss=8.311458, lr=9e-06, time_each_step=3.54s, eta=0:59:18
    2021-08-11 13:50:57 [INFO]	[TRAIN] Epoch=1/100, Step=8/10, loss=5.678126, lr=1.3e-05, time_each_step=2.77s, eta=0:46:19
    2021-08-11 13:50:58 [INFO]	[TRAIN] Epoch=1/100, Step=10/10, loss=4.159629, lr=1.7e-05, time_each_step=2.36s, eta=0:39:20
    2021-08-11 13:50:58 [INFO]	[TRAIN] Epoch 1 finished, loss=5.571951, lr=8e-06 .
    2021-08-11 13:51:09 [INFO]	[TRAIN] Epoch=2/100, Step=2/10, loss=4.970394, lr=2.1e-05, time_each_step=2.85s, eta=0:39:24
    2021-08-11 13:51:11 [INFO]	[TRAIN] Epoch=2/100, Step=4/10, loss=4.622717, lr=2.4e-05, time_each_step=2.59s, eta=0:39:14
    2021-08-11 13:51:13 [INFO]	[TRAIN] Epoch=2/100, Step=6/10, loss=4.616879, lr=2.8e-05, time_each_step=2.41s, eta=0:39:7
    2021-08-11 13:51:15 [INFO]	[TRAIN] Epoch=2/100, Step=8/10, loss=6.409221, lr=3.2e-05, time_each_step=2.23s, eta=0:39:0
    2021-08-11 13:51:16 [INFO]	[TRAIN] Epoch=2/100, Step=10/10, loss=6.691083, lr=3.6e-05, time_each_step=2.09s, eta=0:38:54
    2021-08-11 13:51:16 [INFO]	[TRAIN] Epoch 2 finished, loss=5.753575, lr=2.7e-05 .
    2021-08-11 13:51:27 [INFO]	[TRAIN] Epoch=3/100, Step=2/10, loss=5.567945, lr=3.9e-05, time_each_step=1.77s, eta=0:30:4
    2021-08-11 13:51:30 [INFO]	[TRAIN] Epoch=3/100, Step=4/10, loss=5.867038, lr=4.3e-05, time_each_step=1.81s, eta=0:30:1
    2021-08-11 13:51:32 [INFO]	[TRAIN] Epoch=3/100, Step=6/10, loss=4.603502, lr=4.7e-05, time_each_step=1.8s, eta=0:29:58
    2021-08-11 13:51:33 [INFO]	[TRAIN] Epoch=3/100, Step=8/10, loss=5.137157, lr=5.1e-05, time_each_step=1.81s, eta=0:29:54
    2021-08-11 13:51:35 [INFO]	[TRAIN] Epoch=3/100, Step=10/10, loss=5.689548, lr=5.4e-05, time_each_step=1.82s, eta=0:29:51
    2021-08-11 13:51:35 [INFO]	[TRAIN] Epoch 3 finished, loss=5.284009, lr=4.6e-05 .
    2021-08-11 13:51:44 [INFO]	[TRAIN] Epoch=4/100, Step=2/10, loss=4.282058, lr=5.8e-05, time_each_step=1.77s, eta=0:29:28
    2021-08-11 13:51:47 [INFO]	[TRAIN] Epoch=4/100, Step=4/10, loss=5.633337, lr=6.2e-05, time_each_step=1.83s, eta=0:29:26
    2021-08-11 13:51:49 [INFO]	[TRAIN] Epoch=4/100, Step=6/10, loss=6.758499, lr=6.6e-05, time_each_step=1.81s, eta=0:29:22
    2021-08-11 13:51:50 [INFO]	[TRAIN] Epoch=4/100, Step=8/10, loss=3.860156, lr=6.9e-05, time_each_step=1.79s, eta=0:29:18
    2021-08-11 13:51:52 [INFO]	[TRAIN] Epoch=4/100, Step=10/10, loss=4.464845, lr=7.3e-05, time_each_step=1.78s, eta=0:29:14
    2021-08-11 13:51:52 [INFO]	[TRAIN] Epoch 4 finished, loss=5.060145, lr=6.5e-05 .
    2021-08-11 13:52:01 [INFO]	[TRAIN] Epoch=5/100, Step=2/10, loss=6.193164, lr=7.7e-05, time_each_step=1.67s, eta=0:28:8
    2021-08-11 13:52:04 [INFO]	[TRAIN] Epoch=5/100, Step=4/10, loss=5.496, lr=8.1e-05, time_each_step=1.66s, eta=0:28:4
    2021-08-11 13:52:06 [INFO]	[TRAIN] Epoch=5/100, Step=6/10, loss=10.098552, lr=8.4e-05, time_each_step=1.69s, eta=0:28:1
    2021-08-11 13:52:07 [INFO]	[TRAIN] Epoch=5/100, Step=8/10, loss=4.9667, lr=8.8e-05, time_each_step=1.69s, eta=0:27:58
    2021-08-11 13:52:08 [INFO]	[TRAIN] Epoch=5/100, Step=10/10, loss=5.40681, lr=9.2e-05, time_each_step=1.67s, eta=0:27:55
    2021-08-11 13:52:08 [INFO]	[TRAIN] Epoch 5 finished, loss=5.875248, lr=8.3e-05 .
    2021-08-11 13:52:17 [INFO]	[TRAIN] Epoch=6/100, Step=2/10, loss=6.48116, lr=9.6e-05, time_each_step=1.65s, eta=0:25:24
    2021-08-11 13:52:20 [INFO]	[TRAIN] Epoch=6/100, Step=4/10, loss=5.24794, lr=9.9e-05, time_each_step=1.64s, eta=0:25:21
    2021-08-11 13:52:22 [INFO]	[TRAIN] Epoch=6/100, Step=6/10, loss=5.007981, lr=0.000103, time_each_step=1.61s, eta=0:25:17
    2021-08-11 13:52:23 [INFO]	[TRAIN] Epoch=6/100, Step=8/10, loss=4.347806, lr=0.000107, time_each_step=1.65s, eta=0:25:14
    2021-08-11 13:52:25 [INFO]	[TRAIN] Epoch=6/100, Step=10/10, loss=5.045381, lr=0.000111, time_each_step=1.64s, eta=0:25:11
    2021-08-11 13:52:25 [INFO]	[TRAIN] Epoch 6 finished, loss=5.161837, lr=0.000102 .
    2021-08-11 13:52:34 [INFO]	[TRAIN] Epoch=7/100, Step=2/10, loss=3.550832, lr=0.000114, time_each_step=1.66s, eta=0:26:41
    2021-08-11 13:52:36 [INFO]	[TRAIN] Epoch=7/100, Step=4/10, loss=3.778273, lr=0.000118, time_each_step=1.63s, eta=0:26:38
    2021-08-11 13:52:38 [INFO]	[TRAIN] Epoch=7/100, Step=6/10, loss=4.050705, lr=0.000122, time_each_step=1.61s, eta=0:26:34
    2021-08-11 13:52:40 [INFO]	[TRAIN] Epoch=7/100, Step=8/10, loss=6.162713, lr=0.000126, time_each_step=1.65s, eta=0:26:31
    2021-08-11 13:52:42 [INFO]	[TRAIN] Epoch=7/100, Step=10/10, loss=5.854975, lr=0.000129, time_each_step=1.68s, eta=0:26:28
    2021-08-11 13:52:42 [INFO]	[TRAIN] Epoch 7 finished, loss=5.203119, lr=0.000121 .
    2021-08-11 13:52:50 [INFO]	[TRAIN] Epoch=8/100, Step=2/10, loss=5.56421, lr=0.000133, time_each_step=1.66s, eta=0:26:11
    2021-08-11 13:52:52 [INFO]	[TRAIN] Epoch=8/100, Step=4/10, loss=7.741617, lr=0.000137, time_each_step=1.61s, eta=0:26:7
    2021-08-11 13:52:54 [INFO]	[TRAIN] Epoch=8/100, Step=6/10, loss=3.697447, lr=0.000141, time_each_step=1.63s, eta=0:26:4
    2021-08-11 13:52:55 [INFO]	[TRAIN] Epoch=8/100, Step=8/10, loss=5.229725, lr=0.000144, time_each_step=1.59s, eta=0:26:0
    2021-08-11 13:52:56 [INFO]	[TRAIN] Epoch=8/100, Step=10/10, loss=3.781785, lr=0.000148, time_each_step=1.57s, eta=0:25:57
    2021-08-11 13:52:56 [INFO]	[TRAIN] Epoch 8 finished, loss=5.079801, lr=0.00014 .
    2021-08-11 13:53:10 [INFO]	[TRAIN] Epoch=9/100, Step=2/10, loss=4.585756, lr=0.000152, time_each_step=1.78s, eta=0:22:46
    2021-08-11 13:53:13 [INFO]	[TRAIN] Epoch=9/100, Step=4/10, loss=5.298633, lr=0.000156, time_each_step=1.83s, eta=0:22:43
    2021-08-11 13:53:15 [INFO]	[TRAIN] Epoch=9/100, Step=6/10, loss=7.462276, lr=0.000159, time_each_step=1.86s, eta=0:22:40
    2021-08-11 13:53:16 [INFO]	[TRAIN] Epoch=9/100, Step=8/10, loss=3.95061, lr=0.000163, time_each_step=1.83s, eta=0:22:36
    2021-08-11 13:53:18 [INFO]	[TRAIN] Epoch=9/100, Step=10/10, loss=5.093475, lr=0.000167, time_each_step=1.81s, eta=0:22:32
    2021-08-11 13:53:18 [INFO]	[TRAIN] Epoch 9 finished, loss=5.316053, lr=0.000158 .
    2021-08-11 13:53:27 [INFO]	[TRAIN] Epoch=10/100, Step=2/10, loss=7.18825, lr=0.000171, time_each_step=1.86s, eta=0:32:58
    2021-08-11 13:53:30 [INFO]	[TRAIN] Epoch=10/100, Step=4/10, loss=5.255402, lr=0.000174, time_each_step=1.9s, eta=0:32:55
    2021-08-11 13:53:33 [INFO]	[TRAIN] Epoch=10/100, Step=6/10, loss=7.961398, lr=0.000178, time_each_step=1.92s, eta=0:32:52
    2021-08-11 13:53:34 [INFO]	[TRAIN] Epoch=10/100, Step=8/10, loss=5.397244, lr=0.000182, time_each_step=1.94s, eta=0:32:48
    2021-08-11 13:53:35 [INFO]	[TRAIN] Epoch=10/100, Step=10/10, loss=3.81525, lr=0.000186, time_each_step=1.93s, eta=0:32:44
    2021-08-11 13:53:35 [INFO]	[TRAIN] Epoch 10 finished, loss=5.389094, lr=0.000177 .
    2021-08-11 13:53:45 [INFO]	[TRAIN] Epoch=11/100, Step=2/10, loss=6.684228, lr=0.000189, time_each_step=1.76s, eta=0:25:52
    2021-08-11 13:53:47 [INFO]	[TRAIN] Epoch=11/100, Step=4/10, loss=5.193317, lr=0.000193, time_each_step=1.73s, eta=0:25:48
    2021-08-11 13:53:50 [INFO]	[TRAIN] Epoch=11/100, Step=6/10, loss=4.069492, lr=0.000197, time_each_step=1.77s, eta=0:25:45
    2021-08-11 13:53:52 [INFO]	[TRAIN] Epoch=11/100, Step=8/10, loss=4.807329, lr=0.000201, time_each_step=1.79s, eta=0:25:42
    2021-08-11 13:53:53 [INFO]	[TRAIN] Epoch=11/100, Step=10/10, loss=6.294554, lr=0.000204, time_each_step=1.77s, eta=0:25:38
    2021-08-11 13:53:53 [INFO]	[TRAIN] Epoch 11 finished, loss=5.197126, lr=0.000196 .
    2021-08-11 13:54:02 [INFO]	[TRAIN] Epoch=12/100, Step=2/10, loss=5.698996, lr=0.000208, time_each_step=1.7s, eta=0:27:17
    2021-08-11 13:54:04 [INFO]	[TRAIN] Epoch=12/100, Step=4/10, loss=5.993181, lr=0.000212, time_each_step=1.68s, eta=0:27:13
    2021-08-11 13:54:06 [INFO]	[TRAIN] Epoch=12/100, Step=6/10, loss=7.765477, lr=0.000216, time_each_step=1.67s, eta=0:27:10
    2021-08-11 13:54:07 [INFO]	[TRAIN] Epoch=12/100, Step=8/10, loss=4.822263, lr=0.000219, time_each_step=1.66s, eta=0:27:6
    2021-08-11 13:54:09 [INFO]	[TRAIN] Epoch=12/100, Step=10/10, loss=5.704455, lr=0.000223, time_each_step=1.7s, eta=0:27:3
    2021-08-11 13:54:09 [INFO]	[TRAIN] Epoch 12 finished, loss=5.397884, lr=0.000215 .
    2021-08-11 13:54:19 [INFO]	[TRAIN] Epoch=13/100, Step=2/10, loss=6.184244, lr=0.000227, time_each_step=1.7s, eta=0:23:12
    2021-08-11 13:54:23 [INFO]	[TRAIN] Epoch=13/100, Step=4/10, loss=6.893554, lr=0.000231, time_each_step=1.77s, eta=0:23:10
    2021-08-11 13:54:24 [INFO]	[TRAIN] Epoch=13/100, Step=6/10, loss=4.406001, lr=0.000234, time_each_step=1.68s, eta=0:23:5
    2021-08-11 13:54:25 [INFO]	[TRAIN] Epoch=13/100, Step=8/10, loss=5.075153, lr=0.000238, time_each_step=1.66s, eta=0:23:1
    2021-08-11 13:54:27 [INFO]	[TRAIN] Epoch=13/100, Step=10/10, loss=4.091408, lr=0.000242, time_each_step=1.69s, eta=0:22:58
    2021-08-11 13:54:27 [INFO]	[TRAIN] Epoch 13 finished, loss=5.481491, lr=0.000233 .
    2021-08-11 13:54:39 [INFO]	[TRAIN] Epoch=14/100, Step=2/10, loss=5.69324, lr=0.000246, time_each_step=1.87s, eta=0:26:27
    2021-08-11 13:54:42 [INFO]	[TRAIN] Epoch=14/100, Step=4/10, loss=4.204391, lr=0.000249, time_each_step=1.89s, eta=0:26:23
    2021-08-11 13:54:44 [INFO]	[TRAIN] Epoch=14/100, Step=6/10, loss=3.805285, lr=0.000253, time_each_step=1.91s, eta=0:26:20
    2021-08-11 13:54:46 [INFO]	[TRAIN] Epoch=14/100, Step=8/10, loss=5.961231, lr=0.000257, time_each_step=1.93s, eta=0:26:16
    2021-08-11 13:54:47 [INFO]	[TRAIN] Epoch=14/100, Step=10/10, loss=5.370362, lr=0.000261, time_each_step=1.93s, eta=0:26:12
    2021-08-11 13:54:47 [INFO]	[TRAIN] Epoch 14 finished, loss=5.433533, lr=0.000252 .
    2021-08-11 13:54:55 [INFO]	[TRAIN] Epoch=15/100, Step=2/10, loss=6.409556, lr=0.000264, time_each_step=1.81s, eta=0:29:42
    2021-08-11 13:54:58 [INFO]	[TRAIN] Epoch=15/100, Step=4/10, loss=7.584604, lr=0.000268, time_each_step=1.78s, eta=0:29:38
    2021-08-11 13:55:01 [INFO]	[TRAIN] Epoch=15/100, Step=6/10, loss=3.278593, lr=0.000272, time_each_step=1.82s, eta=0:29:35
    2021-08-11 13:55:02 [INFO]	[TRAIN] Epoch=15/100, Step=8/10, loss=5.143621, lr=0.000276, time_each_step=1.81s, eta=0:29:31
    2021-08-11 13:55:02 [INFO]	[TRAIN] Epoch=15/100, Step=10/10, loss=5.391383, lr=0.000279, time_each_step=1.77s, eta=0:29:27
    2021-08-11 13:55:02 [INFO]	[TRAIN] Epoch 15 finished, loss=5.330545, lr=0.000271 .
    2021-08-11 13:55:18 [INFO]	[TRAIN] Epoch=16/100, Step=2/10, loss=5.313156, lr=0.000283, time_each_step=1.97s, eta=0:21:24
    2021-08-11 13:55:20 [INFO]	[TRAIN] Epoch=16/100, Step=4/10, loss=5.998614, lr=0.000287, time_each_step=1.9s, eta=0:21:19
    2021-08-11 13:55:22 [INFO]	[TRAIN] Epoch=16/100, Step=6/10, loss=4.817003, lr=0.000291, time_each_step=1.86s, eta=0:21:15
    2021-08-11 13:55:23 [INFO]	[TRAIN] Epoch=16/100, Step=8/10, loss=7.966981, lr=0.000294, time_each_step=1.87s, eta=0:21:11
    2021-08-11 13:55:24 [INFO]	[TRAIN] Epoch=16/100, Step=10/10, loss=5.416558, lr=0.000298, time_each_step=1.83s, eta=0:21:7
    2021-08-11 13:55:24 [INFO]	[TRAIN] Epoch 16 finished, loss=5.830021, lr=0.00029 .
    2021-08-11 13:55:34 [INFO]	[TRAIN] Epoch=17/100, Step=2/10, loss=6.658456, lr=0.000302, time_each_step=1.95s, eta=0:30:33
    2021-08-11 13:55:36 [INFO]	[TRAIN] Epoch=17/100, Step=4/10, loss=6.518124, lr=0.000306, time_each_step=1.87s, eta=0:30:28
    2021-08-11 13:55:38 [INFO]	[TRAIN] Epoch=17/100, Step=6/10, loss=6.627895, lr=0.000309, time_each_step=1.87s, eta=0:30:24
    2021-08-11 13:55:40 [INFO]	[TRAIN] Epoch=17/100, Step=8/10, loss=6.042002, lr=0.000313, time_each_step=1.91s, eta=0:30:21
    2021-08-11 13:55:41 [INFO]	[TRAIN] Epoch=17/100, Step=10/10, loss=5.573548, lr=0.000317, time_each_step=1.94s, eta=0:30:18
    2021-08-11 13:55:41 [INFO]	[TRAIN] Epoch 17 finished, loss=6.356688, lr=0.000308 .
    2021-08-11 13:55:50 [INFO]	[TRAIN] Epoch=18/100, Step=2/10, loss=6.792327, lr=0.000321, time_each_step=1.56s, eta=0:23:59
    2021-08-11 13:55:52 [INFO]	[TRAIN] Epoch=18/100, Step=4/10, loss=5.207848, lr=0.000324, time_each_step=1.61s, eta=0:23:56
    2021-08-11 13:55:54 [INFO]	[TRAIN] Epoch=18/100, Step=6/10, loss=6.490379, lr=0.000328, time_each_step=1.61s, eta=0:23:53
    2021-08-11 13:55:55 [INFO]	[TRAIN] Epoch=18/100, Step=8/10, loss=6.456594, lr=0.000332, time_each_step=1.6s, eta=0:23:50
    2021-08-11 13:55:56 [INFO]	[TRAIN] Epoch=18/100, Step=10/10, loss=5.867396, lr=0.000336, time_each_step=1.61s, eta=0:23:47
    2021-08-11 13:55:56 [INFO]	[TRAIN] Epoch 18 finished, loss=5.679137, lr=0.000327 .
    2021-08-11 13:56:08 [INFO]	[TRAIN] Epoch=19/100, Step=2/10, loss=4.796065, lr=0.000339, time_each_step=1.67s, eta=0:20:47
    2021-08-11 13:56:11 [INFO]	[TRAIN] Epoch=19/100, Step=4/10, loss=6.462193, lr=0.000343, time_each_step=1.74s, eta=0:20:45
    2021-08-11 13:56:13 [INFO]	[TRAIN] Epoch=19/100, Step=6/10, loss=8.101998, lr=0.000347, time_each_step=1.73s, eta=0:20:42
    2021-08-11 13:56:13 [INFO]	[TRAIN] Epoch=19/100, Step=8/10, loss=4.61841, lr=0.000351, time_each_step=1.69s, eta=0:20:38
    2021-08-11 13:56:15 [INFO]	[TRAIN] Epoch=19/100, Step=10/10, loss=5.086509, lr=0.000354, time_each_step=1.68s, eta=0:20:34
    2021-08-11 13:56:15 [INFO]	[TRAIN] Epoch 19 finished, loss=5.524935, lr=0.000346 .
    2021-08-11 13:56:30 [INFO]	[TRAIN] Epoch=20/100, Step=2/10, loss=4.780834, lr=0.000358, time_each_step=2.0s, eta=0:25:23
    2021-08-11 13:56:31 [INFO]	[TRAIN] Epoch=20/100, Step=4/10, loss=5.669965, lr=0.000362, time_each_step=1.94s, eta=0:25:18
    2021-08-11 13:56:33 [INFO]	[TRAIN] Epoch=20/100, Step=6/10, loss=7.625172, lr=0.000366, time_each_step=1.94s, eta=0:25:14
    2021-08-11 13:56:34 [INFO]	[TRAIN] Epoch=20/100, Step=8/10, loss=4.372506, lr=0.000369, time_each_step=1.93s, eta=0:25:10
    2021-08-11 13:56:36 [INFO]	[TRAIN] Epoch=20/100, Step=10/10, loss=6.564197, lr=0.000373, time_each_step=1.97s, eta=0:25:7
    2021-08-11 13:56:36 [INFO]	[TRAIN] Epoch 20 finished, loss=5.980058, lr=0.000365 .
    2021-08-11 13:56:36 [INFO]	Start to evaluating(total_samples=32, total_steps=2)...


    100%|██████████| 2/2 [00:05<00:00,  2.87s/it]


    2021-08-11 13:56:42 [INFO]	[EVAL] Finished, Epoch=20, bbox_map=83.345099 .
    2021-08-11 13:56:52 [INFO]	Model saved in output/ppyolo/best_model.
    2021-08-11 13:57:01 [INFO]	Model saved in output/ppyolo/epoch_20.
    2021-08-11 13:57:01 [INFO]	Current evaluated best model in eval_dataset is epoch_20, bbox_map=83.34509871441689
    2021-08-11 13:57:10 [INFO]	[TRAIN] Epoch=21/100, Step=2/10, loss=8.900047, lr=0.000375, time_each_step=1.83s, eta=0:29:19
    2021-08-11 13:57:12 [INFO]	[TRAIN] Epoch=21/100, Step=4/10, loss=6.885113, lr=0.000375, time_each_step=1.79s, eta=0:29:15
    2021-08-11 13:57:14 [INFO]	[TRAIN] Epoch=21/100, Step=6/10, loss=5.879904, lr=0.000375, time_each_step=1.82s, eta=0:29:12
    2021-08-11 13:57:16 [INFO]	[TRAIN] Epoch=21/100, Step=8/10, loss=9.515551, lr=0.000375, time_each_step=1.86s, eta=0:29:8
    2021-08-11 13:57:18 [INFO]	[TRAIN] Epoch=21/100, Step=10/10, loss=4.966006, lr=0.000375, time_each_step=1.87s, eta=0:29:4
    2021-08-11 13:57:18 [INFO]	[TRAIN] Epoch 21 finished, loss=6.48548, lr=0.000375 .
    2021-08-11 13:57:25 [INFO]	[TRAIN] Epoch=22/100, Step=2/10, loss=6.273699, lr=0.000375, time_each_step=1.51s, eta=0:23:25
    2021-08-11 13:57:27 [INFO]	[TRAIN] Epoch=22/100, Step=4/10, loss=5.123774, lr=0.000375, time_each_step=1.54s, eta=0:23:22
    2021-08-11 13:57:29 [INFO]	[TRAIN] Epoch=22/100, Step=6/10, loss=5.821889, lr=0.000375, time_each_step=1.53s, eta=0:23:19
    2021-08-11 13:57:30 [INFO]	[TRAIN] Epoch=22/100, Step=8/10, loss=3.609881, lr=0.000375, time_each_step=1.53s, eta=0:23:16
    2021-08-11 13:57:30 [INFO]	[TRAIN] Epoch=22/100, Step=10/10, loss=5.631444, lr=0.000375, time_each_step=1.47s, eta=0:23:13
    2021-08-11 13:57:30 [INFO]	[TRAIN] Epoch 22 finished, loss=5.64679, lr=0.000375 .
    2021-08-11 13:57:39 [INFO]	[TRAIN] Epoch=23/100, Step=2/10, loss=4.27913, lr=0.000375, time_each_step=1.46s, eta=0:18:22
    2021-08-11 13:57:42 [INFO]	[TRAIN] Epoch=23/100, Step=4/10, loss=5.001262, lr=0.000375, time_each_step=1.49s, eta=0:18:19
    2021-08-11 13:57:43 [INFO]	[TRAIN] Epoch=23/100, Step=6/10, loss=9.925722, lr=0.000375, time_each_step=1.45s, eta=0:18:16
    2021-08-11 13:57:45 [INFO]	[TRAIN] Epoch=23/100, Step=8/10, loss=6.944975, lr=0.000375, time_each_step=1.44s, eta=0:18:13
    2021-08-11 13:57:46 [INFO]	[TRAIN] Epoch=23/100, Step=10/10, loss=6.474109, lr=0.000375, time_each_step=1.42s, eta=0:18:10
    2021-08-11 13:57:46 [INFO]	[TRAIN] Epoch 23 finished, loss=6.354113, lr=0.000375 .
    2021-08-11 13:57:56 [INFO]	[TRAIN] Epoch=24/100, Step=2/10, loss=6.673894, lr=0.000375, time_each_step=1.54s, eta=0:21:30
    2021-08-11 13:57:58 [INFO]	[TRAIN] Epoch=24/100, Step=4/10, loss=4.885168, lr=0.000375, time_each_step=1.55s, eta=0:21:27
    2021-08-11 13:58:00 [INFO]	[TRAIN] Epoch=24/100, Step=6/10, loss=5.4672, lr=0.000375, time_each_step=1.58s, eta=0:21:24
    2021-08-11 13:58:02 [INFO]	[TRAIN] Epoch=24/100, Step=8/10, loss=6.393391, lr=0.000375, time_each_step=1.62s, eta=0:21:21
    2021-08-11 13:58:03 [INFO]	[TRAIN] Epoch=24/100, Step=10/10, loss=6.527786, lr=0.000375, time_each_step=1.65s, eta=0:21:18
    2021-08-11 13:58:03 [INFO]	[TRAIN] Epoch 24 finished, loss=6.056887, lr=0.000375 .
    2021-08-11 13:58:14 [INFO]	[TRAIN] Epoch=25/100, Step=2/10, loss=5.439453, lr=0.000375, time_each_step=1.73s, eta=0:23:49
    2021-08-11 13:58:15 [INFO]	[TRAIN] Epoch=25/100, Step=4/10, loss=5.382199, lr=0.000375, time_each_step=1.69s, eta=0:23:45
    2021-08-11 13:58:18 [INFO]	[TRAIN] Epoch=25/100, Step=6/10, loss=5.335594, lr=0.000375, time_each_step=1.71s, eta=0:23:42
    2021-08-11 13:58:19 [INFO]	[TRAIN] Epoch=25/100, Step=8/10, loss=5.799549, lr=0.000375, time_each_step=1.7s, eta=0:23:38
    2021-08-11 13:58:20 [INFO]	[TRAIN] Epoch=25/100, Step=10/10, loss=10.867015, lr=0.000375, time_each_step=1.71s, eta=0:23:35
    2021-08-11 13:58:20 [INFO]	[TRAIN] Epoch 25 finished, loss=6.401453, lr=0.000375 .
    2021-08-11 13:58:30 [INFO]	[TRAIN] Epoch=26/100, Step=2/10, loss=5.255134, lr=0.000375, time_each_step=1.69s, eta=0:22:32
    2021-08-11 13:58:32 [INFO]	[TRAIN] Epoch=26/100, Step=4/10, loss=6.599524, lr=0.000375, time_each_step=1.67s, eta=0:22:29
    2021-08-11 13:58:34 [INFO]	[TRAIN] Epoch=26/100, Step=6/10, loss=6.47836, lr=0.000375, time_each_step=1.71s, eta=0:22:25
    2021-08-11 13:58:36 [INFO]	[TRAIN] Epoch=26/100, Step=8/10, loss=6.162186, lr=0.000375, time_each_step=1.72s, eta=0:22:22
    2021-08-11 13:58:37 [INFO]	[TRAIN] Epoch=26/100, Step=10/10, loss=6.177959, lr=0.000375, time_each_step=1.69s, eta=0:22:19
    2021-08-11 13:58:37 [INFO]	[TRAIN] Epoch 26 finished, loss=6.10443, lr=0.000375 .
    2021-08-11 13:58:48 [INFO]	[TRAIN] Epoch=27/100, Step=2/10, loss=5.190049, lr=0.000375, time_each_step=1.7s, eta=0:22:48
    2021-08-11 13:58:51 [INFO]	[TRAIN] Epoch=27/100, Step=4/10, loss=6.705993, lr=0.000375, time_each_step=1.79s, eta=0:22:46
    2021-08-11 13:58:53 [INFO]	[TRAIN] Epoch=27/100, Step=6/10, loss=5.616297, lr=0.000375, time_each_step=1.75s, eta=0:22:42
    2021-08-11 13:58:54 [INFO]	[TRAIN] Epoch=27/100, Step=8/10, loss=4.876744, lr=0.000375, time_each_step=1.77s, eta=0:22:38
    2021-08-11 13:58:55 [INFO]	[TRAIN] Epoch=27/100, Step=10/10, loss=7.103469, lr=0.000375, time_each_step=1.76s, eta=0:22:35
    2021-08-11 13:58:55 [INFO]	[TRAIN] Epoch 27 finished, loss=5.862784, lr=0.000375 .
    2021-08-11 13:59:08 [INFO]	[TRAIN] Epoch=28/100, Step=2/10, loss=4.457793, lr=0.000375, time_each_step=1.9s, eta=0:23:34
    2021-08-11 13:59:10 [INFO]	[TRAIN] Epoch=28/100, Step=4/10, loss=6.429227, lr=0.000375, time_each_step=1.91s, eta=0:23:30
    2021-08-11 13:59:12 [INFO]	[TRAIN] Epoch=28/100, Step=6/10, loss=6.394192, lr=0.000375, time_each_step=1.87s, eta=0:23:26
    2021-08-11 13:59:13 [INFO]	[TRAIN] Epoch=28/100, Step=8/10, loss=6.245507, lr=0.000375, time_each_step=1.85s, eta=0:23:23
    2021-08-11 13:59:14 [INFO]	[TRAIN] Epoch=28/100, Step=10/10, loss=6.07069, lr=0.000375, time_each_step=1.85s, eta=0:23:19
    2021-08-11 13:59:14 [INFO]	[TRAIN] Epoch 28 finished, loss=5.801753, lr=0.000375 .
    2021-08-11 13:59:26 [INFO]	[TRAIN] Epoch=29/100, Step=2/10, loss=5.211729, lr=0.000375, time_each_step=1.92s, eta=0:24:23
    2021-08-11 13:59:28 [INFO]	[TRAIN] Epoch=29/100, Step=4/10, loss=7.161747, lr=0.000375, time_each_step=1.85s, eta=0:24:19
    2021-08-11 13:59:31 [INFO]	[TRAIN] Epoch=29/100, Step=6/10, loss=5.966367, lr=0.000375, time_each_step=1.92s, eta=0:24:15
    2021-08-11 13:59:32 [INFO]	[TRAIN] Epoch=29/100, Step=8/10, loss=4.690214, lr=0.000375, time_each_step=1.89s, eta=0:24:11
    2021-08-11 13:59:33 [INFO]	[TRAIN] Epoch=29/100, Step=10/10, loss=5.937825, lr=0.000375, time_each_step=1.89s, eta=0:24:8
    2021-08-11 13:59:33 [INFO]	[TRAIN] Epoch 29 finished, loss=5.871708, lr=0.000375 .
    2021-08-11 13:59:43 [INFO]	[TRAIN] Epoch=30/100, Step=2/10, loss=5.694438, lr=0.000375, time_each_step=1.77s, eta=0:23:57
    2021-08-11 13:59:45 [INFO]	[TRAIN] Epoch=30/100, Step=4/10, loss=5.801013, lr=0.000375, time_each_step=1.76s, eta=0:23:53
    2021-08-11 13:59:48 [INFO]	[TRAIN] Epoch=30/100, Step=6/10, loss=6.369396, lr=0.000375, time_each_step=1.79s, eta=0:23:50
    2021-08-11 13:59:49 [INFO]	[TRAIN] Epoch=30/100, Step=8/10, loss=6.285254, lr=0.000375, time_each_step=1.78s, eta=0:23:46
    2021-08-11 13:59:51 [INFO]	[TRAIN] Epoch=30/100, Step=10/10, loss=5.671293, lr=0.000375, time_each_step=1.81s, eta=0:23:42
    2021-08-11 13:59:51 [INFO]	[TRAIN] Epoch 30 finished, loss=6.047874, lr=0.000375 .
    2021-08-11 14:00:01 [INFO]	[TRAIN] Epoch=31/100, Step=2/10, loss=6.780419, lr=0.000375, time_each_step=1.72s, eta=0:21:52
    2021-08-11 14:00:03 [INFO]	[TRAIN] Epoch=31/100, Step=4/10, loss=4.537371, lr=0.000375, time_each_step=1.75s, eta=0:21:49
    2021-08-11 14:00:04 [INFO]	[TRAIN] Epoch=31/100, Step=6/10, loss=6.628172, lr=0.000375, time_each_step=1.67s, eta=0:21:45
    2021-08-11 14:00:06 [INFO]	[TRAIN] Epoch=31/100, Step=8/10, loss=6.300207, lr=0.000375, time_each_step=1.7s, eta=0:21:42
    2021-08-11 14:00:07 [INFO]	[TRAIN] Epoch=31/100, Step=10/10, loss=7.431544, lr=0.000375, time_each_step=1.7s, eta=0:21:38
    2021-08-11 14:00:07 [INFO]	[TRAIN] Epoch 31 finished, loss=5.91307, lr=0.000375 .
    2021-08-11 14:00:16 [INFO]	[TRAIN] Epoch=32/100, Step=2/10, loss=6.663104, lr=0.000375, time_each_step=1.65s, eta=0:20:44
    2021-08-11 14:00:19 [INFO]	[TRAIN] Epoch=32/100, Step=4/10, loss=4.985381, lr=0.000375, time_each_step=1.66s, eta=0:20:40
    2021-08-11 14:00:20 [INFO]	[TRAIN] Epoch=32/100, Step=6/10, loss=5.767044, lr=0.000375, time_each_step=1.61s, eta=0:20:37
    2021-08-11 14:00:21 [INFO]	[TRAIN] Epoch=32/100, Step=8/10, loss=6.954692, lr=0.000375, time_each_step=1.62s, eta=0:20:34
    2021-08-11 14:00:23 [INFO]	[TRAIN] Epoch=32/100, Step=10/10, loss=4.976643, lr=0.000375, time_each_step=1.61s, eta=0:20:30
    2021-08-11 14:00:23 [INFO]	[TRAIN] Epoch 32 finished, loss=5.515681, lr=0.000375 .
    2021-08-11 14:00:31 [INFO]	[TRAIN] Epoch=33/100, Step=2/10, loss=5.472959, lr=0.000375, time_each_step=1.53s, eta=0:19:20
    2021-08-11 14:00:35 [INFO]	[TRAIN] Epoch=33/100, Step=4/10, loss=5.089534, lr=0.000375, time_each_step=1.57s, eta=0:19:17
    2021-08-11 14:00:37 [INFO]	[TRAIN] Epoch=33/100, Step=6/10, loss=4.848749, lr=0.000375, time_each_step=1.62s, eta=0:19:14
    2021-08-11 14:00:38 [INFO]	[TRAIN] Epoch=33/100, Step=8/10, loss=4.77129, lr=0.000375, time_each_step=1.58s, eta=0:19:11
    2021-08-11 14:00:39 [INFO]	[TRAIN] Epoch=33/100, Step=10/10, loss=5.595109, lr=0.000375, time_each_step=1.59s, eta=0:19:8
    2021-08-11 14:00:39 [INFO]	[TRAIN] Epoch 33 finished, loss=5.435225, lr=0.000375 .
    2021-08-11 14:00:52 [INFO]	[TRAIN] Epoch=34/100, Step=2/10, loss=5.246715, lr=0.000375, time_each_step=1.79s, eta=0:19:43
    2021-08-11 14:00:55 [INFO]	[TRAIN] Epoch=34/100, Step=4/10, loss=5.834596, lr=0.000375, time_each_step=1.81s, eta=0:19:40
    2021-08-11 14:00:57 [INFO]	[TRAIN] Epoch=34/100, Step=6/10, loss=6.585064, lr=0.000375, time_each_step=1.84s, eta=0:19:36
    2021-08-11 14:00:58 [INFO]	[TRAIN] Epoch=34/100, Step=8/10, loss=6.256711, lr=0.000375, time_each_step=1.83s, eta=0:19:32
    2021-08-11 14:00:59 [INFO]	[TRAIN] Epoch=34/100, Step=10/10, loss=5.009776, lr=0.000375, time_each_step=1.82s, eta=0:19:29
    2021-08-11 14:00:59 [INFO]	[TRAIN] Epoch 34 finished, loss=5.652921, lr=0.000375 .
    2021-08-11 14:01:08 [INFO]	[TRAIN] Epoch=35/100, Step=2/10, loss=5.982295, lr=0.000375, time_each_step=1.86s, eta=0:23:52
    2021-08-11 14:01:11 [INFO]	[TRAIN] Epoch=35/100, Step=4/10, loss=7.723389, lr=0.000375, time_each_step=1.79s, eta=0:23:48
    2021-08-11 14:01:13 [INFO]	[TRAIN] Epoch=35/100, Step=6/10, loss=7.141055, lr=0.000375, time_each_step=1.82s, eta=0:23:44
    2021-08-11 14:01:14 [INFO]	[TRAIN] Epoch=35/100, Step=8/10, loss=5.085944, lr=0.000375, time_each_step=1.83s, eta=0:23:41
    2021-08-11 14:01:15 [INFO]	[TRAIN] Epoch=35/100, Step=10/10, loss=5.38625, lr=0.000375, time_each_step=1.83s, eta=0:23:37
    2021-08-11 14:01:15 [INFO]	[TRAIN] Epoch 35 finished, loss=6.151454, lr=0.000375 .
    2021-08-11 14:01:26 [INFO]	[TRAIN] Epoch=36/100, Step=2/10, loss=3.779325, lr=0.000375, time_each_step=1.68s, eta=0:19:16
    2021-08-11 14:01:28 [INFO]	[TRAIN] Epoch=36/100, Step=4/10, loss=4.539853, lr=0.000375, time_each_step=1.64s, eta=0:19:13
    2021-08-11 14:01:29 [INFO]	[TRAIN] Epoch=36/100, Step=6/10, loss=5.023216, lr=0.000375, time_each_step=1.62s, eta=0:19:9
    2021-08-11 14:01:31 [INFO]	[TRAIN] Epoch=36/100, Step=8/10, loss=6.849054, lr=0.000375, time_each_step=1.63s, eta=0:19:6
    2021-08-11 14:01:32 [INFO]	[TRAIN] Epoch=36/100, Step=10/10, loss=5.842547, lr=0.000375, time_each_step=1.63s, eta=0:19:3
    2021-08-11 14:01:32 [INFO]	[TRAIN] Epoch 36 finished, loss=5.342432, lr=0.000375 .
    2021-08-11 14:01:41 [INFO]	[TRAIN] Epoch=37/100, Step=2/10, loss=5.581987, lr=0.000375, time_each_step=1.63s, eta=0:19:5
    2021-08-11 14:01:43 [INFO]	[TRAIN] Epoch=37/100, Step=4/10, loss=6.027231, lr=0.000375, time_each_step=1.63s, eta=0:19:2
    2021-08-11 14:01:45 [INFO]	[TRAIN] Epoch=37/100, Step=6/10, loss=4.775254, lr=0.000375, time_each_step=1.59s, eta=0:18:58
    2021-08-11 14:01:47 [INFO]	[TRAIN] Epoch=37/100, Step=8/10, loss=5.393937, lr=0.000375, time_each_step=1.62s, eta=0:18:55
    2021-08-11 14:01:48 [INFO]	[TRAIN] Epoch=37/100, Step=10/10, loss=5.929992, lr=0.000375, time_each_step=1.62s, eta=0:18:52
    2021-08-11 14:01:48 [INFO]	[TRAIN] Epoch 37 finished, loss=5.38257, lr=0.000375 .
    2021-08-11 14:02:01 [INFO]	[TRAIN] Epoch=38/100, Step=2/10, loss=6.572901, lr=0.000375, time_each_step=1.77s, eta=0:18:33
    2021-08-11 14:02:03 [INFO]	[TRAIN] Epoch=38/100, Step=4/10, loss=4.965165, lr=0.000375, time_each_step=1.75s, eta=0:18:30
    2021-08-11 14:02:04 [INFO]	[TRAIN] Epoch=38/100, Step=6/10, loss=6.869942, lr=0.000375, time_each_step=1.74s, eta=0:18:26
    2021-08-11 14:02:06 [INFO]	[TRAIN] Epoch=38/100, Step=8/10, loss=5.819637, lr=0.000375, time_each_step=1.74s, eta=0:18:23
    2021-08-11 14:02:07 [INFO]	[TRAIN] Epoch=38/100, Step=10/10, loss=8.095419, lr=0.000375, time_each_step=1.75s, eta=0:18:19
    2021-08-11 14:02:07 [INFO]	[TRAIN] Epoch 38 finished, loss=6.035417, lr=0.000375 .
    2021-08-11 14:02:18 [INFO]	[TRAIN] Epoch=39/100, Step=2/10, loss=4.175033, lr=0.000375, time_each_step=1.87s, eta=0:21:12
    2021-08-11 14:02:20 [INFO]	[TRAIN] Epoch=39/100, Step=4/10, loss=6.214126, lr=0.000375, time_each_step=1.84s, eta=0:21:8
    2021-08-11 14:02:23 [INFO]	[TRAIN] Epoch=39/100, Step=6/10, loss=5.424626, lr=0.000375, time_each_step=1.91s, eta=0:21:5
    2021-08-11 14:02:24 [INFO]	[TRAIN] Epoch=39/100, Step=8/10, loss=6.256781, lr=0.000375, time_each_step=1.87s, eta=0:21:1
    2021-08-11 14:02:25 [INFO]	[TRAIN] Epoch=39/100, Step=10/10, loss=5.860181, lr=0.000375, time_each_step=1.84s, eta=0:20:57
    2021-08-11 14:02:25 [INFO]	[TRAIN] Epoch 39 finished, loss=5.751397, lr=0.000375 .
    2021-08-11 14:02:34 [INFO]	[TRAIN] Epoch=40/100, Step=2/10, loss=7.478776, lr=0.000375, time_each_step=1.66s, eta=0:19:49
    2021-08-11 14:02:36 [INFO]	[TRAIN] Epoch=40/100, Step=4/10, loss=6.421268, lr=0.000375, time_each_step=1.66s, eta=0:19:45
    2021-08-11 14:02:37 [INFO]	[TRAIN] Epoch=40/100, Step=6/10, loss=5.944387, lr=0.000375, time_each_step=1.66s, eta=0:19:42
    2021-08-11 14:02:39 [INFO]	[TRAIN] Epoch=40/100, Step=8/10, loss=5.118117, lr=0.000375, time_each_step=1.66s, eta=0:19:39
    2021-08-11 14:02:40 [INFO]	[TRAIN] Epoch=40/100, Step=10/10, loss=5.187707, lr=0.000375, time_each_step=1.65s, eta=0:19:35
    2021-08-11 14:02:40 [INFO]	[TRAIN] Epoch 40 finished, loss=5.901137, lr=0.000375 .
    2021-08-11 14:02:40 [INFO]	Start to evaluating(total_samples=32, total_steps=2)...


    100%|██████████| 2/2 [00:08<00:00,  4.14s/it]


    2021-08-11 14:02:49 [INFO]	[EVAL] Finished, Epoch=40, bbox_map=84.924242 .
    2021-08-11 14:02:58 [INFO]	Model saved in output/ppyolo/best_model.
    2021-08-11 14:03:06 [INFO]	Model saved in output/ppyolo/epoch_40.
    2021-08-11 14:03:06 [INFO]	Current evaluated best model in eval_dataset is epoch_40, bbox_map=84.92424242424244
    2021-08-11 14:03:16 [INFO]	[TRAIN] Epoch=41/100, Step=2/10, loss=6.797449, lr=0.000375, time_each_step=1.6s, eta=0:16:22
    2021-08-11 14:03:19 [INFO]	[TRAIN] Epoch=41/100, Step=4/10, loss=5.031097, lr=0.000375, time_each_step=1.63s, eta=0:16:19
    2021-08-11 14:03:21 [INFO]	[TRAIN] Epoch=41/100, Step=6/10, loss=5.591798, lr=0.000375, time_each_step=1.58s, eta=0:16:16
    2021-08-11 14:03:23 [INFO]	[TRAIN] Epoch=41/100, Step=8/10, loss=5.285811, lr=0.000375, time_each_step=1.62s, eta=0:16:13
    2021-08-11 14:03:24 [INFO]	[TRAIN] Epoch=41/100, Step=10/10, loss=6.179473, lr=0.000375, time_each_step=1.65s, eta=0:16:10
    2021-08-11 14:03:24 [INFO]	[TRAIN] Epoch 41 finished, loss=5.610913, lr=0.000375 .
    2021-08-11 14:03:33 [INFO]	[TRAIN] Epoch=42/100, Step=2/10, loss=5.591914, lr=0.000375, time_each_step=1.64s, eta=0:18:48
    2021-08-11 14:03:35 [INFO]	[TRAIN] Epoch=42/100, Step=4/10, loss=5.012931, lr=0.000375, time_each_step=1.67s, eta=0:18:45
    2021-08-11 14:03:38 [INFO]	[TRAIN] Epoch=42/100, Step=6/10, loss=3.748342, lr=0.000375, time_each_step=1.71s, eta=0:18:42
    2021-08-11 14:03:39 [INFO]	[TRAIN] Epoch=42/100, Step=8/10, loss=5.660268, lr=0.000375, time_each_step=1.72s, eta=0:18:38
    2021-08-11 14:03:41 [INFO]	[TRAIN] Epoch=42/100, Step=10/10, loss=7.18984, lr=0.000375, time_each_step=1.74s, eta=0:18:35
    2021-08-11 14:03:41 [INFO]	[TRAIN] Epoch 42 finished, loss=5.956876, lr=0.000375 .
    2021-08-11 14:03:52 [INFO]	[TRAIN] Epoch=43/100, Step=2/10, loss=5.163554, lr=0.000375, time_each_step=1.77s, eta=0:17:39
    2021-08-11 14:03:54 [INFO]	[TRAIN] Epoch=43/100, Step=4/10, loss=5.234483, lr=0.000375, time_each_step=1.77s, eta=0:17:35
    2021-08-11 14:03:56 [INFO]	[TRAIN] Epoch=43/100, Step=6/10, loss=6.594441, lr=0.000375, time_each_step=1.77s, eta=0:17:32
    2021-08-11 14:03:57 [INFO]	[TRAIN] Epoch=43/100, Step=8/10, loss=6.178859, lr=0.000375, time_each_step=1.74s, eta=0:17:28
    2021-08-11 14:03:58 [INFO]	[TRAIN] Epoch=43/100, Step=10/10, loss=5.816396, lr=0.000375, time_each_step=1.73s, eta=0:17:25
    2021-08-11 14:03:58 [INFO]	[TRAIN] Epoch 43 finished, loss=6.132243, lr=0.000375 .
    2021-08-11 14:04:09 [INFO]	[TRAIN] Epoch=44/100, Step=2/10, loss=7.81228, lr=0.000375, time_each_step=1.77s, eta=0:17:56
    2021-08-11 14:04:11 [INFO]	[TRAIN] Epoch=44/100, Step=4/10, loss=5.725795, lr=0.000375, time_each_step=1.79s, eta=0:17:52
    2021-08-11 14:04:13 [INFO]	[TRAIN] Epoch=44/100, Step=6/10, loss=5.947436, lr=0.000375, time_each_step=1.79s, eta=0:17:49
    2021-08-11 14:04:15 [INFO]	[TRAIN] Epoch=44/100, Step=8/10, loss=6.205566, lr=0.000375, time_each_step=1.79s, eta=0:17:45
    2021-08-11 14:04:16 [INFO]	[TRAIN] Epoch=44/100, Step=10/10, loss=4.696877, lr=0.000375, time_each_step=1.77s, eta=0:17:42
    2021-08-11 14:04:16 [INFO]	[TRAIN] Epoch 44 finished, loss=5.489244, lr=0.000375 .
    2021-08-11 14:04:28 [INFO]	[TRAIN] Epoch=45/100, Step=2/10, loss=7.044242, lr=0.000375, time_each_step=1.83s, eta=0:17:55
    2021-08-11 14:04:30 [INFO]	[TRAIN] Epoch=45/100, Step=4/10, loss=7.599012, lr=0.000375, time_each_step=1.81s, eta=0:17:51
    2021-08-11 14:04:33 [INFO]	[TRAIN] Epoch=45/100, Step=6/10, loss=4.763524, lr=0.000375, time_each_step=1.83s, eta=0:17:48
    2021-08-11 14:04:34 [INFO]	[TRAIN] Epoch=45/100, Step=8/10, loss=7.138796, lr=0.000375, time_each_step=1.85s, eta=0:17:44
    2021-08-11 14:04:35 [INFO]	[TRAIN] Epoch=45/100, Step=10/10, loss=7.022514, lr=0.000375, time_each_step=1.85s, eta=0:17:40
    2021-08-11 14:04:35 [INFO]	[TRAIN] Epoch 45 finished, loss=6.180852, lr=0.000375 .
    2021-08-11 14:04:46 [INFO]	[TRAIN] Epoch=46/100, Step=2/10, loss=5.183098, lr=0.000375, time_each_step=1.89s, eta=0:18:50
    2021-08-11 14:04:49 [INFO]	[TRAIN] Epoch=46/100, Step=4/10, loss=4.898947, lr=0.000375, time_each_step=1.87s, eta=0:18:46
    2021-08-11 14:04:50 [INFO]	[TRAIN] Epoch=46/100, Step=6/10, loss=6.562926, lr=0.000375, time_each_step=1.85s, eta=0:18:43
    2021-08-11 14:04:52 [INFO]	[TRAIN] Epoch=46/100, Step=8/10, loss=6.397, lr=0.000375, time_each_step=1.86s, eta=0:18:39
    2021-08-11 14:04:53 [INFO]	[TRAIN] Epoch=46/100, Step=10/10, loss=5.482986, lr=0.000375, time_each_step=1.83s, eta=0:18:35
    2021-08-11 14:04:53 [INFO]	[TRAIN] Epoch 46 finished, loss=5.326253, lr=0.000375 .
    2021-08-11 14:05:05 [INFO]	[TRAIN] Epoch=47/100, Step=2/10, loss=5.329001, lr=0.000375, time_each_step=1.84s, eta=0:16:56
    2021-08-11 14:05:07 [INFO]	[TRAIN] Epoch=47/100, Step=4/10, loss=5.463839, lr=0.000375, time_each_step=1.84s, eta=0:16:52
    2021-08-11 14:05:10 [INFO]	[TRAIN] Epoch=47/100, Step=6/10, loss=5.680036, lr=0.000375, time_each_step=1.86s, eta=0:16:48
    2021-08-11 14:05:11 [INFO]	[TRAIN] Epoch=47/100, Step=8/10, loss=6.54626, lr=0.000375, time_each_step=1.83s, eta=0:16:45
    2021-08-11 14:05:12 [INFO]	[TRAIN] Epoch=47/100, Step=10/10, loss=4.8162, lr=0.000375, time_each_step=1.84s, eta=0:16:41
    2021-08-11 14:05:12 [INFO]	[TRAIN] Epoch 47 finished, loss=5.750719, lr=0.000375 .
    2021-08-11 14:05:23 [INFO]	[TRAIN] Epoch=48/100, Step=2/10, loss=4.919949, lr=0.000375, time_each_step=1.82s, eta=0:18:19
    2021-08-11 14:05:26 [INFO]	[TRAIN] Epoch=48/100, Step=4/10, loss=5.574605, lr=0.000375, time_each_step=1.86s, eta=0:18:15
    2021-08-11 14:05:27 [INFO]	[TRAIN] Epoch=48/100, Step=6/10, loss=5.553604, lr=0.000375, time_each_step=1.85s, eta=0:18:11
    2021-08-11 14:05:29 [INFO]	[TRAIN] Epoch=48/100, Step=8/10, loss=5.962599, lr=0.000375, time_each_step=1.82s, eta=0:18:8
    2021-08-11 14:05:30 [INFO]	[TRAIN] Epoch=48/100, Step=10/10, loss=4.852974, lr=0.000375, time_each_step=1.84s, eta=0:18:4
    2021-08-11 14:05:30 [INFO]	[TRAIN] Epoch 48 finished, loss=5.465607, lr=0.000375 .
    2021-08-11 14:05:39 [INFO]	[TRAIN] Epoch=49/100, Step=2/10, loss=4.382111, lr=0.000375, time_each_step=1.7s, eta=0:16:22
    2021-08-11 14:05:42 [INFO]	[TRAIN] Epoch=49/100, Step=4/10, loss=5.76201, lr=0.000375, time_each_step=1.72s, eta=0:16:19
    2021-08-11 14:05:43 [INFO]	[TRAIN] Epoch=49/100, Step=6/10, loss=6.699893, lr=0.000375, time_each_step=1.65s, eta=0:16:15
    2021-08-11 14:05:45 [INFO]	[TRAIN] Epoch=49/100, Step=8/10, loss=3.906954, lr=0.000375, time_each_step=1.67s, eta=0:16:12
    2021-08-11 14:05:46 [INFO]	[TRAIN] Epoch=49/100, Step=10/10, loss=4.844017, lr=0.000375, time_each_step=1.66s, eta=0:16:9
    2021-08-11 14:05:46 [INFO]	[TRAIN] Epoch 49 finished, loss=5.024908, lr=0.000375 .
    2021-08-11 14:05:57 [INFO]	[TRAIN] Epoch=50/100, Step=2/10, loss=5.166418, lr=0.000375, time_each_step=1.7s, eta=0:14:40
    2021-08-11 14:05:59 [INFO]	[TRAIN] Epoch=50/100, Step=4/10, loss=4.598126, lr=0.000375, time_each_step=1.66s, eta=0:14:37
    2021-08-11 14:06:01 [INFO]	[TRAIN] Epoch=50/100, Step=6/10, loss=5.080957, lr=0.000375, time_each_step=1.66s, eta=0:14:33
    2021-08-11 14:06:02 [INFO]	[TRAIN] Epoch=50/100, Step=8/10, loss=5.14549, lr=0.000375, time_each_step=1.67s, eta=0:14:30
    2021-08-11 14:06:03 [INFO]	[TRAIN] Epoch=50/100, Step=10/10, loss=5.758663, lr=0.000375, time_each_step=1.69s, eta=0:14:27
    2021-08-11 14:06:03 [INFO]	[TRAIN] Epoch 50 finished, loss=4.95996, lr=0.000375 .
    2021-08-11 14:06:14 [INFO]	[TRAIN] Epoch=51/100, Step=2/10, loss=5.139686, lr=0.000375, time_each_step=1.75s, eta=0:16:10
    2021-08-11 14:06:16 [INFO]	[TRAIN] Epoch=51/100, Step=4/10, loss=4.485341, lr=0.000375, time_each_step=1.73s, eta=0:16:7
    2021-08-11 14:06:19 [INFO]	[TRAIN] Epoch=51/100, Step=6/10, loss=5.576372, lr=0.000375, time_each_step=1.78s, eta=0:16:3
    2021-08-11 14:06:20 [INFO]	[TRAIN] Epoch=51/100, Step=8/10, loss=4.847356, lr=0.000375, time_each_step=1.76s, eta=0:16:0
    2021-08-11 14:06:21 [INFO]	[TRAIN] Epoch=51/100, Step=10/10, loss=4.114304, lr=0.000375, time_each_step=1.76s, eta=0:15:56
    2021-08-11 14:06:21 [INFO]	[TRAIN] Epoch 51 finished, loss=5.154345, lr=0.000375 .
    2021-08-11 14:06:29 [INFO]	[TRAIN] Epoch=52/100, Step=2/10, loss=3.590966, lr=0.000375, time_each_step=1.62s, eta=0:15:22
    2021-08-11 14:06:31 [INFO]	[TRAIN] Epoch=52/100, Step=4/10, loss=5.612742, lr=0.000375, time_each_step=1.61s, eta=0:15:19
    2021-08-11 14:06:33 [INFO]	[TRAIN] Epoch=52/100, Step=6/10, loss=6.147297, lr=0.000375, time_each_step=1.65s, eta=0:15:16
    2021-08-11 14:06:34 [INFO]	[TRAIN] Epoch=52/100, Step=8/10, loss=5.919348, lr=0.000375, time_each_step=1.62s, eta=0:15:13
    2021-08-11 14:06:36 [INFO]	[TRAIN] Epoch=52/100, Step=10/10, loss=6.351866, lr=0.000375, time_each_step=1.63s, eta=0:15:9
    2021-08-11 14:06:36 [INFO]	[TRAIN] Epoch 52 finished, loss=5.579618, lr=0.000375 .
    2021-08-11 14:06:44 [INFO]	[TRAIN] Epoch=53/100, Step=2/10, loss=3.517222, lr=0.000375, time_each_step=1.52s, eta=0:13:29
    2021-08-11 14:06:47 [INFO]	[TRAIN] Epoch=53/100, Step=4/10, loss=5.309067, lr=0.000375, time_each_step=1.54s, eta=0:13:26
    2021-08-11 14:06:49 [INFO]	[TRAIN] Epoch=53/100, Step=6/10, loss=4.274227, lr=0.000375, time_each_step=1.52s, eta=0:13:23
    2021-08-11 14:06:51 [INFO]	[TRAIN] Epoch=53/100, Step=8/10, loss=5.613493, lr=0.000375, time_each_step=1.55s, eta=0:13:20
    2021-08-11 14:06:52 [INFO]	[TRAIN] Epoch=53/100, Step=10/10, loss=4.485391, lr=0.000375, time_each_step=1.56s, eta=0:13:17
    2021-08-11 14:06:52 [INFO]	[TRAIN] Epoch 53 finished, loss=5.063865, lr=0.000375 .
    2021-08-11 14:07:07 [INFO]	[TRAIN] Epoch=54/100, Step=2/10, loss=5.5873, lr=0.000375, time_each_step=1.87s, eta=0:13:43
    2021-08-11 14:07:09 [INFO]	[TRAIN] Epoch=54/100, Step=4/10, loss=5.846558, lr=0.000375, time_each_step=1.88s, eta=0:13:39
    2021-08-11 14:07:11 [INFO]	[TRAIN] Epoch=54/100, Step=6/10, loss=7.420309, lr=0.000375, time_each_step=1.87s, eta=0:13:35
    2021-08-11 14:07:13 [INFO]	[TRAIN] Epoch=54/100, Step=8/10, loss=5.709741, lr=0.000375, time_each_step=1.91s, eta=0:13:32
    2021-08-11 14:07:14 [INFO]	[TRAIN] Epoch=54/100, Step=10/10, loss=4.947237, lr=0.000375, time_each_step=1.9s, eta=0:13:28
    2021-08-11 14:07:14 [INFO]	[TRAIN] Epoch 54 finished, loss=5.960489, lr=0.000375 .
    2021-08-11 14:07:23 [INFO]	[TRAIN] Epoch=55/100, Step=2/10, loss=5.3751, lr=0.000375, time_each_step=1.94s, eta=0:18:12
    2021-08-11 14:07:25 [INFO]	[TRAIN] Epoch=55/100, Step=4/10, loss=4.326932, lr=0.000375, time_each_step=1.91s, eta=0:18:8
    2021-08-11 14:07:28 [INFO]	[TRAIN] Epoch=55/100, Step=6/10, loss=6.233858, lr=0.000375, time_each_step=1.93s, eta=0:18:4
    2021-08-11 14:07:29 [INFO]	[TRAIN] Epoch=55/100, Step=8/10, loss=4.356949, lr=0.000375, time_each_step=1.91s, eta=0:18:0
    2021-08-11 14:07:30 [INFO]	[TRAIN] Epoch=55/100, Step=10/10, loss=4.146162, lr=0.000375, time_each_step=1.92s, eta=0:17:56
    2021-08-11 14:07:30 [INFO]	[TRAIN] Epoch 55 finished, loss=5.136355, lr=0.000375 .
    2021-08-11 14:07:40 [INFO]	[TRAIN] Epoch=56/100, Step=2/10, loss=5.014103, lr=0.000375, time_each_step=1.65s, eta=0:13:24
    2021-08-11 14:07:42 [INFO]	[TRAIN] Epoch=56/100, Step=4/10, loss=6.408309, lr=0.000375, time_each_step=1.67s, eta=0:13:21
    2021-08-11 14:07:44 [INFO]	[TRAIN] Epoch=56/100, Step=6/10, loss=4.467912, lr=0.000375, time_each_step=1.65s, eta=0:13:18
    2021-08-11 14:07:46 [INFO]	[TRAIN] Epoch=56/100, Step=8/10, loss=5.21498, lr=0.000375, time_each_step=1.65s, eta=0:13:15
    2021-08-11 14:07:47 [INFO]	[TRAIN] Epoch=56/100, Step=10/10, loss=6.758101, lr=0.000375, time_each_step=1.63s, eta=0:13:11
    2021-08-11 14:07:47 [INFO]	[TRAIN] Epoch 56 finished, loss=5.64501, lr=0.000375 .
    2021-08-11 14:07:55 [INFO]	[TRAIN] Epoch=57/100, Step=2/10, loss=2.944651, lr=0.000375, time_each_step=1.58s, eta=0:13:18
    2021-08-11 14:07:57 [INFO]	[TRAIN] Epoch=57/100, Step=4/10, loss=6.181894, lr=0.000375, time_each_step=1.57s, eta=0:13:15
    2021-08-11 14:07:59 [INFO]	[TRAIN] Epoch=57/100, Step=6/10, loss=5.975866, lr=0.000375, time_each_step=1.58s, eta=0:13:12
    2021-08-11 14:08:01 [INFO]	[TRAIN] Epoch=57/100, Step=8/10, loss=5.79298, lr=0.000375, time_each_step=1.58s, eta=0:13:9
    2021-08-11 14:08:02 [INFO]	[TRAIN] Epoch=57/100, Step=10/10, loss=5.566347, lr=0.000375, time_each_step=1.6s, eta=0:13:6
    2021-08-11 14:08:02 [INFO]	[TRAIN] Epoch 57 finished, loss=5.326846, lr=0.000375 .
    2021-08-11 14:08:15 [INFO]	[TRAIN] Epoch=58/100, Step=2/10, loss=4.549653, lr=0.000375, time_each_step=1.77s, eta=0:12:21
    2021-08-11 14:08:17 [INFO]	[TRAIN] Epoch=58/100, Step=4/10, loss=7.1229, lr=0.000375, time_each_step=1.73s, eta=0:12:17
    2021-08-11 14:08:19 [INFO]	[TRAIN] Epoch=58/100, Step=6/10, loss=4.575138, lr=0.000375, time_each_step=1.73s, eta=0:12:14
    2021-08-11 14:08:20 [INFO]	[TRAIN] Epoch=58/100, Step=8/10, loss=5.0939, lr=0.000375, time_each_step=1.71s, eta=0:12:10
    2021-08-11 14:08:21 [INFO]	[TRAIN] Epoch=58/100, Step=10/10, loss=6.559875, lr=0.000375, time_each_step=1.72s, eta=0:12:7
    2021-08-11 14:08:21 [INFO]	[TRAIN] Epoch 58 finished, loss=5.146384, lr=0.000375 .
    2021-08-11 14:08:32 [INFO]	[TRAIN] Epoch=59/100, Step=2/10, loss=4.114136, lr=0.000375, time_each_step=1.87s, eta=0:14:25
    2021-08-11 14:08:34 [INFO]	[TRAIN] Epoch=59/100, Step=4/10, loss=5.966861, lr=0.000375, time_each_step=1.86s, eta=0:14:22
    2021-08-11 14:08:36 [INFO]	[TRAIN] Epoch=59/100, Step=6/10, loss=3.979118, lr=0.000375, time_each_step=1.84s, eta=0:14:18
    2021-08-11 14:08:38 [INFO]	[TRAIN] Epoch=59/100, Step=8/10, loss=4.487059, lr=0.000375, time_each_step=1.85s, eta=0:14:14
    2021-08-11 14:08:39 [INFO]	[TRAIN] Epoch=59/100, Step=10/10, loss=5.374398, lr=0.000375, time_each_step=1.83s, eta=0:14:10
    2021-08-11 14:08:39 [INFO]	[TRAIN] Epoch 59 finished, loss=4.557798, lr=0.000375 .
    2021-08-11 14:08:53 [INFO]	[TRAIN] Epoch=60/100, Step=2/10, loss=3.733939, lr=0.000375, time_each_step=1.87s, eta=0:13:21
    2021-08-11 14:08:55 [INFO]	[TRAIN] Epoch=60/100, Step=4/10, loss=4.01683, lr=0.000375, time_each_step=1.9s, eta=0:13:17
    2021-08-11 14:08:57 [INFO]	[TRAIN] Epoch=60/100, Step=6/10, loss=6.246803, lr=0.000375, time_each_step=1.92s, eta=0:13:13
    2021-08-11 14:08:58 [INFO]	[TRAIN] Epoch=60/100, Step=8/10, loss=6.539658, lr=0.000375, time_each_step=1.92s, eta=0:13:10
    2021-08-11 14:09:00 [INFO]	[TRAIN] Epoch=60/100, Step=10/10, loss=3.462115, lr=0.000375, time_each_step=1.92s, eta=0:13:6
    2021-08-11 14:09:00 [INFO]	[TRAIN] Epoch 60 finished, loss=5.058534, lr=0.000375 .
    2021-08-11 14:09:00 [INFO]	Start to evaluating(total_samples=32, total_steps=2)...


    100%|██████████| 2/2 [00:05<00:00,  2.99s/it]


    2021-08-11 14:09:06 [INFO]	[EVAL] Finished, Epoch=60, bbox_map=83.244315 .
    2021-08-11 14:09:14 [INFO]	Model saved in output/ppyolo/epoch_60.
    2021-08-11 14:09:14 [INFO]	Current evaluated best model in eval_dataset is epoch_40, bbox_map=84.92424242424244
    2021-08-11 14:09:22 [INFO]	[TRAIN] Epoch=61/100, Step=2/10, loss=4.130944, lr=0.000375, time_each_step=1.79s, eta=0:14:10
    2021-08-11 14:09:24 [INFO]	[TRAIN] Epoch=61/100, Step=4/10, loss=4.716241, lr=0.000375, time_each_step=1.78s, eta=0:14:7
    2021-08-11 14:09:26 [INFO]	[TRAIN] Epoch=61/100, Step=6/10, loss=6.616401, lr=0.000375, time_each_step=1.8s, eta=0:14:3
    2021-08-11 14:09:28 [INFO]	[TRAIN] Epoch=61/100, Step=8/10, loss=5.05357, lr=0.000375, time_each_step=1.79s, eta=0:14:0
    2021-08-11 14:09:29 [INFO]	[TRAIN] Epoch=61/100, Step=10/10, loss=4.904676, lr=0.000375, time_each_step=1.8s, eta=0:13:56
    2021-08-11 14:09:29 [INFO]	[TRAIN] Epoch 61 finished, loss=5.324993, lr=0.000375 .
    2021-08-11 14:09:40 [INFO]	[TRAIN] Epoch=62/100, Step=2/10, loss=5.996356, lr=0.000375, time_each_step=1.66s, eta=0:10:22
    2021-08-11 14:09:42 [INFO]	[TRAIN] Epoch=62/100, Step=4/10, loss=4.327214, lr=0.000375, time_each_step=1.63s, eta=0:10:18
    2021-08-11 14:09:43 [INFO]	[TRAIN] Epoch=62/100, Step=6/10, loss=3.864747, lr=0.000375, time_each_step=1.62s, eta=0:10:15
    2021-08-11 14:09:44 [INFO]	[TRAIN] Epoch=62/100, Step=8/10, loss=5.998805, lr=0.000375, time_each_step=1.6s, eta=0:10:12
    2021-08-11 14:09:46 [INFO]	[TRAIN] Epoch=62/100, Step=10/10, loss=5.002974, lr=0.000375, time_each_step=1.62s, eta=0:10:8
    2021-08-11 14:09:46 [INFO]	[TRAIN] Epoch 62 finished, loss=4.921657, lr=0.000375 .
    2021-08-11 14:09:59 [INFO]	[TRAIN] Epoch=63/100, Step=2/10, loss=6.103065, lr=0.000375, time_each_step=1.87s, eta=0:11:16
    2021-08-11 14:10:01 [INFO]	[TRAIN] Epoch=63/100, Step=4/10, loss=4.829631, lr=0.000375, time_each_step=1.88s, eta=0:11:12
    2021-08-11 14:10:03 [INFO]	[TRAIN] Epoch=63/100, Step=6/10, loss=4.714514, lr=0.000375, time_each_step=1.85s, eta=0:11:8
    2021-08-11 14:10:04 [INFO]	[TRAIN] Epoch=63/100, Step=8/10, loss=5.854705, lr=0.000375, time_each_step=1.83s, eta=0:11:5
    2021-08-11 14:10:05 [INFO]	[TRAIN] Epoch=63/100, Step=10/10, loss=4.431303, lr=0.000375, time_each_step=1.8s, eta=0:11:1
    2021-08-11 14:10:05 [INFO]	[TRAIN] Epoch 63 finished, loss=4.950118, lr=0.000375 .
    2021-08-11 14:10:16 [INFO]	[TRAIN] Epoch=64/100, Step=2/10, loss=4.282463, lr=0.000375, time_each_step=1.81s, eta=0:12:9
    2021-08-11 14:10:19 [INFO]	[TRAIN] Epoch=64/100, Step=4/10, loss=5.716094, lr=0.000375, time_each_step=1.85s, eta=0:12:5
    2021-08-11 14:10:21 [INFO]	[TRAIN] Epoch=64/100, Step=6/10, loss=7.90618, lr=0.000375, time_each_step=1.85s, eta=0:12:2
    2021-08-11 14:10:22 [INFO]	[TRAIN] Epoch=64/100, Step=8/10, loss=4.475323, lr=0.000375, time_each_step=1.87s, eta=0:11:58
    2021-08-11 14:10:23 [INFO]	[TRAIN] Epoch=64/100, Step=10/10, loss=9.028851, lr=0.000375, time_each_step=1.82s, eta=0:11:54
    2021-08-11 14:10:23 [INFO]	[TRAIN] Epoch 64 finished, loss=5.938872, lr=0.000375 .
    2021-08-11 14:10:34 [INFO]	[TRAIN] Epoch=65/100, Step=2/10, loss=7.644, lr=0.000375, time_each_step=1.7s, eta=0:11:1
    2021-08-11 14:10:36 [INFO]	[TRAIN] Epoch=65/100, Step=4/10, loss=5.569735, lr=0.000375, time_each_step=1.7s, eta=0:10:58
    2021-08-11 14:10:38 [INFO]	[TRAIN] Epoch=65/100, Step=6/10, loss=4.038908, lr=0.000375, time_each_step=1.71s, eta=0:10:54
    2021-08-11 14:10:39 [INFO]	[TRAIN] Epoch=65/100, Step=8/10, loss=4.951239, lr=0.000375, time_each_step=1.71s, eta=0:10:51
    2021-08-11 14:10:40 [INFO]	[TRAIN] Epoch=65/100, Step=10/10, loss=7.205132, lr=0.000375, time_each_step=1.72s, eta=0:10:47
    2021-08-11 14:10:40 [INFO]	[TRAIN] Epoch 65 finished, loss=5.735185, lr=0.000375 .
    2021-08-11 14:10:51 [INFO]	[TRAIN] Epoch=66/100, Step=2/10, loss=5.071855, lr=0.000375, time_each_step=1.76s, eta=0:10:12
    2021-08-11 14:10:53 [INFO]	[TRAIN] Epoch=66/100, Step=4/10, loss=5.797879, lr=0.000375, time_each_step=1.74s, eta=0:10:8
    2021-08-11 14:10:56 [INFO]	[TRAIN] Epoch=66/100, Step=6/10, loss=4.047138, lr=0.000375, time_each_step=1.77s, eta=0:10:5
    2021-08-11 14:10:57 [INFO]	[TRAIN] Epoch=66/100, Step=8/10, loss=4.627327, lr=0.000375, time_each_step=1.77s, eta=0:10:1
    2021-08-11 14:10:58 [INFO]	[TRAIN] Epoch=66/100, Step=10/10, loss=5.347733, lr=0.000375, time_each_step=1.77s, eta=0:9:58
    2021-08-11 14:10:58 [INFO]	[TRAIN] Epoch 66 finished, loss=5.417384, lr=0.000375 .
    2021-08-11 14:11:08 [INFO]	[TRAIN] Epoch=67/100, Step=2/10, loss=6.454225, lr=0.000375, time_each_step=1.71s, eta=0:10:55
    2021-08-11 14:11:10 [INFO]	[TRAIN] Epoch=67/100, Step=4/10, loss=5.576407, lr=0.000375, time_each_step=1.73s, eta=0:10:51
    2021-08-11 14:11:12 [INFO]	[TRAIN] Epoch=67/100, Step=6/10, loss=6.211793, lr=0.000375, time_each_step=1.73s, eta=0:10:48
    2021-08-11 14:11:14 [INFO]	[TRAIN] Epoch=67/100, Step=8/10, loss=4.467885, lr=0.000375, time_each_step=1.76s, eta=0:10:45
    2021-08-11 14:11:15 [INFO]	[TRAIN] Epoch=67/100, Step=10/10, loss=3.967767, lr=0.000375, time_each_step=1.76s, eta=0:10:41
    2021-08-11 14:11:15 [INFO]	[TRAIN] Epoch 67 finished, loss=5.826022, lr=0.000375 .
    2021-08-11 14:11:27 [INFO]	[TRAIN] Epoch=68/100, Step=2/10, loss=4.372994, lr=0.000375, time_each_step=1.76s, eta=0:9:36
    2021-08-11 14:11:30 [INFO]	[TRAIN] Epoch=68/100, Step=4/10, loss=4.884226, lr=0.000375, time_each_step=1.81s, eta=0:9:33
    2021-08-11 14:11:31 [INFO]	[TRAIN] Epoch=68/100, Step=6/10, loss=4.537798, lr=0.000375, time_each_step=1.76s, eta=0:9:29
    2021-08-11 14:11:33 [INFO]	[TRAIN] Epoch=68/100, Step=8/10, loss=4.408709, lr=0.000375, time_each_step=1.77s, eta=0:9:25
    2021-08-11 14:11:34 [INFO]	[TRAIN] Epoch=68/100, Step=10/10, loss=6.590888, lr=0.000375, time_each_step=1.81s, eta=0:9:22
    2021-08-11 14:11:34 [INFO]	[TRAIN] Epoch 68 finished, loss=5.213618, lr=0.000375 .
    2021-08-11 14:11:43 [INFO]	[TRAIN] Epoch=69/100, Step=2/10, loss=4.270731, lr=0.000375, time_each_step=1.75s, eta=0:10:44
    2021-08-11 14:11:46 [INFO]	[TRAIN] Epoch=69/100, Step=4/10, loss=4.599145, lr=0.000375, time_each_step=1.77s, eta=0:10:40
    2021-08-11 14:11:47 [INFO]	[TRAIN] Epoch=69/100, Step=6/10, loss=6.073941, lr=0.000375, time_each_step=1.77s, eta=0:10:37
    2021-08-11 14:11:49 [INFO]	[TRAIN] Epoch=69/100, Step=8/10, loss=5.20995, lr=0.000375, time_each_step=1.76s, eta=0:10:33
    2021-08-11 14:11:51 [INFO]	[TRAIN] Epoch=69/100, Step=10/10, loss=4.015016, lr=0.000375, time_each_step=1.79s, eta=0:10:30
    2021-08-11 14:11:51 [INFO]	[TRAIN] Epoch 69 finished, loss=4.958107, lr=0.000375 .
    2021-08-11 14:12:02 [INFO]	[TRAIN] Epoch=70/100, Step=2/10, loss=4.429461, lr=0.000375, time_each_step=1.74s, eta=0:8:51
    2021-08-11 14:12:05 [INFO]	[TRAIN] Epoch=70/100, Step=4/10, loss=4.425426, lr=0.000375, time_each_step=1.74s, eta=0:8:48
    2021-08-11 14:12:07 [INFO]	[TRAIN] Epoch=70/100, Step=6/10, loss=4.336034, lr=0.000375, time_each_step=1.76s, eta=0:8:44
    2021-08-11 14:12:08 [INFO]	[TRAIN] Epoch=70/100, Step=8/10, loss=8.884144, lr=0.000375, time_each_step=1.77s, eta=0:8:41
    2021-08-11 14:12:10 [INFO]	[TRAIN] Epoch=70/100, Step=10/10, loss=4.623989, lr=0.000375, time_each_step=1.76s, eta=0:8:37
    2021-08-11 14:12:10 [INFO]	[TRAIN] Epoch 70 finished, loss=4.97598, lr=0.000375 .
    2021-08-11 14:12:21 [INFO]	[TRAIN] Epoch=71/100, Step=2/10, loss=4.653337, lr=0.000375, time_each_step=1.93s, eta=0:9:51
    2021-08-11 14:12:24 [INFO]	[TRAIN] Epoch=71/100, Step=4/10, loss=4.323842, lr=0.000375, time_each_step=1.91s, eta=0:9:47
    2021-08-11 14:12:26 [INFO]	[TRAIN] Epoch=71/100, Step=6/10, loss=6.176281, lr=0.000375, time_each_step=1.91s, eta=0:9:43
    2021-08-11 14:12:27 [INFO]	[TRAIN] Epoch=71/100, Step=8/10, loss=3.981862, lr=0.000375, time_each_step=1.91s, eta=0:9:39
    2021-08-11 14:12:28 [INFO]	[TRAIN] Epoch=71/100, Step=10/10, loss=4.211247, lr=0.000375, time_each_step=1.88s, eta=0:9:35
    2021-08-11 14:12:28 [INFO]	[TRAIN] Epoch 71 finished, loss=4.613172, lr=0.000375 .
    2021-08-11 14:12:40 [INFO]	[TRAIN] Epoch=72/100, Step=2/10, loss=7.923995, lr=0.000375, time_each_step=1.9s, eta=0:9:25
    2021-08-11 14:12:42 [INFO]	[TRAIN] Epoch=72/100, Step=4/10, loss=3.472786, lr=0.000375, time_each_step=1.85s, eta=0:9:21
    2021-08-11 14:12:44 [INFO]	[TRAIN] Epoch=72/100, Step=6/10, loss=6.670425, lr=0.000375, time_each_step=1.86s, eta=0:9:17
    2021-08-11 14:12:45 [INFO]	[TRAIN] Epoch=72/100, Step=8/10, loss=4.784012, lr=0.000375, time_each_step=1.83s, eta=0:9:14
    2021-08-11 14:12:47 [INFO]	[TRAIN] Epoch=72/100, Step=10/10, loss=6.644579, lr=0.000375, time_each_step=1.86s, eta=0:9:10
    2021-08-11 14:12:47 [INFO]	[TRAIN] Epoch 72 finished, loss=5.803234, lr=0.000375 .
    2021-08-11 14:12:57 [INFO]	[TRAIN] Epoch=73/100, Step=2/10, loss=4.680008, lr=0.000375, time_each_step=1.75s, eta=0:9:2
    2021-08-11 14:12:59 [INFO]	[TRAIN] Epoch=73/100, Step=4/10, loss=4.942163, lr=0.000375, time_each_step=1.78s, eta=0:8:59
    2021-08-11 14:13:01 [INFO]	[TRAIN] Epoch=73/100, Step=6/10, loss=4.51618, lr=0.000375, time_each_step=1.77s, eta=0:8:55
    2021-08-11 14:13:02 [INFO]	[TRAIN] Epoch=73/100, Step=8/10, loss=5.43662, lr=0.000375, time_each_step=1.76s, eta=0:8:52
    2021-08-11 14:13:03 [INFO]	[TRAIN] Epoch=73/100, Step=10/10, loss=6.549082, lr=0.000375, time_each_step=1.75s, eta=0:8:48
    2021-08-11 14:13:03 [INFO]	[TRAIN] Epoch 73 finished, loss=5.226538, lr=0.000375 .
    2021-08-11 14:13:16 [INFO]	[TRAIN] Epoch=74/100, Step=2/10, loss=3.998796, lr=0.000375, time_each_step=1.81s, eta=0:7:53
    2021-08-11 14:13:19 [INFO]	[TRAIN] Epoch=74/100, Step=4/10, loss=7.112694, lr=0.000375, time_each_step=1.86s, eta=0:7:50
    2021-08-11 14:13:21 [INFO]	[TRAIN] Epoch=74/100, Step=6/10, loss=6.097783, lr=0.000375, time_each_step=1.85s, eta=0:7:46
    2021-08-11 14:13:23 [INFO]	[TRAIN] Epoch=74/100, Step=8/10, loss=4.560379, lr=0.000375, time_each_step=1.89s, eta=0:7:42
    2021-08-11 14:13:24 [INFO]	[TRAIN] Epoch=74/100, Step=10/10, loss=4.047375, lr=0.000375, time_each_step=1.88s, eta=0:7:39
    2021-08-11 14:13:24 [INFO]	[TRAIN] Epoch 74 finished, loss=4.819697, lr=0.000375 .
    2021-08-11 14:13:35 [INFO]	[TRAIN] Epoch=75/100, Step=2/10, loss=3.754968, lr=0.000375, time_each_step=1.92s, eta=0:9:27
    2021-08-11 14:13:38 [INFO]	[TRAIN] Epoch=75/100, Step=4/10, loss=5.499151, lr=0.000375, time_each_step=1.92s, eta=0:9:23
    2021-08-11 14:13:39 [INFO]	[TRAIN] Epoch=75/100, Step=6/10, loss=5.211643, lr=0.000375, time_each_step=1.91s, eta=0:9:19
    2021-08-11 14:13:41 [INFO]	[TRAIN] Epoch=75/100, Step=8/10, loss=6.077746, lr=0.000375, time_each_step=1.93s, eta=0:9:15
    2021-08-11 14:13:42 [INFO]	[TRAIN] Epoch=75/100, Step=10/10, loss=4.684812, lr=0.000375, time_each_step=1.93s, eta=0:9:11
    2021-08-11 14:13:42 [INFO]	[TRAIN] Epoch 75 finished, loss=5.261519, lr=0.000375 .
    2021-08-11 14:13:54 [INFO]	[TRAIN] Epoch=76/100, Step=2/10, loss=3.505874, lr=0.000375, time_each_step=1.91s, eta=0:7:45
    2021-08-11 14:13:56 [INFO]	[TRAIN] Epoch=76/100, Step=4/10, loss=5.341751, lr=0.000375, time_each_step=1.84s, eta=0:7:41
    2021-08-11 14:13:58 [INFO]	[TRAIN] Epoch=76/100, Step=6/10, loss=5.355132, lr=0.000375, time_each_step=1.87s, eta=0:7:37
    2021-08-11 14:14:00 [INFO]	[TRAIN] Epoch=76/100, Step=8/10, loss=5.009925, lr=0.000375, time_each_step=1.85s, eta=0:7:34
    2021-08-11 14:14:01 [INFO]	[TRAIN] Epoch=76/100, Step=10/10, loss=4.290647, lr=0.000375, time_each_step=1.83s, eta=0:7:30
    2021-08-11 14:14:01 [INFO]	[TRAIN] Epoch 76 finished, loss=4.692477, lr=0.000375 .
    2021-08-11 14:14:11 [INFO]	[TRAIN] Epoch=77/100, Step=2/10, loss=4.988263, lr=0.000375, time_each_step=1.8s, eta=0:8:2
    2021-08-11 14:14:13 [INFO]	[TRAIN] Epoch=77/100, Step=4/10, loss=4.901721, lr=0.000375, time_each_step=1.78s, eta=0:7:58
    2021-08-11 14:14:15 [INFO]	[TRAIN] Epoch=77/100, Step=6/10, loss=3.375571, lr=0.000375, time_each_step=1.79s, eta=0:7:54
    2021-08-11 14:14:16 [INFO]	[TRAIN] Epoch=77/100, Step=8/10, loss=7.150967, lr=0.000375, time_each_step=1.76s, eta=0:7:51
    2021-08-11 14:14:17 [INFO]	[TRAIN] Epoch=77/100, Step=10/10, loss=4.742794, lr=0.000375, time_each_step=1.78s, eta=0:7:47
    2021-08-11 14:14:17 [INFO]	[TRAIN] Epoch 77 finished, loss=5.009276, lr=0.000375 .
    2021-08-11 14:14:27 [INFO]	[TRAIN] Epoch=78/100, Step=2/10, loss=4.450627, lr=0.000375, time_each_step=1.63s, eta=0:6:42
    2021-08-11 14:14:29 [INFO]	[TRAIN] Epoch=78/100, Step=4/10, loss=5.443079, lr=0.000375, time_each_step=1.69s, eta=0:6:39
    2021-08-11 14:14:32 [INFO]	[TRAIN] Epoch=78/100, Step=6/10, loss=5.490295, lr=0.000375, time_each_step=1.68s, eta=0:6:35
    2021-08-11 14:14:33 [INFO]	[TRAIN] Epoch=78/100, Step=8/10, loss=5.584933, lr=0.000375, time_each_step=1.68s, eta=0:6:32
    2021-08-11 14:14:35 [INFO]	[TRAIN] Epoch=78/100, Step=10/10, loss=7.486967, lr=0.000375, time_each_step=1.69s, eta=0:6:29
    2021-08-11 14:14:35 [INFO]	[TRAIN] Epoch 78 finished, loss=5.609997, lr=0.000375 .
    2021-08-11 14:14:45 [INFO]	[TRAIN] Epoch=79/100, Step=2/10, loss=3.86279, lr=0.000375, time_each_step=1.69s, eta=0:6:47
    2021-08-11 14:14:48 [INFO]	[TRAIN] Epoch=79/100, Step=4/10, loss=5.337456, lr=0.000375, time_each_step=1.72s, eta=0:6:43
    2021-08-11 14:14:49 [INFO]	[TRAIN] Epoch=79/100, Step=6/10, loss=5.006182, lr=0.000375, time_each_step=1.69s, eta=0:6:40
    2021-08-11 14:14:50 [INFO]	[TRAIN] Epoch=79/100, Step=8/10, loss=4.918838, lr=0.000375, time_each_step=1.69s, eta=0:6:36
    2021-08-11 14:14:51 [INFO]	[TRAIN] Epoch=79/100, Step=10/10, loss=4.706255, lr=0.000375, time_each_step=1.7s, eta=0:6:33
    2021-08-11 14:14:51 [INFO]	[TRAIN] Epoch 79 finished, loss=4.965272, lr=0.000375 .
    2021-08-11 14:15:00 [INFO]	[TRAIN] Epoch=80/100, Step=2/10, loss=3.86696, lr=0.000375, time_each_step=1.64s, eta=0:6:12
    2021-08-11 14:15:01 [INFO]	[TRAIN] Epoch=80/100, Step=4/10, loss=7.716565, lr=0.000375, time_each_step=1.59s, eta=0:6:9
    2021-08-11 14:15:03 [INFO]	[TRAIN] Epoch=80/100, Step=6/10, loss=6.012624, lr=0.000375, time_each_step=1.55s, eta=0:6:5
    2021-08-11 14:15:05 [INFO]	[TRAIN] Epoch=80/100, Step=8/10, loss=5.028444, lr=0.000375, time_each_step=1.59s, eta=0:6:2
    2021-08-11 14:15:06 [INFO]	[TRAIN] Epoch=80/100, Step=10/10, loss=4.371298, lr=0.000375, time_each_step=1.58s, eta=0:5:59
    2021-08-11 14:15:06 [INFO]	[TRAIN] Epoch 80 finished, loss=5.014085, lr=0.000375 .
    2021-08-11 14:15:07 [INFO]	Start to evaluating(total_samples=32, total_steps=2)...


    100%|██████████| 2/2 [00:04<00:00,  2.28s/it]


    2021-08-11 14:15:12 [INFO]	[EVAL] Finished, Epoch=80, bbox_map=84.540043 .
    2021-08-11 14:15:19 [INFO]	Model saved in output/ppyolo/epoch_80.
    2021-08-11 14:15:20 [INFO]	Current evaluated best model in eval_dataset is epoch_40, bbox_map=84.92424242424244
    2021-08-11 14:15:35 [INFO]	[TRAIN] Epoch=81/100, Step=2/10, loss=3.577727, lr=0.000375, time_each_step=1.88s, eta=0:5:14
    2021-08-11 14:15:38 [INFO]	[TRAIN] Epoch=81/100, Step=4/10, loss=3.874129, lr=0.000375, time_each_step=1.85s, eta=0:5:10
    2021-08-11 14:15:40 [INFO]	[TRAIN] Epoch=81/100, Step=6/10, loss=6.184743, lr=0.000375, time_each_step=1.87s, eta=0:5:6
    2021-08-11 14:15:41 [INFO]	[TRAIN] Epoch=81/100, Step=8/10, loss=4.035531, lr=0.000375, time_each_step=1.87s, eta=0:5:3
    2021-08-11 14:15:42 [INFO]	[TRAIN] Epoch=81/100, Step=10/10, loss=4.87326, lr=0.000375, time_each_step=1.89s, eta=0:4:59
    2021-08-11 14:15:42 [INFO]	[TRAIN] Epoch 81 finished, loss=4.841197, lr=0.000375 .
    2021-08-11 14:15:55 [INFO]	[TRAIN] Epoch=82/100, Step=2/10, loss=4.442707, lr=0.000375, time_each_step=2.11s, eta=0:7:19
    2021-08-11 14:15:57 [INFO]	[TRAIN] Epoch=82/100, Step=4/10, loss=5.63632, lr=0.000375, time_each_step=2.15s, eta=0:7:15
    2021-08-11 14:15:59 [INFO]	[TRAIN] Epoch=82/100, Step=6/10, loss=4.520019, lr=0.000375, time_each_step=2.16s, eta=0:7:11
    2021-08-11 14:16:01 [INFO]	[TRAIN] Epoch=82/100, Step=8/10, loss=4.761708, lr=0.000375, time_each_step=2.14s, eta=0:7:6
    2021-08-11 14:16:02 [INFO]	[TRAIN] Epoch=82/100, Step=10/10, loss=5.607872, lr=0.000375, time_each_step=2.12s, eta=0:7:2
    2021-08-11 14:16:02 [INFO]	[TRAIN] Epoch 82 finished, loss=5.260741, lr=0.000375 .
    2021-08-11 14:16:12 [INFO]	[TRAIN] Epoch=83/100, Step=2/10, loss=4.610402, lr=0.000375, time_each_step=1.82s, eta=0:6:2
    2021-08-11 14:16:15 [INFO]	[TRAIN] Epoch=83/100, Step=4/10, loss=4.911575, lr=0.000375, time_each_step=1.84s, eta=0:5:58
    2021-08-11 14:16:16 [INFO]	[TRAIN] Epoch=83/100, Step=6/10, loss=5.634469, lr=0.000375, time_each_step=1.84s, eta=0:5:55
    2021-08-11 14:16:18 [INFO]	[TRAIN] Epoch=83/100, Step=8/10, loss=5.500196, lr=0.000375, time_each_step=1.85s, eta=0:5:51
    2021-08-11 14:16:19 [INFO]	[TRAIN] Epoch=83/100, Step=10/10, loss=3.535293, lr=0.000375, time_each_step=1.83s, eta=0:5:47
    2021-08-11 14:16:19 [INFO]	[TRAIN] Epoch 83 finished, loss=5.074029, lr=0.000375 .
    2021-08-11 14:16:29 [INFO]	[TRAIN] Epoch=84/100, Step=2/10, loss=6.23606, lr=0.000375, time_each_step=1.69s, eta=0:4:56
    2021-08-11 14:16:31 [INFO]	[TRAIN] Epoch=84/100, Step=4/10, loss=5.555074, lr=0.000375, time_each_step=1.67s, eta=0:4:52
    2021-08-11 14:16:33 [INFO]	[TRAIN] Epoch=84/100, Step=6/10, loss=5.717344, lr=0.000375, time_each_step=1.67s, eta=0:4:49
    2021-08-11 14:16:34 [INFO]	[TRAIN] Epoch=84/100, Step=8/10, loss=3.931832, lr=0.000375, time_each_step=1.67s, eta=0:4:46
    2021-08-11 14:16:36 [INFO]	[TRAIN] Epoch=84/100, Step=10/10, loss=4.971541, lr=0.000375, time_each_step=1.7s, eta=0:4:42
    2021-08-11 14:16:36 [INFO]	[TRAIN] Epoch 84 finished, loss=5.058032, lr=0.000375 .
    2021-08-11 14:16:46 [INFO]	[TRAIN] Epoch=85/100, Step=2/10, loss=3.77813, lr=0.000375, time_each_step=1.7s, eta=0:4:45
    2021-08-11 14:16:48 [INFO]	[TRAIN] Epoch=85/100, Step=4/10, loss=4.335384, lr=0.000375, time_each_step=1.69s, eta=0:4:41
    2021-08-11 14:16:50 [INFO]	[TRAIN] Epoch=85/100, Step=6/10, loss=8.319116, lr=0.000375, time_each_step=1.68s, eta=0:4:38
    2021-08-11 14:16:52 [INFO]	[TRAIN] Epoch=85/100, Step=8/10, loss=5.97696, lr=0.000375, time_each_step=1.7s, eta=0:4:35
    2021-08-11 14:16:53 [INFO]	[TRAIN] Epoch=85/100, Step=10/10, loss=4.876006, lr=0.000375, time_each_step=1.71s, eta=0:4:31
    2021-08-11 14:16:53 [INFO]	[TRAIN] Epoch 85 finished, loss=5.301139, lr=0.000375 .
    2021-08-11 14:17:01 [INFO]	[TRAIN] Epoch=86/100, Step=2/10, loss=5.866423, lr=0.000375, time_each_step=1.63s, eta=0:4:24
    2021-08-11 14:17:04 [INFO]	[TRAIN] Epoch=86/100, Step=4/10, loss=4.85606, lr=0.000375, time_each_step=1.66s, eta=0:4:21
    2021-08-11 14:17:06 [INFO]	[TRAIN] Epoch=86/100, Step=6/10, loss=7.268117, lr=0.000375, time_each_step=1.66s, eta=0:4:18
    2021-08-11 14:17:07 [INFO]	[TRAIN] Epoch=86/100, Step=8/10, loss=8.944476, lr=0.000375, time_each_step=1.63s, eta=0:4:14
    2021-08-11 14:17:09 [INFO]	[TRAIN] Epoch=86/100, Step=10/10, loss=4.430507, lr=0.000375, time_each_step=1.63s, eta=0:4:11
    2021-08-11 14:17:09 [INFO]	[TRAIN] Epoch 86 finished, loss=5.62612, lr=0.000375 .
    2021-08-11 14:17:20 [INFO]	[TRAIN] Epoch=87/100, Step=2/10, loss=5.093325, lr=0.000375, time_each_step=1.69s, eta=0:3:49
    2021-08-11 14:17:21 [INFO]	[TRAIN] Epoch=87/100, Step=4/10, loss=4.292297, lr=0.000375, time_each_step=1.65s, eta=0:3:46
    2021-08-11 14:17:24 [INFO]	[TRAIN] Epoch=87/100, Step=6/10, loss=4.395084, lr=0.000375, time_each_step=1.68s, eta=0:3:43
    2021-08-11 14:17:25 [INFO]	[TRAIN] Epoch=87/100, Step=8/10, loss=4.231086, lr=0.000375, time_each_step=1.65s, eta=0:3:39
    2021-08-11 14:17:27 [INFO]	[TRAIN] Epoch=87/100, Step=10/10, loss=5.32079, lr=0.000375, time_each_step=1.68s, eta=0:3:36
    2021-08-11 14:17:27 [INFO]	[TRAIN] Epoch 87 finished, loss=5.187581, lr=0.000375 .
    2021-08-11 14:17:41 [INFO]	[TRAIN] Epoch=88/100, Step=2/10, loss=4.924653, lr=0.000375, time_each_step=2.01s, eta=0:4:4
    2021-08-11 14:17:44 [INFO]	[TRAIN] Epoch=88/100, Step=4/10, loss=3.468965, lr=0.000375, time_each_step=1.97s, eta=0:4:0
    2021-08-11 14:17:45 [INFO]	[TRAIN] Epoch=88/100, Step=6/10, loss=6.972976, lr=0.000375, time_each_step=1.97s, eta=0:3:56
    2021-08-11 14:17:47 [INFO]	[TRAIN] Epoch=88/100, Step=8/10, loss=4.497399, lr=0.000375, time_each_step=2.01s, eta=0:3:52
    2021-08-11 14:17:48 [INFO]	[TRAIN] Epoch=88/100, Step=10/10, loss=8.52327, lr=0.000375, time_each_step=1.98s, eta=0:3:48
    2021-08-11 14:17:48 [INFO]	[TRAIN] Epoch 88 finished, loss=5.399937, lr=0.000375 .
    2021-08-11 14:18:01 [INFO]	[TRAIN] Epoch=89/100, Step=2/10, loss=4.487467, lr=0.000375, time_each_step=2.06s, eta=0:4:27
    2021-08-11 14:18:04 [INFO]	[TRAIN] Epoch=89/100, Step=4/10, loss=5.634811, lr=0.000375, time_each_step=2.12s, eta=0:4:23
    2021-08-11 14:18:06 [INFO]	[TRAIN] Epoch=89/100, Step=6/10, loss=3.645438, lr=0.000375, time_each_step=2.09s, eta=0:4:19
    2021-08-11 14:18:07 [INFO]	[TRAIN] Epoch=89/100, Step=8/10, loss=6.547072, lr=0.000375, time_each_step=2.08s, eta=0:4:15
    2021-08-11 14:18:08 [INFO]	[TRAIN] Epoch=89/100, Step=10/10, loss=4.385708, lr=0.000375, time_each_step=2.04s, eta=0:4:10
    2021-08-11 14:18:08 [INFO]	[TRAIN] Epoch 89 finished, loss=5.020385, lr=0.000375 .
    2021-08-11 14:18:19 [INFO]	[TRAIN] Epoch=90/100, Step=2/10, loss=3.945306, lr=0.000375, time_each_step=1.9s, eta=0:3:40
    2021-08-11 14:18:21 [INFO]	[TRAIN] Epoch=90/100, Step=4/10, loss=3.84485, lr=0.000375, time_each_step=1.89s, eta=0:3:37
    2021-08-11 14:18:23 [INFO]	[TRAIN] Epoch=90/100, Step=6/10, loss=4.181958, lr=0.000375, time_each_step=1.89s, eta=0:3:33
    2021-08-11 14:18:25 [INFO]	[TRAIN] Epoch=90/100, Step=8/10, loss=4.647915, lr=0.000375, time_each_step=1.89s, eta=0:3:29
    2021-08-11 14:18:26 [INFO]	[TRAIN] Epoch=90/100, Step=10/10, loss=5.897745, lr=0.000375, time_each_step=1.9s, eta=0:3:25
    2021-08-11 14:18:26 [INFO]	[TRAIN] Epoch 90 finished, loss=4.711035, lr=0.000375 .
    2021-08-11 14:18:36 [INFO]	[TRAIN] Epoch=91/100, Step=2/10, loss=5.288058, lr=0.000375, time_each_step=1.75s, eta=0:3:16
    2021-08-11 14:18:38 [INFO]	[TRAIN] Epoch=91/100, Step=4/10, loss=4.354756, lr=0.000375, time_each_step=1.7s, eta=0:3:12
    2021-08-11 14:18:40 [INFO]	[TRAIN] Epoch=91/100, Step=6/10, loss=4.664855, lr=0.000375, time_each_step=1.7s, eta=0:3:9
    2021-08-11 14:18:41 [INFO]	[TRAIN] Epoch=91/100, Step=8/10, loss=4.06677, lr=0.000375, time_each_step=1.72s, eta=0:3:5
    2021-08-11 14:18:43 [INFO]	[TRAIN] Epoch=91/100, Step=10/10, loss=7.122146, lr=0.000375, time_each_step=1.75s, eta=0:3:2
    2021-08-11 14:18:43 [INFO]	[TRAIN] Epoch 91 finished, loss=4.797936, lr=0.000375 .
    2021-08-11 14:18:52 [INFO]	[TRAIN] Epoch=92/100, Step=2/10, loss=6.001173, lr=0.000375, time_each_step=1.65s, eta=0:2:36
    2021-08-11 14:18:55 [INFO]	[TRAIN] Epoch=92/100, Step=4/10, loss=4.555372, lr=0.000375, time_each_step=1.67s, eta=0:2:33
    2021-08-11 14:18:56 [INFO]	[TRAIN] Epoch=92/100, Step=6/10, loss=7.228217, lr=0.000375, time_each_step=1.65s, eta=0:2:29
    2021-08-11 14:18:58 [INFO]	[TRAIN] Epoch=92/100, Step=8/10, loss=4.399006, lr=0.000375, time_each_step=1.62s, eta=0:2:26
    2021-08-11 14:18:58 [INFO]	[TRAIN] Epoch=92/100, Step=10/10, loss=3.765206, lr=0.000375, time_each_step=1.6s, eta=0:2:23
    2021-08-11 14:18:58 [INFO]	[TRAIN] Epoch 92 finished, loss=5.147553, lr=0.000375 .
    2021-08-11 14:19:08 [INFO]	[TRAIN] Epoch=93/100, Step=2/10, loss=4.290242, lr=0.000375, time_each_step=1.58s, eta=0:2:15
    2021-08-11 14:19:10 [INFO]	[TRAIN] Epoch=93/100, Step=4/10, loss=4.432199, lr=0.000375, time_each_step=1.59s, eta=0:2:12
    2021-08-11 14:19:11 [INFO]	[TRAIN] Epoch=93/100, Step=6/10, loss=5.315999, lr=0.000375, time_each_step=1.59s, eta=0:2:9
    2021-08-11 14:19:13 [INFO]	[TRAIN] Epoch=93/100, Step=8/10, loss=3.919027, lr=0.000375, time_each_step=1.58s, eta=0:2:6
    2021-08-11 14:19:14 [INFO]	[TRAIN] Epoch=93/100, Step=10/10, loss=4.098766, lr=0.000375, time_each_step=1.58s, eta=0:2:3
    2021-08-11 14:19:14 [INFO]	[TRAIN] Epoch 93 finished, loss=4.751909, lr=0.000375 .
    2021-08-11 14:19:28 [INFO]	[TRAIN] Epoch=94/100, Step=2/10, loss=4.090639, lr=0.000375, time_each_step=1.76s, eta=0:2:3
    2021-08-11 14:19:30 [INFO]	[TRAIN] Epoch=94/100, Step=4/10, loss=5.427155, lr=0.000375, time_each_step=1.74s, eta=0:1:59
    2021-08-11 14:19:32 [INFO]	[TRAIN] Epoch=94/100, Step=6/10, loss=4.877515, lr=0.000375, time_each_step=1.77s, eta=0:1:56
    2021-08-11 14:19:33 [INFO]	[TRAIN] Epoch=94/100, Step=8/10, loss=4.791552, lr=0.000375, time_each_step=1.76s, eta=0:1:52
    2021-08-11 14:19:34 [INFO]	[TRAIN] Epoch=94/100, Step=10/10, loss=5.871588, lr=0.000375, time_each_step=1.76s, eta=0:1:49
    2021-08-11 14:19:34 [INFO]	[TRAIN] Epoch 94 finished, loss=4.859658, lr=0.000375 .
    2021-08-11 14:19:47 [INFO]	[TRAIN] Epoch=95/100, Step=2/10, loss=3.529526, lr=0.000375, time_each_step=1.96s, eta=0:2:5
    2021-08-11 14:19:49 [INFO]	[TRAIN] Epoch=95/100, Step=4/10, loss=5.060755, lr=0.000375, time_each_step=1.96s, eta=0:2:1
    2021-08-11 14:19:51 [INFO]	[TRAIN] Epoch=95/100, Step=6/10, loss=6.588706, lr=0.000375, time_each_step=1.95s, eta=0:1:57
    2021-08-11 14:19:52 [INFO]	[TRAIN] Epoch=95/100, Step=8/10, loss=4.718854, lr=0.000375, time_each_step=1.96s, eta=0:1:53
    2021-08-11 14:19:53 [INFO]	[TRAIN] Epoch=95/100, Step=10/10, loss=4.855227, lr=0.000375, time_each_step=1.94s, eta=0:1:49
    2021-08-11 14:19:53 [INFO]	[TRAIN] Epoch 95 finished, loss=5.182212, lr=0.000375 .
    2021-08-11 14:20:05 [INFO]	[TRAIN] Epoch=96/100, Step=2/10, loss=5.038533, lr=0.000375, time_each_step=1.87s, eta=0:1:46
    2021-08-11 14:20:07 [INFO]	[TRAIN] Epoch=96/100, Step=4/10, loss=5.894141, lr=0.000375, time_each_step=1.88s, eta=0:1:42
    2021-08-11 14:20:09 [INFO]	[TRAIN] Epoch=96/100, Step=6/10, loss=4.218745, lr=0.000375, time_each_step=1.89s, eta=0:1:38
    2021-08-11 14:20:10 [INFO]	[TRAIN] Epoch=96/100, Step=8/10, loss=4.868777, lr=0.000375, time_each_step=1.87s, eta=0:1:34
    2021-08-11 14:20:11 [INFO]	[TRAIN] Epoch=96/100, Step=10/10, loss=5.186242, lr=0.000375, time_each_step=1.88s, eta=0:1:31
    2021-08-11 14:20:11 [INFO]	[TRAIN] Epoch 96 finished, loss=4.992473, lr=0.000375 .
    2021-08-11 14:20:20 [INFO]	[TRAIN] Epoch=97/100, Step=2/10, loss=5.607459, lr=0.000375, time_each_step=1.66s, eta=0:1:20
    2021-08-11 14:20:21 [INFO]	[TRAIN] Epoch=97/100, Step=4/10, loss=6.138831, lr=0.000375, time_each_step=1.63s, eta=0:1:16
    2021-08-11 14:20:23 [INFO]	[TRAIN] Epoch=97/100, Step=6/10, loss=5.258049, lr=0.000375, time_each_step=1.65s, eta=0:1:13
    2021-08-11 14:20:25 [INFO]	[TRAIN] Epoch=97/100, Step=8/10, loss=3.784684, lr=0.000375, time_each_step=1.65s, eta=0:1:10
    2021-08-11 14:20:27 [INFO]	[TRAIN] Epoch=97/100, Step=10/10, loss=5.46702, lr=0.000375, time_each_step=1.67s, eta=0:1:7
    2021-08-11 14:20:27 [INFO]	[TRAIN] Epoch 97 finished, loss=5.023387, lr=0.000375 .
    2021-08-11 14:20:38 [INFO]	[TRAIN] Epoch=98/100, Step=2/10, loss=5.71131, lr=0.000375, time_each_step=1.65s, eta=0:0:57
    2021-08-11 14:20:40 [INFO]	[TRAIN] Epoch=98/100, Step=4/10, loss=4.918693, lr=0.000375, time_each_step=1.65s, eta=0:0:54
    2021-08-11 14:20:42 [INFO]	[TRAIN] Epoch=98/100, Step=6/10, loss=5.254423, lr=0.000375, time_each_step=1.65s, eta=0:0:50
    2021-08-11 14:20:44 [INFO]	[TRAIN] Epoch=98/100, Step=8/10, loss=5.194932, lr=0.000375, time_each_step=1.68s, eta=0:0:47
    2021-08-11 14:20:45 [INFO]	[TRAIN] Epoch=98/100, Step=10/10, loss=7.269807, lr=0.000375, time_each_step=1.68s, eta=0:0:44
    2021-08-11 14:20:45 [INFO]	[TRAIN] Epoch 98 finished, loss=4.814892, lr=0.000375 .
    2021-08-11 14:20:55 [INFO]	[TRAIN] Epoch=99/100, Step=2/10, loss=4.936164, lr=0.000375, time_each_step=1.76s, eta=0:0:45
    2021-08-11 14:20:57 [INFO]	[TRAIN] Epoch=99/100, Step=4/10, loss=5.551133, lr=0.000375, time_each_step=1.8s, eta=0:0:42
    2021-08-11 14:21:00 [INFO]	[TRAIN] Epoch=99/100, Step=6/10, loss=4.671172, lr=0.000375, time_each_step=1.81s, eta=0:0:38
    2021-08-11 14:21:01 [INFO]	[TRAIN] Epoch=99/100, Step=8/10, loss=6.969788, lr=0.000375, time_each_step=1.8s, eta=0:0:34
    2021-08-11 14:21:02 [INFO]	[TRAIN] Epoch=99/100, Step=10/10, loss=5.490433, lr=0.000375, time_each_step=1.79s, eta=0:0:31
    2021-08-11 14:21:02 [INFO]	[TRAIN] Epoch 99 finished, loss=5.288741, lr=0.000375 .
    2021-08-11 14:21:12 [INFO]	[TRAIN] Epoch=100/100, Step=2/10, loss=8.036883, lr=0.000375, time_each_step=1.68s, eta=0:0:26
    2021-08-11 14:21:14 [INFO]	[TRAIN] Epoch=100/100, Step=4/10, loss=4.93237, lr=0.000375, time_each_step=1.67s, eta=0:0:23
    2021-08-11 14:21:16 [INFO]	[TRAIN] Epoch=100/100, Step=6/10, loss=4.463372, lr=0.000375, time_each_step=1.66s, eta=0:0:19
    2021-08-11 14:21:17 [INFO]	[TRAIN] Epoch=100/100, Step=8/10, loss=6.12758, lr=0.000375, time_each_step=1.66s, eta=0:0:16
    2021-08-11 14:21:18 [INFO]	[TRAIN] Epoch=100/100, Step=10/10, loss=3.703371, lr=0.000375, time_each_step=1.68s, eta=0:0:13
    2021-08-11 14:21:18 [INFO]	[TRAIN] Epoch 100 finished, loss=4.982371, lr=0.000375 .
    2021-08-11 14:21:19 [INFO]	Start to evaluating(total_samples=32, total_steps=2)...


    100%|██████████| 2/2 [00:07<00:00,  3.95s/it]


    2021-08-11 14:21:27 [INFO]	[EVAL] Finished, Epoch=100, bbox_map=86.0488 .
    2021-08-11 14:21:35 [INFO]	Model saved in output/ppyolo/best_model.
    2021-08-11 14:21:43 [INFO]	Model saved in output/ppyolo/epoch_100.
    2021-08-11 14:21:43 [INFO]	Current evaluated best model in eval_dataset is epoch_100, bbox_map=86.04879968516332


## 预测一下


```python
import paddlex as pdx

model = pdx.load_model('output/ppyolo/best_model')
with open('objDataset/MidAutumn/test_list.txt') as f:
    for line in f:
        test_jpg='objDataset/MidAutumn/' + line.split()[0]
        result = model.predict(test_jpg)
        pdx.det.visualize(test_jpg, result, threshold=0.5, save_dir='./output/predict', color=None)

```

    2021-08-11 09:33:42 [INFO]	Model[PPYOLO] loaded.


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/__init__.py:107: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import MutableMapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/rcsetup.py:20: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Iterable, Mapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/colors.py:53: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Sized
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:2349: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      if isinstance(obj, collections.Iterator):


    2021-08-11 09:33:43 [INFO]	The visualized result is saved as ./output/predict/visualize_303.jpg
    2021-08-11 09:33:43 [INFO]	The visualized result is saved as ./output/predict/visualize_319.jpg
    2021-08-11 09:33:44 [INFO]	The visualized result is saved as ./output/predict/visualize_146.jpg
    2021-08-11 09:33:44 [INFO]	The visualized result is saved as ./output/predict/visualize_136.jpg
    2021-08-11 09:33:44 [INFO]	The visualized result is saved as ./output/predict/visualize_172.jpg
    2021-08-11 09:33:44 [INFO]	The visualized result is saved as ./output/predict/visualize_112.jpg
    2021-08-11 09:33:44 [INFO]	The visualized result is saved as ./output/predict/visualize_158.jpg
    2021-08-11 09:33:44 [INFO]	The visualized result is saved as ./output/predict/visualize_9.jpg
    2021-08-11 09:33:44 [INFO]	The visualized result is saved as ./output/predict/visualize_189.jpg
    2021-08-11 09:33:44 [INFO]	The visualized result is saved as ./output/predict/visualize_224.jpg
    2021-08-11 09:33:44 [INFO]	The visualized result is saved as ./output/predict/visualize_66.jpg
    2021-08-11 09:33:44 [INFO]	The visualized result is saved as ./output/predict/visualize_314.jpg
    2021-08-11 09:33:44 [INFO]	The visualized result is saved as ./output/predict/visualize_81.jpg
    2021-08-11 09:33:44 [INFO]	The visualized result is saved as ./output/predict/visualize_208.jpg
    2021-08-11 09:33:44 [INFO]	The visualized result is saved as ./output/predict/visualize_127.jpg
    2021-08-11 09:33:45 [INFO]	The visualized result is saved as ./output/predict/visualize_220.jpg
    2021-08-11 09:33:45 [INFO]	The visualized result is saved as ./output/predict/visualize_192.jpg
    2021-08-11 09:33:45 [INFO]	The visualized result is saved as ./output/predict/visualize_206.jpg
    2021-08-11 09:33:45 [INFO]	The visualized result is saved as ./output/predict/visualize_269.jpg
    2021-08-11 09:33:45 [INFO]	The visualized result is saved as ./output/predict/visualize_3.jpg
    2021-08-11 09:33:45 [INFO]	The visualized result is saved as ./output/predict/visualize_283.jpg
    2021-08-11 09:33:45 [INFO]	The visualized result is saved as ./output/predict/visualize_230.jpg
    2021-08-11 09:33:45 [INFO]	The visualized result is saved as ./output/predict/visualize_251.jpg
    2021-08-11 09:33:45 [INFO]	The visualized result is saved as ./output/predict/visualize_153.jpg
    2021-08-11 09:33:45 [INFO]	The visualized result is saved as ./output/predict/visualize_204.jpg
    2021-08-11 09:33:45 [INFO]	The visualized result is saved as ./output/predict/visualize_94.jpg
    2021-08-11 09:33:45 [INFO]	The visualized result is saved as ./output/predict/visualize_231.jpg
    2021-08-11 09:33:45 [INFO]	The visualized result is saved as ./output/predict/visualize_86.jpg
    2021-08-11 09:33:46 [INFO]	The visualized result is saved as ./output/predict/visualize_170.jpg
    2021-08-11 09:33:46 [INFO]	The visualized result is saved as ./output/predict/visualize_151.jpg
    2021-08-11 09:33:46 [INFO]	The visualized result is saved as ./output/predict/visualize_218.jpg
    2021-08-11 09:33:46 [INFO]	The visualized result is saved as ./output/predict/visualize_35.jpg


__识别错误示例__
> ![识别错误](https://ai-studio-static-online.cdn.bcebos.com/3366e078f5f04d068f7dd715a4c23e63f0cb725cf1194f52aecdab66d67cfa7f)

__未识别__ 看起来训练时加RandomCrop能解决
> ![未识别](https://ai-studio-static-online.cdn.bcebos.com/32407d3f2e714c3888d58b8c9792d7205fc6e3ea5d9b4b298fa57e37a835c45e)



```python
# 看看RandomCrop加入后有没用
test_jpgs = ['objDataset/MidAutumn/JPEGImages/151.jpg', 'objDataset/MidAutumn/JPEGImages/283.jpg']
model = pdx.load_model('output/ppyolo/best_model')
results = model.batch_predict(test_jpgs)
print(results)
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/framework.py:2341: UserWarning: The Attr(force_cpu) of Op(fill_constant) will be deprecated in the future, please use 'device_guard' instead. 'device_guard' has higher priority when they are used at the same time.
      "used at the same time." % type)
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/iou_aware.py:64
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:299: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/iou_aware.py:40
    The behavior of expression A / B has been unified with elementwise_div(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_div(X, Y, axis=0) instead of A / B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/io.py:2358: UserWarning: This list is not set, Because of Paramerter not found in program. There are: create_parameter_0.w_0 create_parameter_1.w_0 create_parameter_2.w_0 create_parameter_3.w_0 create_parameter_4.w_0 create_parameter_5.w_0 create_parameter_6.w_0 create_parameter_7.w_0 create_parameter_8.w_0 create_parameter_9.w_0 create_parameter_10.w_0 create_parameter_11.w_0 create_parameter_12.w_0 create_parameter_13.w_0 create_parameter_14.w_0 create_parameter_15.w_0 create_parameter_16.w_0 create_parameter_17.w_0 create_parameter_18.w_0 create_parameter_19.w_0 create_parameter_20.w_0 create_parameter_21.w_0 create_parameter_22.w_0 create_parameter_23.w_0 create_parameter_24.w_0 create_parameter_25.w_0 create_parameter_26.w_0 create_parameter_27.w_0 create_parameter_28.w_0 create_parameter_29.w_0 create_parameter_30.w_0 create_parameter_31.w_0 create_parameter_32.w_0 create_parameter_33.w_0 create_parameter_34.w_0 create_parameter_35.w_0 create_parameter_36.w_0 create_parameter_37.w_0 create_parameter_38.w_0 create_parameter_39.w_0 create_parameter_40.w_0 create_parameter_41.w_0 create_parameter_42.w_0 create_parameter_43.w_0 create_parameter_44.w_0 create_parameter_45.w_0 create_parameter_46.w_0 create_parameter_47.w_0
      format(" ".join(unused_para_list)))


    2021-08-11 14:36:52 [INFO]	Model[PPYOLO] loaded.
    [[{'category_id': 1, 'bbox': [113.70600128173828, 69.43547821044922, 148.0114974975586, 151.04447174072266], 'score': 0.5471319556236267, 'category': 'lantern'}, {'category_id': 1, 'bbox': [313.7374267578125, 157.62261962890625, 182.73345947265625, 235.01019287109375], 'score': 0.5253264904022217, 'category': 'lantern'}, {'category_id': 1, 'bbox': [2.7886123657226562, 153.80401611328125, 161.46041107177734, 196.55465698242188], 'score': 0.49936842918395996, 'category': 'lantern'}, {'category_id': 1, 'bbox': [122.21456909179688, 224.22613525390625, 190.7928466796875, 202.3616943359375], 'score': 0.36321571469306946, 'category': 'lantern'}, {'category_id': 1, 'bbox': [264.94696044921875, 87.75943756103516, 148.45538330078125, 156.89769744873047], 'score': 0.34916815161705017, 'category': 'lantern'}, {'category_id': 0, 'bbox': [107.39799499511719, 70.83119201660156, 150.52723693847656, 145.61276245117188], 'score': 0.26368603110313416, 'category': 'MoonCake'}, {'category_id': 0, 'bbox': [315.8815612792969, 176.14389038085938, 178.744873046875, 205.06378173828125], 'score': 0.25928473472595215, 'category': 'MoonCake'}, {'category_id': 0, 'bbox': [0.5540618896484375, 161.87158203125, 150.6817626953125, 186.37246704101562], 'score': 0.2490052580833435, 'category': 'MoonCake'}, {'category_id': 0, 'bbox': [122.21456909179688, 224.22613525390625, 190.7928466796875, 202.3616943359375], 'score': 0.1969166100025177, 'category': 'MoonCake'}, {'category_id': 2, 'bbox': [315.8815612792969, 176.14389038085938, 178.744873046875, 205.06378173828125], 'score': 0.11163917928934097, 'category': 'moon'}, {'category_id': 1, 'bbox': [90.40335083007812, 196.10012817382812, 231.30194091796875, 248.62445068359375], 'score': 0.09166758507490158, 'category': 'lantern'}, {'category_id': 1, 'bbox': [112.53569030761719, 166.40481567382812, 208.8236846923828, 282.6964111328125], 'score': 0.08720655739307404, 'category': 'lantern'}, {'category_id': 1, 'bbox': [111.62007141113281, 204.270263671875, 210.08970642089844, 233.78021240234375], 'score': 0.08360783755779266, 'category': 'lantern'}, {'category_id': 1, 'bbox': [315.8815612792969, 176.14389038085938, 178.744873046875, 205.06378173828125], 'score': 0.0741078183054924, 'category': 'lantern'}, {'category_id': 0, 'bbox': [264.94696044921875, 87.75943756103516, 148.45538330078125, 156.89769744873047], 'score': 0.06933795660734177, 'category': 'MoonCake'}, {'category_id': 1, 'bbox': [269.7106628417969, 83.3282241821289, 152.75872802734375, 144.9989242553711], 'score': 0.06292831152677536, 'category': 'lantern'}, {'category_id': 1, 'bbox': [93.42622375488281, 175.0662841796875, 224.9387664794922, 265.418212890625], 'score': 0.057463984936475754, 'category': 'lantern'}, {'category_id': 2, 'bbox': [107.39799499511719, 70.83119201660156, 150.52723693847656, 145.61276245117188], 'score': 0.051510803401470184, 'category': 'moon'}, {'category_id': 1, 'bbox': [265.2386779785156, 78.9996566772461, 159.60418701171875, 171.5219497680664], 'score': 0.047403734177351, 'category': 'lantern'}, {'category_id': 1, 'bbox': [107.39799499511719, 70.83119201660156, 150.52723693847656, 145.61276245117188], 'score': 0.04216286167502403, 'category': 'lantern'}, {'category_id': 1, 'bbox': [0.5540618896484375, 161.87158203125, 150.6817626953125, 186.37246704101562], 'score': 0.04085206985473633, 'category': 'lantern'}, {'category_id': 1, 'bbox': [2.0085678100585938, 124.42078399658203, 185.6087417602539, 253.43608856201172], 'score': 0.039442576467990875, 'category': 'lantern'}, {'category_id': 1, 'bbox': [308.1505126953125, 196.32708740234375, 191.8494873046875, 187.80255126953125], 'score': 0.03882254287600517, 'category': 'lantern'}, {'category_id': 1, 'bbox': [127.06106567382812, 208.22637939453125, 177.67007446289062, 208.89569091796875], 'score': 0.03781065717339516, 'category': 'lantern'}, {'category_id': 3, 'bbox': [313.7374267578125, 157.62261962890625, 182.73345947265625, 235.01019287109375], 'score': 0.03443710505962372, 'category': 'rabbit'}, {'category_id': 1, 'bbox': [322.2142028808594, 165.18597412109375, 165.8079833984375, 195.603271484375], 'score': 0.03244703635573387, 'category': 'lantern'}, {'category_id': 0, 'bbox': [12.019794464111328, 4.890235900878906, 109.06342697143555, 89.84092712402344], 'score': 0.03062533773481846, 'category': 'MoonCake'}, {'category_id': 0, 'bbox': [111.62007141113281, 204.270263671875, 210.08970642089844, 233.78021240234375], 'score': 0.03044005110859871, 'category': 'MoonCake'}, {'category_id': 0, 'bbox': [2.7886123657226562, 153.80401611328125, 161.46041107177734, 196.55465698242188], 'score': 0.029838189482688904, 'category': 'MoonCake'}, {'category_id': 1, 'bbox': [306.671630859375, 120.48115539550781, 192.17645263671875, 278.6589813232422], 'score': 0.029690569266676903, 'category': 'lantern'}, {'category_id': 1, 'bbox': [255.93212890625, 48.646827697753906, 191.8814697265625, 256.0033187866211], 'score': 0.02726791240274906, 'category': 'lantern'}, {'category_id': 0, 'bbox': [313.7374267578125, 157.62261962890625, 182.73345947265625, 235.01019287109375], 'score': 0.026782386004924774, 'category': 'MoonCake'}, {'category_id': 1, 'bbox': [95.57437896728516, 98.24079895019531, 243.77254486083984, 370.03868103027344], 'score': 0.02579040266573429, 'category': 'lantern'}, {'category_id': 1, 'bbox': [254.54708862304688, 19.187408447265625, 199.94586181640625, 361.6241149902344], 'score': 0.025781553238630295, 'category': 'lantern'}, {'category_id': 0, 'bbox': [113.70600128173828, 69.43547821044922, 148.0114974975586, 151.04447174072266], 'score': 0.025214213877916336, 'category': 'MoonCake'}, {'category_id': 1, 'bbox': [307.7259521484375, 178.89356994628906, 173.76507568359375, 203.74851989746094], 'score': 0.02473750151693821, 'category': 'lantern'}, {'category_id': 1, 'bbox': [104.80072021484375, 44.817970275878906, 174.0972900390625, 206.43242645263672], 'score': 0.024412060156464577, 'category': 'lantern'}, {'category_id': 0, 'bbox': [90.40335083007812, 196.10012817382812, 231.30194091796875, 248.62445068359375], 'score': 0.02415824495255947, 'category': 'MoonCake'}, {'category_id': 1, 'bbox': [85.48632049560547, 48.54600524902344, 183.96308135986328, 197.51260375976562], 'score': 0.02368617057800293, 'category': 'lantern'}, {'category_id': 2, 'bbox': [269.7106628417969, 83.3282241821289, 152.75872802734375, 144.9989242553711], 'score': 0.023442987352609634, 'category': 'moon'}, {'category_id': 1, 'bbox': [345.34423828125, 222.98434448242188, 122.57177734375, 109.79315185546875], 'score': 0.023031890392303467, 'category': 'lantern'}, {'category_id': 0, 'bbox': [112.53569030761719, 166.40481567382812, 208.8236846923828, 282.6964111328125], 'score': 0.022571789100766182, 'category': 'MoonCake'}, {'category_id': 1, 'bbox': [243.20285034179688, 35.52583312988281, 241.31890869140625, 380.17430114746094], 'score': 0.022097021341323853, 'category': 'lantern'}, {'category_id': 1, 'bbox': [281.92498779296875, 94.99856567382812, 131.1363525390625, 99.42062377929688], 'score': 0.021520249545574188, 'category': 'lantern'}, {'category_id': 1, 'bbox': [64.41456604003906, 53.3406982421875, 271.55052185058594, 404.2166442871094], 'score': 0.0210143830627203, 'category': 'lantern'}, {'category_id': 1, 'bbox': [78.8923110961914, 115.90150451660156, 253.1662826538086, 335.7020721435547], 'score': 0.020457057282328606, 'category': 'lantern'}, {'category_id': 1, 'bbox': [125.9757308959961, 83.646240234375, 130.9166030883789, 118.17453002929688], 'score': 0.02015901915729046, 'category': 'lantern'}, {'category_id': 0, 'bbox': [127.06106567382812, 208.22637939453125, 177.67007446289062, 208.89569091796875], 'score': 0.02012527361512184, 'category': 'MoonCake'}, {'category_id': 1, 'bbox': [13.165950775146484, 0.35488128662109375, 112.0406379699707, 136.83698272705078], 'score': 0.02003541775047779, 'category': 'lantern'}, {'category_id': 0, 'bbox': [345.34423828125, 222.98434448242188, 122.57177734375, 109.79315185546875], 'score': 0.02002573385834694, 'category': 'MoonCake'}, {'category_id': 1, 'bbox': [234.9596405029297, 60.895179748535156, 205.2967071533203, 226.9392318725586], 'score': 0.019797328859567642, 'category': 'lantern'}, {'category_id': 1, 'bbox': [107.49456787109375, 80.9762954711914, 159.871826171875, 154.43810272216797], 'score': 0.019316770136356354, 'category': 'lantern'}, {'category_id': 1, 'bbox': [7.359466552734375, 140.78469848632812, 153.97679138183594, 189.647705078125], 'score': 0.01835433393716812, 'category': 'lantern'}, {'category_id': 1, 'bbox': [271.0038146972656, 117.68156433105469, 228.99618530273438, 281.5734405517578], 'score': 0.018287647515535355, 'category': 'lantern'}, {'category_id': 1, 'bbox': [0.0, 126.84523010253906, 201.87646484375, 282.02757263183594], 'score': 0.018189622089266777, 'category': 'lantern'}, {'category_id': 0, 'bbox': [308.1505126953125, 196.32708740234375, 191.8494873046875, 187.80255126953125], 'score': 0.01790156029164791, 'category': 'MoonCake'}, {'category_id': 0, 'bbox': [281.92498779296875, 94.99856567382812, 131.1363525390625, 99.42062377929688], 'score': 0.01725774258375168, 'category': 'MoonCake'}, {'category_id': 0, 'bbox': [118.60305786132812, 84.47394561767578, 122.36563110351562, 112.14159393310547], 'score': 0.017045047134160995, 'category': 'MoonCake'}, {'category_id': 1, 'bbox': [0.0, 97.06553649902344, 246.40682983398438, 305.18312072753906], 'score': 0.01694054715335369, 'category': 'lantern'}, {'category_id': 1, 'bbox': [113.80924224853516, 220.71612548828125, 186.78914642333984, 209.8818359375], 'score': 0.01620548777282238, 'category': 'lantern'}, {'category_id': 1, 'bbox': [305.31048583984375, 178.44674682617188, 194.68951416015625, 225.02325439453125], 'score': 0.015917254611849785, 'category': 'lantern'}, {'category_id': 1, 'bbox': [277.941162109375, 108.56312561035156, 111.176513671875, 109.30007934570312], 'score': 0.015813574194908142, 'category': 'lantern'}, {'category_id': 0, 'bbox': [93.42622375488281, 175.0662841796875, 224.9387664794922, 265.418212890625], 'score': 0.015584597364068031, 'category': 'MoonCake'}, {'category_id': 0, 'bbox': [277.941162109375, 108.56312561035156, 111.176513671875, 109.30007934570312], 'score': 0.015394606627523899, 'category': 'MoonCake'}, {'category_id': 2, 'bbox': [111.62007141113281, 204.270263671875, 210.08970642089844, 233.78021240234375], 'score': 0.015283796004951, 'category': 'moon'}, {'category_id': 2, 'bbox': [313.7374267578125, 157.62261962890625, 182.73345947265625, 235.01019287109375], 'score': 0.015216727741062641, 'category': 'moon'}, {'category_id': 0, 'bbox': [7.359466552734375, 140.78469848632812, 153.97679138183594, 189.647705078125], 'score': 0.015114080160856247, 'category': 'MoonCake'}, {'category_id': 3, 'bbox': [107.39799499511719, 70.83119201660156, 150.52723693847656, 145.61276245117188], 'score': 0.015031620860099792, 'category': 'rabbit'}, {'category_id': 1, 'bbox': [103.43877410888672, 81.80195617675781, 157.96369171142578, 153.15313720703125], 'score': 0.01492425799369812, 'category': 'lantern'}, {'category_id': 0, 'bbox': [269.7106628417969, 83.3282241821289, 152.75872802734375, 144.9989242553711], 'score': 0.014909351244568825, 'category': 'MoonCake'}, {'category_id': 1, 'bbox': [118.60305786132812, 84.47394561767578, 122.36563110351562, 112.14159393310547], 'score': 0.014757947064936161, 'category': 'lantern'}, {'category_id': 1, 'bbox': [265.5732116699219, 85.03367614746094, 150.87103271484375, 144.42666625976562], 'score': 0.014737814664840698, 'category': 'lantern'}, {'category_id': 1, 'bbox': [291.82415771484375, 155.91217041015625, 202.44940185546875, 235.52935791015625], 'score': 0.0146766547113657, 'category': 'lantern'}, {'category_id': 2, 'bbox': [2.7886123657226562, 153.80401611328125, 161.46041107177734, 196.55465698242188], 'score': 0.014445473439991474, 'category': 'moon'}, {'category_id': 0, 'bbox': [322.2142028808594, 165.18597412109375, 165.8079833984375, 195.603271484375], 'score': 0.014323219656944275, 'category': 'MoonCake'}, {'category_id': 1, 'bbox': [121.6591567993164, 205.1031494140625, 173.56328582763672, 214.0556640625], 'score': 0.014166213572025299, 'category': 'lantern'}, {'category_id': 0, 'bbox': [125.9757308959961, 83.646240234375, 130.9166030883789, 118.17453002929688], 'score': 0.014137323014438152, 'category': 'MoonCake'}, {'category_id': 1, 'bbox': [232.56321716308594, 24.1824951171875, 212.66835021972656, 396.214111328125], 'score': 0.013756233267486095, 'category': 'lantern'}, {'category_id': 1, 'bbox': [259.1598205566406, 46.00341796875, 240.84017944335938, 372.1521911621094], 'score': 0.01355776283890009, 'category': 'lantern'}, {'category_id': 1, 'bbox': [0.0, 175.75991821289062, 156.91574096679688, 178.586669921875], 'score': 0.012881875038146973, 'category': 'lantern'}, {'category_id': 1, 'bbox': [0.0, 167.2722930908203, 169.52639770507812, 195.1943817138672], 'score': 0.012499733828008175, 'category': 'lantern'}, {'category_id': 1, 'bbox': [24.860366821289062, 60.23846435546875, 306.50364685058594, 386.50128173828125], 'score': 0.012476446107029915, 'category': 'lantern'}, {'category_id': 1, 'bbox': [73.94486999511719, 7.639678955078125, 243.8792266845703, 435.3338317871094], 'score': 0.011993810534477234, 'category': 'lantern'}, {'category_id': 1, 'bbox': [22.03585433959961, 191.3307342529297, 125.97494888305664, 117.15287780761719], 'score': 0.011564241722226143, 'category': 'lantern'}, {'category_id': 0, 'bbox': [2.0085678100585938, 124.42078399658203, 185.6087417602539, 253.43608856201172], 'score': 0.0110622588545084, 'category': 'MoonCake'}, {'category_id': 1, 'bbox': [254.49966430664062, 33.34093475341797, 191.73004150390625, 239.3302993774414], 'score': 0.01088219415396452, 'category': 'lantern'}, {'category_id': 1, 'bbox': [245.783203125, 39.994873046875, 186.76177978515625, 316.0360412597656], 'score': 0.010874275118112564, 'category': 'lantern'}, {'category_id': 1, 'bbox': [65.45402526855469, 55.552520751953125, 304.90077209472656, 396.6572570800781], 'score': 0.010855386964976788, 'category': 'lantern'}, {'category_id': 0, 'bbox': [347.498046875, 202.99107360839844, 118.01287841796875, 100.12117004394531], 'score': 0.010829844512045383, 'category': 'MoonCake'}, {'category_id': 0, 'bbox': [279.5215759277344, 101.71286010742188, 117.64581298828125, 95.75942993164062], 'score': 0.010700964368879795, 'category': 'MoonCake'}, {'category_id': 2, 'bbox': [281.92498779296875, 94.99856567382812, 131.1363525390625, 99.42062377929688], 'score': 0.010696358978748322, 'category': 'moon'}, {'category_id': 0, 'bbox': [306.671630859375, 120.48115539550781, 192.17645263671875, 278.6589813232422], 'score': 0.01010228507220745, 'category': 'MoonCake'}, {'category_id': 0, 'bbox': [14.84991455078125, 190.90994262695312, 110.24382019042969, 119.12896728515625], 'score': 0.010037063620984554, 'category': 'MoonCake'}], [{'category_id': 0, 'bbox': [129.51882934570312, 197.06991577148438, 270.4811706542969, 245.2752685546875], 'score': 0.9472435712814331, 'category': 'MoonCake'}, {'category_id': 0, 'bbox': [142.30661010742188, 190.2030029296875, 248.6337890625, 235.64654541015625], 'score': 0.10435958206653595, 'category': 'MoonCake'}, {'category_id': 0, 'bbox': [155.71514892578125, 183.09898376464844, 239.06268310546875, 250.0854034423828], 'score': 0.056450605392456055, 'category': 'MoonCake'}, {'category_id': 0, 'bbox': [143.13302612304688, 191.90577697753906, 256.8669738769531, 253.64598083496094], 'score': 0.05545849725604057, 'category': 'MoonCake'}, {'category_id': 0, 'bbox': [121.18202209472656, 205.06655883789062, 262.87266540527344, 235.0037841796875], 'score': 0.03693016245961189, 'category': 'MoonCake'}, {'category_id': 0, 'bbox': [134.28428649902344, 215.6526641845703, 259.0821075439453, 246.1855926513672], 'score': 0.033221665769815445, 'category': 'MoonCake'}, {'category_id': 0, 'bbox': [40.07283020019531, 366.4900207519531, 151.3300323486328, 147.95144653320312], 'score': 0.028222493827342987, 'category': 'MoonCake'}, {'category_id': 0, 'bbox': [250.63729858398438, 5.049247741699219, 143.724365234375, 153.8122329711914], 'score': 0.02287311665713787, 'category': 'MoonCake'}, {'category_id': 0, 'bbox': [161.2657470703125, 210.27227783203125, 203.53570556640625, 213.13983154296875], 'score': 0.020308125764131546, 'category': 'MoonCake'}, {'category_id': 2, 'bbox': [129.51882934570312, 197.06991577148438, 270.4811706542969, 245.2752685546875], 'score': 0.020018605515360832, 'category': 'moon'}, {'category_id': 1, 'bbox': [129.51882934570312, 197.06991577148438, 270.4811706542969, 245.2752685546875], 'score': 0.013731495477259159, 'category': 'lantern'}, {'category_id': 0, 'bbox': [204.3587188720703, 259.6850280761719, 115.91517639160156, 129.3387451171875], 'score': 0.013494670391082764, 'category': 'MoonCake'}, {'category_id': 0, 'bbox': [145.01358032226562, 220.04843139648438, 203.40438842773438, 197.87823486328125], 'score': 0.011097232811152935, 'category': 'MoonCake'}, {'category_id': 0, 'bbox': [312.6652526855469, 9.613468170166016, 62.46759033203125, 126.54058456420898], 'score': 0.010330360382795334, 'category': 'MoonCake'}]]



```python
pdx.det.visualize(test_jpgs[0], results[0], threshold=0.5, save_dir='./output/predict_crop', color=None)
pdx.det.visualize(test_jpgs[1], results[1], threshold=0.5, save_dir='./output/predict_crop', color=None)
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:2349: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      if isinstance(obj, collections.Iterator):


    2021-08-11 14:38:17 [INFO]	The visualized result is saved as ./output/predict_crop/visualize_151.jpg
    2021-08-11 14:38:17 [INFO]	The visualized result is saved as ./output/predict_crop/visualize_283.jpg


__更新结果__ 可惜边上的灯还是没认出来
> ![](https://ai-studio-static-online.cdn.bcebos.com/f61936a2263347da9a1a7e148088a6ab54ddf21a6d1e4c00a704d60f40b34561)
> ![](https://ai-studio-static-online.cdn.bcebos.com/63eb6475d871415f8ff741fbf9c321a26a626759908a43529a21c567fd9d2967)


# 五、作业提交

完成以上四个步骤，并且跑通后，恭喜你！你已经学会使用PaddleX训练目标检测模型了！

首先点击右上方的“**文件**”，点击“**导出Notebook为Markdown**”：
![](https://ai-studio-static-online.cdn.bcebos.com/0b502ce690274eee89c285c702a26a72ff012df791eb45b79b7866a97f549e33)

导出后的文件用于上传至GitHub。

下面请点击左侧“版本”，在出现的新界面中点击“生成新版本”，按如下操作即可生成版本，用于提交作业：
![](https://ai-studio-static-online.cdn.bcebos.com/4e58b131f2db45289119dfa90081eea43b781d6894254069b4f02cffa067ffe1)

生成版本后，回到项目主页，点击“**设置为公开**”：
![](https://ai-studio-static-online.cdn.bcebos.com/4a9229e9eb8146169c848327ccb1f85ea30af431eee14aea9c2e6c9aeb364926)



