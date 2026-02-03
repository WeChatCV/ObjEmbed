# ObjEmbed: 通用多模态物体向量

## 👀 ObjEmbed介绍

<p align="left">
    <img src="./assets/model.png" width="800px">
</p>


将物体与文本描述进行对齐是一个基础的视觉语言任务，具有重要的应用价值。现有的多模态嵌入模型能够将全图编码成一个向量并与全图文本描述进行对齐，但是它们往往不能编码细粒度的物体特征。

在这个工作中，我们提出了ObjEmbed模型，首个基于多模态大模型的物体向量模型。它将整幅图片的每个物体分别编码成一个向量，同时将全图也编码成一个向量。利用编码的向量，该模型支持多种视觉任务，包括：视觉定位，局部图像检索，全局图像检索。

ObjEmbed模型具有三个关键的特征：
- **以物体为中心的表征。** 该模型同时编码物体的语义信息和位置信息，并产生两个互补的向量：一个是物体向量，用于语义匹配；一个是IoU向量，用于评估坐标框的质量。物体与文本描述的匹配分数由语义相似性和预测的IoU分数的乘积最终决定，这样能够完成更加准确的检索。
- **全能性。** 该模型能够同时支持区域级别的任务和全图级别的任务。
- **高效性。** 该模型将全图中的所有物体的特征以及全图的特征在一次前向推理中进行编码，非常高效。

ObjEmbed模型在18个公开数据集上取得领先性能。

## 📈 实验结果

#### 📍 模型库

- [ObjEmbed-2B](https://huggingface.co/fushh7/ObjEmbed-2B)
- [ObjEmbed-4B](https://huggingface.co/fushh7/ObjEmbed-4B)

我们使用[WeDetect-Base-Uni](https://github.com/WeChatCV/WeDetect)作为候选框提取网络，该模型可以在这里进行下载:
- [WeDetect-Base-Uni](https://huggingface.co/fushh7/WeDetect)

#### 📍 结果

<p align="left">
    <img src="./assets/performance1.png" width="800px">
</p>
<p align="left">
    <img src="./assets/performance2.png" width="800px">
</p>
<p align="left">
    <img src="./assets/performance3.png" width="400px">
</p>
<p align="left">
    <img src="./assets/performance4.png" width="800px">
</p>

## 🔧 安装环境

#### 基本库

```
pytorch==2.6.1+cu124
transformers==4.57.1
trl==0.17.0
accelerate==1.10.0
```

- 请按照下列的指令安装环境

```
pip install transformers==4.57.1 trl==0.17.0 accelerate==1.10.0 -i https://mirrors.cloud.tencent.com/pypi/simple
pip install pycocotools terminaltables jsonlines tabulate ddd-dataset torchmetrics lvis -i https://mirrors.cloud.tencent.com/pypi/simple
```
- 如果您需要使用lvis库，请确保`numpy<=1.24`.


## ⭐ Demo

#### 📍 Referring Expression Comprehension 
```
# output the top1 prediction
python infer_objembed.py --objembed_checkpoint /PATH/TO/OBJEMBED --wedetect_uni_checkpoint /PATH/TO/WEDETECT_UNI --image assets/demo.jpg --query "The car's license plate in HAWAII" --task rec --visualize
```
<p align="left">
    <img src="./assets/pred.png" width="500px">
</p>


#### 📍 Image Retrieval

```
python infer_objembed.py --objembed_checkpoint /PATH/TO/OBJEMBED --wedetect_uni_checkpoint /PATH/TO/WEDETECT_UNI --image image1.jpg image2.jpg image3.jpg --query "YOUR_QUERY" --task retrieval_by_image
```


## 📏 评测
#### 📍 Visual Grounding
```
cd eval_grounding
export PYTHONPATH=../

# coco / coco_o / lvis / FG-OVD / d3 / odinw13
torchrun --nproc-per-node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=29500 eval.py --checkpoint /PATH/TO/OBJEMBED --dataset coco --nms --task_specific_visual_prompt

# refcoco
torchrun --nproc-per-node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=29500 eval.py --checkpoint /PATH/TO/OBJEMBED --dataset refcoco --num_select 20 --task_specific_visual_prompt
```
- 请您修改`eval_grounding/eval.py`中第`47-417`行的数据路径。
- 对于每一个数据集，您首先需要为每张图片提出候选框，并保存在json文件中。您可以参照`generate_proposal.py`这份代码进行提取。我们提供了refcoco的[提取结果](https://huggingface.co/datasets/fushh7/eval_refcoco)。


#### 📍 Image Retrieval
```
cd eval_retrieval
export PYTHONPATH=../

# sharegpt4v / dci / coco / coco_cn / d3 / flickr30k / flickr30k_cn
# sorce_1k / reircoco / ilias / ilias_i2i
torchrun --nproc-per-node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=29500 eval.py --checkpoint /PATH/TO/OBJEMBED --dataset sorce_1k
```
- 请您修改`eval_retrieval/eval.py`中第`19-90`行的数据路径。
- 对于每一个数据集，您首先需要为每张图片提出候选框，并保存在json文件中。您可以参照`generate_proposal.py`这份代码进行提取。我们提供了refcoco的[提取结果](https://huggingface.co/datasets/fushh7/eval_refcoco)。



## 🙏 致谢

- 本项目基于[WeDetect](https://github.com/WeChatCV/WeDetect)、[transformers](https://github.com/huggingface/transformers)、[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) 等项目开发，感谢这些优秀的开源项目。

## ✒️ 引用

如果您觉得我们的工作对您的研究有帮助，请您引用我们的工作：   

```bibtex
@article{fu2026objembed,
  title={ObjEmbed: Towards Universal Multimodal Object Embeddings},
  author={Fu, Shenghao and Su, Yukun and Rao, Fengyun and LYU, Jing and Xie, Xiaohua and Zheng, Wei-Shi},
  journal={arXiv preprint arXiv:2602.01753},
  year={2026}
}
```

## 📜 协议

- 我们的模型和代码在Apache 2.0协议下开源。

