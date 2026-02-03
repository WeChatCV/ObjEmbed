import os
import time
import json
import random
import argparse
import itertools
import re
import torch
import torch.nn.functional as F
import copy
from pycocotools.coco import COCO

from PIL import Image
from tqdm import tqdm
import numpy as np



ds_collections = {
    'sorce_1k': {
        'ann_path': 'datasets/sorce-1k/dataset.jsonl',
        'visual_prompt': 'Locate the specific object being described by analyzing its unique instance-level attributes, its spatial position, and its relationship with surrounding objects. ',
        'text_prompt': "Find an object that matches the given caption. %s",
        'img_path': 'datasets/sorce-1k/full_res/',
        'proposals': 'datasets/wedetect_ref/eval_proposals/sorce_proposals_all.json',
    },
    'coco': {
        'ann_path': 'datasets/coco/annotations/captions_val2017.json',
        'visual_prompt': 'Detect all objects in the image by identifying the common visual features of their respective classes. ',
        'text_prompt': "Find an image that matches the given caption. %s",
        'img_path': 'datasets/coco/val2017/',
        'proposals': 'datasets/wedetect_ref/eval_proposals/coco_proposals_all.json',
    },
    'coco_cn': {
        'ann_path': 'coco_cn.txt',
        'visual_prompt': 'Detect all objects in the image by identifying the common visual features of their respective classes. ',
        'text_prompt': "Find an image that matches the given caption. %s",
        'img_path': 'datasets/coco2014/',
        'proposals': 'datasets/wedetect_ref/eval_proposals/coco_cn_proposals_all.json',
    },
    'flickr30k': {
        'ann_path': 'datasets/flickr/flickr30k_test.json',
        'visual_prompt': 'Detect all objects in the image by identifying the common visual features of their respective classes. ',
        'text_prompt': "Find an image that matches the given caption. %s",
        'img_path': 'datasets/flickr/flickr/',
        'proposals': 'datasets/wedetect_ref/eval_proposals/flickr30k_proposals_all.json',
    },
    'flickr30k_cn': {
        'ann_path': 'datasets/flickr/flickr30k_cn_test_texts.jsonl',
        'visual_prompt': 'Detect all objects in the image by identifying the common visual features of their respective classes. ',
        'text_prompt': "Find an image that matches the given caption. %s",
        'img_path': 'datasets/flickr/flickr/',
        'proposals': 'datasets/wedetect_ref/eval_proposals/flickr30k_cn_proposals_all.json',
    },
    'sharegpt4v': {
        'ann_path': 'datasets/sharegpt4v/share-captioner_coco_lcs_sam_1246k_1107.json',
        'visual_prompt': 'Locate the specific object being described by analyzing its unique instance-level attributes, its spatial position, and its relationship with surrounding objects. ',
        'text_prompt': "Find an image that matches the given caption. %s",
        'img_path': 'datasets/sharegpt4v/images/',
        'proposals': 'datasets/wedetect_ref/eval_proposals/sharegpt4v_proposals_all.json',
    },
    'dci': {
        'ann_path': 'datasets/DCI/anno.json',
        'visual_prompt': 'Locate the specific object being described by analyzing its unique instance-level attributes, its spatial position, and its relationship with surrounding objects. ',
        'text_prompt': "Find an image that matches the given caption. %s",
        'img_path': 'datasets/DCI/images/',
        'proposals': 'datasets/wedetect_ref/eval_proposals/dci_proposals_all.json',
    },
    'reircoco': {
        'ann_path': 'datasets/REIRCOCO/reircoco_val.json',
        'visual_prompt': 'Locate the specific object being described by analyzing its unique instance-level attributes, its spatial position, and its relationship with surrounding objects. ',
        'text_prompt': "Find an image that matches the given caption. %s",
        'img_path': 'datasets/coco2014/train2014/',
        'proposals': 'datasets/wedetect_ref/eval_proposals/reircoco_proposals_all.json',
    },
    'ilias': {
        'ann_path': 'datasets/ILIAS/ilias_data.json',
        'visual_prompt': 'Locate the specific object being described by analyzing its unique instance-level attributes, its spatial position, and its relationship with surrounding objects. ',
        'text_prompt': "Find an image that matches the given caption. %s",
        'img_path': 'datasets/ILIAS/',
        'proposals': 'datasets/wedetect_ref/eval_proposals/ilias_proposals_all.json',
    },
    'ilias_i2i': {
        'ann_path': 'datasets/ILIAS/ilias_data.json',
        'visual_prompt': 'Locate the specific object being described by analyzing its unique instance-level attributes, its spatial position, and its relationship with surrounding objects. ',
        'text_prompt': "Find an image that matches the given caption. %s",
        'img_path': 'datasets/ILIAS/',
        'proposals': 'datasets/wedetect_ref/eval_proposals/ilias_proposals_all.json',
    },
}

class RetrievalDataset(torch.utils.data.Dataset):
    

    def __init__(
        self,
        dataset,
    ):
        super().__init__()
        self.dataset = dataset
        self.proposals = json.load(open(ds_collections[dataset]['proposals']))
        self.anns = []
        if dataset == 'sorce_1k':
            with open(ds_collections[dataset]['ann_path'], 'r') as f:
                data = [json.loads(line) for line in f] # caption, image
            for da in data:
                new_da = {
                    'image': ds_collections[self.dataset]['img_path'] + da['image'],
                    'visual_prompt': copy.deepcopy(ds_collections[self.dataset]['visual_prompt']),
                    'text_prompt': [copy.deepcopy(ds_collections[self.dataset]['text_prompt']) % da['caption']],
                }
                self.anns.append(new_da)
        elif dataset == 'coco':
            images = json.load(open(ds_collections[dataset]['ann_path']))['images']
            coco = COCO(ds_collections[dataset]['ann_path'])
            for ann in images:
                ann_ids = coco.getAnnIds(imgIds=ann['id'])
                ann_infos = coco.loadAnns(ann_ids)
                assert len(ann_infos) >= 5, ann_infos
                ann_infos = ann_infos[:5]
                item = {
                    'image': ds_collections[dataset]['img_path'] + ann['file_name'],
                    'visual_prompt': copy.deepcopy(ds_collections[self.dataset]['visual_prompt']),
                    'text_prompt': [copy.deepcopy(ds_collections[self.dataset]['text_prompt']) % ann_info['caption'] for ann_info in ann_infos],
                }
                self.anns.append(item)
        elif dataset == 'coco_cn':
            pattern = re.compile(r'COCO_(train|val)2014_(\d{12})#zhm#\d\s+(.*)')
            image_to_texts_dict = {}
            with open(ds_collections[dataset]['ann_path'], 'r', encoding='utf-8') as file:
                for line in file:
                    match = pattern.match(line.strip())
                    if match:

                        image_id = f"COCO_{match.group(1)}2014_{match.group(2)}.jpg"

                        text_description = match.group(3)

                        if image_id not in image_to_texts_dict:
                            image_to_texts_dict[image_id] = []
                        image_to_texts_dict[image_id].append(text_description)
            for imagename, captions in image_to_texts_dict.items():
                if "train" in imagename:
                    fullname = ds_collections[dataset]['img_path'] + "train2014/" + imagename
                elif "val" in imagename:
                    fullname = ds_collections[dataset]['img_path'] + "val2014/" + imagename
                
                item = {
                    'image': fullname,
                    'visual_prompt': copy.deepcopy(ds_collections[self.dataset]['visual_prompt']),
                    'text_prompt': [copy.deepcopy(ds_collections[self.dataset]['text_prompt']) % caption for caption in captions],
                }
                self.anns.append(item)
        elif dataset == 'flickr30k':
            images = json.load(open(ds_collections[dataset]['ann_path']))
            for ann in images:
                item = {
                    'image': ds_collections[dataset]['img_path'] + ann['filename'],
                    'visual_prompt': copy.deepcopy(ds_collections[self.dataset]['visual_prompt']),
                    'text_prompt': [copy.deepcopy(ds_collections[self.dataset]['text_prompt']) % caption for caption in ann["raw"]],
                }
                self.anns.append(item)
        elif dataset == 'flickr30k_cn':
            with open(ds_collections[dataset]['ann_path']) as f:
                data = [json.loads(line) for line in f]
            image_to_texts_dict = {}
            for da in data:
                if da["image_ids"][0] not in image_to_texts_dict:
                    image_to_texts_dict[da["image_ids"][0]] = []
                image_to_texts_dict[da["image_ids"][0]].append(da['text'])
            for imagename, captions in image_to_texts_dict.items():
                item = {
                    'image': ds_collections[dataset]['img_path'] + str(imagename) + '.jpg',
                    'visual_prompt': copy.deepcopy(ds_collections[self.dataset]['visual_prompt']),
                    'text_prompt': [copy.deepcopy(ds_collections[self.dataset]['text_prompt']) % caption for caption in captions],
                }
                self.anns.append(item)
        elif dataset == 'sharegpt4v':
            images = json.load(open(ds_collections[dataset]['ann_path']))[:1000]
            for ann in images:
                item = {
                    'image': ds_collections[dataset]['img_path'] + ann['image'].split('/')[-1],
                    'visual_prompt': copy.deepcopy(ds_collections[self.dataset]['visual_prompt']),
                    'text_prompt': [copy.deepcopy(ds_collections[self.dataset]['text_prompt']) % ann['conversations'][1]['value']],
                }
                self.anns.append(item)
        elif dataset == 'dci':
            images = json.load(open(ds_collections[dataset]['ann_path']))
            for ann in images:
                item = {
                    'image': ds_collections[dataset]['img_path'] + ann['image'],
                    'visual_prompt': copy.deepcopy(ds_collections[self.dataset]['visual_prompt']),
                    'text_prompt': [copy.deepcopy(ds_collections[self.dataset]['text_prompt']) % ann['extra_caption']],
                }
                self.anns.append(item)
        elif dataset == 'reircoco':
            images = json.load(open(ds_collections[dataset]['ann_path']))
            for ann in images['images']:
                item = {
                    'image': ds_collections[dataset]['img_path'] + ann['file_name'],
                    'visual_prompt': copy.deepcopy(ds_collections[self.dataset]['visual_prompt']),
                    'text_prompt': [copy.deepcopy(ds_collections[self.dataset]['text_prompt']) % ann["expressions"][0]],
                }
                self.anns.append(item)
        elif dataset == 'ilias':
            images = json.load(open(ds_collections[dataset]['ann_path']))
            candidates_idx = []
            query_idx = []
            idx = 0
            for ann in images:
                query_idx.append(idx)
                can_idx = []
                for img in ann["candidates"]:
                    item = {
                        'image': ds_collections[dataset]['img_path'] + img,
                        'visual_prompt': copy.deepcopy(ds_collections[self.dataset]['visual_prompt']),
                        'text_prompt': [copy.deepcopy(ds_collections[self.dataset]['text_prompt']) % ann["query_text"]],
                    }
                    can_idx.append(idx)
                    self.anns.append(item)
                    idx += 1
                candidates_idx.append(can_idx)
            self.candidates_idx = candidates_idx
            self.query_idx = query_idx
        elif dataset == 'ilias_i2i':
            images = json.load(open(ds_collections[dataset]['ann_path']))
            query_idx = []
            candidates_idx = []
            idx = 0
            for ann in images:
                item = {
                    'image': ds_collections[dataset]['img_path'] + ann["queries"][0],
                    'visual_prompt': copy.deepcopy(ds_collections[self.dataset]['visual_prompt']),
                    'text_prompt': [copy.deepcopy(ds_collections[self.dataset]['text_prompt']) % ann["query_text"]],
                    'proposals':  ann["query_boxes"][0],
                }
                assert len(ann["query_boxes"][0]) == 1
                self.anns.append(item)
                query_idx.append(idx)
                idx += 1
            print(idx)
            idx = 0
            for ann in images:
                can_idx = []
                for img in ann["candidates"]:
                    item = {
                        'image': ds_collections[dataset]['img_path'] + img,
                        'visual_prompt': copy.deepcopy(ds_collections[self.dataset]['visual_prompt']),
                        'text_prompt': [copy.deepcopy(ds_collections[self.dataset]['text_prompt']) % ann["query_text"]],
                    }
                    self.anns.append(item)
                    can_idx.append(idx)
                    idx += 1
                candidates_idx.append(can_idx)
            self.candidates_idx = candidates_idx
            self.query_idx = query_idx
        

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]
        data = {}
        data['image'] = Image.open(ann['image']).convert('RGB')
        w, h = data['image'].size
        data['proposals'] = ann['proposals'] if 'proposals' in ann else self.proposals[ann['image']][0][:100]
        # data['proposals'] = []
        for i in range(len(data['proposals'])):
            data['proposals'][i][0] = max(0, min(w, data['proposals'][i][0]))
            data['proposals'][i][1] = max(0, min(h, data['proposals'][i][1]))
            data['proposals'][i][2] = max(0, min(w, data['proposals'][i][2]))
            data['proposals'][i][3] = max(0, min(h, data['proposals'][i][3]))
        data['visual_prompt'] = ann['visual_prompt']
        data['text_prompt'] = ann['text_prompt']
        return data


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def collate_fn(inputs):
    return inputs



def compute_retrieval_recall(scores, k_list=[1, 5, 10], place='diag'):
    """
    返回:
    - recalls: dict, e.g., {'R@1': 0.75, 'R@5': 0.92, 'R@10': 0.96}
    """
    N = scores.size(0)
    # 对每一行按相似度降序排序，得到索引
    _, ranked_indices = scores.sort(dim=1, descending=True)  # shape: (N, M)

    recalls = {}
    for k in k_list:
        # 取前 K 个候选的索引
        top_k_indices = ranked_indices[:, :k]  # shape: (N, k)

        if place == 'diag':
            # 创建对角线上的正样本索引：第 i 行的目标是 i
            target_indices = torch.arange(N, device=scores.device)  # [0, 1, 2, ..., N-1]
        elif place == 'first':
            # 创建全0的正样本索引：第 i 行的目标是 0
            target_indices = torch.zeros(N, device=scores.device, dtype=top_k_indices.dtype)  # [0, 1, 2, ..., N-1]

        # 扩展 target 到 shape (N, k)，便于比较
        target_indices = target_indices.unsqueeze(1).expand(-1, k)  # shape: (N, k)

        # 判断每行的 top-k 是否包含其对应的 target (i == i)
        correct = (top_k_indices == target_indices).any(dim=1)  # shape: (N,), bool
        recall_at_k = correct.float().mean().item()
        recalls[f'R@{k}'] = recall_at_k

    print(recalls)


def eval_coco(similarity, k_list=[1, 5, 10]):
    """
    Args:
        similarity: shape [num_images, num_texts] (e.g., [5000, 25000])
        k_list: list of k values for Recall@k
    """
    # 确保 similarity 是 tensor
    if not isinstance(similarity, torch.Tensor):
        similarity = torch.tensor(similarity)
    
    # 确保在同一设备上计算
    device = similarity.device
    num_images, num_texts = similarity.shape
    # COCO 数据集通常每个图像对应 5 个文本
    captions_per_image = num_texts // num_images 
    
    max_k = max(k_list)

    # ======================= I2T (Image to Text) =======================
    print("I2T")
    
    # 1. 直接获取所有图像的前 max_k 个预测文本的索引
    # values: [num_images, max_k], indices: [num_images, max_k]
    _, topk_indices = torch.topk(similarity, k=max_k, dim=1)
    
    # 2. 构建 Ground Truth 逻辑
    # 对于第 i 张图，正确的文本索引范围是 [5*i, 5*i+4]
    # 我们可以通过整除 5 来判断：如果 (pred_text_index // 5) == image_index，则预测正确
    pred_image_ids = topk_indices // captions_per_image
    gt_image_ids = torch.arange(num_images, device=device).unsqueeze(1) # [num_images, 1]
    
    # 3. 比较预测结果
    # matches: [num_images, max_k] (布尔矩阵)
    matches = (pred_image_ids == gt_image_ids)
    
    # 4. 计算各 K 值的 Recall
    for k in k_list:
        # 只要前 k 列中有一个 True，该样本就算召回成功
        recall = matches[:, :k].any(dim=1).float().mean().item()
        print(f"R@{k} {recall:.6f}")

    # ======================= T2I (Text to Image) =======================
    print("T2I")
    
    # 1. 转置矩阵，变成 [num_texts, num_images]
    similarity_t = similarity.T
    
    # 2. 获取所有文本的前 max_k 个预测图像的索引
    _, topk_indices = torch.topk(similarity_t, k=max_k, dim=1)
    
    # 3. 构建 Ground Truth 逻辑
    # 对于第 i 个文本，正确的图像索引是 i // 5
    gt_image_ids = torch.arange(num_texts, device=device) // captions_per_image
    gt_image_ids = gt_image_ids.unsqueeze(1) # [num_texts, 1]
    
    # 4. 比较预测结果 (topk_indices 里的值就是图像 ID，直接比较)
    matches = (topk_indices == gt_image_ids)
    
    # 5. 计算各 K 值的 Recall
    for k in k_list:
        recall = matches[:, :k].any(dim=1).float().mean().item()
        print(f"R@{k} {recall:.6f}")


import torch

def compute_map_at_k(similarity, true_indices, k=50):
    """
    计算 mAP@K
    
    Args:
        similarity: shape [num_queries, num_candidates] 的相似度矩阵 (Tensor)
        true_indices: 一个列表，长度为 num_queries。
                      每个元素是一个 list 或 tensor，包含该 query 对应的正确 candidate 下标。
                      例如: [[1, 10], [5], [0, 2, 9], ...]
        k: 计算 Top K (默认 50)
        
    Returns:
        mAP@K 分数 (float)
    """
    # 1. 确保 similarity 是 tensor 并在 GPU 上 (如果可用)
    if not isinstance(similarity, torch.Tensor):
        similarity = torch.tensor(similarity)
    device = similarity.device
    num_queries = similarity.shape[0]

    # 2. 并行获取所有 Query 的前 K 个预测索引
    # topk_indices: [num_queries, k]
    _, topk_indices = torch.topk(similarity, k=k, dim=1)

    average_precisions = []

    # 3. 遍历每个 Query 计算 AP (由于 GT 长度不一，这里使用循环处理逻辑，但计算仍在 GPU)
    for i in range(num_queries):
        # 获取当前 query 的预测结果 (长度为 k)
        pred_indices = topk_indices[i] 
        
        # 获取当前 query 的真实标签
        # 如果输入是 list，转为 tensor 并移动到同一设备
        gt_indices = true_indices[i]
        if not isinstance(gt_indices, torch.Tensor):
            gt_indices = torch.tensor(gt_indices, device=device, dtype=torch.long)
        else:
            gt_indices = gt_indices.to(device)
            
        num_gt = len(gt_indices)
        
        # 如果该 Query 没有对应的正确答案，跳过或记为 0
        if num_gt == 0:
            average_precisions.append(0.0)
            continue

        # 4. 判断命中情况 (核心加速点)
        # torch.isin 检查 pred_indices 中的每个元素是否存在于 gt_indices 中
        # hits: [k] (布尔 tensor, True 表示命中)
        if hasattr(torch, 'isin'): # PyTorch 1.10+
            hits = torch.isin(pred_indices, gt_indices)
        else:
            # 兼容旧版本 PyTorch
            hits = (pred_indices.unsqueeze(1) == gt_indices.unsqueeze(0)).any(dim=1)

        # 如果前 K 个都没命中，AP 为 0
        if hits.sum() == 0:
            average_precisions.append(0.0)
            continue

        # 5. 计算 AP@K
        hits = hits.float()
        
        # cumsum: 计算截止到当前 rank 的命中总数 -> [1, 1, 2, 2, 3...]
        cumsum_hits = torch.cumsum(hits, dim=0)
        
        # ranks: 当前的位置 -> [1, 2, 3, 4, 5...]
        ranks = torch.arange(1, k + 1, device=device, dtype=torch.float)
        
        # precision_at_i: 截止到每个位置的精度
        precision_at_i = cumsum_hits / ranks
        
        # 只累加命中位置的精度 (AP 定义)
        # AP = (P@1 * rel@1 + P@2 * rel@2 + ... + P@k * rel@k) / min(num_gt, k)
        # 注意：分母通常有两种定义：
        # 定义 A (标准): 除以总的正确答案数量 (num_gt)。这会惩罚没被检索到的正确答案。
        # 定义 B (截断): 除以 min(num_gt, k)。这只关注前 K 个位置的表现。
        # 检索任务中通常使用 定义 A。
        
        score = (precision_at_i * hits).sum() / num_gt
        # 如果你想用定义 B，请取消下面这行的注释并注释上面那行
        # score = (precision_at_i * hits).sum() / min(num_gt, k)
        
        average_precisions.append(score.item())

    # 6. 计算 mAP (所有 Query 的平均值)
    mAP = sum(average_precisions) / len(average_precisions)
    return mAP



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    # dataset = RetrievalDataset(args.dataset)

    from datetime import timedelta
    timeout = timedelta(seconds=7200)
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
        timeout=timeout,
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    from models.vision_process import process_vision_info
    from transformers import AutoProcessor

    # Model initialization
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map='cuda',
    )

    from models.qwen3vl_objembed import ObjectEmbed
    model = ObjectEmbed.from_pretrained(args.checkpoint, **model_kwargs)
    model = model.eval()

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    object_token_index = processor.tokenizer.convert_tokens_to_ids("<object>")
    local_text_id = processor.tokenizer.convert_tokens_to_ids("<local_text>")
    model.model.object_token_id = object_token_index
    global_id = None
    global_text_id = None

    if model.use_global_caption:
        global_id = processor.tokenizer.convert_tokens_to_ids("<global>")
        global_text_id = processor.tokenizer.convert_tokens_to_ids("<global_text>")

    random.seed(args.seed)
    dataset = RetrievalDataset(args.dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    query_embedding = []
    candicate_object_embedding = []
    candicate_image_embedding = []
    objectnesses = []
    visualize_idx = 0
    for inputs in tqdm(dataloader, disable=torch.distributed.get_rank() != 0):
        image = inputs[0]['image']
        ori_shape = [image.size]
        proposals = copy.deepcopy(inputs[0]['proposals'])
        proposals = [torch.tensor(proposals).cuda().to(model.dtype)]
        
        if args.dataset != 'ilias_i2i':
            for query in inputs[0]['text_prompt']:
                if args.dataset == 'sorce_1k' or args.dataset == 'reircoco' or args.dataset == 'ilias':
                    messages = [
                        {
                            "role": "user", 
                            "content": 
                            [
                                {"type": "text", "text": query + " <local_text>"}
                            ]
                        }
                    ]
                else:
                    messages = [
                        {
                            "role": "user", 
                            "content": 
                            [
                                {"type": "text", "text": query + " <global_text>"}
                            ]
                        }
                    ]
                texts = [processor.apply_chat_template(messages, tokenize=False).strip()]
                model_inputs = processor(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    do_resize=False,
                )
                model_inputs = model_inputs.to(model.device)
                
                with torch.inference_mode():
                    pred = model(
                        text_processed=model_inputs,
                        global_id=global_id,
                        local_text_id=local_text_id,
                        global_text_id=global_text_id
                    )
                if args.dataset == 'sorce_1k' or args.dataset == 'reircoco' or args.dataset == 'ilias':
                    query_embedding.append(pred['local_text_embeddings'].cpu())
                else:
                    query_embedding.append(pred['global_text_embeddings'].cpu())
                # query_embedding.append(pred['text_embeddings'].cpu())


        obj_str = ""
        for j in range(proposals[0].shape[0]):
            obj_str += "Object %d: <object><object>. " % j
            
        if model.use_two_tokens == 0:
            obj_str = obj_str + "The global image is <global>"
        elif model.use_two_tokens == 1:
            obj_str = "The global image is <global>. " + obj_str + "The global image is <global>"
        else:
            obj_str = "The coarse global image is <global>. " + obj_str + " The detailed global image is <global>. "
            
        messages = [
            {
                "role": "user", 
                "content": 
                [
                    {"type": "image", "image": image}, 
                    {"type": "text", "text": inputs[0]['visual_prompt'] + obj_str}
                ]
            }
        ]
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)

        texts = [processor.apply_chat_template(messages, tokenize=False).strip()]
        model_inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            do_resize=False,
        )
        model_inputs = model_inputs.to(model.device)
        
        with torch.inference_mode():
            pred = model(
                **model_inputs,
                bboxes=copy.deepcopy(proposals),
                ori_shapes=ori_shape,
                bboxes_id=object_token_index,
                global_id=global_id,
                local_text_id=local_text_id,
                global_text_id=global_text_id,
            )

        object_embeddings = pred['object_embeddings']
        objectness = pred['objness'].sigmoid().cpu().float()
        candicate_object_embedding.append(object_embeddings.cpu())
        if model.use_two_tokens > 0:
            candicate_image_embedding.append(pred['full_image_embeddings'][:, 0, :].cpu())
        else:
            candicate_image_embedding.append(pred['full_image_embeddings'].cpu())
        objectnesses.append(objectness)

        
    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_query_embedding = [None for _ in range(world_size)]
    merged_candicate_object_embedding = [None for _ in range(world_size)]
    merged_candicate_image_embedding = [None for _ in range(world_size)]
    merged_objectnesses = [None for _ in range(world_size)]
    if args.dataset != 'ilias_i2i':
        torch.distributed.all_gather_object(merged_query_embedding, query_embedding)
    torch.distributed.all_gather_object(merged_candicate_object_embedding, candicate_object_embedding)
    torch.distributed.all_gather_object(merged_candicate_image_embedding, candicate_image_embedding)
    torch.distributed.all_gather_object(merged_objectnesses, objectnesses)
    if args.dataset != 'ilias_i2i':
        merged_query_embedding = [_ for _ in itertools.chain.from_iterable(merged_query_embedding)]
    merged_candicate_object_embedding = [_ for _ in itertools.chain.from_iterable(merged_candicate_object_embedding)]
    merged_candicate_image_embedding = [_ for _ in itertools.chain.from_iterable(merged_candicate_image_embedding)]
    merged_objectnesses = [_ for _ in itertools.chain.from_iterable(merged_objectnesses)]

    with torch.no_grad():
        if torch.distributed.get_rank() == 0:
            print(f"Evaluating {args.dataset} ...")
            if args.dataset == 'sorce_1k' or args.dataset == 'reircoco':
                merged_query_embedding = torch.cat(merged_query_embedding, dim=0).to(model.device)
                merged_candicate_object_embedding = torch.cat(merged_candicate_object_embedding, dim=0).to(model.device)
                merged_query_embedding = F.normalize(merged_query_embedding)
                merged_candicate_object_embedding = F.normalize(merged_candicate_object_embedding)
                pred_scores = merged_query_embedding @ merged_candicate_object_embedding.transpose(-1, -2)
                pred_scores = pred_scores * model.logit_log_scale.exp()
                pred_scores = pred_scores + model.logit_bias
                pred_scores = pred_scores.float().sigmoid().cpu()
                
                pred_scores = pred_scores.reshape(len(merged_query_embedding), len(merged_query_embedding), 100)
                pred_scores, index = torch.max(pred_scores, dim=-1)
                compute_retrieval_recall(pred_scores, place='diag')
            if args.dataset == 'coco' or args.dataset == 'coco_cn' or args.dataset == 'flickr30k' or args.dataset == 'flickr30k_cn':
                merged_query_embedding = torch.cat(merged_query_embedding, dim=0).to(model.device)
                merged_candicate_image_embedding = torch.cat(merged_candicate_image_embedding, dim=0).to(model.device)
                merged_query_embedding = F.normalize(merged_query_embedding)
                merged_candicate_image_embedding = F.normalize(merged_candicate_image_embedding)
                pred_scores = merged_query_embedding @ merged_candicate_image_embedding.transpose(-1, -2)
                pred_scores = pred_scores.to(model.device) * model.logit_image_log_scale.exp()
                pred_scores = pred_scores + model.logit_image_bias
                pred_scores = pred_scores.float().sigmoid().cpu()
                eval_coco(pred_scores.permute(1, 0))
            if args.dataset == 'sharegpt4v' or args.dataset == 'dci':
                merged_query_embedding = torch.cat(merged_query_embedding, dim=0).to(model.device)
                merged_candicate_image_embedding = torch.cat(merged_candicate_image_embedding, dim=0).to(model.device)
                merged_query_embedding = F.normalize(merged_query_embedding)
                merged_candicate_image_embedding = F.normalize(merged_candicate_image_embedding)
                pred_scores = merged_query_embedding @ merged_candicate_image_embedding.transpose(-1, -2)
                pred_scores = pred_scores.to(model.device) * model.logit_image_log_scale.exp()
                pred_scores = pred_scores + model.logit_image_bias
                pred_scores = pred_scores.float().sigmoid().cpu()
                print("I2T")
                compute_retrieval_recall(pred_scores.permute(1, 0), place='diag')
                print("T2I")
                compute_retrieval_recall(pred_scores, place='diag')
            if args.dataset == 'ilias':
                merged_query_embedding = torch.cat(merged_query_embedding, dim=0).to(model.device)
                merged_candicate_object_embedding = torch.cat(merged_candicate_object_embedding, dim=0)
                merged_query_embedding = F.normalize(merged_query_embedding)
                pred_scores = []
                for i in range(len(merged_candicate_object_embedding) // 100):
                    merged_candicate_object_embedding_i = F.normalize(merged_candicate_object_embedding[i*100:(i+1)*100].to(model.device))
                    pred_scores_i = merged_query_embedding @ merged_candicate_object_embedding_i.transpose(-1, -2)
                    pred_scores_i = pred_scores_i * model.logit_log_scale.exp()
                    pred_scores_i = pred_scores_i + model.logit_bias
                    pred_scores_i = pred_scores_i.float().sigmoid().cpu()
                    pred_scores.append(pred_scores_i)
                pred_scores = torch.cat(pred_scores, dim=1)
                pred_scores = pred_scores.reshape(len(merged_query_embedding), len(merged_candicate_object_embedding) // 100, 100)
                pred_scores = torch.max(pred_scores, dim=-1)[0]
                true_indices = dataset.candidates_idx
                query_indices = dataset.query_idx
                selected_pred_scores = pred_scores[query_indices]
                mAP50 = compute_map_at_k(selected_pred_scores, true_indices, k=50)
                print(f"mAP@50: {mAP50:.6f}")
            if args.dataset == 'ilias_i2i':
                merged_query_embedding = torch.cat(merged_candicate_image_embedding[:len(dataset.query_idx)], dim=0).to(model.device)
                merged_query_embedding = F.normalize(merged_query_embedding)
                merged_candicate_object_embedding = torch.cat(merged_candicate_object_embedding[len(dataset.query_idx):], dim=0)
                pred_scores = []
                for i in range(len(merged_candicate_object_embedding) // 100):
                    merged_candicate_object_embedding_i = F.normalize(merged_candicate_object_embedding[i*100:(i+1)*100].to(model.device))
                    pred_scores_i = merged_query_embedding @ merged_candicate_object_embedding_i.transpose(-1, -2)
                    pred_scores_i = pred_scores_i * model.logit_log_scale.exp()
                    pred_scores_i = pred_scores_i + model.logit_bias
                    pred_scores_i = pred_scores_i.float().sigmoid().cpu()
                    pred_scores.append(pred_scores_i)
                pred_scores = torch.cat(pred_scores, dim=1)
                pred_scores = pred_scores.reshape(len(merged_query_embedding), len(merged_candicate_object_embedding) // 100, 100)
                pred_scores = torch.max(pred_scores, dim=-1)[0]
                true_indices = dataset.candidates_idx
                mAP50 = compute_map_at_k(pred_scores, true_indices, k=50)
                print(f"mAP@50: {mAP50:.6f}")
            
            
        torch.distributed.barrier()
