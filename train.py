import os
import bisect
import json
import random
import torch
from torch.utils.data import DataLoader, Sampler, ConcatDataset
from transformers import AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_sagemaker_mp_enabled, logging
import numpy as np
import transformers
from transformers import Trainer
from models.vision_process import process_vision_info
from models.qwen3vl_objembed import ObjectEmbed
from torch.utils.data import Dataset

from typing import List, Dict, Any, Iterator
import numpy as np
from torchvision.ops.boxes import box_area
from PIL import Image, ImageDraw, ImageFont
import copy
import math

logger = logging.get_logger(__name__)


from dataclasses import dataclass, field

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    freeze_vision_modules: bool = False
    freeze_llm_modules: bool = False

    per_image_train_text_batch_size: int = field(
        default=10,
        metadata={"help": "number of text per image"},
    )

    num_classes: int = field(
        default=80,
        metadata={"help": "number of text per image"},
    )

    model_name_or_path: str = None
    dataset_name: str = 'mixture'
    use_task_prompt: bool = False
    use_global_caption: bool = False
    use_two_tokens: int = 0
    use_two_captions: bool = False



class GroupedBatchSampler(Sampler[List[int]]):
    """
    自定义的 Batch Sampler，确保每个批次的数据都来自同一个数据集。

    它与 ConcatDataset 一起使用。
    """
    def __init__(self, dataset: ConcatDataset, batch_size: int, shuffle: bool = True):
        """
        Args:
            dataset (ConcatDataset): 必须是 ConcatDataset。
            batch_size (int): 每个批次的大小。
            shuffle (bool): 是否在每个 epoch 开始时打乱批次的顺序。
        """
        if not isinstance(dataset, ConcatDataset):
            raise TypeError("Dataset must be a ConcatDataset")
            
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_device = torch.distributed.get_world_size()
        self.total_batch_size = self.batch_size * self.num_device

        # 计算每个数据集中有多少个完整的批次
        self.dataset_batch_sizes = [
            len(d) // self.total_batch_size for d in self.dataset.datasets
        ]
        self.num_batches = sum(self.dataset_batch_sizes)
        
        # 预先生成所有批次的索引，以便快速迭代
        self._generate_batches()

    def _generate_batches(self):
        self.batches = []
        # 遍历每个子数据集
        for dataset_idx, single_dataset in enumerate(self.dataset.datasets):
            # 获取当前数据集在 ConcatDataset 中的起始索引
            start_index = self.dataset.cumulative_sizes[dataset_idx-1] if dataset_idx > 0 else 0
            
            # 为当前数据集生成索引池
            indices = np.arange(len(single_dataset))
            if self.shuffle:
                np.random.shuffle(indices)
            
            # 将索引池切分为批次
            num_batches_in_dataset = len(indices) // self.total_batch_size
            for i in range(num_batches_in_dataset):
                batch_indices = indices[i * self.total_batch_size : (i + 1) * self.total_batch_size]
                # 将局部索引转换为在 ConcatDataset 中的全局索引
                global_indices = (start_index + batch_indices).tolist()
                self.batches.append(global_indices)
        
        # 如果需要，打乱所有批次的顺序
        if self.shuffle:
            np.random.shuffle(self.batches)

    def __iter__(self) -> Iterator[List[int]]:
        # 在每个 epoch 开始时，重新生成并打乱批次
        self._generate_batches()

        for chunk in self.batches:
            rank = torch.distributed.get_rank()
            yield chunk[rank*self.batch_size: (rank+1)*self.batch_size]

    def __len__(self) -> int:
        # 返回总的批次数量
        return self.num_batches


# 自定义 Trainer
class CustomTrainer(Trainer):
    """
    一个支持直接传入train_dataloader的Trainer类，继承自BaseHgTrainer。

    """
    
    def get_train_dataloader(self) -> DataLoader:
        """
        返回训练用的 DataLoader。
        这里是我们插入自定义 BatchSampler 的地方。
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # 确保 train_dataset 是 ConcatDataset
        if not isinstance(self.train_dataset, ConcatDataset):
            raise TypeError(
                "This custom trainer requires the train_dataset to be a ConcatDataset."
            )

        # 创建我们的自定义 Batch Sampler
        batch_sampler = GroupedBatchSampler(
            dataset=self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True
        )

        # 创建 DataLoader
        # 注意：当提供了 batch_sampler 时，batch_size, shuffle, sampler, drop_last 都必须为 None
        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def create_optimizer(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        # 如果已经存在优化器，跳过
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and 'visual' in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * 0.1,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and 'visual' in n)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate * 0.1,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and 'visual' not in n and 'logit' not in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and 'visual' not in n and 'logit' not in n)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and 'logit' in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * 10,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and "logit" in n)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate * 10,
                },
            ]
            
            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if "bitsandbytes" in str(optimizer_cls) and optimizer_kwargs.get("optim_bits", None) == 8:
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2**20}M params")

        if is_sagemaker_mp_enabled():
            import smdistributed.modelparallel.torch as smp
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

 
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou


LVIS_CATEGORY_IMAGE_COUNT = [{'id': 1, 'image_count': 64}, {'id': 2, 'image_count': 364}, {'id': 3, 'image_count': 1911}, {'id': 4, 'image_count': 149}, {'id': 5, 'image_count': 29}, {'id': 6, 'image_count': 26}, {'id': 7, 'image_count': 59}, {'id': 8, 'image_count': 22}, {'id': 9, 'image_count': 12}, {'id': 10, 'image_count': 28}, {'id': 11, 'image_count': 505}, {'id': 12, 'image_count': 1207}, {'id': 13, 'image_count': 4}, {'id': 14, 'image_count': 10}, {'id': 15, 'image_count': 500}, {'id': 16, 'image_count': 33}, {'id': 17, 'image_count': 3}, {'id': 18, 'image_count': 44}, {'id': 19, 'image_count': 561}, {'id': 20, 'image_count': 8}, {'id': 21, 'image_count': 9}, {'id': 22, 'image_count': 33}, {'id': 23, 'image_count': 1883}, {'id': 24, 'image_count': 98}, {'id': 25, 'image_count': 70}, {'id': 26, 'image_count': 46}, {'id': 27, 'image_count': 117}, {'id': 28, 'image_count': 41}, {'id': 29, 'image_count': 1395}, {'id': 30, 'image_count': 7}, {'id': 31, 'image_count': 1}, {'id': 32, 'image_count': 314}, {'id': 33, 'image_count': 31}, {'id': 34, 'image_count': 1905}, {'id': 35, 'image_count': 1859}, {'id': 36, 'image_count': 1623}, {'id': 37, 'image_count': 47}, {'id': 38, 'image_count': 3}, {'id': 39, 'image_count': 3}, {'id': 40, 'image_count': 1}, {'id': 41, 'image_count': 305}, {'id': 42, 'image_count': 6}, {'id': 43, 'image_count': 210}, {'id': 44, 'image_count': 36}, {'id': 45, 'image_count': 1787}, {'id': 46, 'image_count': 17}, {'id': 47, 'image_count': 51}, {'id': 48, 'image_count': 138}, {'id': 49, 'image_count': 3}, {'id': 50, 'image_count': 1470}, {'id': 51, 'image_count': 3}, {'id': 52, 'image_count': 2}, {'id': 53, 'image_count': 186}, {'id': 54, 'image_count': 76}, {'id': 55, 'image_count': 26}, {'id': 56, 'image_count': 303}, {'id': 57, 'image_count': 738}, {'id': 58, 'image_count': 1799}, {'id': 59, 'image_count': 1934}, {'id': 60, 'image_count': 1609}, {'id': 61, 'image_count': 1622}, {'id': 62, 'image_count': 41}, {'id': 63, 'image_count': 4}, {'id': 64, 'image_count': 11}, {'id': 65, 'image_count': 270}, {'id': 66, 'image_count': 349}, {'id': 67, 'image_count': 42}, {'id': 68, 'image_count': 823}, {'id': 69, 'image_count': 6}, {'id': 70, 'image_count': 48}, {'id': 71, 'image_count': 3}, {'id': 72, 'image_count': 42}, {'id': 73, 'image_count': 24}, {'id': 74, 'image_count': 16}, {'id': 75, 'image_count': 605}, {'id': 76, 'image_count': 646}, {'id': 77, 'image_count': 1765}, {'id': 78, 'image_count': 2}, {'id': 79, 'image_count': 125}, {'id': 80, 'image_count': 1420}, {'id': 81, 'image_count': 140}, {'id': 82, 'image_count': 4}, {'id': 83, 'image_count': 322}, {'id': 84, 'image_count': 60}, {'id': 85, 'image_count': 2}, {'id': 86, 'image_count': 231}, {'id': 87, 'image_count': 333}, {'id': 88, 'image_count': 1941}, {'id': 89, 'image_count': 367}, {'id': 90, 'image_count': 1922}, {'id': 91, 'image_count': 18}, {'id': 92, 'image_count': 81}, {'id': 93, 'image_count': 1}, {'id': 94, 'image_count': 1852}, {'id': 95, 'image_count': 430}, {'id': 96, 'image_count': 247}, {'id': 97, 'image_count': 94}, {'id': 98, 'image_count': 21}, {'id': 99, 'image_count': 1821}, {'id': 100, 'image_count': 16}, {'id': 101, 'image_count': 12}, {'id': 102, 'image_count': 25}, {'id': 103, 'image_count': 41}, {'id': 104, 'image_count': 244}, {'id': 105, 'image_count': 7}, {'id': 106, 'image_count': 1}, {'id': 107, 'image_count': 40}, {'id': 108, 'image_count': 40}, {'id': 109, 'image_count': 104}, {'id': 110, 'image_count': 1671}, {'id': 111, 'image_count': 49}, {'id': 112, 'image_count': 243}, {'id': 113, 'image_count': 2}, {'id': 114, 'image_count': 242}, {'id': 115, 'image_count': 271}, {'id': 116, 'image_count': 104}, {'id': 117, 'image_count': 8}, {'id': 118, 'image_count': 1758}, {'id': 119, 'image_count': 1}, {'id': 120, 'image_count': 48}, {'id': 121, 'image_count': 14}, {'id': 122, 'image_count': 40}, {'id': 123, 'image_count': 1}, {'id': 124, 'image_count': 37}, {'id': 125, 'image_count': 1510}, {'id': 126, 'image_count': 6}, {'id': 127, 'image_count': 1903}, {'id': 128, 'image_count': 70}, {'id': 129, 'image_count': 86}, {'id': 130, 'image_count': 7}, {'id': 131, 'image_count': 5}, {'id': 132, 'image_count': 1406}, {'id': 133, 'image_count': 1901}, {'id': 134, 'image_count': 15}, {'id': 135, 'image_count': 28}, {'id': 136, 'image_count': 6}, {'id': 137, 'image_count': 494}, {'id': 138, 'image_count': 234}, {'id': 139, 'image_count': 1922}, {'id': 140, 'image_count': 1}, {'id': 141, 'image_count': 35}, {'id': 142, 'image_count': 5}, {'id': 143, 'image_count': 1828}, {'id': 144, 'image_count': 8}, {'id': 145, 'image_count': 63}, {'id': 146, 'image_count': 1668}, {'id': 147, 'image_count': 4}, {'id': 148, 'image_count': 95}, {'id': 149, 'image_count': 17}, {'id': 150, 'image_count': 1567}, {'id': 151, 'image_count': 2}, {'id': 152, 'image_count': 103}, {'id': 153, 'image_count': 50}, {'id': 154, 'image_count': 1309}, {'id': 155, 'image_count': 6}, {'id': 156, 'image_count': 92}, {'id': 157, 'image_count': 19}, {'id': 158, 'image_count': 37}, {'id': 159, 'image_count': 4}, {'id': 160, 'image_count': 709}, {'id': 161, 'image_count': 9}, {'id': 162, 'image_count': 82}, {'id': 163, 'image_count': 15}, {'id': 164, 'image_count': 3}, {'id': 165, 'image_count': 61}, {'id': 166, 'image_count': 51}, {'id': 167, 'image_count': 5}, {'id': 168, 'image_count': 13}, {'id': 169, 'image_count': 642}, {'id': 170, 'image_count': 24}, {'id': 171, 'image_count': 255}, {'id': 172, 'image_count': 9}, {'id': 173, 'image_count': 1808}, {'id': 174, 'image_count': 31}, {'id': 175, 'image_count': 158}, {'id': 176, 'image_count': 80}, {'id': 177, 'image_count': 1884}, {'id': 178, 'image_count': 158}, {'id': 179, 'image_count': 2}, {'id': 180, 'image_count': 12}, {'id': 181, 'image_count': 1659}, {'id': 182, 'image_count': 7}, {'id': 183, 'image_count': 834}, {'id': 184, 'image_count': 57}, {'id': 185, 'image_count': 174}, {'id': 186, 'image_count': 95}, {'id': 187, 'image_count': 27}, {'id': 188, 'image_count': 22}, {'id': 189, 'image_count': 1391}, {'id': 190, 'image_count': 90}, {'id': 191, 'image_count': 40}, {'id': 192, 'image_count': 445}, {'id': 193, 'image_count': 21}, {'id': 194, 'image_count': 1132}, {'id': 195, 'image_count': 177}, {'id': 196, 'image_count': 4}, {'id': 197, 'image_count': 17}, {'id': 198, 'image_count': 84}, {'id': 199, 'image_count': 55}, {'id': 200, 'image_count': 30}, {'id': 201, 'image_count': 25}, {'id': 202, 'image_count': 2}, {'id': 203, 'image_count': 125}, {'id': 204, 'image_count': 1135}, {'id': 205, 'image_count': 19}, {'id': 206, 'image_count': 72}, {'id': 207, 'image_count': 1926}, {'id': 208, 'image_count': 159}, {'id': 209, 'image_count': 7}, {'id': 210, 'image_count': 1}, {'id': 211, 'image_count': 13}, {'id': 212, 'image_count': 35}, {'id': 213, 'image_count': 18}, {'id': 214, 'image_count': 8}, {'id': 215, 'image_count': 6}, {'id': 216, 'image_count': 35}, {'id': 217, 'image_count': 1222}, {'id': 218, 'image_count': 103}, {'id': 219, 'image_count': 28}, {'id': 220, 'image_count': 63}, {'id': 221, 'image_count': 28}, {'id': 222, 'image_count': 5}, {'id': 223, 'image_count': 7}, {'id': 224, 'image_count': 14}, {'id': 225, 'image_count': 1918}, {'id': 226, 'image_count': 133}, {'id': 227, 'image_count': 16}, {'id': 228, 'image_count': 27}, {'id': 229, 'image_count': 110}, {'id': 230, 'image_count': 1895}, {'id': 231, 'image_count': 4}, {'id': 232, 'image_count': 1927}, {'id': 233, 'image_count': 8}, {'id': 234, 'image_count': 1}, {'id': 235, 'image_count': 263}, {'id': 236, 'image_count': 10}, {'id': 237, 'image_count': 2}, {'id': 238, 'image_count': 3}, {'id': 239, 'image_count': 87}, {'id': 240, 'image_count': 9}, {'id': 241, 'image_count': 71}, {'id': 242, 'image_count': 13}, {'id': 243, 'image_count': 18}, {'id': 244, 'image_count': 2}, {'id': 245, 'image_count': 5}, {'id': 246, 'image_count': 45}, {'id': 247, 'image_count': 1}, {'id': 248, 'image_count': 23}, {'id': 249, 'image_count': 32}, {'id': 250, 'image_count': 4}, {'id': 251, 'image_count': 1}, {'id': 252, 'image_count': 858}, {'id': 253, 'image_count': 661}, {'id': 254, 'image_count': 168}, {'id': 255, 'image_count': 210}, {'id': 256, 'image_count': 65}, {'id': 257, 'image_count': 4}, {'id': 258, 'image_count': 2}, {'id': 259, 'image_count': 159}, {'id': 260, 'image_count': 31}, {'id': 261, 'image_count': 811}, {'id': 262, 'image_count': 1}, {'id': 263, 'image_count': 42}, {'id': 264, 'image_count': 27}, {'id': 265, 'image_count': 2}, {'id': 266, 'image_count': 5}, {'id': 267, 'image_count': 95}, {'id': 268, 'image_count': 32}, {'id': 269, 'image_count': 1}, {'id': 270, 'image_count': 1}, {'id': 271, 'image_count': 1844}, {'id': 272, 'image_count': 897}, {'id': 273, 'image_count': 31}, {'id': 274, 'image_count': 23}, {'id': 275, 'image_count': 1}, {'id': 276, 'image_count': 202}, {'id': 277, 'image_count': 746}, {'id': 278, 'image_count': 44}, {'id': 279, 'image_count': 14}, {'id': 280, 'image_count': 26}, {'id': 281, 'image_count': 1}, {'id': 282, 'image_count': 2}, {'id': 283, 'image_count': 25}, {'id': 284, 'image_count': 238}, {'id': 285, 'image_count': 592}, {'id': 286, 'image_count': 26}, {'id': 287, 'image_count': 5}, {'id': 288, 'image_count': 42}, {'id': 289, 'image_count': 13}, {'id': 290, 'image_count': 46}, {'id': 291, 'image_count': 1}, {'id': 292, 'image_count': 8}, {'id': 293, 'image_count': 34}, {'id': 294, 'image_count': 5}, {'id': 295, 'image_count': 1}, {'id': 296, 'image_count': 1871}, {'id': 297, 'image_count': 717}, {'id': 298, 'image_count': 1010}, {'id': 299, 'image_count': 679}, {'id': 300, 'image_count': 3}, {'id': 301, 'image_count': 4}, {'id': 302, 'image_count': 1}, {'id': 303, 'image_count': 166}, {'id': 304, 'image_count': 2}, {'id': 305, 'image_count': 266}, {'id': 306, 'image_count': 101}, {'id': 307, 'image_count': 6}, {'id': 308, 'image_count': 14}, {'id': 309, 'image_count': 133}, {'id': 310, 'image_count': 2}, {'id': 311, 'image_count': 38}, {'id': 312, 'image_count': 95}, {'id': 313, 'image_count': 1}, {'id': 314, 'image_count': 12}, {'id': 315, 'image_count': 49}, {'id': 316, 'image_count': 5}, {'id': 317, 'image_count': 5}, {'id': 318, 'image_count': 16}, {'id': 319, 'image_count': 216}, {'id': 320, 'image_count': 12}, {'id': 321, 'image_count': 1}, {'id': 322, 'image_count': 54}, {'id': 323, 'image_count': 5}, {'id': 324, 'image_count': 245}, {'id': 325, 'image_count': 12}, {'id': 326, 'image_count': 7}, {'id': 327, 'image_count': 35}, {'id': 328, 'image_count': 36}, {'id': 329, 'image_count': 32}, {'id': 330, 'image_count': 1027}, {'id': 331, 'image_count': 10}, {'id': 332, 'image_count': 12}, {'id': 333, 'image_count': 1}, {'id': 334, 'image_count': 67}, {'id': 335, 'image_count': 71}, {'id': 336, 'image_count': 30}, {'id': 337, 'image_count': 48}, {'id': 338, 'image_count': 249}, {'id': 339, 'image_count': 13}, {'id': 340, 'image_count': 29}, {'id': 341, 'image_count': 14}, {'id': 342, 'image_count': 236}, {'id': 343, 'image_count': 15}, {'id': 344, 'image_count': 1521}, {'id': 345, 'image_count': 25}, {'id': 346, 'image_count': 249}, {'id': 347, 'image_count': 139}, {'id': 348, 'image_count': 2}, {'id': 349, 'image_count': 2}, {'id': 350, 'image_count': 1890}, {'id': 351, 'image_count': 1240}, {'id': 352, 'image_count': 1}, {'id': 353, 'image_count': 9}, {'id': 354, 'image_count': 1}, {'id': 355, 'image_count': 3}, {'id': 356, 'image_count': 11}, {'id': 357, 'image_count': 4}, {'id': 358, 'image_count': 236}, {'id': 359, 'image_count': 44}, {'id': 360, 'image_count': 19}, {'id': 361, 'image_count': 1100}, {'id': 362, 'image_count': 7}, {'id': 363, 'image_count': 69}, {'id': 364, 'image_count': 2}, {'id': 365, 'image_count': 8}, {'id': 366, 'image_count': 5}, {'id': 367, 'image_count': 227}, {'id': 368, 'image_count': 6}, {'id': 369, 'image_count': 106}, {'id': 370, 'image_count': 81}, {'id': 371, 'image_count': 17}, {'id': 372, 'image_count': 134}, {'id': 373, 'image_count': 312}, {'id': 374, 'image_count': 8}, {'id': 375, 'image_count': 271}, {'id': 376, 'image_count': 2}, {'id': 377, 'image_count': 103}, {'id': 378, 'image_count': 1938}, {'id': 379, 'image_count': 574}, {'id': 380, 'image_count': 120}, {'id': 381, 'image_count': 2}, {'id': 382, 'image_count': 2}, {'id': 383, 'image_count': 13}, {'id': 384, 'image_count': 29}, {'id': 385, 'image_count': 1710}, {'id': 386, 'image_count': 66}, {'id': 387, 'image_count': 1008}, {'id': 388, 'image_count': 1}, {'id': 389, 'image_count': 3}, {'id': 390, 'image_count': 1942}, {'id': 391, 'image_count': 19}, {'id': 392, 'image_count': 1488}, {'id': 393, 'image_count': 46}, {'id': 394, 'image_count': 106}, {'id': 395, 'image_count': 115}, {'id': 396, 'image_count': 19}, {'id': 397, 'image_count': 2}, {'id': 398, 'image_count': 1}, {'id': 399, 'image_count': 28}, {'id': 400, 'image_count': 9}, {'id': 401, 'image_count': 192}, {'id': 402, 'image_count': 12}, {'id': 403, 'image_count': 21}, {'id': 404, 'image_count': 247}, {'id': 405, 'image_count': 6}, {'id': 406, 'image_count': 64}, {'id': 407, 'image_count': 7}, {'id': 408, 'image_count': 40}, {'id': 409, 'image_count': 542}, {'id': 410, 'image_count': 2}, {'id': 411, 'image_count': 1898}, {'id': 412, 'image_count': 36}, {'id': 413, 'image_count': 4}, {'id': 414, 'image_count': 1}, {'id': 415, 'image_count': 191}, {'id': 416, 'image_count': 6}, {'id': 417, 'image_count': 41}, {'id': 418, 'image_count': 39}, {'id': 419, 'image_count': 46}, {'id': 420, 'image_count': 1}, {'id': 421, 'image_count': 1451}, {'id': 422, 'image_count': 1878}, {'id': 423, 'image_count': 11}, {'id': 424, 'image_count': 82}, {'id': 425, 'image_count': 18}, {'id': 426, 'image_count': 1}, {'id': 427, 'image_count': 7}, {'id': 428, 'image_count': 3}, {'id': 429, 'image_count': 575}, {'id': 430, 'image_count': 1907}, {'id': 431, 'image_count': 8}, {'id': 432, 'image_count': 4}, {'id': 433, 'image_count': 32}, {'id': 434, 'image_count': 11}, {'id': 435, 'image_count': 4}, {'id': 436, 'image_count': 54}, {'id': 437, 'image_count': 202}, {'id': 438, 'image_count': 32}, {'id': 439, 'image_count': 3}, {'id': 440, 'image_count': 130}, {'id': 441, 'image_count': 119}, {'id': 442, 'image_count': 141}, {'id': 443, 'image_count': 29}, {'id': 444, 'image_count': 525}, {'id': 445, 'image_count': 1323}, {'id': 446, 'image_count': 2}, {'id': 447, 'image_count': 113}, {'id': 448, 'image_count': 16}, {'id': 449, 'image_count': 7}, {'id': 450, 'image_count': 35}, {'id': 451, 'image_count': 1908}, {'id': 452, 'image_count': 353}, {'id': 453, 'image_count': 18}, {'id': 454, 'image_count': 14}, {'id': 455, 'image_count': 77}, {'id': 456, 'image_count': 8}, {'id': 457, 'image_count': 37}, {'id': 458, 'image_count': 1}, {'id': 459, 'image_count': 346}, {'id': 460, 'image_count': 19}, {'id': 461, 'image_count': 1779}, {'id': 462, 'image_count': 23}, {'id': 463, 'image_count': 25}, {'id': 464, 'image_count': 67}, {'id': 465, 'image_count': 19}, {'id': 466, 'image_count': 28}, {'id': 467, 'image_count': 4}, {'id': 468, 'image_count': 27}, {'id': 469, 'image_count': 1861}, {'id': 470, 'image_count': 11}, {'id': 471, 'image_count': 13}, {'id': 472, 'image_count': 13}, {'id': 473, 'image_count': 32}, {'id': 474, 'image_count': 1767}, {'id': 475, 'image_count': 42}, {'id': 476, 'image_count': 17}, {'id': 477, 'image_count': 128}, {'id': 478, 'image_count': 1}, {'id': 479, 'image_count': 9}, {'id': 480, 'image_count': 10}, {'id': 481, 'image_count': 4}, {'id': 482, 'image_count': 9}, {'id': 483, 'image_count': 18}, {'id': 484, 'image_count': 41}, {'id': 485, 'image_count': 28}, {'id': 486, 'image_count': 3}, {'id': 487, 'image_count': 65}, {'id': 488, 'image_count': 9}, {'id': 489, 'image_count': 23}, {'id': 490, 'image_count': 24}, {'id': 491, 'image_count': 1}, {'id': 492, 'image_count': 2}, {'id': 493, 'image_count': 59}, {'id': 494, 'image_count': 48}, {'id': 495, 'image_count': 17}, {'id': 496, 'image_count': 1877}, {'id': 497, 'image_count': 18}, {'id': 498, 'image_count': 1920}, {'id': 499, 'image_count': 50}, {'id': 500, 'image_count': 1890}, {'id': 501, 'image_count': 99}, {'id': 502, 'image_count': 1530}, {'id': 503, 'image_count': 3}, {'id': 504, 'image_count': 11}, {'id': 505, 'image_count': 19}, {'id': 506, 'image_count': 3}, {'id': 507, 'image_count': 63}, {'id': 508, 'image_count': 5}, {'id': 509, 'image_count': 6}, {'id': 510, 'image_count': 233}, {'id': 511, 'image_count': 54}, {'id': 512, 'image_count': 36}, {'id': 513, 'image_count': 10}, {'id': 514, 'image_count': 124}, {'id': 515, 'image_count': 101}, {'id': 516, 'image_count': 3}, {'id': 517, 'image_count': 363}, {'id': 518, 'image_count': 3}, {'id': 519, 'image_count': 30}, {'id': 520, 'image_count': 18}, {'id': 521, 'image_count': 199}, {'id': 522, 'image_count': 97}, {'id': 523, 'image_count': 32}, {'id': 524, 'image_count': 121}, {'id': 525, 'image_count': 16}, {'id': 526, 'image_count': 12}, {'id': 527, 'image_count': 2}, {'id': 528, 'image_count': 214}, {'id': 529, 'image_count': 48}, {'id': 530, 'image_count': 26}, {'id': 531, 'image_count': 13}, {'id': 532, 'image_count': 4}, {'id': 533, 'image_count': 11}, {'id': 534, 'image_count': 123}, {'id': 535, 'image_count': 7}, {'id': 536, 'image_count': 200}, {'id': 537, 'image_count': 91}, {'id': 538, 'image_count': 9}, {'id': 539, 'image_count': 72}, {'id': 540, 'image_count': 1886}, {'id': 541, 'image_count': 4}, {'id': 542, 'image_count': 1}, {'id': 543, 'image_count': 1}, {'id': 544, 'image_count': 1932}, {'id': 545, 'image_count': 4}, {'id': 546, 'image_count': 56}, {'id': 547, 'image_count': 854}, {'id': 548, 'image_count': 755}, {'id': 549, 'image_count': 1843}, {'id': 550, 'image_count': 96}, {'id': 551, 'image_count': 7}, {'id': 552, 'image_count': 74}, {'id': 553, 'image_count': 66}, {'id': 554, 'image_count': 57}, {'id': 555, 'image_count': 44}, {'id': 556, 'image_count': 1905}, {'id': 557, 'image_count': 4}, {'id': 558, 'image_count': 90}, {'id': 559, 'image_count': 1635}, {'id': 560, 'image_count': 8}, {'id': 561, 'image_count': 5}, {'id': 562, 'image_count': 50}, {'id': 563, 'image_count': 545}, {'id': 564, 'image_count': 20}, {'id': 565, 'image_count': 193}, {'id': 566, 'image_count': 285}, {'id': 567, 'image_count': 3}, {'id': 568, 'image_count': 1}, {'id': 569, 'image_count': 1904}, {'id': 570, 'image_count': 294}, {'id': 571, 'image_count': 3}, {'id': 572, 'image_count': 5}, {'id': 573, 'image_count': 24}, {'id': 574, 'image_count': 2}, {'id': 575, 'image_count': 2}, {'id': 576, 'image_count': 16}, {'id': 577, 'image_count': 8}, {'id': 578, 'image_count': 154}, {'id': 579, 'image_count': 66}, {'id': 580, 'image_count': 1}, {'id': 581, 'image_count': 24}, {'id': 582, 'image_count': 1}, {'id': 583, 'image_count': 4}, {'id': 584, 'image_count': 75}, {'id': 585, 'image_count': 6}, {'id': 586, 'image_count': 126}, {'id': 587, 'image_count': 24}, {'id': 588, 'image_count': 22}, {'id': 589, 'image_count': 1872}, {'id': 590, 'image_count': 16}, {'id': 591, 'image_count': 423}, {'id': 592, 'image_count': 1927}, {'id': 593, 'image_count': 38}, {'id': 594, 'image_count': 3}, {'id': 595, 'image_count': 1945}, {'id': 596, 'image_count': 35}, {'id': 597, 'image_count': 1}, {'id': 598, 'image_count': 13}, {'id': 599, 'image_count': 9}, {'id': 600, 'image_count': 14}, {'id': 601, 'image_count': 37}, {'id': 602, 'image_count': 3}, {'id': 603, 'image_count': 4}, {'id': 604, 'image_count': 100}, {'id': 605, 'image_count': 195}, {'id': 606, 'image_count': 1}, {'id': 607, 'image_count': 12}, {'id': 608, 'image_count': 24}, {'id': 609, 'image_count': 489}, {'id': 610, 'image_count': 10}, {'id': 611, 'image_count': 1689}, {'id': 612, 'image_count': 42}, {'id': 613, 'image_count': 81}, {'id': 614, 'image_count': 894}, {'id': 615, 'image_count': 1868}, {'id': 616, 'image_count': 7}, {'id': 617, 'image_count': 1567}, {'id': 618, 'image_count': 10}, {'id': 619, 'image_count': 8}, {'id': 620, 'image_count': 7}, {'id': 621, 'image_count': 629}, {'id': 622, 'image_count': 89}, {'id': 623, 'image_count': 15}, {'id': 624, 'image_count': 134}, {'id': 625, 'image_count': 4}, {'id': 626, 'image_count': 1802}, {'id': 627, 'image_count': 595}, {'id': 628, 'image_count': 1210}, {'id': 629, 'image_count': 48}, {'id': 630, 'image_count': 418}, {'id': 631, 'image_count': 1846}, {'id': 632, 'image_count': 5}, {'id': 633, 'image_count': 221}, {'id': 634, 'image_count': 10}, {'id': 635, 'image_count': 7}, {'id': 636, 'image_count': 76}, {'id': 637, 'image_count': 22}, {'id': 638, 'image_count': 10}, {'id': 639, 'image_count': 341}, {'id': 640, 'image_count': 1}, {'id': 641, 'image_count': 705}, {'id': 642, 'image_count': 1900}, {'id': 643, 'image_count': 188}, {'id': 644, 'image_count': 227}, {'id': 645, 'image_count': 861}, {'id': 646, 'image_count': 6}, {'id': 647, 'image_count': 115}, {'id': 648, 'image_count': 5}, {'id': 649, 'image_count': 43}, {'id': 650, 'image_count': 14}, {'id': 651, 'image_count': 6}, {'id': 652, 'image_count': 15}, {'id': 653, 'image_count': 1167}, {'id': 654, 'image_count': 15}, {'id': 655, 'image_count': 994}, {'id': 656, 'image_count': 28}, {'id': 657, 'image_count': 2}, {'id': 658, 'image_count': 338}, {'id': 659, 'image_count': 334}, {'id': 660, 'image_count': 15}, {'id': 661, 'image_count': 102}, {'id': 662, 'image_count': 1}, {'id': 663, 'image_count': 8}, {'id': 664, 'image_count': 1}, {'id': 665, 'image_count': 1}, {'id': 666, 'image_count': 28}, {'id': 667, 'image_count': 91}, {'id': 668, 'image_count': 260}, {'id': 669, 'image_count': 131}, {'id': 670, 'image_count': 128}, {'id': 671, 'image_count': 3}, {'id': 672, 'image_count': 10}, {'id': 673, 'image_count': 39}, {'id': 674, 'image_count': 2}, {'id': 675, 'image_count': 925}, {'id': 676, 'image_count': 354}, {'id': 677, 'image_count': 31}, {'id': 678, 'image_count': 10}, {'id': 679, 'image_count': 215}, {'id': 680, 'image_count': 71}, {'id': 681, 'image_count': 43}, {'id': 682, 'image_count': 28}, {'id': 683, 'image_count': 34}, {'id': 684, 'image_count': 16}, {'id': 685, 'image_count': 273}, {'id': 686, 'image_count': 2}, {'id': 687, 'image_count': 999}, {'id': 688, 'image_count': 4}, {'id': 689, 'image_count': 107}, {'id': 690, 'image_count': 2}, {'id': 691, 'image_count': 1}, {'id': 692, 'image_count': 454}, {'id': 693, 'image_count': 9}, {'id': 694, 'image_count': 1901}, {'id': 695, 'image_count': 61}, {'id': 696, 'image_count': 91}, {'id': 697, 'image_count': 46}, {'id': 698, 'image_count': 1402}, {'id': 699, 'image_count': 74}, {'id': 700, 'image_count': 421}, {'id': 701, 'image_count': 226}, {'id': 702, 'image_count': 10}, {'id': 703, 'image_count': 1720}, {'id': 704, 'image_count': 261}, {'id': 705, 'image_count': 1337}, {'id': 706, 'image_count': 293}, {'id': 707, 'image_count': 62}, {'id': 708, 'image_count': 814}, {'id': 709, 'image_count': 407}, {'id': 710, 'image_count': 6}, {'id': 711, 'image_count': 16}, {'id': 712, 'image_count': 7}, {'id': 713, 'image_count': 1791}, {'id': 714, 'image_count': 2}, {'id': 715, 'image_count': 1915}, {'id': 716, 'image_count': 1940}, {'id': 717, 'image_count': 13}, {'id': 718, 'image_count': 16}, {'id': 719, 'image_count': 448}, {'id': 720, 'image_count': 12}, {'id': 721, 'image_count': 18}, {'id': 722, 'image_count': 4}, {'id': 723, 'image_count': 71}, {'id': 724, 'image_count': 189}, {'id': 725, 'image_count': 74}, {'id': 726, 'image_count': 103}, {'id': 727, 'image_count': 3}, {'id': 728, 'image_count': 110}, {'id': 729, 'image_count': 5}, {'id': 730, 'image_count': 9}, {'id': 731, 'image_count': 15}, {'id': 732, 'image_count': 25}, {'id': 733, 'image_count': 7}, {'id': 734, 'image_count': 647}, {'id': 735, 'image_count': 824}, {'id': 736, 'image_count': 100}, {'id': 737, 'image_count': 47}, {'id': 738, 'image_count': 121}, {'id': 739, 'image_count': 731}, {'id': 740, 'image_count': 73}, {'id': 741, 'image_count': 49}, {'id': 742, 'image_count': 23}, {'id': 743, 'image_count': 4}, {'id': 744, 'image_count': 62}, {'id': 745, 'image_count': 118}, {'id': 746, 'image_count': 99}, {'id': 747, 'image_count': 40}, {'id': 748, 'image_count': 1036}, {'id': 749, 'image_count': 105}, {'id': 750, 'image_count': 21}, {'id': 751, 'image_count': 229}, {'id': 752, 'image_count': 7}, {'id': 753, 'image_count': 72}, {'id': 754, 'image_count': 9}, {'id': 755, 'image_count': 10}, {'id': 756, 'image_count': 328}, {'id': 757, 'image_count': 468}, {'id': 758, 'image_count': 1}, {'id': 759, 'image_count': 2}, {'id': 760, 'image_count': 24}, {'id': 761, 'image_count': 11}, {'id': 762, 'image_count': 72}, {'id': 763, 'image_count': 17}, {'id': 764, 'image_count': 10}, {'id': 765, 'image_count': 17}, {'id': 766, 'image_count': 489}, {'id': 767, 'image_count': 47}, {'id': 768, 'image_count': 93}, {'id': 769, 'image_count': 1}, {'id': 770, 'image_count': 12}, {'id': 771, 'image_count': 228}, {'id': 772, 'image_count': 5}, {'id': 773, 'image_count': 76}, {'id': 774, 'image_count': 71}, {'id': 775, 'image_count': 30}, {'id': 776, 'image_count': 109}, {'id': 777, 'image_count': 14}, {'id': 778, 'image_count': 1}, {'id': 779, 'image_count': 8}, {'id': 780, 'image_count': 26}, {'id': 781, 'image_count': 339}, {'id': 782, 'image_count': 153}, {'id': 783, 'image_count': 2}, {'id': 784, 'image_count': 3}, {'id': 785, 'image_count': 8}, {'id': 786, 'image_count': 47}, {'id': 787, 'image_count': 8}, {'id': 788, 'image_count': 6}, {'id': 789, 'image_count': 116}, {'id': 790, 'image_count': 69}, {'id': 791, 'image_count': 13}, {'id': 792, 'image_count': 6}, {'id': 793, 'image_count': 1928}, {'id': 794, 'image_count': 79}, {'id': 795, 'image_count': 14}, {'id': 796, 'image_count': 7}, {'id': 797, 'image_count': 20}, {'id': 798, 'image_count': 114}, {'id': 799, 'image_count': 221}, {'id': 800, 'image_count': 502}, {'id': 801, 'image_count': 62}, {'id': 802, 'image_count': 87}, {'id': 803, 'image_count': 4}, {'id': 804, 'image_count': 1912}, {'id': 805, 'image_count': 7}, {'id': 806, 'image_count': 186}, {'id': 807, 'image_count': 18}, {'id': 808, 'image_count': 4}, {'id': 809, 'image_count': 3}, {'id': 810, 'image_count': 7}, {'id': 811, 'image_count': 1413}, {'id': 812, 'image_count': 7}, {'id': 813, 'image_count': 12}, {'id': 814, 'image_count': 248}, {'id': 815, 'image_count': 4}, {'id': 816, 'image_count': 1881}, {'id': 817, 'image_count': 529}, {'id': 818, 'image_count': 1932}, {'id': 819, 'image_count': 50}, {'id': 820, 'image_count': 3}, {'id': 821, 'image_count': 28}, {'id': 822, 'image_count': 10}, {'id': 823, 'image_count': 5}, {'id': 824, 'image_count': 5}, {'id': 825, 'image_count': 18}, {'id': 826, 'image_count': 14}, {'id': 827, 'image_count': 1890}, {'id': 828, 'image_count': 660}, {'id': 829, 'image_count': 8}, {'id': 830, 'image_count': 25}, {'id': 831, 'image_count': 10}, {'id': 832, 'image_count': 218}, {'id': 833, 'image_count': 36}, {'id': 834, 'image_count': 16}, {'id': 835, 'image_count': 808}, {'id': 836, 'image_count': 479}, {'id': 837, 'image_count': 1404}, {'id': 838, 'image_count': 307}, {'id': 839, 'image_count': 57}, {'id': 840, 'image_count': 28}, {'id': 841, 'image_count': 80}, {'id': 842, 'image_count': 11}, {'id': 843, 'image_count': 92}, {'id': 844, 'image_count': 20}, {'id': 845, 'image_count': 194}, {'id': 846, 'image_count': 23}, {'id': 847, 'image_count': 52}, {'id': 848, 'image_count': 673}, {'id': 849, 'image_count': 2}, {'id': 850, 'image_count': 2}, {'id': 851, 'image_count': 1}, {'id': 852, 'image_count': 2}, {'id': 853, 'image_count': 8}, {'id': 854, 'image_count': 80}, {'id': 855, 'image_count': 3}, {'id': 856, 'image_count': 3}, {'id': 857, 'image_count': 15}, {'id': 858, 'image_count': 2}, {'id': 859, 'image_count': 10}, {'id': 860, 'image_count': 386}, {'id': 861, 'image_count': 65}, {'id': 862, 'image_count': 3}, {'id': 863, 'image_count': 35}, {'id': 864, 'image_count': 5}, {'id': 865, 'image_count': 180}, {'id': 866, 'image_count': 99}, {'id': 867, 'image_count': 49}, {'id': 868, 'image_count': 28}, {'id': 869, 'image_count': 1}, {'id': 870, 'image_count': 52}, {'id': 871, 'image_count': 36}, {'id': 872, 'image_count': 70}, {'id': 873, 'image_count': 6}, {'id': 874, 'image_count': 29}, {'id': 875, 'image_count': 24}, {'id': 876, 'image_count': 1115}, {'id': 877, 'image_count': 61}, {'id': 878, 'image_count': 18}, {'id': 879, 'image_count': 18}, {'id': 880, 'image_count': 665}, {'id': 881, 'image_count': 1096}, {'id': 882, 'image_count': 29}, {'id': 883, 'image_count': 8}, {'id': 884, 'image_count': 14}, {'id': 885, 'image_count': 1622}, {'id': 886, 'image_count': 2}, {'id': 887, 'image_count': 3}, {'id': 888, 'image_count': 32}, {'id': 889, 'image_count': 55}, {'id': 890, 'image_count': 1}, {'id': 891, 'image_count': 10}, {'id': 892, 'image_count': 10}, {'id': 893, 'image_count': 47}, {'id': 894, 'image_count': 3}, {'id': 895, 'image_count': 29}, {'id': 896, 'image_count': 342}, {'id': 897, 'image_count': 25}, {'id': 898, 'image_count': 1469}, {'id': 899, 'image_count': 521}, {'id': 900, 'image_count': 347}, {'id': 901, 'image_count': 35}, {'id': 902, 'image_count': 7}, {'id': 903, 'image_count': 207}, {'id': 904, 'image_count': 108}, {'id': 905, 'image_count': 2}, {'id': 906, 'image_count': 34}, {'id': 907, 'image_count': 12}, {'id': 908, 'image_count': 10}, {'id': 909, 'image_count': 13}, {'id': 910, 'image_count': 361}, {'id': 911, 'image_count': 1023}, {'id': 912, 'image_count': 782}, {'id': 913, 'image_count': 2}, {'id': 914, 'image_count': 5}, {'id': 915, 'image_count': 247}, {'id': 916, 'image_count': 221}, {'id': 917, 'image_count': 4}, {'id': 918, 'image_count': 8}, {'id': 919, 'image_count': 158}, {'id': 920, 'image_count': 3}, {'id': 921, 'image_count': 752}, {'id': 922, 'image_count': 64}, {'id': 923, 'image_count': 707}, {'id': 924, 'image_count': 143}, {'id': 925, 'image_count': 1}, {'id': 926, 'image_count': 49}, {'id': 927, 'image_count': 126}, {'id': 928, 'image_count': 76}, {'id': 929, 'image_count': 11}, {'id': 930, 'image_count': 11}, {'id': 931, 'image_count': 4}, {'id': 932, 'image_count': 39}, {'id': 933, 'image_count': 11}, {'id': 934, 'image_count': 13}, {'id': 935, 'image_count': 91}, {'id': 936, 'image_count': 14}, {'id': 937, 'image_count': 5}, {'id': 938, 'image_count': 3}, {'id': 939, 'image_count': 10}, {'id': 940, 'image_count': 18}, {'id': 941, 'image_count': 9}, {'id': 942, 'image_count': 6}, {'id': 943, 'image_count': 951}, {'id': 944, 'image_count': 2}, {'id': 945, 'image_count': 1}, {'id': 946, 'image_count': 19}, {'id': 947, 'image_count': 1942}, {'id': 948, 'image_count': 1916}, {'id': 949, 'image_count': 139}, {'id': 950, 'image_count': 43}, {'id': 951, 'image_count': 1969}, {'id': 952, 'image_count': 5}, {'id': 953, 'image_count': 134}, {'id': 954, 'image_count': 74}, {'id': 955, 'image_count': 381}, {'id': 956, 'image_count': 1}, {'id': 957, 'image_count': 381}, {'id': 958, 'image_count': 6}, {'id': 959, 'image_count': 1826}, {'id': 960, 'image_count': 28}, {'id': 961, 'image_count': 1635}, {'id': 962, 'image_count': 1967}, {'id': 963, 'image_count': 16}, {'id': 964, 'image_count': 1926}, {'id': 965, 'image_count': 1789}, {'id': 966, 'image_count': 401}, {'id': 967, 'image_count': 1968}, {'id': 968, 'image_count': 1167}, {'id': 969, 'image_count': 1}, {'id': 970, 'image_count': 56}, {'id': 971, 'image_count': 17}, {'id': 972, 'image_count': 1}, {'id': 973, 'image_count': 58}, {'id': 974, 'image_count': 9}, {'id': 975, 'image_count': 8}, {'id': 976, 'image_count': 1124}, {'id': 977, 'image_count': 31}, {'id': 978, 'image_count': 16}, {'id': 979, 'image_count': 491}, {'id': 980, 'image_count': 432}, {'id': 981, 'image_count': 1945}, {'id': 982, 'image_count': 1899}, {'id': 983, 'image_count': 5}, {'id': 984, 'image_count': 28}, {'id': 985, 'image_count': 7}, {'id': 986, 'image_count': 146}, {'id': 987, 'image_count': 1}, {'id': 988, 'image_count': 25}, {'id': 989, 'image_count': 22}, {'id': 990, 'image_count': 1}, {'id': 991, 'image_count': 10}, {'id': 992, 'image_count': 9}, {'id': 993, 'image_count': 308}, {'id': 994, 'image_count': 4}, {'id': 995, 'image_count': 1969}, {'id': 996, 'image_count': 45}, {'id': 997, 'image_count': 12}, {'id': 998, 'image_count': 1}, {'id': 999, 'image_count': 85}, {'id': 1000, 'image_count': 1127}, {'id': 1001, 'image_count': 11}, {'id': 1002, 'image_count': 60}, {'id': 1003, 'image_count': 1}, {'id': 1004, 'image_count': 16}, {'id': 1005, 'image_count': 1}, {'id': 1006, 'image_count': 65}, {'id': 1007, 'image_count': 13}, {'id': 1008, 'image_count': 655}, {'id': 1009, 'image_count': 51}, {'id': 1010, 'image_count': 1}, {'id': 1011, 'image_count': 673}, {'id': 1012, 'image_count': 5}, {'id': 1013, 'image_count': 36}, {'id': 1014, 'image_count': 54}, {'id': 1015, 'image_count': 5}, {'id': 1016, 'image_count': 8}, {'id': 1017, 'image_count': 305}, {'id': 1018, 'image_count': 297}, {'id': 1019, 'image_count': 1053}, {'id': 1020, 'image_count': 223}, {'id': 1021, 'image_count': 1037}, {'id': 1022, 'image_count': 63}, {'id': 1023, 'image_count': 1881}, {'id': 1024, 'image_count': 507}, {'id': 1025, 'image_count': 333}, {'id': 1026, 'image_count': 1911}, {'id': 1027, 'image_count': 1765}, {'id': 1028, 'image_count': 1}, {'id': 1029, 'image_count': 5}, {'id': 1030, 'image_count': 1}, {'id': 1031, 'image_count': 9}, {'id': 1032, 'image_count': 2}, {'id': 1033, 'image_count': 151}, {'id': 1034, 'image_count': 82}, {'id': 1035, 'image_count': 1931}, {'id': 1036, 'image_count': 41}, {'id': 1037, 'image_count': 1895}, {'id': 1038, 'image_count': 24}, {'id': 1039, 'image_count': 22}, {'id': 1040, 'image_count': 35}, {'id': 1041, 'image_count': 69}, {'id': 1042, 'image_count': 962}, {'id': 1043, 'image_count': 588}, {'id': 1044, 'image_count': 21}, {'id': 1045, 'image_count': 825}, {'id': 1046, 'image_count': 52}, {'id': 1047, 'image_count': 5}, {'id': 1048, 'image_count': 5}, {'id': 1049, 'image_count': 5}, {'id': 1050, 'image_count': 1860}, {'id': 1051, 'image_count': 56}, {'id': 1052, 'image_count': 1582}, {'id': 1053, 'image_count': 7}, {'id': 1054, 'image_count': 2}, {'id': 1055, 'image_count': 1562}, {'id': 1056, 'image_count': 1885}, {'id': 1057, 'image_count': 1}, {'id': 1058, 'image_count': 5}, {'id': 1059, 'image_count': 137}, {'id': 1060, 'image_count': 1094}, {'id': 1061, 'image_count': 134}, {'id': 1062, 'image_count': 29}, {'id': 1063, 'image_count': 22}, {'id': 1064, 'image_count': 522}, {'id': 1065, 'image_count': 50}, {'id': 1066, 'image_count': 68}, {'id': 1067, 'image_count': 16}, {'id': 1068, 'image_count': 40}, {'id': 1069, 'image_count': 35}, {'id': 1070, 'image_count': 135}, {'id': 1071, 'image_count': 1413}, {'id': 1072, 'image_count': 772}, {'id': 1073, 'image_count': 50}, {'id': 1074, 'image_count': 1015}, {'id': 1075, 'image_count': 1}, {'id': 1076, 'image_count': 65}, {'id': 1077, 'image_count': 1900}, {'id': 1078, 'image_count': 1302}, {'id': 1079, 'image_count': 1977}, {'id': 1080, 'image_count': 2}, {'id': 1081, 'image_count': 29}, {'id': 1082, 'image_count': 36}, {'id': 1083, 'image_count': 138}, {'id': 1084, 'image_count': 4}, {'id': 1085, 'image_count': 67}, {'id': 1086, 'image_count': 26}, {'id': 1087, 'image_count': 25}, {'id': 1088, 'image_count': 33}, {'id': 1089, 'image_count': 37}, {'id': 1090, 'image_count': 50}, {'id': 1091, 'image_count': 270}, {'id': 1092, 'image_count': 12}, {'id': 1093, 'image_count': 316}, {'id': 1094, 'image_count': 41}, {'id': 1095, 'image_count': 224}, {'id': 1096, 'image_count': 105}, {'id': 1097, 'image_count': 1925}, {'id': 1098, 'image_count': 1021}, {'id': 1099, 'image_count': 1213}, {'id': 1100, 'image_count': 172}, {'id': 1101, 'image_count': 28}, {'id': 1102, 'image_count': 745}, {'id': 1103, 'image_count': 187}, {'id': 1104, 'image_count': 147}, {'id': 1105, 'image_count': 136}, {'id': 1106, 'image_count': 34}, {'id': 1107, 'image_count': 41}, {'id': 1108, 'image_count': 636}, {'id': 1109, 'image_count': 570}, {'id': 1110, 'image_count': 1149}, {'id': 1111, 'image_count': 61}, {'id': 1112, 'image_count': 1890}, {'id': 1113, 'image_count': 18}, {'id': 1114, 'image_count': 143}, {'id': 1115, 'image_count': 1517}, {'id': 1116, 'image_count': 7}, {'id': 1117, 'image_count': 943}, {'id': 1118, 'image_count': 6}, {'id': 1119, 'image_count': 1}, {'id': 1120, 'image_count': 11}, {'id': 1121, 'image_count': 101}, {'id': 1122, 'image_count': 1909}, {'id': 1123, 'image_count': 800}, {'id': 1124, 'image_count': 1}, {'id': 1125, 'image_count': 44}, {'id': 1126, 'image_count': 3}, {'id': 1127, 'image_count': 44}, {'id': 1128, 'image_count': 31}, {'id': 1129, 'image_count': 7}, {'id': 1130, 'image_count': 20}, {'id': 1131, 'image_count': 11}, {'id': 1132, 'image_count': 13}, {'id': 1133, 'image_count': 1924}, {'id': 1134, 'image_count': 113}, {'id': 1135, 'image_count': 2}, {'id': 1136, 'image_count': 139}, {'id': 1137, 'image_count': 12}, {'id': 1138, 'image_count': 37}, {'id': 1139, 'image_count': 1866}, {'id': 1140, 'image_count': 47}, {'id': 1141, 'image_count': 1468}, {'id': 1142, 'image_count': 729}, {'id': 1143, 'image_count': 24}, {'id': 1144, 'image_count': 1}, {'id': 1145, 'image_count': 10}, {'id': 1146, 'image_count': 3}, {'id': 1147, 'image_count': 14}, {'id': 1148, 'image_count': 4}, {'id': 1149, 'image_count': 29}, {'id': 1150, 'image_count': 4}, {'id': 1151, 'image_count': 70}, {'id': 1152, 'image_count': 46}, {'id': 1153, 'image_count': 14}, {'id': 1154, 'image_count': 48}, {'id': 1155, 'image_count': 1855}, {'id': 1156, 'image_count': 113}, {'id': 1157, 'image_count': 1}, {'id': 1158, 'image_count': 1}, {'id': 1159, 'image_count': 10}, {'id': 1160, 'image_count': 54}, {'id': 1161, 'image_count': 1923}, {'id': 1162, 'image_count': 630}, {'id': 1163, 'image_count': 31}, {'id': 1164, 'image_count': 69}, {'id': 1165, 'image_count': 7}, {'id': 1166, 'image_count': 11}, {'id': 1167, 'image_count': 1}, {'id': 1168, 'image_count': 30}, {'id': 1169, 'image_count': 50}, {'id': 1170, 'image_count': 45}, {'id': 1171, 'image_count': 28}, {'id': 1172, 'image_count': 114}, {'id': 1173, 'image_count': 193}, {'id': 1174, 'image_count': 21}, {'id': 1175, 'image_count': 91}, {'id': 1176, 'image_count': 31}, {'id': 1177, 'image_count': 1469}, {'id': 1178, 'image_count': 1924}, {'id': 1179, 'image_count': 87}, {'id': 1180, 'image_count': 77}, {'id': 1181, 'image_count': 11}, {'id': 1182, 'image_count': 47}, {'id': 1183, 'image_count': 21}, {'id': 1184, 'image_count': 47}, {'id': 1185, 'image_count': 70}, {'id': 1186, 'image_count': 1838}, {'id': 1187, 'image_count': 19}, {'id': 1188, 'image_count': 531}, {'id': 1189, 'image_count': 11}, {'id': 1190, 'image_count': 941}, {'id': 1191, 'image_count': 113}, {'id': 1192, 'image_count': 26}, {'id': 1193, 'image_count': 5}, {'id': 1194, 'image_count': 56}, {'id': 1195, 'image_count': 73}, {'id': 1196, 'image_count': 32}, {'id': 1197, 'image_count': 128}, {'id': 1198, 'image_count': 623}, {'id': 1199, 'image_count': 12}, {'id': 1200, 'image_count': 52}, {'id': 1201, 'image_count': 11}, {'id': 1202, 'image_count': 1674}, {'id': 1203, 'image_count': 81}]  # noqa
LVIS_CLASSES = [
        'aerosol_can', 'air_conditioner', 'airplane', 'alarm_clock', 'alcohol',
        'alligator', 'almond', 'ambulance', 'amplifier', 'anklet', 'antenna',
        'apple', 'applesauce', 'apricot', 'apron', 'aquarium',
        'arctic_(type_of_shoe)', 'armband', 'armchair', 'armoire', 'armor',
        'artichoke', 'trash_can', 'ashtray', 'asparagus', 'atomizer',
        'avocado', 'award', 'awning', 'ax', 'baboon', 'baby_buggy',
        'basketball_backboard', 'backpack', 'handbag', 'suitcase', 'bagel',
        'bagpipe', 'baguet', 'bait', 'ball', 'ballet_skirt', 'balloon',
        'bamboo', 'banana', 'Band_Aid', 'bandage', 'bandanna', 'banjo',
        'banner', 'barbell', 'barge', 'barrel', 'barrette', 'barrow',
        'baseball_base', 'baseball', 'baseball_bat', 'baseball_cap',
        'baseball_glove', 'basket', 'basketball', 'bass_horn', 'bat_(animal)',
        'bath_mat', 'bath_towel', 'bathrobe', 'bathtub', 'batter_(food)',
        'battery', 'beachball', 'bead', 'bean_curd', 'beanbag', 'beanie',
        'bear', 'bed', 'bedpan', 'bedspread', 'cow', 'beef_(food)', 'beeper',
        'beer_bottle', 'beer_can', 'beetle', 'bell', 'bell_pepper', 'belt',
        'belt_buckle', 'bench', 'beret', 'bib', 'Bible', 'bicycle', 'visor',
        'billboard', 'binder', 'binoculars', 'bird', 'birdfeeder', 'birdbath',
        'birdcage', 'birdhouse', 'birthday_cake', 'birthday_card',
        'pirate_flag', 'black_sheep', 'blackberry', 'blackboard', 'blanket',
        'blazer', 'blender', 'blimp', 'blinker', 'blouse', 'blueberry',
        'gameboard', 'boat', 'bob', 'bobbin', 'bobby_pin', 'boiled_egg',
        'bolo_tie', 'deadbolt', 'bolt', 'bonnet', 'book', 'bookcase',
        'booklet', 'bookmark', 'boom_microphone', 'boot', 'bottle',
        'bottle_opener', 'bouquet', 'bow_(weapon)', 'bow_(decorative_ribbons)',
        'bow-tie', 'bowl', 'pipe_bowl', 'bowler_hat', 'bowling_ball', 'box',
        'boxing_glove', 'suspenders', 'bracelet', 'brass_plaque', 'brassiere',
        'bread-bin', 'bread', 'breechcloth', 'bridal_gown', 'briefcase',
        'broccoli', 'broach', 'broom', 'brownie', 'brussels_sprouts',
        'bubble_gum', 'bucket', 'horse_buggy', 'bull', 'bulldog', 'bulldozer',
        'bullet_train', 'bulletin_board', 'bulletproof_vest', 'bullhorn',
        'bun', 'bunk_bed', 'buoy', 'burrito', 'bus_(vehicle)', 'business_card',
        'butter', 'butterfly', 'button', 'cab_(taxi)', 'cabana', 'cabin_car',
        'cabinet', 'locker', 'cake', 'calculator', 'calendar', 'calf',
        'camcorder', 'camel', 'camera', 'camera_lens', 'camper_(vehicle)',
        'can', 'can_opener', 'candle', 'candle_holder', 'candy_bar',
        'candy_cane', 'walking_cane', 'canister', 'canoe', 'cantaloup',
        'canteen', 'cap_(headwear)', 'bottle_cap', 'cape', 'cappuccino',
        'car_(automobile)', 'railcar_(part_of_a_train)', 'elevator_car',
        'car_battery', 'identity_card', 'card', 'cardigan', 'cargo_ship',
        'carnation', 'horse_carriage', 'carrot', 'tote_bag', 'cart', 'carton',
        'cash_register', 'casserole', 'cassette', 'cast', 'cat', 'cauliflower',
        'cayenne_(spice)', 'CD_player', 'celery', 'cellular_telephone',
        'chain_mail', 'chair', 'chaise_longue', 'chalice', 'chandelier',
        'chap', 'checkbook', 'checkerboard', 'cherry', 'chessboard',
        'chicken_(animal)', 'chickpea', 'chili_(vegetable)', 'chime',
        'chinaware', 'crisp_(potato_chip)', 'poker_chip', 'chocolate_bar',
        'chocolate_cake', 'chocolate_milk', 'chocolate_mousse', 'choker',
        'chopping_board', 'chopstick', 'Christmas_tree', 'slide', 'cider',
        'cigar_box', 'cigarette', 'cigarette_case', 'cistern', 'clarinet',
        'clasp', 'cleansing_agent', 'cleat_(for_securing_rope)', 'clementine',
        'clip', 'clipboard', 'clippers_(for_plants)', 'cloak', 'clock',
        'clock_tower', 'clothes_hamper', 'clothespin', 'clutch_bag', 'coaster',
        'coat', 'coat_hanger', 'coatrack', 'cock', 'cockroach',
        'cocoa_(beverage)', 'coconut', 'coffee_maker', 'coffee_table',
        'coffeepot', 'coil', 'coin', 'colander', 'coleslaw',
        'coloring_material', 'combination_lock', 'pacifier', 'comic_book',
        'compass', 'computer_keyboard', 'condiment', 'cone', 'control',
        'convertible_(automobile)', 'sofa_bed', 'cooker', 'cookie',
        'cooking_utensil', 'cooler_(for_food)', 'cork_(bottle_plug)',
        'corkboard', 'corkscrew', 'edible_corn', 'cornbread', 'cornet',
        'cornice', 'cornmeal', 'corset', 'costume', 'cougar', 'coverall',
        'cowbell', 'cowboy_hat', 'crab_(animal)', 'crabmeat', 'cracker',
        'crape', 'crate', 'crayon', 'cream_pitcher', 'crescent_roll', 'crib',
        'crock_pot', 'crossbar', 'crouton', 'crow', 'crowbar', 'crown',
        'crucifix', 'cruise_ship', 'police_cruiser', 'crumb', 'crutch',
        'cub_(animal)', 'cube', 'cucumber', 'cufflink', 'cup', 'trophy_cup',
        'cupboard', 'cupcake', 'hair_curler', 'curling_iron', 'curtain',
        'cushion', 'cylinder', 'cymbal', 'dagger', 'dalmatian', 'dartboard',
        'date_(fruit)', 'deck_chair', 'deer', 'dental_floss', 'desk',
        'detergent', 'diaper', 'diary', 'die', 'dinghy', 'dining_table', 'tux',
        'dish', 'dish_antenna', 'dishrag', 'dishtowel', 'dishwasher',
        'dishwasher_detergent', 'dispenser', 'diving_board', 'Dixie_cup',
        'dog', 'dog_collar', 'doll', 'dollar', 'dollhouse', 'dolphin',
        'domestic_ass', 'doorknob', 'doormat', 'doughnut', 'dove', 'dragonfly',
        'drawer', 'underdrawers', 'dress', 'dress_hat', 'dress_suit',
        'dresser', 'drill', 'drone', 'dropper', 'drum_(musical_instrument)',
        'drumstick', 'duck', 'duckling', 'duct_tape', 'duffel_bag', 'dumbbell',
        'dumpster', 'dustpan', 'eagle', 'earphone', 'earplug', 'earring',
        'easel', 'eclair', 'eel', 'egg', 'egg_roll', 'egg_yolk', 'eggbeater',
        'eggplant', 'electric_chair', 'refrigerator', 'elephant', 'elk',
        'envelope', 'eraser', 'escargot', 'eyepatch', 'falcon', 'fan',
        'faucet', 'fedora', 'ferret', 'Ferris_wheel', 'ferry', 'fig_(fruit)',
        'fighter_jet', 'figurine', 'file_cabinet', 'file_(tool)', 'fire_alarm',
        'fire_engine', 'fire_extinguisher', 'fire_hose', 'fireplace',
        'fireplug', 'first-aid_kit', 'fish', 'fish_(food)', 'fishbowl',
        'fishing_rod', 'flag', 'flagpole', 'flamingo', 'flannel', 'flap',
        'flash', 'flashlight', 'fleece', 'flip-flop_(sandal)',
        'flipper_(footwear)', 'flower_arrangement', 'flute_glass', 'foal',
        'folding_chair', 'food_processor', 'football_(American)',
        'football_helmet', 'footstool', 'fork', 'forklift', 'freight_car',
        'French_toast', 'freshener', 'frisbee', 'frog', 'fruit_juice',
        'frying_pan', 'fudge', 'funnel', 'futon', 'gag', 'garbage',
        'garbage_truck', 'garden_hose', 'gargle', 'gargoyle', 'garlic',
        'gasmask', 'gazelle', 'gelatin', 'gemstone', 'generator',
        'giant_panda', 'gift_wrap', 'ginger', 'giraffe', 'cincture',
        'glass_(drink_container)', 'globe', 'glove', 'goat', 'goggles',
        'goldfish', 'golf_club', 'golfcart', 'gondola_(boat)', 'goose',
        'gorilla', 'gourd', 'grape', 'grater', 'gravestone', 'gravy_boat',
        'green_bean', 'green_onion', 'griddle', 'grill', 'grits', 'grizzly',
        'grocery_bag', 'guitar', 'gull', 'gun', 'hairbrush', 'hairnet',
        'hairpin', 'halter_top', 'ham', 'hamburger', 'hammer', 'hammock',
        'hamper', 'hamster', 'hair_dryer', 'hand_glass', 'hand_towel',
        'handcart', 'handcuff', 'handkerchief', 'handle', 'handsaw',
        'hardback_book', 'harmonium', 'hat', 'hatbox', 'veil', 'headband',
        'headboard', 'headlight', 'headscarf', 'headset',
        'headstall_(for_horses)', 'heart', 'heater', 'helicopter', 'helmet',
        'heron', 'highchair', 'hinge', 'hippopotamus', 'hockey_stick', 'hog',
        'home_plate_(baseball)', 'honey', 'fume_hood', 'hook', 'hookah',
        'hornet', 'horse', 'hose', 'hot-air_balloon', 'hotplate', 'hot_sauce',
        'hourglass', 'houseboat', 'hummingbird', 'hummus', 'polar_bear',
        'icecream', 'popsicle', 'ice_maker', 'ice_pack', 'ice_skate',
        'igniter', 'inhaler', 'iPod', 'iron_(for_clothing)', 'ironing_board',
        'jacket', 'jam', 'jar', 'jean', 'jeep', 'jelly_bean', 'jersey',
        'jet_plane', 'jewel', 'jewelry', 'joystick', 'jumpsuit', 'kayak',
        'keg', 'kennel', 'kettle', 'key', 'keycard', 'kilt', 'kimono',
        'kitchen_sink', 'kitchen_table', 'kite', 'kitten', 'kiwi_fruit',
        'knee_pad', 'knife', 'knitting_needle', 'knob', 'knocker_(on_a_door)',
        'koala', 'lab_coat', 'ladder', 'ladle', 'ladybug', 'lamb_(animal)',
        'lamb-chop', 'lamp', 'lamppost', 'lampshade', 'lantern', 'lanyard',
        'laptop_computer', 'lasagna', 'latch', 'lawn_mower', 'leather',
        'legging_(clothing)', 'Lego', 'legume', 'lemon', 'lemonade', 'lettuce',
        'license_plate', 'life_buoy', 'life_jacket', 'lightbulb',
        'lightning_rod', 'lime', 'limousine', 'lion', 'lip_balm', 'liquor',
        'lizard', 'log', 'lollipop', 'speaker_(stero_equipment)', 'loveseat',
        'machine_gun', 'magazine', 'magnet', 'mail_slot', 'mailbox_(at_home)',
        'mallard', 'mallet', 'mammoth', 'manatee', 'mandarin_orange', 'manger',
        'manhole', 'map', 'marker', 'martini', 'mascot', 'mashed_potato',
        'masher', 'mask', 'mast', 'mat_(gym_equipment)', 'matchbox',
        'mattress', 'measuring_cup', 'measuring_stick', 'meatball', 'medicine',
        'melon', 'microphone', 'microscope', 'microwave_oven', 'milestone',
        'milk', 'milk_can', 'milkshake', 'minivan', 'mint_candy', 'mirror',
        'mitten', 'mixer_(kitchen_tool)', 'money',
        'monitor_(computer_equipment) computer_monitor', 'monkey', 'motor',
        'motor_scooter', 'motor_vehicle', 'motorcycle', 'mound_(baseball)',
        'mouse_(computer_equipment)', 'mousepad', 'muffin', 'mug', 'mushroom',
        'music_stool', 'musical_instrument', 'nailfile', 'napkin',
        'neckerchief', 'necklace', 'necktie', 'needle', 'nest', 'newspaper',
        'newsstand', 'nightshirt', 'nosebag_(for_animals)',
        'noseband_(for_animals)', 'notebook', 'notepad', 'nut', 'nutcracker',
        'oar', 'octopus_(food)', 'octopus_(animal)', 'oil_lamp', 'olive_oil',
        'omelet', 'onion', 'orange_(fruit)', 'orange_juice', 'ostrich',
        'ottoman', 'oven', 'overalls_(clothing)', 'owl', 'packet', 'inkpad',
        'pad', 'paddle', 'padlock', 'paintbrush', 'painting', 'pajamas',
        'palette', 'pan_(for_cooking)', 'pan_(metal_container)', 'pancake',
        'pantyhose', 'papaya', 'paper_plate', 'paper_towel', 'paperback_book',
        'paperweight', 'parachute', 'parakeet', 'parasail_(sports)', 'parasol',
        'parchment', 'parka', 'parking_meter', 'parrot',
        'passenger_car_(part_of_a_train)', 'passenger_ship', 'passport',
        'pastry', 'patty_(food)', 'pea_(food)', 'peach', 'peanut_butter',
        'pear', 'peeler_(tool_for_fruit_and_vegetables)', 'wooden_leg',
        'pegboard', 'pelican', 'pen', 'pencil', 'pencil_box',
        'pencil_sharpener', 'pendulum', 'penguin', 'pennant', 'penny_(coin)',
        'pepper', 'pepper_mill', 'perfume', 'persimmon', 'person', 'pet',
        'pew_(church_bench)', 'phonebook', 'phonograph_record', 'piano',
        'pickle', 'pickup_truck', 'pie', 'pigeon', 'piggy_bank', 'pillow',
        'pin_(non_jewelry)', 'pineapple', 'pinecone', 'ping-pong_ball',
        'pinwheel', 'tobacco_pipe', 'pipe', 'pistol', 'pita_(bread)',
        'pitcher_(vessel_for_liquid)', 'pitchfork', 'pizza', 'place_mat',
        'plate', 'platter', 'playpen', 'pliers', 'plow_(farm_equipment)',
        'plume', 'pocket_watch', 'pocketknife', 'poker_(fire_stirring_tool)',
        'pole', 'polo_shirt', 'poncho', 'pony', 'pool_table', 'pop_(soda)',
        'postbox_(public)', 'postcard', 'poster', 'pot', 'flowerpot', 'potato',
        'potholder', 'pottery', 'pouch', 'power_shovel', 'prawn', 'pretzel',
        'printer', 'projectile_(weapon)', 'projector', 'propeller', 'prune',
        'pudding', 'puffer_(fish)', 'puffin', 'pug-dog', 'pumpkin', 'puncher',
        'puppet', 'puppy', 'quesadilla', 'quiche', 'quilt', 'rabbit',
        'race_car', 'racket', 'radar', 'radiator', 'radio_receiver', 'radish',
        'raft', 'rag_doll', 'raincoat', 'ram_(animal)', 'raspberry', 'rat',
        'razorblade', 'reamer_(juicer)', 'rearview_mirror', 'receipt',
        'recliner', 'record_player', 'reflector', 'remote_control',
        'rhinoceros', 'rib_(food)', 'rifle', 'ring', 'river_boat', 'road_map',
        'robe', 'rocking_chair', 'rodent', 'roller_skate', 'Rollerblade',
        'rolling_pin', 'root_beer', 'router_(computer_equipment)',
        'rubber_band', 'runner_(carpet)', 'plastic_bag',
        'saddle_(on_an_animal)', 'saddle_blanket', 'saddlebag', 'safety_pin',
        'sail', 'salad', 'salad_plate', 'salami', 'salmon_(fish)',
        'salmon_(food)', 'salsa', 'saltshaker', 'sandal_(type_of_shoe)',
        'sandwich', 'satchel', 'saucepan', 'saucer', 'sausage', 'sawhorse',
        'saxophone', 'scale_(measuring_instrument)', 'scarecrow', 'scarf',
        'school_bus', 'scissors', 'scoreboard', 'scraper', 'screwdriver',
        'scrubbing_brush', 'sculpture', 'seabird', 'seahorse', 'seaplane',
        'seashell', 'sewing_machine', 'shaker', 'shampoo', 'shark',
        'sharpener', 'Sharpie', 'shaver_(electric)', 'shaving_cream', 'shawl',
        'shears', 'sheep', 'shepherd_dog', 'sherbert', 'shield', 'shirt',
        'shoe', 'shopping_bag', 'shopping_cart', 'short_pants', 'shot_glass',
        'shoulder_bag', 'shovel', 'shower_head', 'shower_cap',
        'shower_curtain', 'shredder_(for_paper)', 'signboard', 'silo', 'sink',
        'skateboard', 'skewer', 'ski', 'ski_boot', 'ski_parka', 'ski_pole',
        'skirt', 'skullcap', 'sled', 'sleeping_bag', 'sling_(bandage)',
        'slipper_(footwear)', 'smoothie', 'snake', 'snowboard', 'snowman',
        'snowmobile', 'soap', 'soccer_ball', 'sock', 'sofa', 'softball',
        'solar_array', 'sombrero', 'soup', 'soup_bowl', 'soupspoon',
        'sour_cream', 'soya_milk', 'space_shuttle', 'sparkler_(fireworks)',
        'spatula', 'spear', 'spectacles', 'spice_rack', 'spider', 'crawfish',
        'sponge', 'spoon', 'sportswear', 'spotlight', 'squid_(food)',
        'squirrel', 'stagecoach', 'stapler_(stapling_machine)', 'starfish',
        'statue_(sculpture)', 'steak_(food)', 'steak_knife', 'steering_wheel',
        'stepladder', 'step_stool', 'stereo_(sound_system)', 'stew', 'stirrer',
        'stirrup', 'stool', 'stop_sign', 'brake_light', 'stove', 'strainer',
        'strap', 'straw_(for_drinking)', 'strawberry', 'street_sign',
        'streetlight', 'string_cheese', 'stylus', 'subwoofer', 'sugar_bowl',
        'sugarcane_(plant)', 'suit_(clothing)', 'sunflower', 'sunglasses',
        'sunhat', 'surfboard', 'sushi', 'mop', 'sweat_pants', 'sweatband',
        'sweater', 'sweatshirt', 'sweet_potato', 'swimsuit', 'sword',
        'syringe', 'Tabasco_sauce', 'table-tennis_table', 'table',
        'table_lamp', 'tablecloth', 'tachometer', 'taco', 'tag', 'taillight',
        'tambourine', 'army_tank', 'tank_(storage_vessel)',
        'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)', 'tape_measure',
        'tapestry', 'tarp', 'tartan', 'tassel', 'tea_bag', 'teacup',
        'teakettle', 'teapot', 'teddy_bear', 'telephone', 'telephone_booth',
        'telephone_pole', 'telephoto_lens', 'television_camera',
        'television_set', 'tennis_ball', 'tennis_racket', 'tequila',
        'thermometer', 'thermos_bottle', 'thermostat', 'thimble', 'thread',
        'thumbtack', 'tiara', 'tiger', 'tights_(clothing)', 'timer', 'tinfoil',
        'tinsel', 'tissue_paper', 'toast_(food)', 'toaster', 'toaster_oven',
        'toilet', 'toilet_tissue', 'tomato', 'tongs', 'toolbox', 'toothbrush',
        'toothpaste', 'toothpick', 'cover', 'tortilla', 'tow_truck', 'towel',
        'towel_rack', 'toy', 'tractor_(farm_equipment)', 'traffic_light',
        'dirt_bike', 'trailer_truck', 'train_(railroad_vehicle)', 'trampoline',
        'tray', 'trench_coat', 'triangle_(musical_instrument)', 'tricycle',
        'tripod', 'trousers', 'truck', 'truffle_(chocolate)', 'trunk', 'vat',
        'turban', 'turkey_(food)', 'turnip', 'turtle', 'turtleneck_(clothing)',
        'typewriter', 'umbrella', 'underwear', 'unicycle', 'urinal', 'urn',
        'vacuum_cleaner', 'vase', 'vending_machine', 'vent', 'vest',
        'videotape', 'vinegar', 'violin', 'vodka', 'volleyball', 'vulture',
        'waffle', 'waffle_iron', 'wagon', 'wagon_wheel', 'walking_stick',
        'wall_clock', 'wall_socket', 'wallet', 'walrus', 'wardrobe',
        'washbasin', 'automatic_washer', 'watch', 'water_bottle',
        'water_cooler', 'water_faucet', 'water_heater', 'water_jug',
        'water_gun', 'water_scooter', 'water_ski', 'water_tower',
        'watering_can', 'watermelon', 'weathervane', 'webcam', 'wedding_cake',
        'wedding_ring', 'wet_suit', 'wheel', 'wheelchair', 'whipped_cream',
        'whistle', 'wig', 'wind_chime', 'windmill', 'window_box_(for_plants)',
        'windshield_wiper', 'windsock', 'wine_bottle', 'wine_bucket',
        'wineglass', 'blinder_(for_horses)', 'wok', 'wolf', 'wooden_spoon',
        'wreath', 'wrench', 'wristband', 'wristlet', 'yacht', 'yogurt',
        'yoke_(animal_equipment)', 'zebra', 'zucchini']

# fmt: on

def get_fed_loss_cls_weights(freq_weight_power=0.5):
    """
    Get frequency weight for each class sorted by class id.
    We now calcualte freqency weight using image_count to the power freq_weight_power.

    Args:
        dataset_names: list of dataset names
        freq_weight_power: power value
    """

    class_freq_meta = LVIS_CATEGORY_IMAGE_COUNT
    class_freq = torch.tensor(
        [c["image_count"] for c in sorted(class_freq_meta, key=lambda x: x["id"])]
    )
    class_freq_weight = class_freq.float() ** freq_weight_power
    return class_freq_weight



def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)



DETECTION_PROMPT = [
    'Detect all objects in the image by identifying the common visual features of their respective classes. ',
    'Localize each object by matching it to the archetypal visual form of its category. ',
    'Detect all objects in the image by recognizing the shared visual attributes of their respective categories. ',
    'Identify every object in the scene based on the core visual characteristics that define its class. ',
    'Locate all objects by using the fundamental visual properties common to their object class. ',
    'Perform object detection by referencing the shared visual patterns that characterize each class. ',
    'Find every object in the picture by matching it to the defining visual traits of its category. ',
    'Identify all objects present by their class-defining visual features, which are common across all instances. ',
    'Detect every object based on the visual essence shared by all members of its class. ',
    'Localize each object in the image according to the general visual blueprint of its category. ',
    'Identify all objects by applying the common visual criteria that define their respective classes. '
]

REC_PROMPT = [
    'Locate the specific object being described by analyzing its unique instance-level attributes, its spatial position, and its relationship with surrounding objects. ',
    'Identify the single instance mentioned in the text by considering its distinct visual features, its location within the scene, and its context relative to nearby items. ',
    "Ground the referring expression by pinpointing the object that matches the description's details regarding its appearance, placement, and interaction with other elements. ",
    'Find the objects that correspond to the given description, paying close attention to its specific details, its position, and how it relates to its neighbors. ',
    'Disambiguate and find the correct object by carefully examining the provided description of its instance-specific properties, its coordinates in the image, and its spatial arrangement with other objects. ',
    'Resolve the reference by identifying the object that uniquely matches the specified details, including its appearance, its place in the scene, and its connections to adjacent objects. ',
    'Pinpoint the described instance by evaluating its specific visual traits, its spatial context, and its relational properties with other objects in the image. ',
    'To locate the referred object, you must analyze three things from the description: 1) its unique visual details (e.g., color, texture), 2) its precise location, and 3) its relationship to the objects around it. ',
    'Find the specific object the text is referring to by synthesizing information about its individual characteristics, its location, and its interactions within the scene. ',
    'Based on the textual description, identify the target object by considering its instance-level appearance, its absolute and relative position, and its interplay with the surrounding environment. '
]


CELEBRITY_PROMPT = [
    'Identify the famous person depicted in this image ',
    'Recognize and name the public figure featured in this picture. ',
    'Please provide the name of the well-known individual in this image. ',
    'Identify all recognizable celebrities in this image. ',
    'Name the celebrity whose face is most prominent in this picture. ',
    'State the name of the celebrity shown. '
]

HUMANREF_PROMPT = [
    'Locate the specific person being described by analyzing his unique appearance, his spatial position, and his action. ',
    'Determine which person matches the description by analyzing their physical features, spatial relationship to others, and what they are doing. ',
    'Find the target person in the scene based on their unique attire, position relative to objects, and specific action they are performing. ',
    'Locate the person being referred to by combining clues about their appearance, where they are standing, and the action they are engaged in. ',
    'Pinpoint the exact individual by evaluating their facial characteristics, proximity to key landmarks, and visible behavior. ',
    'Match the description to the correct person by assessing their height, clothing color, location in the frame, and dynamic motion. ',
    'Single out the described person by integrating visual cues such as hairstyle, spatial context, and interaction with surrounding objects or people. ',
    'Detect the person of interest using a combination of their outfit, position in the environment, and whether they are moving or stationary. ',
    'Trace the individual who fits the profile: analyze appearance details, relative position (e.g., left/right, front/back), and active behavior. ',
    'Select the correct person from the group by cross-referencing their described look, exact location, and current action in the scene. ',
]


CELEBRITY = set([
    'Eleven from the Strange Things', 'Obi-Wan Kenobi', 'Captain America', 'Queen Maeve', 'Buzz Lightyear', 'Mary Poppins', 'John Rambo', 'Hermione Granger', 'Rick Grimes', 'Ferris Bueller', 'Sheldon Cooper', 'Sarah Connor', 'Negan Smith', 'Amy Farrah Fowler', 'Professor Severus Snape', 'Shrek', 'Ada Shelby', 'Palpatine', 'Jorah Mormont', 'Thomas Shelby', 'Cersei Lannister', 'Luke Skywalker', 'Maximus Decimus Meridius', 'Sansa Stark', 'John Wick', 'Atticus Finch', 'Harry Potter', 'Wolverine', 'Mr. Bean', 'Ronald Weasley', 'Melisandre', 'Ross Geller', 'Bilbo Baggins', 'Samwise Gamgee', 'Maggie Greene', 'Jon Snow', 'Wednesday Addams', 'Hans Gruber', 'Rocky Balboa', 'Michael Corleone', 'Leonard Hofstadter', 'Chandler Bing', 'Littlefinger', 'Barbossa', 'Rachel Green', 'Howard Wolowitz', 'Jaime Lannister', 'Legolas', 'Axel Foley', 'James T. Kirk', 'Kevin McCallister', 'Margaery Tyrell', 'Leia Organa', 'Forrest Gump', 'Marty McFly', 'Han Solo', 'Billy Butcher', 'Mr. Kesuke Miyagi', 'Professor Albus Dumbledore', 'Steve Harrington', 'Sirius Black', 'Gus Fring', 'Jack Sparrow', 'Inigo Montoya', 'Professor Minerva McGonagall', 'Homelander', 'Arthur Shelby', 'Superman', 'Indiana Jones', 'Beetlejuice', 'Polly Gray', 'Hughie Campbell', 'Peter Venkman', 'Sandor Clegane', 'Soldier Boy', 'Joey Tribbiani', 'Rajesh Koothrappali', 'Gollum', 'Vito Corleone', 'Daryl Dixon', 'Aragorn', 'Andy Dufresne', 'Jesse Pinkman', 'Penny Hofstadter', 'Jean-Luc Picard', 'Lord Voldemort', 'Oberyn Martell', 'Luna Lovegood', 'Carol Peletier', 'Monica Geller', 'Alfred Pennyworth', 'Bernadette Rostenkowski-Wolowitz', 'Darth Maul', 'Thor', 'Neo', 'John McClane', 'Phoebe Buffay', 'Spock', 'Dorothy Gale',
    'Michael Bublé', 'Demi Lovato', 'Aerosmith', 'Drake', 'ZAYN', 'Jennifer Lopez', 'Olivia Rodrigo', 'Kali Uchis', 'Flo Rida', 'Doja Cat', 'Skrillex', 'Chris Brown', 'Ellie Goulding', '50 Cent', 'Katy Perry', 'Gucci Mane', 'Charli XCX', 'Avril Lavigne', 'Shawn Mendes', 'Lil Wayne', 'J. Cole', 'Linkin Park', 'Migos', 'Taylor Swift', 'DJ Khaled', 'Red Hot Chili Peppers', 'Justin Bieber', 'Calvin Harris', '2 Chainz', 'Frank Ocean', 'Jason Mraz', 'Alicia Keys', 'Miley Cyrus', 'Childish Gambino', 'Meghan Trainor', 'Tyga', 'Usher', 'SZA', 'Bad Bunny', 'Labrinth', 'Diplo', 'Jack Johnson', 'Halsey', 'Young Thug', 'Bon Jovi', 'Post Malone', 'Christina Aguilera', 'The Kooks', 'John Mayer', 'The 1975', 'Akon', 'G-Eazy', 'Panic! At the Disco', 'Eminem', 'Ed Sheeran', 'Maroon 5', 'Ne-Yo', 'Zedd', 'Dr. Dre', 'Queen', 'Nelly Furtado', 'Steve Lacy', 'Imagine Dragons', 'Sia', 'Mac DeMarco', 'Big Sean', 'Martin Garrix', 'Camila Cabello', 'The Rolling Stones', 'Khalid', 'Harry Styles', 'Charlie Puth', 'Kanye West', 'The Weeknd', 'Kendrick Lamar', 'Travis Scott', 'Kesha', 'Nelly', 'Tyler', 'The Creator', 'Billie Eilish', 'Metro Boomin', 'Gwen Stefani', 'Sean Paul', 'Vampire Weekend', 'Jay-Z', 'Kelly Clarkson', 'Stevie Wonder', 'Adele', 'Arctic Monkeys', 'Lorde', 'Britney Spears', 'Selena Gomez', 'Daddy Yankee', '21 Savage', 'David Guetta', 'J Balvin', 'The Cure', 'Bruno Mars', 'Dua Lipa', 'Bruce Springsteen', 'Snoop Dogg', 'B.o.B', 'OutKast', 'Lady Gaga', 'Hozier', 'Wiz Khalifa', 'Foo Fighters', 'Lana Del Rey', 'Beyoncé', 'Madonna', 'Shakira', 'John Legend', 'Mark Ronson', 'Sam Smith', 'Billy Joel', 'Jeremih', 'Paramore', 'Chance the Rapper', 'DJ Snake', 'Sabrina Carpenter', 'Kid Cudi', 'Trey Songz', 'Kings of Leon', 'Enrique Iglesias', 'Pharrell Williams', 'Arcade Fire', 'Jessie J', 'Lil Uzi Vert', 'Bob Dylan',
    'Tom Wilkinson', 'Al Pacino', 'Kevin Costner', 'Franco Nero', 'Philip Seymour Hoffman', 'Alan Rickman', 'Leonardo DiCaprio', 'Ben Affleck', 'William Hurt', 'Mark Wahlberg', 'Jonah Hill', 'Shia LaBeouf', 'Don Cheadle', 'Orlando Bloom', 'Jeff Goldblum', 'Denzel Washington', 'Alec Baldwin', 'Bradley Cooper', 'Ed Harris', 'Jason Clarke', 'Mahershala Ali', 'Viggo Mortensen', 'Owen Wilson', 'Alan Arkin', 'James Caan', 'Nicolas Cage', 'Samuel L', 'David Strathairn', 'Matt Damon', 'George Clooney', 'Giovanni Ribisi', 'Jared Leto', 'Kevin Spacey', 'Matthew McConaughey', 'Gary Sinise', 'Pete Postlethwaite', 'Keanu Reeves', 'Timothy Spall', 'Harry Dean Stanton', 'John Carroll Lynch', 'Chiwetel Ejiofor', 'Woody Harrelson', 'Ryan Gosling', 'Joaquin Phoenix', 'Donald Sutherland', 'Paul Dano', 'Chris Hemsworth', 'David Oyelowo', 'Tom Hardy', 'Barry Pepper', 'Kurt Russell', 'Christian Bale', 'Jeff Daniels', 'Ben Whishaw', 'Sterling Hayden', 'Edward Norton', 'Sam Shepard', 'Andy Garcia', 'Harvey Keitel', 'Benicio Del Toro', 'Gene Hackman', 'Bruce Willis', 'Guy Pearce', 'Jonathan Pryce', 'Michael Fassbender', 'James Stewart', 'Zach Galifianakis', 'Forest Whitaker', 'Vincent Cassel', 'Michael Sheen', 'Tom Berenger', 'Jim Carrey', 'Steve Buscemi', 'Joe Pesci', 'Christian Berkel', 'Rutger Hauer', 'Mel Gibson', 'Elliott Gould', 'Tim Robbins', 'Daniel Craig', 'Jeffrey Wright', 'Matthew Modine', 'Domhnall Gleeson', 'Brendan Gleeson', 'John Hurt', 'Michael Stuhlbarg', 'Hugo Weaving', 'John Goodman', 'Mark Hamill', 'Colin Farrell', 'Ken Watanabe', 'Clint Eastwood', 'Ralph Fiennes', 'Val Kilmer', 'John Hawkes', 'Ben Kingsley', 'Seth Rogen', 'Robert Duvall', 'Brad Pitt', 'Max von Sydow', 'Stanley Tucci', 'Tom Cruise', 'Christopher Lloyd', 'Tommy Lee Jones', 'Jason Statham', 'Michael Caine', 'Paul Giamatti', 'Josh Hutcherson', 'Adrien Brody', 'Michael J', 'Jeremy Renner', 'Liam Neeson', 'Mark Ruffalo', 'Terrence Howard', 'John Cleese', 'Harrison Ford', 'Clive Owen', 'Jake Gyllenhaal', 'Will Smith', 'Danny DeVito', 'Elijah Wood', 'Sean Connery', 'Tom Sizemore', 'Stellan Skarsgård', 'Robin Williams', 'Hugh Jackman', 'John Lithgow', 'Benedict Cumberbatch', 'Mykelti Williamson', 'John Malkovich', 'Gary Oldman', 'Johnny Depp', 'Jeff Bridges', 'Hugh Grant', 'Jean Reno', 'Aaron Eckhart', 'Michael Madsen', 'Jude Law', 'J.K', 'Jon Voight', 'Casey Affleck', 'Robert Pattinson', 'Daniel Brühl', 'Billy Bob Thornton', 'Russell Crowe', 'Ewan McGregor', 'Christopher Walken', 'Morgan Freeman', 'Josh Brolin', 'Richard Harris', 'Shea Whigham', 'Bill Murray', 'Christoph Waltz', 'Jamie Foxx', 'Christopher Plummer', 'Ethan Hawke', 'Albert Finney', 'Miles Teller', 'Don Johnson', 'Javier Bardem', 'Bill Paxton', 'Robert De Niro', 'Timothée Chalamet', 'Sam Rockwell', 'Kevin Bacon', 'Simon Pegg', 'Sean Penn', 'Ving Rhames', 'Tom Hanks', 'Anthony Hopkins', 'Heath Ledger', 'Tim Roth', 'Martin Sheen', 'Michael Keaton', 'Joseph Gordon-Levitt', 'Kyle Chandler', 'John Travolta', 'Bruce Dern', 'Steve Carell', 'Dustin Hoffman', 'Oscar Isaac'
    'Dirk Nowitzki', 'Mia Hamm',' Diana Taurasi', 'Allyson Felix', 'Sheryl Swoopes', 'David Ortiz', 'James Harden', 'Mike Trout', 'Mookie Betts', 'Chris Paul', 'Aitana Bonmati', 'Roger Federer', 'Faker', 'Annika Sorenstam', 'Thierry Henry', 'Jimmie Johnson', 'Tom Brady', 'Randy Moss', 'Shohei Ohtani', 'Kevin Durant', 'Zinedine Zidane', 'Calvin Johnson', 'Novak Djokovic', 'LeBron James', 'Alexia Putellas', 'Albert Pujols', 'Max Scherzer', 'Mariano Rivera', 'Ichiro Suzuki', 'Patrick Mahomes', 'Aaron Donald', 'Steve Nash', 'Georges St-Pierre', 'Giannis Antetokounmpo', 'Stephen Curry', 'Floyd Mayweather', 'Clayton Kershaw', 'Manny Pacquiao', 'Andrés Iniesta', 'Barry Bonds', 'Tim Duncan', 'Lauren Jackson', 'Luka Modric', 'Shelly-Ann Fraser Pryce', 'Bryce Harper', 'Ray Lewis', 'Simone Biles', 'Rafael Nadal', 'Derek Jeter', 'Shaun White', 'Michael Schumacher', 'Peyton Manning', 'Candace Parker', 'Nikola Jokic', 'Serena Williams', 'Jason Kidd', 'Andy Murray', 'Mikaela Shiffrin', 'Lewis Hamilton', 'Lisa Leslie', 'Bernard Hopkins', 'Kobe Bryant', 'Justin Verlander', 'Tamika Catchings', 'Alex Rodriguez', 'Jon Jones', 'Tiger Woods', 'Dwyane Wade', 'Kohei Uchimura', 'Michael Phelps', 'Xavi Hernandez', 'Cristiano Ronaldo', 'Usain Bolt', 'Max Verstappen', 'Kawhi Leonard', 'Venus Williams', 'Katie Ledecky', 'Kylian Mbappé', 'Maya Moore', 'Alex Ovechkin', 'Sidney Crosby', 'Phil Mickelson', 'Adrian Beltré', 'Kevin Garnett', 'Miguel Cabrera', 'Lionel Messi',
    'Brian Chesky', 'Garrett Camp', 'Kevin Systrom', 'Sam Walton', 'Larry Ellison', 'Ratan Tata', 'Ritesh Agarwal', 'Ted Turner', 'Steve Jobs', 'Jack Ma', 'Richard Branson', 'Jeff Bezos',
    'Che Guevara', 'Mike Pence', 'Li Ka-shing', 'Woodrow Wilson', 'Dwight D. Eisenhower', 'Abdel Fattah el-Sisi', 'Franklin D. Roosevelt', 'Yasser Arafat', 'Haruhiko Kuroda', 'Donald Trump', 'Bill Clinton', 'Stephen Schwarzman', 'Narendra Modi', 'Gianni Infantino', 'Masayoshi Son', 'Bernard Arnault', 'Hui Ka Yan', 'Benjamin Netanyahu', 'Winston Churchill', 'Vladimir Putin', 'John F. Kennedy', 'Theodore Roosevelt', 'Nelson Mandela', 'Sergey Brin', 'Margaret Thatcher', 'Xi Jinping', 'Ronald Reagan', 'Golda Meir', 'Recep Tayyip Erdogan', 'Charles de Gaulle', 'Jim Yong Kim', 'Warren Buffett', 'Qamar Javed Bajwa', 'Jerome H. Powell', 'Wang Jianlin', 'Lech Wałęsa', 'Michel Temer', 'Doug McMillon', 'Mohammed bin Salman Al Saud', 'Lloyd Blankfein', 'Lee Hsien Loong', 'Jawaharlal Nehru', 'Shinzo Abe', 'Michael Bloomberg', 'Tony Blair', 'Li Keqiang', 'Rodrigo Duterte', 'Justin Trudeau', 'Hu Jintao', 'Bob Iger', 'Mario Draghi', 'Khalifa bin Zayed Al-Nahyan', 'Bashar al-Assad', 'Ayatollah Khomeini', 'Ali Hoseini-Khamenei', 'Indira Gandhi', 'Deng Xiaoping', 'Kim Jong-un', 'Ma Huateng', 'Joseph Stalin', 'Mikhail Gorbachev', 'Moon Jae-in', 'Mary Barra', 'Mahatma Gandhi', 'Christine Lagarde', 'Jokowi Widodo', 'Mao Zedong', 'Ken Griffin', 'Mustafa Kemal Atatürk', 'Theresa May', 'Aliko Dangote', 'Darren Woods', 'Jiang Zemin', 'Rupert Murdoch', 'Fidel Castro', 'Jean-Claude Juncker', 'Robert Mueller', 'Enrique Pena Nieto', 'Carlos Slim Helu', 'Tim Cook', 'Robin Li', 'Antonio Guterres', 'Larry Fink',
    'lbert Einstein', 'Fei-Fei Li', 'Jennifer Doudna', 'Yann LeCun', 'Thomas Edison', 'Gregor Mendel', 'Geoffrey Hinton', 'John von Neumann', 'Max Planck', 'Sam Altman', 'Tim Berners-Lee', 'Andrew Ng', 'Rosalind Franklin', 'Andre Geim', 'Marie Curie', 'Ilya Sutskever', 'Kip Thorne', 'James Watson', 'Srinivasa Ramanujan', 'Nikola Tesla', 'Niels Bohr', 'Enrico Fermi', 'Rachel Carson', 'Yoshua Bengio', 'Alan Turing', 'Stephen Hawking', 'Francis Crick', 'Linus Pauling', 'Demis Hassabis', 'Werner Heisenberg', 'Barbara McClintock',
])




GENERAL_PROMPT = "Represent each object into a single token. "


class LazySupervisedDataset(Dataset):

    def __init__(self, data_path: str, dataset_name, dataset_type, proposal_path, per_image_train_text_batch_size, per_device_train_batch_size, use_task_prompt=False, use_global_caption=False, use_two_captions=False, use_two_tokens=0):
        super(LazySupervisedDataset, self).__init__()
        if data_path.endswith('.json'):
            with open(data_path, "r") as file:
                self.list_data_dict = json.load(file)
        elif data_path.endswith('.jsonl'):
            with open(data_path, "r") as file:
                self.list_data_dict = [json.loads(line) for line in file]
        if dataset_name == 'self_sam_list':
            for i in range(len(self.list_data_dict)):
                for obj in self.list_data_dict[i]['objects']:
                    obj['label'] = obj['label'][0]

        if type(proposal_path) == list:
            temp_proposals = {}
            for path in proposal_path:
                temp_proposals.update(json.load(open(path, "r")))
            self.proposals = temp_proposals
        else:
            self.proposals = json.load(open(proposal_path, "r"))
        self.dataset_type = dataset_type
        self.dataset_name = dataset_name
        self.per_image_train_text_batch_size = per_image_train_text_batch_size
        self.num_candicate_text = (per_device_train_batch_size - 1) * per_image_train_text_batch_size
        self.use_task_prompt = use_task_prompt
        self.use_global_caption = use_global_caption
        self.use_two_captions = use_two_captions
        self.use_two_tokens = use_two_tokens
        if self.dataset_name == 'lvis':
            total_texts = LVIS_CLASSES
            class_weights = get_fed_loss_cls_weights()
        elif self.dataset_name == 'human_ref':
            with open('datasets/processed_reircoco_data.json') as f:
                temp_list_data_dict = json.load(f)
            total_texts = set()
            for source in temp_list_data_dict:
                for objects in source['objects']:
                    flag = 0
                    for name in ['person', 'people', 'individual', 'human', 'man', 'woman', 'boy', 'girl', 'male', 'female', 'guy', 'lady', 'child', 'kid', 'toddler', 'infant', 'adult', 'elder', 'senior', 'teenager']:
                        if name in objects['label'].lower():
                            flag = 1
                    if flag == 0:
                        total_texts.add(objects['label'])
            total_texts = list(total_texts)
            class_weights = torch.ones((len(total_texts),))
            del temp_list_data_dict
        else:
            total_texts = set()
            for source in self.list_data_dict:
                for objects in source['objects']:
                    total_texts.add(objects['label'])
                if 'negs' in source:
                    for text in source['negs']:
                        total_texts.add(text)
            total_texts = list(total_texts)
            class_weights = torch.ones((len(total_texts),))
        self.total_texts = total_texts
        self.class_weights = class_weights
        print("There are %d classes in %s dataset" % (len(self.total_texts), self.dataset_name))


    def __len__(self):
        return len(self.list_data_dict)


    def __getitem__(self, i):
        # Format into conversation
        num_base_retries = 5
        try:
            return self._get_item(i)
        except Exception as e:
            print(e)
            print(i)


        for attempt_idx in range(num_base_retries):
            try:
                sample_idx = random.choice(range(len(self)))
                sample = self._get_item(sample_idx)
                return sample
            except Exception as e:
                # no need to sleep
                print(f'[try other #{attempt_idx}] Failed to fetch sample {sample_idx}. Exception:', e)
                pass
        

    def _get_item(self, i):
        source = copy.deepcopy(self.list_data_dict[i])
        # print(self.dataset_name)
        # open image
        image = Image.open(source["image"]).convert("RGB")

        # load proposal
        proposals = torch.tensor(self.proposals[source["image"]]).float().reshape(-1, 4)
        proposals = proposals[torch.randperm(proposals.size(0))]
        
        # load text labels and we need to ensure that the number of text label on each GPU is the same
        text_labels = list(set([objects["label"] for objects in source["objects"]]))
        negs = copy.deepcopy(source['negs']) if 'negs' in source else []
        text_labels += negs

        
        prob = self.class_weights.float().clone()
        if self.dataset_name != 'human_ref' and len(text_labels) > 0:
            text_labels_idx = torch.tensor([self.total_texts.index(label) for label in text_labels], dtype=torch.long)
            prob[text_labels_idx] = 0

        if len(text_labels) > self.per_image_train_text_batch_size:
            random.shuffle(text_labels)
            text_labels = text_labels[:self.per_image_train_text_batch_size]
            selected_objects = [objects for objects in source["objects"] if objects['label'] in text_labels]
            source["objects"] = selected_objects
        
        sampled_negative_classes = torch.multinomial(
            prob, self.num_candicate_text + (self.per_image_train_text_batch_size - len(text_labels)), replacement=False
        )
        neg_texts = [self.total_texts[idx] for idx in sampled_negative_classes.tolist()]
        candicates_labels = neg_texts[self.per_image_train_text_batch_size - len(text_labels):]
        text_labels += neg_texts[:self.per_image_train_text_batch_size - len(text_labels)]

        gt_boxes = torch.tensor([objects["bbox"] for objects in source["objects"]]).float().reshape(-1, 4)

        if len(gt_boxes) > 0:
            ious = box_iou(gt_boxes, proposals)
            ious = torch.max(ious, dim=1)[0]
            # add gt bboxes
            proposals = torch.cat([proposals, gt_boxes[ious < 0.5]], dim=0)
            proposals = proposals[torch.randperm(proposals.size(0))]
        
        proposals[:, 0::2] = proposals[:, 0::2].clamp(min=0.0, max=image.width)
        proposals[:, 1::2] = proposals[:, 1::2].clamp(min=0.0, max=image.height)

        if self.use_task_prompt:
            if self.dataset_type == 'detection':
                prompt = random.choice(DETECTION_PROMPT)
            elif self.dataset_type == 'rec' and self.dataset_name == 'human_ref':
                if len(source['objects']) > 0 and source['objects'][0]['label'].strip() in CELEBRITY:
                    prompt = random.choice(CELEBRITY_PROMPT)
                else:
                    prompt = random.choice(HUMANREF_PROMPT)
            else:
                prompt = random.choice(REC_PROMPT)
        else:
            prompt = GENERAL_PROMPT

        global_caption = None
        if self.use_two_captions:
            global_caption = {
                'short_caption': source['captions']['short_caption'],
                'long_caption': source['captions']['long_caption'],
            }
        else:
            if self.use_global_caption and self.dataset_type =='detection':
                global_caption = source['captions']['short_caption']
            elif self.use_global_caption and self.dataset_type =='rec':
                global_caption = source['captions']['long_caption']

        return {
            'source': source,
            'image': image,
            'proposals': proposals,
            'dataset_name': self.dataset_name,
            'dataset_type': self.dataset_type,
            'text_labels': text_labels,
            'candicates_labels': candicates_labels,
            'total_num_text_label': self.per_image_train_text_batch_size + self.num_candicate_text,
            'idx': i,
            'prompt': prompt,
            'global_caption': global_caption,
            'use_two_tokens': self.use_two_tokens,
        }
            


def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate batch of examples for training."""

    sources = []
    images = []
    proposals = []
    dataset_names = []
    dataset_types = []
    total_num_text_label = 0
    text_labels = []
    candicates_labels = []
    idxs = []
    prompts = []
    global_captions = []
    use_two_tokens = []

    for i, example in enumerate(examples):
        sources.append(example['source'])
        images.append(example['image'])
        proposals.append(example['proposals'])
        dataset_names.append(example['dataset_name'])
        dataset_types.append(example['dataset_type'])
        total_num_text_label = example['total_num_text_label']
        candicates_labels += example['candicates_labels']
        text_labels += example['text_labels']
        idxs.append(example['idx'])
        prompts.append(example['prompt'])
        use_two_tokens.append(example['use_two_tokens'])
        if type(example['global_caption']) == dict:
            global_captions.append(example['global_caption']['short_caption'])
            global_captions.append(example['global_caption']['long_caption'])
        else:
            global_captions.append(example['global_caption'])


    if dataset_names[0] != 'human_ref':
        text_labels = list(set(text_labels))
        candicates_labels = list(set(candicates_labels) - set(text_labels))
        if len(text_labels) < total_num_text_label:
            text_labels += candicates_labels[:total_num_text_label - len(text_labels)]

    ori_shapes = [img.size for img in images]
    bboxes_labels = []
    bboxes_scores = []
    bboxes_label_masks = []
    image_conversations = []
    all_image_inputs = []
    for i, source in enumerate(sources):
        gt_labels = torch.tensor([text_labels.index(objects['label']) for objects in source['objects']])
        gt_bboxes = torch.tensor([objects["bbox"] for objects in source["objects"]]).float().reshape(-1, 4)

        grounding_label = torch.ones(proposals[i].shape[0], dtype=torch.long) * -1
        grounding_label_score = torch.zeros(proposals[i].shape[0], dtype=torch.float)
        grounding_label_mask = torch.zeros(proposals[i].shape[0], dtype=torch.float)

        if len(gt_bboxes) > 0:
            normalized_gt_bboxes = gt_bboxes / torch.tensor([ori_shapes[i][0], ori_shapes[i][1], ori_shapes[i][0], ori_shapes[i][1]])
            normalized_gt_bboxes = box_xyxy_to_cxcywh(normalized_gt_bboxes)
            ious = box_iou(gt_bboxes, proposals[i])
            ious_per_gt, idx = torch.max(ious, dim=1)
            ious_per_proposal = torch.max(ious, dim=0)[0]
            # only select one best proposal for each gt
            grounding_label[idx[ious_per_gt >= 0.5]] = gt_labels[ious_per_gt >= 0.5]
            grounding_label_score[ious_per_proposal >= 0.5] = ious_per_proposal[ious_per_proposal >= 0.5]
            grounding_label_mask[idx[ious_per_gt >= 0.5]] = 1
            grounding_label_mask[ious_per_proposal < 0.2] = 1

        bboxes_labels.append(grounding_label)
        bboxes_scores.append(grounding_label_score)
        bboxes_label_masks.append(grounding_label_mask)
        obj_str = ""
        for j in range(proposals[i].shape[0]):
            obj_str += "Object %d: <object><object>. " % j

        if use_two_tokens[i] == 0:
            obj_str = obj_str + "The global image is <global>"
        elif use_two_tokens[i] == 1:
            obj_str = "The global image is <global>. " + obj_str + "The global image is <global>"
        else:
            obj_str = "The coarse global image is <global>. " + obj_str + " The detailed global image is <global>. "
        messages = [
                {
                    "role": "user", 
                    "content": 
                    [
                        {"type": "image", "image": images[i]}, 
                        {"type": "text", "text": prompts[i] + obj_str}
                    ]
                }
            ]
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
        all_image_inputs += image_inputs
        image_conversations.append(processor.apply_chat_template(messages, tokenize=False).strip())
    # print(image_conversations)
    image_conversation_inputs = processor(
        text=image_conversations,
        images=all_image_inputs,
        return_tensors="pt",
        padding=True,
        do_resize=False,
    )
    
    text_conversations = []
    for label in text_labels:
        messages = [
                {
                    "role": "user", 
                    "content": 
                    [
                        {"type": "text", "text": "Find an object that matches the given caption. %s <local_text>" % (label)}
                    ]
                }
            ]

        text_conversations.append(processor.apply_chat_template(messages, tokenize=False).strip())
    
    if global_captions[0] is not None:
        for global_caption in global_captions:
            messages = [
                {
                    "role": "user", 
                    "content": 
                    [
                        {"type": "text", "text": "Find an image that matches the given caption. %s <global_text>" % (global_caption)}
                    ]
                }
            ]

            text_conversations.append(processor.apply_chat_template(messages, tokenize=False).strip())
    
    # print(text_conversations)
    text_conversation_inputs = processor(
        text=text_conversations,
        return_tensors="pt",
        padding=True,
        do_resize=False,
    )
    
    object_token_index = processor.tokenizer.convert_tokens_to_ids('<object>')
    local_text_index = processor.tokenizer.convert_tokens_to_ids('<local_text>')
    if global_captions[0] is not None:
        global_token_index = processor.tokenizer.convert_tokens_to_ids('<global>')
        global_text_index = processor.tokenizer.convert_tokens_to_ids('<global_text>')
    else:
        global_token_index = None
        global_text_index = None
    labels = image_conversation_inputs["input_ids"].clone().float()
    labels[:] = -100
 
    for i, bboxes_label in enumerate(bboxes_labels):
        labels[i, image_conversation_inputs["input_ids"][i] == object_token_index] = bboxes_label.clone().float().unsqueeze(1).repeat(1, 2).flatten(0, 1)

    inputs = {
        "text_processed": text_conversation_inputs,
        "labels": labels,
        "text_labels": text_labels,
        "bboxes": proposals,
        "ori_shapes": ori_shapes,
        "bboxes_labels": bboxes_labels,
        "bboxes_scores": bboxes_scores,
        "bboxes_id": object_token_index,  
        'dataset_names': dataset_names,
        'bboxes_label_masks': bboxes_label_masks,
        'idxs': idxs,
        'global_captions': global_captions,
        'global_id': global_token_index,
        'local_text_id': local_text_index,
        'global_text_id': global_text_index,
    }

    return {**inputs, **image_conversation_inputs}

if __name__ == "__main__":
    # Parse arguments
    parser = transformers.HfArgumentParser((TrainingArguments, ))
    training_args, = parser.parse_args_into_dataclasses()
    
    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    from transformers import Qwen3VLConfig
    config = Qwen3VLConfig.from_pretrained(training_args.model_name_or_path)
    config.num_classes = training_args.num_classes
    config.use_global_caption = training_args.use_global_caption
    config.use_two_tokens = training_args.use_two_tokens
    config.use_two_captions = training_args.use_two_captions
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = ObjectEmbed.from_pretrained(training_args.model_name_or_path, config=config, **model_kwargs)
    model.init_new_paramters()
    processor = AutoProcessor.from_pretrained(training_args.model_name_or_path)
    processor.tokenizer.padding_side = 'right'
    processor.tokenizer.add_tokens(['<object>'], special_tokens=True)
    processor.tokenizer.add_tokens(['<local_text>'], special_tokens=True)
    object_token_index = processor.tokenizer.convert_tokens_to_ids('<object>')

    model.model.object_token_id = object_token_index
    if training_args.use_global_caption:
        processor.tokenizer.add_tokens(['<global>'], special_tokens=True)
        processor.tokenizer.add_tokens(['<global_text>'], special_tokens=True)
    

    # Prepare dataset
    if training_args.dataset_name == 'mixture_with_caption':
        dataset_files = [
            ['datasets/processed_FineCops_Ref_data_with_caption.jsonl', 'FineCops_Ref', 'rec', 'datasets/FineCops_Ref_proposals.json'],
            ['datasets/processed_grefcoco_data_with_caption.jsonl', 'grefcoco', 'rec', 'datasets/grefcoco_proposals.json'],
            ['datasets/processed_human_ref_data_with_caption.jsonl', 'human_ref', 'rec', 'datasets/human_ref_proposals.json'],
            ['datasets/processed_refcoco_data_with_caption.jsonl', 'refcoco', 'rec', 'datasets/refcoco_proposals.json'],
            ['datasets/processed_reircoco_data_with_caption.jsonl', 'reircoco', 'rec', 'datasets/reircoco_proposals.json'],
            ['datasets/processed_Ref_L4_data_with_caption.jsonl', 'Ref_L4', 'rec', 'datasets/Ref_L4_proposals.json'],

            ['datasets/processed_refcoco_data_with_caption.jsonl', 'refcocov2', 'rec', 'datasets/refcoco_proposals.json'],

            ['datasets/processed_v3det_data_with_caption.jsonl', 'v3det', 'detection', 'datasets/v3det_proposals.json'],
            ['datasets/processed_coco_data_with_caption.jsonl', 'coco', 'detection', 'datasets/coco_proposals.json'],
            ['datasets/processed_lvis_data_with_caption.jsonl', 'lvis', 'detection', 'datasets/lvis_proposals.json'],

            ['datasets/processed_dam_LVIS_v1_with_caption.jsonl', 'dam_LVIS', 'rec', 'datasets/coco_proposals.json'],
            ['datasets/processed_dam_OpenImages_v1_with_caption.jsonl', 'dam_OpenImages', 'rec', 'datasets/OpenImages_proposals.json'],
            ['datasets/processed_sam_300k_with_captionv2.jsonl', 'self_sam_list', 'rec', 'datasets/self_sam_300k_proposals.json'],

            ['datasets/processed_fgovd_1_attributes_with_caption.jsonl', 'fgovd_1_attributes', 'rec', 'datasets/fgovd_1_attributes_proposals.json'],
            ['datasets/processed_fgovd_2_attributes_with_caption.jsonl', 'fgovd_2_attributes', 'rec', 'datasets/fgovd_2_attributes_proposals.json'],
            ['datasets/processed_fgovd_3_attributes_with_caption.jsonl', 'fgovd_3_attributes', 'rec', 'datasets/fgovd_3_attributes_proposals.json'],
            ['datasets/processed_fgovd_color_with_caption.jsonl', 'fgovd_color', 'rec', 'datasets/fgovd_color_proposals.json'],
            ['datasets/processed_fgovd_material_with_caption.jsonl', 'fgovd_material', 'rec', 'datasets/fgovd_material_proposals.json'],
            ['datasets/processed_fgovd_pattern_with_caption.jsonl', 'fgovd_pattern', 'rec', 'datasets/fgovd_pattern_proposals.json'],
            ['datasets/processed_fgovd_shuffle_negatives_with_caption.jsonl', 'fgovd_shuffle_negatives', 'rec', 'datasets/fgovd_shuffle_negatives_proposals.json'],
            ['datasets/processed_fgovd_transparency_with_caption.jsonl', 'fgovd_transparency', 'rec', 'datasets/fgovd_transparency_proposals.json'],
        ]
    
    datasets = [LazySupervisedDataset(file[0], file[1], file[2], file[3], training_args.per_image_train_text_batch_size, training_args.per_device_train_batch_size, use_task_prompt=training_args.use_task_prompt, use_global_caption=training_args.use_global_caption, use_two_tokens=training_args.use_two_tokens, use_two_captions=training_args.use_two_captions) for file in dataset_files]
    concat_dataset = ConcatDataset(datasets)
    # prepared_dataset = [prepare_dataset(example) for example in dataset['train']]

    if training_args.freeze_vision_modules:
        print("Freezing vision modules...")
        for n, p in model.named_parameters():
            if any(keyword in n for keyword in ['visual', 'wedetect']):
                p.requires_grad = False
    if training_args.freeze_llm_modules:
        print("Freezing LLM modules...")
        for n, p in model.named_parameters():
            if any(keyword in n for keyword in ['language_model']):
                p.requires_grad = False
        model.lm_head.requires_grad = True

    total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
    trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: ~{total_params/1e6:.2f} MB)")
    print(f"Trainable parameters: ~{trainable_params/1e6:.2f} MB)")

    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=concat_dataset,
        data_collator=collate_fn,
    )

    # Train model
    if training_args.resume_from_checkpoint is not None:
        checkpoint = get_last_checkpoint(training_args.output_dir)
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save final model

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
