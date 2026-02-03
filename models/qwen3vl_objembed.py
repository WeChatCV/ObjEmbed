from dataclasses import dataclass
import torch
import torch.nn as nn
import torchvision
import re
import math
import copy
from transformers.modeling_outputs import ModelOutput
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLConfig, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLPreTrainedModel, Qwen3VLModel, is_torchdynamo_compiling, Cache, Qwen3VLModelOutputWithPast, Qwen3VLCausalLMOutputWithPast
from typing import Optional, List, Union, Tuple
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.ops.boxes import box_area

DEFAULT_OBJECT_TOKEN = '<object>'

def gen_sineembed_for_position(pos_tensor, embedding_dim):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    dim = embedding_dim // pos_tensor.size(-1)
    scale = 2 * math.pi
    dim_t = torch.arange(dim, dtype=pos_tensor.dtype, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / dim)
    x_embed = pos_tensor[:, 0] * scale
    y_embed = pos_tensor[:, 1] * scale
    pos_x = x_embed[:, None] / dim_t
    pos_y = y_embed[:, None] / dim_t
    pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
    pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=1)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, 2] * scale
        pos_w = w_embed[:, None] / dim_t
        pos_w = torch.stack((pos_w[:, 0::2].sin(), pos_w[:, 1::2].cos()), dim=2).flatten(1)

        h_embed = pos_tensor[:, 3] * scale
        pos_h = h_embed[:, None] / dim_t
        pos_h = torch.stack((pos_h[:, 0::2].sin(), pos_h[:, 1::2].cos()), dim=2).flatten(1)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=1)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


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

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)


    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)




def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def sigmoid_focal_loss(inputs, targets, bboxes_label_masks=None, alpha: float = 0.25, gamma: float = 2, use_focal=True):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    if use_focal:
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
    else:
        loss = ce_loss

    # return loss.mean()
    if bboxes_label_masks is not None:
        loss = loss * bboxes_label_masks
    return loss.mean(0).sum() * 2



def loss_boxes(src_boxes, target_boxes, num_boxes):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
    """

    loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

    losses = {}
    losses['loss_bbox'] = loss_bbox.sum() / num_boxes

    loss_giou = 1 - torch.diag(generalized_box_iou(
        box_cxcywh_to_xyxy(src_boxes),
        box_cxcywh_to_xyxy(target_boxes)))
    losses['loss_giou'] = loss_giou.sum() / num_boxes

    return losses



class Qwen3VLModelGrounding(Qwen3VLModel):
    def __init__(self, config):
        super().__init__(config)
        # 这里可以添加新的模块，例如 bbox 编码器
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', 'mlp2x_gelu')
        mlp_depth = int(mlp_gelu_match.group(1))


        modules = [nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size))
        self.image_pos_projector = nn.Sequential(*modules)
        self.image_pos_projector[-1].weight.data.zero_()
        self.image_pos_projector[-1].bias.data.zero_()

        print(config.text_config.hidden_size)
        if config.text_config.hidden_size > 4000: 
            modules = [nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size))
            self.object_vision_projector = nn.Sequential(*modules)
        else:
            modules = [nn.Linear(config.text_config.hidden_size * 7 * 7, config.text_config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size))
            self.object_vision_projector = nn.Sequential(*modules)

        modules = [nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size))
        self.object_pos_projector = nn.Sequential(*modules)
        self.object_pos_projector[-1].weight.data.zero_()
        self.object_pos_projector[-1].bias.data.zero_()

        self.second_scale_conv = nn.ConvTranspose2d(config.text_config.hidden_size, config.text_config.hidden_size // 2, kernel_size=2, stride=2)
        self.first_scale_conv1 = nn.ConvTranspose2d(config.text_config.hidden_size, config.text_config.hidden_size // 2, kernel_size=2, stride=2)
        self.first_scale_norm = nn.LayerNorm(config.text_config.hidden_size // 2)
        self.first_scale_act = nn.GELU()
        self.first_scale_conv2 = nn.ConvTranspose2d(config.text_config.hidden_size // 2, config.text_config.hidden_size // 4, kernel_size=2, stride=2)
        self.merge = nn.Linear(config.text_config.hidden_size // 4 + config.text_config.hidden_size // 2 + config.text_config.hidden_size, config.text_config.hidden_size)


    def generate_coordinate(self, featmap, device='cuda'):
        featmap_sizes = featmap.shape[-2:]
        x_range = torch.linspace(0, int(featmap_sizes[1])-1, int(featmap_sizes[1]), device=device) / int(featmap_sizes[1])
        y_range = torch.linspace(0, int(featmap_sizes[0])-1, int(featmap_sizes[0]), device=device) / int(featmap_sizes[0])
        y, x = torch.meshgrid(y_range, x_range)
        y = y.unsqueeze(-1)
        x = x.unsqueeze(-1)
        coord_feat = torch.cat([x, y], -1)

        return coord_feat
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        bboxes = None,  # 新增参数：边界框
        ori_shapes = None,  # 新增参数：原始图像尺寸
        pos_embeddings = None,
        **kwargs,
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_features = []
            object_features = []
            object_masks = []
            box_coors = []
            split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
            scale3_image_feats = [feat.clone() for feat in image_embeds]
            scale2_image_feats = torch.split(deepstack_image_embeds[-1], split_sizes)
            scale1_image_feats = torch.split(deepstack_image_embeds[-2], split_sizes)
            # extract RoI features based on bboxes
            for i, (scale1_image_feat, scale2_image_feat, scale3_image_feat, ori_shape, feat_shape, bbox) in enumerate(zip(scale1_image_feats, scale2_image_feats, scale3_image_feats, ori_shapes, image_grid_thw, bboxes)):
                feat_shape = feat_shape.cpu().tolist()
                T, H, W = feat_shape
                H = H // self.visual.spatial_merge_size
                W = W // self.visual.spatial_merge_size
                # the first scale
                scale1_image_feat = scale1_image_feat.reshape(T, H, W, self.config.text_config.hidden_size).permute(0, 3, 1, 2).contiguous()
                scale1_image_feat = self.first_scale_conv1(scale1_image_feat).permute(0, 2, 3, 1)
                scale1_image_feat = self.first_scale_act(self.first_scale_norm(scale1_image_feat)).permute(0, 3, 1, 2)
                scale1_image_feat = self.first_scale_conv2(scale1_image_feat)

                # the second scale
                scale2_image_feat = scale2_image_feat.reshape(T, H, W, self.config.text_config.hidden_size).permute(0, 3, 1, 2).contiguous()
                scale2_image_feat = self.second_scale_conv(scale2_image_feat)

                # the third scale
                scale3_image_feat = scale3_image_feat.reshape(T, H, W, self.config.text_config.hidden_size).permute(0, 3, 1, 2).contiguous()

                if len(bbox) == 0:
                    gt_bbox = torch.tensor([[0, 0, W * 32, H * 32]], device=scale3_image_feat.device, dtype=scale3_image_feat.dtype)
                    object_masks.append(torch.tensor([0], device=scale3_image_feat.device, dtype=torch.bool))
                else:
                    gt_bbox = torch.tensor(bbox, device=scale3_image_feat.device, dtype=scale3_image_feat.dtype) / (torch.tensor([ori_shape[0], ori_shape[1], ori_shape[0], ori_shape[1]], device=scale3_image_feat.device, dtype=scale3_image_feat.dtype) / torch.tensor([W * 32, H * 32, W * 32, H * 32], device=scale3_image_feat.device, dtype=scale3_image_feat.dtype))
                    object_masks.append(torch.tensor([1]*len(bbox), device=scale3_image_feat.device, dtype=torch.bool))
                
                roi_feats1 = torchvision.ops.roi_align(scale1_image_feat.float(), [gt_bbox.float()], 7, 1/8).to(scale3_image_feat.dtype)
                roi_feats2 = torchvision.ops.roi_align(scale2_image_feat.float(), [gt_bbox.float()], 7, 1/16).to(scale3_image_feat.dtype)
                roi_feats3 = torchvision.ops.roi_align(scale3_image_feat.float(), [gt_bbox.float()], 7, 1/32).to(scale3_image_feat.dtype)

                # image_feats
                image_coor = (self.generate_coordinate(scale3_image_feat) + 0.5).to(scale3_image_feat.dtype)
                image_coor = self.image_pos_projector(gen_sineembed_for_position(image_coor.flatten(0, 1), self.config.text_config.hidden_size))
                image_features.append(image_embeds[i] + image_coor)

                # object_feats
                roi_feats = torch.cat([roi_feats1, roi_feats2, roi_feats3], dim=1).permute(0, 2, 3, 1)
                roi_feats = self.merge(roi_feats)
                if self.config.text_config.hidden_size > 4000: 
                    roi_feats = roi_feats.flatten(1, 2)
                    roi_feats = torch.mean(roi_feats, dim=1)
                    roi_feats = self.object_vision_projector(roi_feats)
                else:
                    roi_feats = self.object_vision_projector(roi_feats.flatten(1))
                box_coor = box_xyxy_to_cxcywh(gt_bbox) / torch.tensor([W * 32, H * 32, W * 32, H * 32], device=gt_bbox.device, dtype=gt_bbox.dtype)
                box_coor = self.object_pos_projector(gen_sineembed_for_position(box_coor, self.config.text_config.hidden_size))
                object_features.append(roi_feats + box_coor)
                box_coors.append(box_coor)

            image_embeds = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            object_masks = torch.cat(object_masks, dim=0)
            object_id_mask = (input_ids == self.object_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            object_features = torch.cat(object_features, dim=0)[object_masks]
            box_coors = torch.cat(box_coors, dim=0)[object_masks]
            pos_embeddings += box_coors
            object_features = torch.cat([object_features.unsqueeze(1), pos_embeddings.unsqueeze(1)], dim=1)
            # object_features = object_features.unsqueeze(1).repeat(1, 2, 1)
            # object_features[:, 1, :] += pos_embeddings
            object_features = object_features.flatten(0, 1)
            inputs_embeds = inputs_embeds.masked_scatter(object_id_mask, object_features)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )



@dataclass
class ObjectEmbedOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    full_image_embeddings: Optional[torch.FloatTensor] = None
    object_embeddings: Optional[tuple[torch.FloatTensor]] = None
    global_text_embeddings: Optional[tuple[torch.FloatTensor]] = None
    local_text_embeddings: Optional[tuple[torch.FloatTensor]] = None
    objness: Optional[torch.FloatTensor] = None


def gather_and_deduplicate_negatives_simple(
    local_embeddings: torch.Tensor, 
    local_texts: List[str],
    datasetname,
    bs,
) -> Tuple[torch.Tensor, List[str]]:
    """
    在DDP环境中，收集所有卡的负样本（embeddings 和 texts），并进行去重。
    此版本不保留来自其他卡的 embeddings 的梯度，更简单高效。
    
    Args:
        local_embeddings (torch.Tensor): 当前卡上的文本嵌入，形状为 (N, D)。
        local_texts (List[str]): 当前卡上对应的原始文本列表，长度为 N。

    Returns:
        Tuple[torch.Tensor, List[str]]:
            - all_dedup_embeddings (torch.Tensor): 去重后的所有负样本嵌入。
            - all_dedup_texts (List[str]): 去重后的所有负样本原始文本。
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if world_size == 1:
        return local_embeddings, local_texts
    
    # 1. 收集所有卡的 texts (无需梯度)
    all_texts_list = [None] * world_size
    dist.all_gather_object(all_texts_list, local_texts)
    all_texts_flat = [text for sublist in all_texts_list for text in sublist]

    # 2. 收集所有卡的 embeddings (无需梯度)
    all_embeddings = torch.cat(torch.distributed.nn.all_gather(local_embeddings), dim=0)

    if datasetname != 'human_ref':
        # 3. 基于文本进行去重，并记录原始索引
        unique_texts = []
        original_indices = []
        seen_texts = []
        for i, text in enumerate(all_texts_flat):
            if text not in seen_texts and text not in local_texts:
                seen_texts.append(text)
                unique_texts.append(text)
                original_indices.append(i)
        
        selected_embeddings = all_embeddings[torch.tensor(original_indices)]
                
        # 将列表堆叠成最终的 tensor
        all_dedup_embeddings = torch.cat([local_embeddings, selected_embeddings], dim=0)
    else:
        num_text_per_image = len(local_texts) // bs
        text_from_other_device = all_texts_flat[:len(local_texts)*rank] + all_texts_flat[len(local_texts)*(rank+1):]
        seen_texts = []
        for i in range((world_size-1) * bs):
            seen_texts.append(text_from_other_device[i*num_text_per_image+1: (i+1)*num_text_per_image])
        embeddings_from_other_device = torch.cat([all_embeddings[:len(local_texts)*rank], all_embeddings[len(local_texts)*(rank+1):]])
        embeddings_from_other_device = embeddings_from_other_device.reshape((world_size-1) * bs, num_text_per_image, all_embeddings.shape[-1])[:, 1:].flatten(0, 1)
        all_dedup_embeddings = torch.cat([local_embeddings, embeddings_from_other_device], dim=0)
    

    return all_dedup_embeddings, local_texts + seen_texts




class ObjectEmbed(Qwen3VLForConditionalGeneration):
    def __init__(self, config: Qwen3VLConfig):
        super().__init__(config)
        self.model = Qwen3VLModelGrounding(config)
        self.post_init()

        logit_scale_value = 0.05
        self.logit_log_scale = nn.Parameter(
            torch.Tensor([math.log(1 / logit_scale_value)]), requires_grad=True)
        bias_value = -math.log((1 - 0.01) / 0.01)
        self.logit_bias = nn.Parameter(
            torch.Tensor([bias_value]), requires_grad=True)
        self.num_classes = config.num_classes
        self.use_global_caption = getattr(config, 'use_global_caption', False)
        self.use_two_tokens = getattr(config, 'use_two_tokens', 0)
        self.use_two_captions = getattr(config, 'use_two_captions', False)

        if self.use_global_caption:
            self.logit_image_log_scale = nn.Parameter(
                torch.Tensor([math.log(1 / logit_scale_value)]), requires_grad=True)
            self.logit_image_bias = nn.Parameter(
                torch.Tensor([bias_value]), requires_grad=True)

        self.logit_iou_proj = nn.Sequential(
            nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.text_config.hidden_size // 2, 1)
        )

        self.logit_pos_embedding = nn.Embedding(1, config.text_config.hidden_size)
        print(self.logit_log_scale, self.logit_bias)

    def init_new_paramters(self,):
        self.logit_log_scale.data[:] = math.log(1 / 0.05)
        self.logit_bias.data[:] = -math.log((1 - 0.01) / 0.01) * 3
        if self.use_global_caption:
            self.logit_image_log_scale.data[:] = math.log(1 / 0.05)
            self.logit_image_bias.data[:] = -math.log((1 - 0.01) / 0.01) * 3
        print(self.logit_log_scale, self.logit_bias)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        text_processed = None,
        text_labels = None,
        bboxes = None,  # 新增参数：边界框
        ori_shapes = None,  # 新增参数：原始图像尺寸
        bboxes_labels = None,  # 新增参数：边界框标签
        bboxes_scores = None,  # 新增参数：边界框标签分数
        bboxes_label_masks = None,  # 新增参数
        bboxes_id = None,  # 新增参数：边界框对应的 token id
        global_id = None,  # 新增参数：边界框对应的 token id
        local_text_id = None,  # 新增参数：本地文本对应的 token id
        global_text_id = None,  # 新增参数：全局文本对应的 token id
        dataset_names = None,
        idxs = None,
        global_captions = None,
        **kwargs,
    ):
        # generate text_embedding
        local_text_embeddings = None
        global_text_embeddings = None
        if text_processed is not None:
            text_outputs = self.model(**text_processed)
        
            last_hidden_state = text_outputs.last_hidden_state
            local_text_positions = text_processed['input_ids'] == local_text_id
            local_text_embeddings = last_hidden_state[local_text_positions].reshape(-1, last_hidden_state.shape[-1])
            num_classes = len(local_text_embeddings)
            if self.use_global_caption:
                global_text_positions = text_processed['input_ids'] == global_text_id
                global_text_embeddings = last_hidden_state[global_text_positions].reshape(-1, last_hidden_state.shape[-1])
                if self.use_two_captions and self.training:
                    global_text_embeddings = global_text_embeddings.view(-1, 2, last_hidden_state.shape[-1])

        # generate object_embedding
        full_image_embeddings = None
        object_embeddings = None
        object_ious = None
        if input_ids is not None:
            num_boxes = [len(box) for box in bboxes]
            pos_embeddings = self.logit_pos_embedding(torch.zeros((sum(num_boxes),), device=input_ids.device, dtype=torch.long))
            image_outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position, 
                bboxes=bboxes, 
                ori_shapes=ori_shapes,
                pos_embeddings=pos_embeddings,
            )
            last_hidden_state = image_outputs.last_hidden_state
            if self.use_global_caption:
                proposal_positions = input_ids == global_id
                full_image_embeddings = last_hidden_state[proposal_positions].reshape(-1, last_hidden_state.shape[-1])
                if self.use_two_tokens > 0:
                    full_image_embeddings = full_image_embeddings.view(-1, 2, last_hidden_state.shape[-1])
            
            proposal_positions = input_ids == bboxes_id
            object_embeddings_base = last_hidden_state[proposal_positions].reshape(-1, 2, last_hidden_state.shape[-1])

            object_embeddings = object_embeddings_base[:, 0, :]
            object_ious = self.logit_iou_proj(object_embeddings_base[:, 1, :])

        # compute loss
        loss = None
        if self.training:

            # obj cls loss
            num_text_per_image = len(text_labels) // len(bboxes)
            num_boxes_per_image = [len(box) for box in bboxes]
            full_text_embeddings, full_text_labels = gather_and_deduplicate_negatives_simple(local_text_embeddings, text_labels, dataset_names[0], len(bboxes))
            full_text_embeddings = full_text_embeddings[:self.num_classes]
            full_text_labels = full_text_labels[:self.num_classes]

            object_embeddings = F.normalize(object_embeddings)
            full_text_embeddings = F.normalize(full_text_embeddings)
            proposal_logits = object_embeddings @ full_text_embeddings.transpose(-1, -2)
            proposal_logits = proposal_logits * self.logit_log_scale.exp()
            proposal_logits = proposal_logits + self.logit_bias
            proposal_logits = proposal_logits.float()
        
            bboxes_labels = torch.cat(bboxes_labels).to(proposal_logits.device).long().view(-1)
            bboxes_labels[bboxes_labels == -1] = proposal_logits.shape[1]
            bboxes_scores = torch.cat(bboxes_scores).to(proposal_logits.device).float().view(-1)
            bboxes_label_masks = torch.cat(bboxes_label_masks).unsqueeze(-1).repeat(1, full_text_embeddings.shape[0])

            target_classes_onehot = torch.zeros([proposal_logits.shape[0], proposal_logits.shape[1] + 1],
                                            dtype=proposal_logits.dtype, layout=proposal_logits.layout, device=proposal_logits.device)
            target_classes_onehot.scatter_(1, bboxes_labels.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:,:-1]

            if dataset_names[0] == 'human_ref':
                positive_idx = torch.tensor([j*num_text_per_image for j in range(len(bboxes))], device=target_classes_onehot.device, dtype=torch.int)
                num_boxes_per_image = torch.cumsum(torch.tensor([0] + num_boxes_per_image, device=target_classes_onehot.device, dtype=torch.int), dim=0)
                for i in range(len(bboxes)):
                    temp_positive_idx = torch.cat([positive_idx[:i], positive_idx[(i+1):]])
                    bboxes_label_masks[num_boxes_per_image[i]:num_boxes_per_image[i+1], temp_positive_idx] = 0

            loss = sigmoid_focal_loss(proposal_logits, target_classes_onehot, bboxes_label_masks)

            # img cls loss
            if self.use_global_caption:
                if self.use_two_tokens == 0 and not self.use_two_captions:
                    full_global_text_embeddings, full_global_captions = gather_and_deduplicate_negatives_simple(global_text_embeddings, global_captions, 'global', len(bboxes))
                    full_image_embeddings = F.normalize(full_image_embeddings)
                    full_global_text_embeddings = F.normalize(full_global_text_embeddings)
                    
                    global_logits = full_image_embeddings @ full_global_text_embeddings.transpose(-1, -2)
                    global_logits = global_logits * self.logit_image_log_scale.exp()
                    global_logits = global_logits + self.logit_image_bias
                    global_logits = global_logits.float()

                    target_classes_onehot = torch.zeros([len(bboxes), global_logits.shape[1]],
                                                dtype=global_logits.dtype, layout=global_logits.layout, device=global_logits.device)
                    target_classes_onehot[torch.arange(len(bboxes)), torch.arange(len(bboxes))] = 1
                    loss += sigmoid_focal_loss(global_logits, target_classes_onehot) * 0.5
                elif self.use_two_tokens == 1 and not self.use_two_captions:
                    full_global_text_embeddings, full_global_captions = gather_and_deduplicate_negatives_simple(global_text_embeddings, global_captions, 'global', len(bboxes))
                    full_image_embeddings = F.normalize(full_image_embeddings.flatten(0, 1))
                    full_global_text_embeddings = F.normalize(full_global_text_embeddings)
                    global_logits = full_image_embeddings @ full_global_text_embeddings.transpose(-1, -2)
                    global_logits = global_logits * self.logit_image_log_scale.exp()
                    global_logits = global_logits + self.logit_image_bias
                    global_logits = global_logits.float()

                    target_classes_onehot = torch.zeros([len(bboxes), global_logits.shape[1]],
                                                dtype=global_logits.dtype, layout=global_logits.layout, device=global_logits.device)
                    target_classes_onehot[torch.arange(len(bboxes)), torch.arange(len(bboxes))] = 1
                    target_classes_onehot = target_classes_onehot.unsqueeze(1).repeat(1, 2, 1).view(-1, global_logits.shape[1])
                    loss += sigmoid_focal_loss(global_logits, target_classes_onehot) * 0.5
                elif self.use_two_tokens == 0 and self.use_two_captions:
                    global_captions1 = global_captions[::2]
                    global_captions2 = global_captions[1::2]
                    global_text_embeddings1 = global_text_embeddings[:, 0, :]
                    global_text_embeddings2 = global_text_embeddings[:, 1, :]
                    full_global_text_embeddings1, full_global_captions = gather_and_deduplicate_negatives_simple(global_text_embeddings1, global_captions1, 'global', len(bboxes))
                    full_global_text_embeddings2, full_global_captions = gather_and_deduplicate_negatives_simple(global_text_embeddings2, global_captions2, 'global', len(bboxes))
                    full_image_embeddings = F.normalize(full_image_embeddings)
                    min_num = min(full_global_text_embeddings1.shape[0], full_global_text_embeddings2.shape[0])
                    full_global_text_embeddings1 = F.normalize(full_global_text_embeddings1)[:min_num]
                    full_global_text_embeddings2 = F.normalize(full_global_text_embeddings2)[:min_num]

                    global_logits1 = full_image_embeddings @ full_global_text_embeddings1.transpose(-1, -2)
                    global_logits2 = full_image_embeddings @ full_global_text_embeddings2.transpose(-1, -2)
                    global_logits = torch.cat([global_logits1, global_logits2], dim=0)
                    global_logits = global_logits * self.logit_image_log_scale.exp()
                    global_logits = global_logits + self.logit_image_bias
                    global_logits = global_logits.float()

                    target_classes_onehot = torch.zeros([len(bboxes), global_logits.shape[1]],
                                                dtype=global_logits.dtype, layout=global_logits.layout, device=global_logits.device)
                    target_classes_onehot[torch.arange(len(bboxes)), torch.arange(len(bboxes))] = 1
                    target_classes_onehot = target_classes_onehot.unsqueeze(0).repeat(2, 1, 1).view(-1, global_logits.shape[1])
                    loss += sigmoid_focal_loss(global_logits, target_classes_onehot) * 0.5
                elif self.use_two_tokens == 1 and self.use_two_captions:
                    global_captions1 = global_captions[::2]
                    global_captions2 = global_captions[1::2]
                    global_text_embeddings1 = global_text_embeddings[:, 0, :]
                    global_text_embeddings2 = global_text_embeddings[:, 1, :]
                    full_global_text_embeddings1, full_global_captions = gather_and_deduplicate_negatives_simple(global_text_embeddings1, global_captions1, 'global', len(bboxes))
                    full_global_text_embeddings2, full_global_captions = gather_and_deduplicate_negatives_simple(global_text_embeddings2, global_captions2, 'global', len(bboxes))
                    min_num = min(full_global_text_embeddings1.shape[0], full_global_text_embeddings2.shape[0])
                    full_image_embeddings = F.normalize(full_image_embeddings.flatten(0, 1))
                    full_global_text_embeddings1 = F.normalize(full_global_text_embeddings1)[:min_num]
                    full_global_text_embeddings2 = F.normalize(full_global_text_embeddings2)[:min_num]

                    global_logits1 = full_image_embeddings @ full_global_text_embeddings1.transpose(-1, -2)
                    global_logits2 = full_image_embeddings @ full_global_text_embeddings2.transpose(-1, -2)
                    global_logits = torch.cat([global_logits1, global_logits2], dim=0)
                    global_logits = global_logits * self.logit_image_log_scale.exp()
                    global_logits = global_logits + self.logit_image_bias
                    global_logits = global_logits.float()

                    target_classes_onehot = torch.zeros([len(bboxes), global_logits.shape[1]],
                                                dtype=global_logits.dtype, layout=global_logits.layout, device=global_logits.device)
                    target_classes_onehot[torch.arange(len(bboxes)), torch.arange(len(bboxes))] = 1
                    target_classes_onehot = target_classes_onehot.unsqueeze(1).repeat(1, 2, 1).view(-1, global_logits.shape[1])
                    target_classes_onehot = target_classes_onehot.unsqueeze(0).repeat(2, 1, 1).view(-1, global_logits.shape[1])
                    loss += sigmoid_focal_loss(global_logits, target_classes_onehot) * 0.5
                
                elif self.use_two_tokens == 2 and self.use_two_captions:
                    global_captions1 = global_captions[::2]
                    global_captions2 = global_captions[1::2]
                    global_text_embeddings1 = global_text_embeddings[:, 0, :]
                    global_text_embeddings2 = global_text_embeddings[:, 1, :]
                    full_global_text_embeddings1, full_global_captions = gather_and_deduplicate_negatives_simple(global_text_embeddings1, global_captions1, 'global', len(bboxes))
                    full_global_text_embeddings2, full_global_captions = gather_and_deduplicate_negatives_simple(global_text_embeddings2, global_captions2, 'global', len(bboxes))
                    full_image_embeddings1 = F.normalize(full_image_embeddings[:, 0, :])
                    full_image_embeddings2 = F.normalize(full_image_embeddings[:, 1, :])
                    min_num = min(full_global_text_embeddings1.shape[0], full_global_text_embeddings2.shape[0])
                    full_global_text_embeddings1 = F.normalize(full_global_text_embeddings1)[:min_num]
                    full_global_text_embeddings2 = F.normalize(full_global_text_embeddings2)[:min_num]

                    global_logits1 = full_image_embeddings1 @ full_global_text_embeddings1.transpose(-1, -2)
                    global_logits2 = full_image_embeddings2 @ full_global_text_embeddings2.transpose(-1, -2)
                    global_logits = torch.cat([global_logits1, global_logits2], dim=0)
                    global_logits = global_logits * self.logit_image_log_scale.exp()
                    global_logits = global_logits + self.logit_image_bias
                    global_logits = global_logits.float()

                    target_classes_onehot = torch.zeros([len(bboxes), global_logits.shape[1]],
                                                dtype=global_logits.dtype, layout=global_logits.layout, device=global_logits.device)
                    target_classes_onehot[torch.arange(len(bboxes)), torch.arange(len(bboxes))] = 1
                    target_classes_onehot = target_classes_onehot.unsqueeze(0).repeat(2, 1, 1).view(-1, global_logits.shape[1])
                    loss += sigmoid_focal_loss(global_logits, target_classes_onehot) * 0.25

                else:
                    raise NotImplementedError

            # reg loss
            if torch.sum(bboxes_labels < num_classes) == 0:
                loss_iou_head = object_ious.sum() * 0.
                loss += loss_iou_head
            else:
                loss_iou_head = sigmoid_focal_loss(object_ious[bboxes_scores > 0], bboxes_scores[bboxes_scores > 0].unsqueeze(1))
                loss += loss_iou_head

        return ObjectEmbedOutput(
            loss=loss,
            object_embeddings=object_embeddings,
            full_image_embeddings=full_image_embeddings,
            local_text_embeddings=local_text_embeddings,
            global_text_embeddings=global_text_embeddings,
            objness = object_ious,
        )
    
    
        



