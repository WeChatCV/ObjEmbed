from models.qwen3vl_objembed import ObjectEmbed
from models.vision_process import process_vision_info
from transformers import AutoProcessor
from generate_proposal import SimpleYOLOWorldDetector
from vis import plot_bounding_boxes
import argparse
import torch
from PIL import Image
import copy
import torch.nn.functional as F



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--objembed_checkpoint', type=str, default='')
    parser.add_argument('--wedetect_uni_checkpoint', type=str, default='')
    parser.add_argument('--image', nargs='+', type=str, required=True, help='sperate by space')
    parser.add_argument('--query', type=str, default='')
    parser.add_argument('--image_query', type=str, default='')
    parser.add_argument('--task', type=str, default='rec', choices=['rec', 'retrieval_by_object', 'retrieval_by_image'])
    parser.add_argument('--visualize', action='store_true', help='only for rec task')
    args = parser.parse_args()

    print('Input image:', args.image)
    print("query:", args.query)
    if args.task == 'rec':
        assert len(args.image) == 1, "Only support single image for rec task"

    # load detection model
    model_size = 'base' if 'base' in args.wedetect_uni_checkpoint else 'large'
    det_model = SimpleYOLOWorldDetector(backbone_size=model_size, prompt_dim=768, num_prompts=256, num_proposals=100)
    checkpoint = torch.load(args.wedetect_uni_checkpoint, map_location='cpu')
    # backbone
    keys = list(checkpoint.keys())
    for key in keys:
        if 'backbone' in key:
            new_key = key.replace('backbone.image_model.model.', 'backbone.')
            checkpoint[new_key] = checkpoint.pop(key)
    # head
    keys = list(checkpoint.keys())
    for key in keys:
        if 'bbox_head' in key:
            new_key = key.replace('bbox_head.head_module.', 'bbox_head.')
            new_key = new_key.replace('0.2.', '0.6.')
            new_key = new_key.replace('1.2.', '1.6.')
            new_key = new_key.replace('2.2.', '2.6.')
            new_key = new_key.replace('1.bn', '4')
            new_key = new_key.replace('1.conv', '3')
            new_key = new_key.replace('0.bn', '1')
            new_key = new_key.replace('0.conv', '0')
            checkpoint[new_key] = checkpoint.pop(key)
    det_model = det_model.cuda().eval()
    msg = det_model.load_state_dict(checkpoint, strict=False)


    # load qwen model
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    model = ObjectEmbed.from_pretrained(args.objembed_checkpoint, **model_kwargs)
    processor = AutoProcessor.from_pretrained(args.objembed_checkpoint)
    object_token_index = processor.tokenizer.convert_tokens_to_ids("<object>")
    local_text_id = processor.tokenizer.convert_tokens_to_ids("<local_text>")
    model.model.object_token_id = object_token_index
    global_id = None
    global_text_id = None

    if model.use_global_caption:
        global_id = processor.tokenizer.convert_tokens_to_ids("<global>")
        global_text_id = processor.tokenizer.convert_tokens_to_ids("<global_text>")
    model = model.cuda().eval()


    # compute text embedding for query
    if args.query != '':
        if args.task == 'rec' or args.task == 'retrieval_by_object':
            messages = [
                {
                    "role": "user", 
                    "content": 
                    [
                        {"type": "text", "text": "Find an object that matches the given caption. %s <local_text>" % args.query}
                    ]
                }
            ]
        else:
            messages = [
                {
                    "role": "user", 
                    "content": 
                    [
                        {"type": "text", "text": "Find an image that matches the given caption. %s <global_text>" % args.query}
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
        if args.task == 'rec' or args.task == 'retrieval_by_object':
            query_embedding = pred['local_text_embeddings']
        else:
            query_embedding = pred['global_text_embeddings']
    else:
        image = Image.open(args.image_query).convert("RGB")
        ori_shape = [image.size]
        messages = [
            {
                "role": "user", 
                "content": 
                [
                    {"type": "image", "image": copy.deepcopy(image)}, 
                    {"type": "text", "text": "The coarse global image is <global>. The detailed global image is <global>. "}
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
                bboxes=[torch.zeros((0, 4)).cuda().to(model.dtype)],
                ori_shapes=ori_shape,
                bboxes_id=object_token_index,
                global_id=global_id,
                local_text_id=local_text_id,
                global_text_id=global_text_id,
            )

        query_embedding = pred['full_image_embeddings'][:, 0, :]


    # compute object embeddings for each image
    candicate_object_embedding = []
    candicate_image_embedding = []
    objectnesses = []
    for img_path in args.image:
        with torch.no_grad():
            outputs = det_model([img_path])

        proposals = [outputs[0]['bboxes'].float().cpu().tolist()]
        obj_str = ""
        for j in range(len(proposals[0])):
            obj_str += "Object %d: <object><object>. " % j
        if model.use_two_tokens == 0:
            obj_str = obj_str + "The global image is <global>"
        elif model.use_two_tokens == 1:
            obj_str = "The global image is <global>. " + obj_str + "The global image is <global>"
        else:
            obj_str = "The coarse global image is <global>. " + obj_str + " The detailed global image is <global>. "
            
        image = Image.open(img_path).convert("RGB")
        ori_shape = [image.size]
        messages = [
            {
                "role": "user", 
                "content": 
                [
                    {"type": "image", "image": copy.deepcopy(image)}, 
                    {"type": "text", "text": "Locate the specific object being described by analyzing its unique instance-level attributes, its spatial position, and its relationship with surrounding objects. " + obj_str}
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
                bboxes=[torch.tensor(proposals[0]).cuda().to(model.dtype)],
                ori_shapes=ori_shape,
                bboxes_id=object_token_index,
                global_id=global_id,
                local_text_id=local_text_id,
                global_text_id=global_text_id,
            )

        object_embeddings = pred['object_embeddings']
        objectness = pred['objness'].sigmoid().float()
        candicate_object_embedding.append(object_embeddings)
        if model.use_two_tokens > 0:
            candicate_image_embedding.append(pred['full_image_embeddings'][:, 0, :])
        else:
            candicate_image_embedding.append(pred['full_image_embeddings'])
        objectnesses.append(objectness)
    
    candicate_object_embedding = torch.cat(candicate_object_embedding, dim=0)
    candicate_image_embedding = torch.cat(candicate_image_embedding, dim=0)
    objectnesses = torch.cat(objectnesses, dim=0)
    if args.task == 'rec':
        candicate_object_embedding = F.normalize(candicate_object_embedding, dim=-1)
        query_embedding = F.normalize(query_embedding, dim=-1)
        pred_scores = query_embedding @ candicate_object_embedding.transpose(-1, -2)
        pred_scores = pred_scores * model.logit_log_scale.exp()
        pred_scores = pred_scores + model.logit_bias
        pred_scores = pred_scores.float().sigmoid().cpu().flatten()
        pred_scores = pred_scores * objectnesses.cpu().flatten()
    elif args.task == 'retrieval_by_object':
        candicate_object_embedding = F.normalize(candicate_object_embedding, dim=-1)
        query_embedding = F.normalize(query_embedding, dim=-1)
        pred_scores = query_embedding @ candicate_object_embedding.transpose(-1, -2)
        pred_scores = pred_scores * model.logit_log_scale.exp()
        pred_scores = pred_scores + model.logit_bias
        pred_scores = pred_scores.float().sigmoid().cpu()
        pred_scores = pred_scores.reshape(len(args.image), 100)
        pred_scores = torch.max(pred_scores, dim=0)[0]
        print(pred_scores)
    elif args.task == 'retrieval_by_image':
        candicate_image_embedding = F.normalize(candicate_image_embedding, dim=-1)
        query_embedding = F.normalize(query_embedding, dim=-1)
        pred_scores = query_embedding @ candicate_image_embedding.transpose(-1, -2)
        pred_scores = pred_scores * model.logit_image_log_scale.exp()
        pred_scores = pred_scores + model.logit_image_bias
        pred_scores = pred_scores.float().sigmoid().cpu().flatten()
        print(pred_scores)
    
    if args.task == 'rec' and args.visualize:
        max_index = torch.argmax(pred_scores)
        pred_image = plot_bounding_boxes(image, [proposals[0][max_index]])
        # pred_image = plot_bounding_boxes(image, proposals[0])
        pred_image.save("pred.png")  # 你可以自定义保存路径和文件名

        
    

    

