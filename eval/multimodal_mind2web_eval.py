import torch
import ast
import json
import re
import argparse
import os
from PIL import Image
import logging
from tqdm import tqdm
import numpy as np
import collections
import string

from multiprocessing import freeze_support

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModel, AutoImageProcessor
from qwen_vl_utils import process_vision_info


from opencua_utils import (
    analyze_vision_tokens_opencua_multi_images,
    extract_actions
)

from ui_tars_utils import (
    parse_action_to_structure_output, MIN_PIXELS, MAX_PIXELS, IMAGE_FACTOR,
    analyze_vision_tokens_multi_images
)

from attention_helpers import (
    replace_qwen2_5_vl,
    replace_opencua,
    set_attention_implementation,
    configure_accelerate_skip_attention,
    set_kv_cache_budget,
    set_last_vision_indices,
    set_vision_start_idx,
    set_vision_end_idx,
    set_alpha,
    set_window_size,
    set_move_attention_to_cpu,
    set_temperature
)

MM_MIND2WEB_PROMPT =  """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 


## Output Format
```
Thought: ...
Action: ...
```

## Action Space


- type(content='xxx') # First click on the textbox, and then type the content. 
- scroll(point='<point>x1 y1</point>', direction='down or up') # Show more information on the `direction` side.
- select(point='<point>x1 y1</point>', content='xxx') # Select dropdown menu at point x1 y1 and choose entry xxx.
- click(point='<point>x1 y1</point>').   

## Note
- Use English in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part. 
- For `type` action, Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content. 

## Rule
- The `type` action will click + type in a textbox or search box. DO NOT `click` ON search box; otherwise, you will be terminated.

## Previous Actions
{previous_actions}



## User Instruction
{instruction}


"""

MM_MIND2WEB_PROMPT_OPENCUA = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform a series of pyautogui actions to complete the task.

For each step, provide your response in this format:

Thought:
  - Step by Step Progress Assessment:
    - Analyze completed task parts and their contribution to the overall goal
    - Reflect on potential errors, unexpected results, or obstacles
    - If previous action was incorrect, predict a logical recovery step
  - Next Action Analysis:
    - List possible next actions based on current state
    - Evaluate options considering current state and previous actions
    - Propose most logical next action
    - Anticipate consequences of the proposed action
  - For Text Input Actions:
    - Note current cursor position
    - Consolidate repetitive actions (specify count for multiple keypresses)
    - Describe expected final text outcome
    - Use first-person perspective in reasoning

Action:
  Provide clear, concise, and actionable instructions:
  - If the action involves interacting with a specific target:
    - Describe target explicitly without using coordinates
    - Specify element names when possible (use original language if non-English)
    - Describe features (shape, color, position) if name unavailable
    - For window control buttons, identify correctly (minimize "—", maximize "□", close "X")
  - if the action involves keyboard actions like 'press', 'write', 'hotkey':
    - Consolidate repetitive keypresses with count
    - Specify expected text outcome for typing actions

Finally, output the action as PyAutoGUI code or the following functions:
- {{"name": "computer.triple_click", "description": "Triple click on the screen", "parameters": {{"type": "object", "properties": {{"x": {{"type": "number", "description": "The x coordinate of the triple click"}}, "y": {{"type": "number", "description": "The y coordinate of the triple click"}}}}, "required": ["x", "y"]}}}}
- {{"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {{"type": "object", "properties": {{"status": {{"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}}}}, "required": ["status"]}}}}

## Rule
- The `write` action will click + write in a textbox or search box. DO NOT `click` ON search box; otherwise, you will be terminated.


## Previous Actions
{previous_actions}

## User Instruction
{instruction}
"""
######## Metric helper functions ##########

def calculate_f1(pred, label):
    
    pred = set(pred.lower().strip().split())
    label = set(label.lower().strip().split())
    # remove punctuation
    pred = set([x for x in pred if x not in string.punctuation])
    label = set([x for x in label if x not in string.punctuation])
    if len(pred) == 0 and len(label) == 0:
        return 1
    if len(pred) == 0 or len(label) == 0:
        return 0

    tp = len(pred & label)
    fp = len(pred - label)
    fn = len(label - pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1

def is_output_inside_bbox(bboxes, output, scale):
    
    
    try:
        output_x, output_y = output
        output_x /= scale
        output_y /= scale

        for bbox in bboxes:
            bbox_x, bbox_y, bbox_width, bbox_height = bbox
            if bbox_x <= output_x <= bbox_x + bbox_width and bbox_y <= output_y <= bbox_y + bbox_height:
                return True, (output_x, output_y)

        return False, (output_x, output_y)
    except (ValueError, TypeError) as e:
        print(f"Error unpacking output coordinates: {e}, output: {output}")
        return False, (0, 0)


######## End of metric helper functions ##########



logging.basicConfig(level=logging.INFO)
torch.manual_seed(1234)

# Device selection utility
def get_device():
    """Dynamically select the best available device"""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS device (Apple Silicon)")
    else:
        device = "cpu"
        print("Using CPU device")
    return device


def add_box_token(input_string):
    # Step 1: Split the string into individual actions
    if "Action: " in input_string and "start_box=" in input_string:
        suffix = input_string.split("Action: ")[0] + "Action: "
        actions = input_string.split("Action: ")[1:]
        processed_actions = []
        for action in actions:
            action = action.strip()
            # Step 2: Extract coordinates (start_box or end_box) using regex
            coordinates = re.findall(r"(start_box|end_box)='\((\d+),\s*(\d+)\)'", action)
            
            updated_action = action  # Start with the original action
            for coord_type, x, y in coordinates:
                # Convert x and y to integers
                updated_action = updated_action.replace(f"{coord_type}='({x},{y})'", f"{coord_type}='<|box_start|>({x},{y})<|box_end|>'")
            processed_actions.append(updated_action)
        
        # Step 5: Reconstruct the final string
        final_string = suffix + "\n\n".join(processed_actions)
    else:
        final_string = input_string
    return final_string

if __name__ == '__main__':
    freeze_support()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--mm_mind2web_imgs', type=str, default="/fsx/home/kh.huang/gui_agent/data/Multimodal-Mind2Web/release_images/")
    parser.add_argument('--mm_mind2web_test', type=str, default="/fsx/home/kh.huang/gui_agent/UGround/offline_evaluation/Multimodal-Mind2Web/data/samples")
    parser.add_argument('--task', type=str, required=True, choices=["all"])
    parser.add_argument('--debug', default=None, type=int)
    parser.add_argument('--max_new_tokens', type=int, default=500)
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/mps/cpu). If not specified, will auto-detect best available device.')
    parser.add_argument('--model_dtype', type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"], help='Data type to use (auto/bfloat16/float16/float32).')
    parser.add_argument('--attention_implementation', type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"], help='Attention implementation to use (eager/flash_attention_2).')
    parser.add_argument('--kv_cache', type=str, default="original", choices=["original", "pyramid_kv", "vl_cache", "snap_kv", "gui_kv"], help='KV cache method to use (original/pyramid_kv/vl_cache/snap_kv/gui_kv).')
    parser.add_argument('--kv_cache_budget', type=float, default=100, help='KV cache budget in tokens.')
    parser.add_argument('--alpha', type=float, default=None, help='Alpha for GUIKV.')
    parser.add_argument('--window_size', type=int, default=None, help='Window size for GUIKV.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for token information scores.')
    parser.add_argument('--results_dir', type=str, help='Directory to store evaluation results in JSON format.')
    args = parser.parse_args()

    # Get the device to use
    if args.device:
        device = args.device
        print(f"Using user-specified device: {device}")
    else:
        device = get_device()
    
    print(f"Selected device: {device}")
    print(f"Number of CUDA devices available: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    # Validate device availability
    try:
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        elif device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            print("Warning: MPS requested but not available. Falling back to CPU.")
            device = "cpu"
    except Exception as e:
        print(f"Device validation error: {e}. Using CPU as fallback.")
        device = "cpu"

    model_path = args.model_path
    print("model_path: ", model_path)

    if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        tokenizer = None  # Not needed for UI-TARS
    elif args.model_path == "xlangai/OpenCUA-7B":
        processor = AutoImageProcessor.from_pretrained("xlangai/OpenCUA-7B", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("xlangai/OpenCUA-7B", trust_remote_code=True)
    else:
        # Default to UI-TARS processor for backward compatibility
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        tokenizer = None
    
    if args.model_dtype == "float32":
        model_dtype = torch.float32
    elif args.model_dtype == "bfloat16":
        model_dtype = torch.bfloat16
    elif args.model_dtype == "float16":
        model_dtype = torch.float16
    elif args.model_dtype == "auto":
        model_dtype = "auto"
    else:
        raise ValueError(f"Invalid model dtype: {args.model_dtype}")
    
    # replace the attention forward function
    # if args.attention_implementation == "eager":
    
    if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
        replace_qwen2_5_vl(kv_cache_mode=args.kv_cache)
    elif args.model_path == "xlangai/OpenCUA-7B":
        replace_opencua(kv_cache_mode=args.kv_cache)
    else:
        # Default to UI-TARS for backward compatibility
        replace_qwen2_5_vl(kv_cache_mode=args.kv_cache)
    
    
    
        
    # Load model with dynamic device selection
    if device == "cpu":
        if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=model_dtype, device_map="cpu",
                attn_implementation=args.attention_implementation,
            )
        elif args.model_path == "xlangai/OpenCUA-7B":
            model = AutoModel.from_pretrained(
                model_path, torch_dtype=model_dtype, device_map="cpu",
                attn_implementation=args.attention_implementation,
                trust_remote_code=True
            )
        else:
            # Default to UI-TARS for backward compatibility
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=model_dtype, device_map="cpu",
                attn_implementation=args.attention_implementation,
            )
        set_attention_implementation(model, args)
        set_kv_cache_budget(model, args)
        if args.attention_implementation == "eager":
            set_move_attention_to_cpu(model, args)
            # Configure accelerate to skip moving attention tensors back to GPU
            configure_accelerate_skip_attention(model)
    else:
        # Check if we have multiple GPUs
        if torch.cuda.device_count() > 1:
            device_map = "auto"
        else:
            # For single GPU, use explicit device mapping
            device_map = {"": "cuda:0"}  # Map entire model to GPU 0
        
        if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=model_dtype, device_map=device_map, 
                attn_implementation=args.attention_implementation,
            )
        elif args.model_path == "xlangai/OpenCUA-7B":
            model = AutoModel.from_pretrained(
                model_path, torch_dtype=model_dtype, device_map=device_map,
                attn_implementation=args.attention_implementation,
                trust_remote_code=True
            )
        else:
            # Default to UI-TARS for backward compatibility
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=model_dtype, device_map=device_map, 
                attn_implementation=args.attention_implementation,
            )
        set_attention_implementation(model, args)
        set_kv_cache_budget(model, args)
        if args.attention_implementation == "eager":
            set_move_attention_to_cpu(model, args)
            # Configure accelerate to skip moving attention tensors back to GPU
            configure_accelerate_skip_attention(model)

    print("Load Success")

   
    all_element_acc = []
    all_operation_f1 = []
    all_step_acc = []
    sample_to_website = {}
   
    
    if args.task == "all":
        tasks = ["domain", "task", "website"]
    else:
        tasks = [args.task]
    tasks_result = []
    result = []
    all_task_results = []  # Accumulate results from all tasks
    for task in tasks:
        dataset = "cross_" + task + "_blocks.jsonl"
        mm_mind2web_data = []
        with open(os.path.join(args.mm_mind2web_test, dataset), 'r') as f:
            for line in f:
                mm_mind2web_data.append(json.loads(line.strip()))
        if args.debug is not None:
            mm_mind2web_data = mm_mind2web_data[:args.debug]
            print("Num of sample: " + str(len(mm_mind2web_data)) + " (debug mode)")
        else:
            print("Num of sample: " + str(len(mm_mind2web_data)))
    
        for j, sample in tqdm(enumerate(mm_mind2web_data), desc=f"Processing {task} data", total=len(mm_mind2web_data)):
            annotation_action_id = f"{sample['annotation_id']}_{sample['action_uid']}"
            annotation_id = sample['annotation_id']
            
            # we start with 0
            current_block_index = 0        
            
            img_path = os.path.join(args.mm_mind2web_imgs, f"cross_{task}", annotation_action_id, f"{current_block_index}.png")
            
            if not os.path.exists(img_path):
                print("img not found")
                continue
            
            image = Image.open(img_path)
            img_size = image.size
            img_width, img_height = img_size
            instruction = sample["task"]
                
            gt_bboxes = sample["bbox"]
            
            
            # Rescale bboxes to normalized coordinates
            rescaled_gt_bboxes = []
            for bbox in gt_bboxes:
                bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]
                rescaled_gt_bboxes.append(bbox)
            gt_bboxes = rescaled_gt_bboxes

            max_steps = 5
            current_instance_step = 0
            previous_actions = sample["previous_actions"]
            while current_instance_step < max_steps:
                current_instance_step += 1

                if previous_actions:
                    previous_actions_text = "\n".join(previous_actions)

                if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
                    messages = [
                     {
                        "role": "system",
                        "content": "You are a helpful assistant. "
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": img_path, 
                            },
                            {"type": "text", "text": MM_MIND2WEB_PROMPT.format(instruction=instruction, previous_actions=previous_actions_text)},
                        ],
                    }
                ]
                elif args.model_path == "xlangai/OpenCUA-7B":
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img_path},
                                {"type": "text", "text": MM_MIND2WEB_PROMPT_OPENCUA.format(instruction=instruction, previous_actions=previous_actions_text)},
                            ],
                        },
                    ]
                else:
                    # Default to UI-TARS format for backward compatibility
                    messages = [
                         {
                            "role": "system",
                            "content": "You are a helpful assistant. "
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": img_path, 
                                },
                                {"type": "text", "text": MM_MIND2WEB_PROMPT.format(instruction=instruction, previous_actions=previous_actions_text)},
                            ],
                        }
                    ]
                
                # print("messages: ", messages[1]["content"][-1]["text"])
                # Preparation for inference
                if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    

                    image_inputs, video_inputs = process_vision_info(messages)
                    
                    ### HF
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(device)
                elif args.model_path == "xlangai/OpenCUA-7B":
                    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                    
                    image = Image.open(img_path).convert('RGB')
                    info = processor.preprocess(images=[image])
                    pixel_values = torch.tensor(info['pixel_values']).to(dtype=torch.bfloat16, device=model.device)
                    grid_thws = torch.tensor(info['image_grid_thw'])
                    input_ids = torch.tensor([input_ids]).to(model.device)
                else:
                    # Default to UI-TARS processing for backward compatibility
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    image_inputs, video_inputs = process_vision_info(messages)
                    
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(device)

                # Analyze vision tokens for KV cache methods that need it
                if args.kv_cache == "gui_kv" or args.kv_cache == "vl_cache":
                    if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
                        vision_analysis = analyze_vision_tokens_multi_images(processor, image_inputs, video_inputs, text, image_count=1)
                    elif args.model_path == "xlangai/OpenCUA-7B":
                        vision_analysis = analyze_vision_tokens_opencua_multi_images(tokenizer, input_ids, image_grid_thw=info["image_grid_thw"], merge_size=2, image_count=1)
                    else:
                        vision_analysis = analyze_vision_tokens_multi_images(processor, image_inputs, video_inputs, text, image_count=1)
            
                set_window_size(model, args)
                if args.kv_cache == "vl_cache":
                    last_vision_indices = []
                    vision_end_idx = vision_analysis.get('vision_end_idx', 0)
                    last_vision_indices.append(vision_end_idx)
                    set_last_vision_indices(model, last_vision_indices, args)
                elif args.kv_cache == "gui_kv":
                    if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
                        pixel_values = inputs.pixel_values
                        vision_hidden_states = None
                        image_grid_thw = inputs.image_grid_thw
                    elif args.model_path == "xlangai/OpenCUA-7B":
                        vision_hidden_states = None
                        image_grid_thw = grid_thws
                    else:
                        pixel_values = inputs.pixel_values
                        vision_hidden_states = None
                        image_grid_thw = inputs.image_grid_thw

                    set_vision_start_idx(model, vision_analysis['vision_start_idx'], args)
                    set_vision_end_idx(model, vision_analysis['vision_end_idx'], args)
                    set_alpha(model, args)
                    set_temperature(model, args)
                    
                try:
                    if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
                        outputs = model.generate(**inputs,
                                            max_new_tokens=args.max_new_tokens,
                                            pad_token_id=processor.tokenizer.eos_token_id,
                                            output_attentions=False,
                                            use_cache=True,
                                            do_sample=False,
                                            return_dict_in_generate=True)
                        
                        generated_ids = outputs if not hasattr(outputs, 'sequences') else outputs.sequences
                        
                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        hf_output_text = processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )[0]
                        output_text = hf_output_text
                    elif args.model_path == "xlangai/OpenCUA-7B":
                        output_text = ""
                        generated_ids = model.generate(
                                        input_ids, 
                                        pixel_values=pixel_values, 
                                        grid_thws=grid_thws,
                                        max_new_tokens=args.max_new_tokens,
                                        use_cache=True,
                                        do_sample=False,
                                        return_dict_in_generate=False
                                        )
                        
                        prompt_len = input_ids.shape[1]
                        generated_ids = generated_ids[:, prompt_len:]
                        output_text = tokenizer.batch_decode(
                            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )[0]
                    else:
                        raise NotImplementedError("Model not supported")
                except Exception as e:
                    print(f"Error in generation: {e}")
                    print("Using outputs from the previous iteration")
                    print(f"Image dimensions: {image.size}")

                sample_idx = mm_mind2web_data.index(sample)

                try:
                    if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
                        parsed_actions = parse_action_to_structure_output(output_text, \
                            origin_resized_height=img_height, \
                            origin_resized_width=img_width, \
                            max_pixels=MAX_PIXELS, \
                            min_pixels=MIN_PIXELS, \
                            factor=IMAGE_FACTOR, \
                            model_type="qwen25vl")[0]
                        predicted_operation = parsed_actions["action_type"]
                        
                        direction = ""
                        content = ""
                        click_point = ""
                        if predicted_operation == "scroll":
                            click_point = list(parsed_actions["action_inputs"].values())[0]
                            direction = list(parsed_actions["action_inputs"].values())[1]
                        elif predicted_operation == "click":
                            click_point = list(parsed_actions["action_inputs"].values())[0]
                        elif predicted_operation == "type":
                            content = list(parsed_actions["action_inputs"].values())[0]
                        elif predicted_operation == "select":
                            click_point = list(parsed_actions["action_inputs"].values())[0]
                            content = list(parsed_actions["action_inputs"].values())[1]
                        else:
                            raise NotImplementedError("Operation not supported")

                        try:
                            click_point = ast.literal_eval(click_point)
                        except:
                            click_point = ""
                            
                    elif args.model_path == "xlangai/OpenCUA-7B":
                        # Parse OpenCUA's pyautogui output format
                        parsed_actions = extract_actions(output_text,
                            origin_resized_height=img_height,
                            origin_resized_width=img_width,
                            max_pixels=MAX_PIXELS,
                            min_pixels=MIN_PIXELS,
                            factor=IMAGE_FACTOR,
                            model_type="qwen25vl")
                        print("parsed_actions: ", parsed_actions)
                        predicted_operation = ""
                        direction = ""
                        content = ""
                        click_point = ""

                        if parsed_actions and len(parsed_actions) > 0:
                            first_action = parsed_actions[0]
                            if first_action[0] in ["click", "doubleClick", "rightClick", "tripleClick", "moveTo", "dragTo", "triple_click"]:
                                predicted_operation = "click"
                                click_point = first_action[1]

                            if first_action[0] == "scroll":
                                predicted_operation = "scroll"
                                if isinstance(first_action[1], int):
                                    direction = "down" if first_action[1] < 0 else "up"
                                else:
                                    direction = first_action[1]

                            if first_action[0] == "write":
                                content = first_action[1]
                                predicted_operation = "type"

                            if first_action[0] == "press":
                                predicted_operation = "press"
                                content = first_action[1]
                            
                            elif first_action[0] == "terminate":
                                predicted_operation = "finished"
                                if any(word in first_action[1].lower() for word in ["unsuccessful", "infeasible", "impossible", "cannot", "can't", "unable", "fail", "error"]):
                                    content = "infeasible"
                                else:
                                    content = "successful"
                                    
                    else:
                        raise NotImplementedError("Model not supported")
                    
                    prediction_response = {
                        "operation": predicted_operation,
                        "click_point": click_point,
                        "direction": direction,
                        "content": content,
                    }
                    
                except Exception as e:
                    print(output_text)
                    
                    print(e)
                    
                    prediction_response = {
                        "operation": "",
                        "click_point": "",
                        "direction": "",
                        "content": "",
                    }


                if prediction_response["operation"] == "scroll":
                    previous_actions.append(f"Scroll {prediction_response['direction']}.")

                    if prediction_response["direction"] == "down":
                        next_block_index = current_block_index + 1
                        next_img_path = os.path.join(args.mm_mind2web_imgs, f"cross_{task}", annotation_action_id, f"{next_block_index}.png")
                        if os.path.exists(next_img_path):
                            current_block_index = next_block_index
                            img_path = next_img_path
                    elif prediction_response["direction"] == "up":
                        next_block_index = current_block_index - 1
                        if next_block_index >= 0:
                            next_img_path = os.path.join(args.mm_mind2web_imgs, f"cross_{task}", annotation_action_id, f"{next_block_index}.png")
                            if os.path.exists(next_img_path):
                                current_block_index = next_block_index
                                img_path = next_img_path
                    continue
                else:
                    break

            if prediction_response["click_point"]:
                correct, coords = is_output_inside_bbox(gt_bboxes, prediction_response["click_point"][:2], scale=1.0)
                all_element_acc.append([1 if correct else 0, annotation_id])
            else:
                correct = False
                all_element_acc.append([0, annotation_id])
            
            current_action = (sample["operation"], sample["value"])
            pred_action = f"{prediction_response['operation']} {prediction_response['content']}" 
            f1_score = calculate_f1(pred_action, current_action[0]+" "+current_action[1])
            all_operation_f1.append([f1_score, annotation_id])
            all_step_acc.append([1 if (all_operation_f1[-1][0]==1 and all_element_acc[-1][0]==1) else 0, annotation_id])        
            print("current_action", current_action)
            print("pred_action", pred_action)
            print("f1_score: ", f1_score)
            print("element correct: ", correct)
            
        
        total_steps = {sample['annotation_id']: sample['total_steps'] for sample in mm_mind2web_data}
        current_steps = collections.defaultdict(int)
        for _, annotation_id in all_element_acc:
            current_steps[annotation_id] += 1
        for annotation_id, steps in total_steps.items():
            while current_steps[annotation_id] < steps:
                all_element_acc.append([0, annotation_id])
                all_operation_f1.append([0, annotation_id])
                all_step_acc.append([0, annotation_id])
                current_steps[annotation_id] += 1
        
        macro_element_acc = collections.defaultdict(list)
        macro_operation_f1 = collections.defaultdict(list)
        macro_step_acc = collections.defaultdict(list)
        for x in all_element_acc:
            macro_element_acc[x[1]].append(x[0])
        for x in all_operation_f1:
            macro_operation_f1[x[1]].append(x[0])
        for x in all_step_acc:
            macro_step_acc[x[1]].append(x[0])
        
        error_ratio = collections.defaultdict(int)
        
        for annotation_id, x in macro_step_acc.items():
            
            error_count = len([y for y in x if y == 0])
            if error_count <= 3:
                error_ratio[error_count] += 1
            else:
                error_ratio[">3"] += 1
        
        
        total_tasks = len(macro_element_acc)
        error_ratio = {k: v/total_tasks for k, v in error_ratio.items()}
        macro_element_acc = np.mean([np.mean(x) for x in macro_element_acc.values()])
        macro_operation_f1 = np.mean([np.mean(x) for x in macro_operation_f1.values()])
        macro_step_acc = np.mean([np.mean(x) for x in macro_step_acc.values()])

        print("\n" + "=" * 80)
        print("MULTIMODAL MIND2WEB EVALUATION RESULTS")
        print("=" * 80)

        mind2web_metrics = {
            "element_accuracy": macro_element_acc,
            "operation_f1": macro_operation_f1,
            "step_accuracy": macro_step_acc,
            "error_ratio": error_ratio
        }


        print(json.dumps(mind2web_metrics, indent=2, ensure_ascii=False))

        task_result = {
            "task": task,
            "element_accuracy": macro_element_acc,
            "operation_f1": macro_operation_f1,
            "step_accuracy": macro_step_acc,
            "error_ratio": error_ratio,
            "total_tasks": total_tasks
        }
        all_task_results.append(task_result)
        tasks_result.append([macro_element_acc, macro_operation_f1, macro_step_acc])
        
        print("=" * 80)

    if args.results_dir and all_task_results:
        os.makedirs(args.results_dir, exist_ok=True)

        all_element_accs = [result["element_accuracy"] for result in all_task_results]
        all_operation_f1s = [result["operation_f1"] for result in all_task_results]
        all_step_accs = [result["step_accuracy"] for result in all_task_results]
        
        avg_element_acc = sum(all_element_accs) / len(all_element_accs) if all_element_accs else 0
        avg_operation_f1 = sum(all_operation_f1s) / len(all_operation_f1s) if all_operation_f1s else 0
        avg_step_acc = sum(all_step_accs) / len(all_step_accs) if all_step_accs else 0
        overall_total_tasks = sum([result["total_tasks"] for result in all_task_results])

        detailed_results = {
            "model_path": args.model_path,
            "tasks_evaluated": args.task,
            "kv_cache": args.kv_cache,
            "kv_cache_budget": args.kv_cache_budget,
            "attention_implementation": args.attention_implementation,
            "model_dtype": args.model_dtype,
            "max_new_tokens": args.max_new_tokens,
            "overall_metrics": {
                "element_accuracy": avg_element_acc,
                "operation_f1": avg_operation_f1,
                "step_accuracy": avg_step_acc
            },
            "total_tasks": overall_total_tasks,
            "task_breakdown": all_task_results
        }

        detailed_results_path = os.path.join(args.results_dir, 'mind2web_detailed_results.json')
        with open(detailed_results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved Mind2Web detailed results to {detailed_results_path}")

        summary_results = {
            "model_path": args.model_path,
            "tasks_evaluated": args.task,
            "total_tasks": overall_total_tasks,
            "overall_element_accuracy": avg_element_acc,
            "overall_operation_f1": avg_operation_f1,
            "overall_step_accuracy": avg_step_acc,
            "kv_cache": args.kv_cache,
            "kv_cache_budget": args.kv_cache_budget,
            "attention_implementation": args.attention_implementation,
            "model_dtype": args.model_dtype,
            "max_new_tokens": args.max_new_tokens,
            "task_breakdown": []
        }

        for result in all_task_results:
            summary_results["task_breakdown"].append({
                "task": result["task"],
                "element_accuracy": result["element_accuracy"],
                "operation_f1": result["operation_f1"],
                "step_accuracy": result["step_accuracy"],
                "total_tasks": result["total_tasks"]
            })

        summary_results_path = os.path.join(args.results_dir, 'mind2web_summary_results.json')
        with open(summary_results_path, 'w') as f:
            json.dump(summary_results, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved Mind2Web summary results to {summary_results_path}")

        print("\n" + "=" * 80)
        print("FINAL MIND2WEB EVALUATION RESULTS")
        print("=" * 80)

        if len(all_task_results) > 0:
            print(f"\nResults by Task:")
            print(f"{'Task':<15} {'Element Acc':<12} {'Operation F1':<12} {'Step Acc':<12} {'Total Tasks':<12}")
            print("-" * 67)
            for result in all_task_results:
                print(f"{result['task']:<15} {result['element_accuracy']*100:<12.2f}% {result['operation_f1']*100:<12.2f}% {result['step_accuracy']*100:<12.2f}% {result['total_tasks']:<12}")
            print("-" * 67)

        print(f"\nOverall Summary:")
        print(f"{'Metric':<25} {'Accuracy':<15}")
        print("-" * 40)
        print(f"{'Element Accuracy':<25} {avg_element_acc*100:<15.2f}%")
        print(f"{'Operation F1':<25} {avg_operation_f1*100:<15.2f}%")
        print(f"{'Step Accuracy':<25} {avg_step_acc*100:<15.2f}%")
        print(f"{'Total Tasks':<25} {overall_total_tasks:<15}")
        print("-" * 40)
        
        print("=" * 80)