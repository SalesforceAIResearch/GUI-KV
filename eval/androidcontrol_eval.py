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
    set_temperature,
)



logging.basicConfig(level=logging.INFO)
torch.manual_seed(1234)


ANDROIDCONTROL_PROMPT_HIGH = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Thought: ...
Action: ...
```
## Action Space

click(point='<point>x1 y1</point>')
long_press(point='<point>x1 y1</point>')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
press_back()
wait() # Wait for the screen to finish loading.
finished(content='successful|infeasible') # Use `infeasible` if you think the task is not feasible (including cases like you don't have enough information or cannot perform some necessary actions); otherwise, use `successful`.


## Note
- Use English in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.
- Always use `click` if you want to open an app.
- content in `finished` action should only be `successful` or `infeasible`.

## Previous Actions
{previous_actions}

## User Instruction
{goal}

"""

ANDROIDCONTROL_PROMPT_LOW = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Thought: ...
Action: ...
```
## Action Space

click(point='<point>x1 y1</point>')
long_press(point='<point>x1 y1</point>')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
press_back()
wait() # Wait for the screen to finish loading.
finished(content='successful|infeasible') # Use `infeasible` if you think the task is not feasible (including cases like you don't have enough information or cannot perform some necessary actions); otherwise, use `successful`.


## Note
- Use English in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.
- Always use `click` if you want to open an app.
- You must follow User Instruction below.

## User Goal:
{goal}

## Previous Actions
{previous_actions}

## User Instruction
{task}
"""

ANDROIDCONTROL_PROMPT_HIGH_OPENCUA = """
You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.\n\nFor each step, provide your response in this format:\n\nThought:\n  - Step by Step Progress Assessment:\n    - Analyze completed task parts and their contribution to the overall goal\n    - Reflect on potential errors, unexpected results, or obstacles\n    - If previous action was incorrect, predict a logical recovery step\n  - Next Action Analysis:\n    - List possible next actions based on current state\n    - Evaluate options considering current state and previous actions\n    - Propose most logical next action\n    - Anticipate consequences of the proposed action\n  - For Text Input Actions:\n    - Note current cursor position\n    - Consolidate repetitive actions (specify count for multiple keypresses)\n    - Describe expected final text outcome\n    - Use first-person perspective in reasoning\n\nAction:\n  Provide clear, concise, and actionable instructions:\n  - If the action involves interacting with a specific target:\n    - Describe target explicitly without using coordinates\n    - Specify element names when possible (use original language if non-English)\n    - Describe features (shape, color, position) if name unavailable\n    - For window control buttons, identify correctly (minimize "—", maximize "□", close "X")\n  - if the action involves keyboard actions like \'press\', \'write\', \'hotkey\':\n    - Consolidate repetitive keypresses with count\n    - Specify expected text outcome for typing actions\n\nFinally, output the action as PyAutoGUI code or the following functions:\n- {{"name": "computer.triple_click", "description": "Triple click on the screen", "parameters": {{"type": "object", "properties": {{"x": {{"type": "number", "description": "The x coordinate of the triple click"}}, "y": {{"type": "number", "description": "The y coordinate of the triple click"}}}}, "required": ["x", "y"]}}}}\n- {{"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {{"type": "object", "properties": {{"status": {{"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}}}}, "required": ["status"]}}}}'

## Previous Actions
{previous_actions}

## User Instruction
{goal}
"""

ANDROIDCONTROL_PROMPT_LOW_OPENCUA = """
You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.\n\nFor each step, provide your response in this format:\n\nThought:\n  - Step by Step Progress Assessment:\n    - Analyze completed task parts and their contribution to the overall goal\n    - Reflect on potential errors, unexpected results, or obstacles\n    - If previous action was incorrect, predict a logical recovery step\n  - Next Action Analysis:\n    - List possible next actions based on current state\n    - Evaluate options considering current state and previous actions\n    - Propose most logical next action\n    - Anticipate consequences of the proposed action\n  - For Text Input Actions:\n    - Note current cursor position\n    - Consolidate repetitive actions (specify count for multiple keypresses)\n    - Describe expected final text outcome\n    - Use first-person perspective in reasoning\n\nAction:\n  Provide clear, concise, and actionable instructions:\n  - If the action involves interacting with a specific target:\n    - Describe target explicitly without using coordinates\n    - Specify element names when possible (use original language if non-English)\n    - Describe features (shape, color, position) if name unavailable\n    - For window control buttons, identify correctly (minimize "—", maximize "□", close "X")\n  - if the action involves keyboard actions like \'press\', \'write\', \'hotkey\':\n    - Consolidate repetitive keypresses with count\n    - Specify expected text outcome for typing actions\n\nFinally, output the action as PyAutoGUI code or the following functions:\n- {{"name": "computer.triple_click", "description": "Triple click on the screen", "parameters": {{"type": "object", "properties": {{"x": {{"type": "number", "description": "The x coordinate of the triple click"}}, "y": {{"type": "number", "description": "The y coordinate of the triple click"}}}}, "required": ["x", "y"]}}}}\n- {{"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {{"type": "object", "properties": {{"status": {{"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}}}}, "required": ["status"]}}}}'


## Note
- You must follow User Instruction below.

## User Goal
{goal}

## Previous Actions
{previous_actions}

## User Instruction
{task}
"""

def bounding_box_contains_point(bbox, x, y):
    return bbox['x_min'] <= x <= bbox['x_max'] and bbox['y_min'] <= y <= bbox['y_max']

def find_smallest_bbox_node(x, y, tree):
    """
    Find the smallest bounding box node that contains the given coordinates
    Returns a tuple of (node, bbox) if found, (None, None) if not found
    """
    smallest_node = None
    smallest_bbox = None
    smallest_area = float('inf')
    
    for node in tree:
        if isinstance(node, dict):
            bbox = node['bbox_pixels']
            if bounding_box_contains_point(bbox, x, y):
                area = (bbox['x_max'] - bbox['x_min']) * (bbox['y_max'] - bbox['y_min'])
                if area < smallest_area:
                    smallest_area = area
                    smallest_node = node
                    smallest_bbox = bbox
        elif isinstance(node, list):
            child_node, child_bbox = find_smallest_bbox_node(x, y, node)
            if child_node:
                child_area = (child_bbox['x_max'] - child_bbox['x_min']) * (child_bbox['y_max'] - child_bbox['y_min'])
                if child_area < smallest_area:
                    smallest_area = child_area
                    smallest_node = child_node
                    smallest_bbox = child_bbox
    
    return smallest_node, smallest_bbox

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--androidcontrol_imgs', type=str, required=True)
    parser.add_argument('--androidcontrol_test', type=str,  required=True)
    parser.add_argument('--task', type=str, required=True, choices=["all"])
    parser.add_argument('--debug', default=None, type=int)
    parser.add_argument('--max_new_tokens', type=int, default=600)
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/mps/cpu). If not specified, will auto-detect best available device.')
    parser.add_argument('--model_dtype', type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"], help='Data type to use (auto/bfloat16/float16/float32).')
    parser.add_argument('--attention_implementation', type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"], help='Attention implementation to use (eager/flash_attention_2).')
    parser.add_argument('--kv_cache', type=str, default="original", choices=["original", "pyramid_kv", "vl_cache", "snap_kv", "gui_kv"], help='KV cache method to use (original/pyramid_kv/vl_cache/snap_kv/gui_kv).')
    parser.add_argument('--kv_cache_budget', type=float, default=100, help='KV cache budget in tokens.')
    parser.add_argument('--alpha', type=float, default=None, help='Alpha for GUIKV.')
    parser.add_argument('--window_size', type=int, default=None, help='Window size for GUIKV.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for token information scores.')
    parser.add_argument('--results_dir', type=str, help='Directory to store evaluation results in JSON format.')
    parser.add_argument('--instruction_level', type=str, default="high", choices=["high", "low"], help='Instruction level to use (high/low).')
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
        raise NotImplementedError(f"Model {args.model_path} not implemented")
        
    
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

    step_correctness = []
    dataset = "500_steps.json"
    
    with open(os.path.join(args.androidcontrol_test, dataset), 'r') as f:
        androidcontrol_data = json.load(f)

    if args.debug is not None:
        # Limit to 100 examples for quick evaluation
        androidcontrol_data = androidcontrol_data[:args.debug]
        print("Num of sample: " + str(len(androidcontrol_data)) + " (limited to 100 for quick evaluation)")
    else:
        print("Num of sample: " + str(len(androidcontrol_data)))

    for j, sample in tqdm(enumerate(androidcontrol_data), desc=f"Processing data", total=len(androidcontrol_data)):
        img_path = os.path.join(args.androidcontrol_imgs, sample['screenshot'])
        
        if not os.path.exists(img_path):
            print("img not found: ", img_path)
            continue
        
        image = Image.open(img_path)
        img_size = image.size
        img_width, img_height = img_size
        goal = sample["goal"]
        low_level_task = sample["step_instruction"]
        previous_actions = sample.get("previous_actions", [])
        previous_actions_text = "\n".join(previous_actions)

        if args.instruction_level == "high":
            if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
                user_prompt = ANDROIDCONTROL_PROMPT_HIGH.format(goal=goal, previous_actions=previous_actions_text)
            elif args.model_path == "xlangai/OpenCUA-7B":
                user_prompt = ANDROIDCONTROL_PROMPT_HIGH_OPENCUA.format(goal=goal, previous_actions=previous_actions_text)
            else:
                raise ValueError(f"Invalid model path: {args.model_path}")
        elif args.instruction_level == "low":
            if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
                user_prompt = ANDROIDCONTROL_PROMPT_LOW.format(goal=goal, task=low_level_task, previous_actions=previous_actions_text)
            elif args.model_path == "xlangai/OpenCUA-7B":
                user_prompt = ANDROIDCONTROL_PROMPT_LOW_OPENCUA.format(goal=goal, task=low_level_task, previous_actions=previous_actions_text)
            else:
                raise ValueError(f"Invalid model path: {args.model_path}")
        else:
            raise ValueError(f"Invalid instruction level: {args.instruction_level}")

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
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ]
        elif args.model_path == "xlangai/OpenCUA-7B":
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]
        else:
            raise NotImplementedError(f"Model {args.model_path} not implemented")

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
            raise NotImplementedError(f"Model {args.model_path} not implemented")

        # Analyze vision tokens for KV cache methods that require it
        if args.kv_cache == "gui_kv" or args.kv_cache == "vl_cache":
            if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
                vision_analysis = analyze_vision_tokens_multi_images(processor, image_inputs, video_inputs, text)
            elif args.model_path == "xlangai/OpenCUA-7B":
                vision_analysis = analyze_vision_tokens_opencua_multi_images(tokenizer, input_ids, image_grid_thw=info["image_grid_thw"], merge_size=2)
            else:
                raise NotImplementedError(f"Model {args.model_path} not implemented")

        set_window_size(model, args)
        if args.kv_cache == "vl_cache":
            
            last_vision_indices = []
            vision_end_idx = vision_analysis.get('vision_end_idx', 0)
            last_vision_indices.append(vision_end_idx)
            set_last_vision_indices(model, last_vision_indices, args)
        elif args.kv_cache == "gui_kv":
            if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
                # Get pixel values from the processor inputs
                pixel_values = inputs.pixel_values  # Shape: [batch, channels, height, width]
                
                # Use the model's get_image_features method to get vision encoder outputs
                vision_hidden_states = None
                image_grid_thw = inputs.image_grid_thw
            elif args.model_path == "xlangai/OpenCUA-7B":
                # For OpenCUA, pixel_values and grid_thws are already prepared
                vision_hidden_states = None
                image_grid_thw = grid_thws
            else:
                # Default to UI-TARS approach
                pixel_values = inputs.pixel_values
                vision_hidden_states = None
                image_grid_thw = inputs.image_grid_thw

            set_vision_start_idx(model, vision_analysis['vision_start_idx'], args)
            set_vision_end_idx(model, vision_analysis['vision_end_idx'], args)
            set_alpha(model, args)
            set_temperature(model, args)

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
            print("output_text: ", output_text)
        elif args.model_path == "xlangai/OpenCUA-7B":
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
            print("output_text: ", output_text)
        else:
            # Default to UI-TARS generation
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
            print("output_text: ", output_text)

        sample_idx = androidcontrol_data.index(sample)

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
                action_inputs = parsed_actions.get("action_inputs", {})
                
                if predicted_operation == "scroll":
                    click_point = list(parsed_actions["action_inputs"].values())[0]
                    direction = action_inputs.get("direction", "")
                elif predicted_operation in ["click", "long_press"]:
                    click_point = list(parsed_actions["action_inputs"].values())[0]
                elif predicted_operation == "type":
                    content = action_inputs.get("content", "")
                elif predicted_operation == "finished":
                    content = action_inputs.get("content", "")
                    
                    # oftentimes, finished content is open-ended and unconstrained. We need to post-process this
                    if content not in ["successful", "infeasible"]:
                        # parse any sub-string indicating unsuccessful or infeasible in content
                        if any(word in content.lower() for word in ["unsuccessful", "infeasible", "impossible", "cannot", "can't", "unable", "fail", "error"]):
                            content = "infeasible"
                        else:
                            content = "successful"
                elif predicted_operation in ["press_back", "wait"]:
                    pass
                else:
                    print(f"Operation {predicted_operation} not supported")

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
                # Default values
                predicted_operation = ""
                direction = ""
                content = ""
                click_point = ""
                
                # Map OpenCUA actions to androidcontrol format
                if parsed_actions and len(parsed_actions) > 0:
                    first_action = parsed_actions[0]
                    # OpenCUA returns tuples: (action_type, coordinate)
                    if first_action[0] in ["click", "doubleClick", "rightClick", "tripleClick", "moveTo", "dragTo", "triple_click"]:
                        predicted_operation = "click"
                        click_point = first_action[1]
                    
                    if first_action[0] == "scroll":
                        predicted_operation = "scroll"
                        direction = first_action[1]
                    # Parse text content from OpenCUA output for write/type actions
                    if first_action[0] == "write":
                        # Extract content from pyautogui.write() command
                        
                        content = first_action[1]
                        predicted_operation = "type"
                            
                    # Handle special actions
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
            
        
        
        # Compute step accuracy score
        correct_step = 0
        
        # Get ground truth action
        gt_action = sample.get('action', {})
        gt_action_type = gt_action.get('action_type', '')
        
        # Get accessibility tree
        accessibility_tree = sample.get('accessibility_tree', [])
        predicted_operation = prediction_response["operation"]
        # Handle grounding actions (click, long_press, type_text)
        if predicted_operation in ['click', 'long_press', 'type'] and gt_action_type in ['click', 'long_press', 'type_text']:
            # Map 'type' to 'type_text' for comparison
            pred_action_type = 'type_text' if predicted_operation == 'type' else predicted_operation
            
            if pred_action_type == gt_action_type:
                # Find the ground truth bounding box
                if 'x' in gt_action and 'y' in gt_action and accessibility_tree:
                    node, bbox = find_smallest_bbox_node(gt_action['x'], gt_action['y'], accessibility_tree)
                    
                    # Check if predicted point falls within bbox
                    if bbox and click_point and len(click_point) >= 2:
                        # Convert normalized coordinates to pixel coordinates
                        pred_x = click_point[0] * img_width
                        pred_y = click_point[1] * img_height
                        
                        if bounding_box_contains_point(bbox, pred_x, pred_y):
                            # For type_text, also check text matches
                            if pred_action_type == 'type_text':
                                if gt_action.get('text', '') == content:
                                    correct_step = 1
                            else:
                                correct_step = 1
                        else:
                            # Print which coordinate is out of bounds
                            x_in_bounds = bbox["x_min"] <= pred_x <= bbox["x_max"]
                            y_in_bounds = bbox["y_min"] <= pred_y <= bbox["y_max"]
                            
                            if not x_in_bounds and not y_in_bounds:
                                print(f"Both x and y are out of bbox: pred_x={pred_x} (bbox x range: {bbox['x_min']}-{bbox['x_max']}), pred_y={pred_y} (bbox y range: {bbox['y_min']}-{bbox['y_max']})")
                            elif not x_in_bounds:
                                print(f"x is out of bbox: pred_x={pred_x} (bbox x range: {bbox['x_min']}-{bbox['x_max']}), pred_y={pred_y} is within bounds")
                            elif not y_in_bounds:
                                print(f"y is out of bbox: pred_y={pred_y} (bbox y range: {bbox['y_min']}-{bbox['y_max']}), pred_x={pred_x} is within bounds")
        
        # Handle equivalent action: click vs open_app
        elif (predicted_operation == 'click' and gt_action_type == 'open_app') or \
             (predicted_operation == 'open_app' and gt_action_type == 'click'):
            if predicted_operation == 'click' and click_point and len(click_point) >= 2 and accessibility_tree:
                # Convert normalized coordinates to pixel coordinates
                pred_x = click_point[0] * img_width
                pred_y = click_point[1] * img_height
                
                element, _ = find_smallest_bbox_node(pred_x, pred_y, accessibility_tree)
                if element:
                    text = (element.get('text') or "").lower()
                    content_desc = (element.get("content_description") or "").lower()
                    app_name = (gt_action.get("app_name") or "").lower()
                    print("app_name: ", app_name, "text: ", text, "content_desc: ", content_desc)
                    if app_name and ((text and app_name in text) or (content_desc and app_name in content_desc)):
                        correct_step = 1
        
        # Handle equivalent action: click vs navigate_back
        elif (predicted_operation == 'click' and gt_action_type == 'navigate_back') or \
             (predicted_operation == 'press_back' and gt_action_type == 'navigate_back'):
            if predicted_operation == 'click' and click_point and len(click_point) >= 2 and accessibility_tree:
                # Convert normalized coordinates to pixel coordinates
                pred_x = click_point[0] * img_width
                pred_y = click_point[1] * img_height
                
                element, _ = find_smallest_bbox_node(pred_x, pred_y, accessibility_tree)
                if element:
                    text = (element.get('text') or "").lower()
                    content_desc = (element.get("content_description") or "").lower()
                    
                    if (text and "back" in text) or (content_desc and "back" in content_desc):
                        correct_step = 1
            elif predicted_operation == 'press_back':
                correct_step = 1
        
        # Handle other actions (exact matching)
        else:
            # Map operation names to match ground truth format
            operation_mapping = {
                'press_back': 'navigate_back',
                'wait': 'wait',
                'finished': 'status',
                'scroll': 'scroll'
            }
            
            mapped_operation = operation_mapping.get(predicted_operation, predicted_operation)
            print("mapped_operation: ", mapped_operation, "gt_action_type: ", gt_action_type)
            if mapped_operation == gt_action_type:
                # For scroll, check direction
                if mapped_operation == 'scroll':
                    if gt_action.get('direction', '') == direction:
                        correct_step = 1
                # For status (mapped from finished), check content
                elif mapped_operation == 'status':
                    if gt_action.get('goal_status', '') == content:
                        correct_step = 1
                # For wait and navigate_back
                else:
                    correct_step = 1
        print("prediction_response: ", prediction_response)
        print("gt_action_type: ", gt_action_type, "gt_action", gt_action)
        print("correct_step: ", correct_step)
        step_correctness.append(correct_step)
        
    step_accuracy = np.mean(step_correctness) if step_correctness else 0.0
    
    # Count grounding steps (click, long_press, type_text actions)
    grounding_steps = 0
    correct_grounding_steps = 0
    
    for i, sample in enumerate(androidcontrol_data[:len(step_correctness)]):
        gt_action = sample.get('action', {})
        gt_action_type = gt_action.get('action_type', '')
        
        if gt_action_type in ['click', 'long_press', 'type_text']:
            grounding_steps += 1
            if step_correctness[i] == 1:
                correct_grounding_steps += 1
    
    grounding_accuracy = correct_grounding_steps / grounding_steps if grounding_steps > 0 else 0.0
    
    # Print AndroidControl evaluation results
    print("\n" + "=" * 80)
    print("ANDROIDCONTROL EVALUATION RESULTS")
    print("=" * 80)
    
    # Create detailed results dictionary
    androidcontrol_metrics = {
        "step_accuracy": step_accuracy,
        "correct_steps": sum(step_correctness),
        "total_steps": len(step_correctness),
        "grounding_accuracy": grounding_accuracy,
        "correct_grounding_steps": correct_grounding_steps,
        "grounding_steps": grounding_steps
    }
    
    print(f"Step Accuracy: {step_accuracy:.2%} ({sum(step_correctness)}/{len(step_correctness)})")
    print(f"Grounding Accuracy: {grounding_accuracy:.2%} ({correct_grounding_steps}/{grounding_steps})")
    
    print("=" * 80)

    # Save results if requested
    if args.results_dir:
        # Create results directory if it doesn't exist
        os.makedirs(args.results_dir, exist_ok=True)
        
        # Create comprehensive results
        detailed_results = {
            "model_path": args.model_path,
            "dataset": dataset,
            "kv_cache": args.kv_cache,
            "kv_cache_budget": args.kv_cache_budget,
            "attention_implementation": args.attention_implementation,
            "model_dtype": args.model_dtype,
            "max_new_tokens": args.max_new_tokens,
            "metrics": androidcontrol_metrics,
            "debug": args.debug is not None,
            "num_samples_evaluated": len(step_correctness)
        }
        
        # Save results to JSON
        results_filename = f'androidcontrol_results_budget{args.kv_cache_budget}.json'
        results_path = os.path.join(args.results_dir, results_filename)
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved AndroidControl results to {results_path}")