import torch
import json
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
    parse_response_actions_opencua,
    extract_actions
)

from ui_tars_utils import (
    parse_action_to_structure_output, MIN_PIXELS, MAX_PIXELS, IMAGE_FACTOR,
    analyze_vision_tokens_multi_images,
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


ACTION_HISTORY_TEMPLATE = "## Action:\n{action}\n"
STEP_TEMPLATE = "# Step {step_num}:\n"
RESPONSE_TEMPLATE = "## Observation:\n{observation}\n\n## Thought:\n{thought}\n\n## Action:\n{action}\n\n## Code:\n{code}\n"
HISTORY_TEMPLATE = "## Observation:\n{observation}\n\n## Thought:\n{thought}\n\n## Action:\n{action}\n"

AGENTNETBENCH_PROMPT_UI_TARS = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags
After the <think> tags, you should place final answer, which concludes your summarized thought and your action.

For example,
```
<think>detailed reasoning content here</think>
Thought: a small plan and finally summarize your next action (with its target element) in one sentence
Action: ...
```

## Action Space

click(point='<point>x1 y1</point>')
left_double(point='<point>x1 y1</point>')
right_single(point='<point>x1 y1</point>')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
hotkey(key='ctrl c') # Split keys with a space and use lowercase. Also, do not use more than 3 keys in one hotkey action.
type(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content. 
scroll(point='<point>x1 y1</point>', direction='down or up or right or left') # Show more information on the `direction` side.
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='successful|failure') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.

## Output Example
Thought: Let's click ...
Action: click(point='<point>100 200</point>')

## Note
- Use English in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.
- If you have executed several same actions (like repeatedly clicking the same point) but the screen keeps no change, please try to execute a modified action when necessary.
- finished content should be either`successful` or `failure`.

## User Instruction
{instruction}

Please generate the next move according to the screenshot and task instruction.
"""

AGENTNETBENCH_PROMPT_OPENCUA = """You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.

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

# Task Instruction:
{instruction}

Please generate the next move according to the screenshot and task instruction.
"""

def evaluate_action(pred_action, gt_action, alternative_options=None):
    """
    Evaluate a predicted action against ground truth and alternative options.
    Returns a score between 0 and 1.
    """
    
    # Check if predicted action type matches ground truth
    pred_type = pred_action[0].lower() if pred_action else ""
    gt_type = gt_action['type'].lower()
    
    # Special case for triple_click in predictions
    if pred_type == 'triple_click':
        pred_type = 'tripleclick'
    
    # If action types don't match, check alternatives
    if pred_type != gt_type:
        # Check if predicted action matches any alternative
        if alternative_options:
            for alt_actions in alternative_options:
                if alt_actions and len(alt_actions) > 0:
                    alt_type = alt_actions[0]['type'].lower()
                    if pred_type == alt_type:
                        # Use alternative as ground truth
                        return evaluate_action(pred_action, alt_actions[0], None)
        return 0.0
    
    # Action types match, evaluate based on action type
    score = 0.0
    
    if pred_type in ['click', 'doubleclick', 'rightclick', 'tripleclick', 'moveto', 'dragto']:
        # Position-based actions
        if len(pred_action) > 1 and isinstance(pred_action[1], tuple) and len(pred_action[1]) == 2:
            pred_x, pred_y = pred_action[1]
            gt_pos = gt_action['params'].get('position', {})
            gt_x = gt_pos.get('x', 0)
            gt_y = gt_pos.get('y', 0)
            
            # Check if prediction falls within any bounding box
            if 'metadata' in gt_action and 'bboxes' in gt_action['metadata']:
                for bbox_info in gt_action['metadata']['bboxes']:
                    bbox = bbox_info['rel_bbox']
                    # bbox format: [x, y, width, height]
                    if (bbox[0] <= pred_x <= bbox[0] + bbox[2] and
                        bbox[1] <= pred_y <= bbox[1] + bbox[3]):
                        score = 1.0
                        break
            
            # If not in any bbox, calculate distance-based score
            if score == 0.0:
                distance = ((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2) ** 0.5
                # Use threshold of 0.01 * sqrt(2) ≈ 0.0141
                threshold = 0.01 * (2 ** 0.5)
                if distance <= threshold:
                    score = 1.0
                else:
                    # Exponential decay for distances beyond threshold
                    score = np.exp(-120 * (distance - threshold))
    
    elif pred_type == 'write':
        # Text-based actions
        if len(pred_action) > 1:
            pred_text = str(pred_action[1]).lower().strip()
            gt_text = gt_action['params'].get('text', gt_action['params'].get('content', '')).lower().strip()
            
            # Check for trailing newline differences
            pred_has_newline = pred_text.endswith('\n')
            gt_has_newline = gt_text.endswith('\n')
            pred_text = pred_text.rstrip('\n')
            gt_text = gt_text.rstrip('\n')
            
            if pred_text == gt_text:
                # Penalize slightly if newline presence differs
                score = 0.9 if pred_has_newline != gt_has_newline else 1.0
            else:
                # Calculate text similarity using edit distance
                try:
                    import editdistance
                    max_len = max(len(pred_text), len(gt_text))
                    if max_len == 0:
                        score = 1.0
                    else:
                        edit_dist = editdistance.eval(pred_text, gt_text)
                        similarity = 1.0 - (edit_dist / max_len)
                        # Apply threshold
                        if similarity >= 0.8:
                            score = 1.0
                        else:
                            score = similarity / 0.8
                except ImportError:
                    # Fallback to exact match
                    score = 1.0 if pred_text == gt_text else 0.0
    
    elif pred_type in ['press', 'hotkey']:
        # Key-based actions
        if len(pred_action) > 1:
            pred_keys = pred_action[1]
            gt_keys = gt_action['params'].get('keys', [])
            
            if isinstance(pred_keys, list) and isinstance(gt_keys, list):
                # Normalize keys to lowercase
                pred_keys_norm = [k.lower() for k in pred_keys]
                gt_keys_norm = [k.lower() for k in gt_keys]
                
                # Check if key sets match (ignoring order)
                if set(pred_keys_norm) == set(gt_keys_norm):
                    score = 1.0
    
    elif pred_type == 'scroll':
        # For scroll actions, just check that action type matches
        score = 1.0
    
    elif pred_type == 'terminate':
        # Check status matches
        if len(pred_action) > 1:
            pred_status = pred_action[1]
            gt_status = gt_action['params'].get('status', '')
            score = 1.0 if pred_status == gt_status else 0.0
    
    return score

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


def convert_ui_tars_to_agentnetbench_actions(parsed_actions):
    """Convert UI-TARS parsed actions to AgentNetBench format."""
    if not parsed_actions:
        return []

    if isinstance(parsed_actions, list) and len(parsed_actions) > 0:
        action_data = parsed_actions[0]
    else:
        return []
    
    action_type = action_data.get('action_type', '')
    action_inputs = action_data.get('action_inputs', {})

    def normalize_to_absolute(point_str):
        """Convert normalized coordinates string to absolute pixel coordinates."""
        if isinstance(point_str, str):
            try:
                point_str = point_str.strip('[]')
                coords = [float(x.strip()) for x in point_str.split(',')]
                if len(coords) >= 2:
                    x_abs = coords[0]
                    y_abs = coords[1]
                    return (x_abs, y_abs)
            except:
                pass
        elif isinstance(point_str, list) and len(point_str) >= 2:
            x_abs = point_str[0]
            y_abs = point_str[1]
            return (x_abs, y_abs)
        return None

    if action_type == 'click':
        point = action_inputs.get('point', action_inputs.get('start_box', []))
        coords = normalize_to_absolute(point)
        if coords:
            return [('click', coords)]
    
    elif action_type == 'left_double':
        point = action_inputs.get('point', action_inputs.get('start_box', []))
        coords = normalize_to_absolute(point)
        if coords:
            return [('doubleclick', coords)]
    
    elif action_type == 'right_single':
        point = action_inputs.get('point', action_inputs.get('start_box', []))
        coords = normalize_to_absolute(point)
        if coords:
            return [('rightclick', coords)]
    
    elif action_type == 'drag':
        start_point = action_inputs.get('start_box', action_inputs.get('start_point', []))
        end_point = action_inputs.get('end_box', action_inputs.get('end_point', []))
        start_coords = normalize_to_absolute(start_point)
        end_coords = normalize_to_absolute(end_point)
        if start_coords and end_coords:
            return [('dragto', end_coords)]

    elif action_type == 'hotkey':
        keys = action_inputs.get('key', action_inputs.get('keys', ''))
        if keys:
            if isinstance(keys, str):
                key_list = keys.lower().split()
                key_list = [k.replace('ctrl', 'control') for k in key_list]
                return [('hotkey', key_list)]
            elif isinstance(keys, list):
                return [('hotkey', keys)]

    elif action_type == 'long_press':
        point = action_inputs.get('point', action_inputs.get('start_box', []))
        coords = normalize_to_absolute(point)
        if coords:
            return [('click', coords)]

    elif action_type == 'type':
        content = action_inputs.get('content', '')
        if content:
            if content.endswith('\n'):
                return [('write', content[:-1]), ('press', ['enter'])]
            else:
                return [('write', content)]

    elif action_type == 'scroll':
        point = action_inputs.get('point', action_inputs.get('start_box', []))
        direction = action_inputs.get('direction', '')
        coords = normalize_to_absolute(point)
        if coords and direction:
            amount = 5 if direction in ['up', 'right'] else -5
            return [('scroll', amount)]

    elif action_type == 'press_back':
        return [('press', ['escape'])]

    elif action_type == 'wait':
        return []
    
    elif action_type == 'finished':
        content = action_inputs.get('content', '')
        if content and ('success' in content.lower() or 'successful' in content.lower()):
            return [('terminate', 'success')]
        else:
            return [('terminate', 'failure')]

    return []


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--agentnetbench_data', type=str, required=True, help='Path to AgentNetBench test data directory')
    parser.add_argument('--agentnetbench_imgs', type=str, required=True, help='Path to AgentNetBench images directory')
    parser.add_argument('--task', type=str, default="all", choices=["all"])
    parser.add_argument('--debug', default=None, type=int)
    parser.add_argument('--max_new_tokens', type=int, default=1000)
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/mps/cpu). If not specified, will auto-detect best available device.')
    parser.add_argument('--model_dtype', type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"], help='Data type to use (auto/bfloat16/float16/float32).')
    parser.add_argument('--attention_implementation', type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"], help='Attention implementation to use (eager/flash_attention_2).')
    parser.add_argument('--kv_cache', type=str, default="original", choices=["original", "pyramid_kv", "vl_cache", "snap_kv", "gui_kv"], help='KV cache method to use (original/pyramid_kv/vl_cache/snap_kv/gui_kv).')
    parser.add_argument('--kv_cache_budget', type=float, default=100, help='KV cache budget in tokens.')
    parser.add_argument('--alpha', type=float, default=None, help='Alpha for GUIKV.')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta for downweighting redundant tokens in GUIKV.')
    parser.add_argument('--window_size', type=int, default=None, help='Window size for GUIKV.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for token information scores.')
    parser.add_argument('--results_dir', type=str, help='Directory to store evaluation results in JSON format.')
    parser.add_argument('--image_slots', type=int, default=5, help='Number of previous images to include in context')
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

    # use Qwen-VL-Chat
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
        replace_qwen2_5_vl(kv_cache_mode=args.kv_cache)

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
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=model_dtype, device_map="cpu",
                attn_implementation=args.attention_implementation,
            )
        set_attention_implementation(model, args)
        set_kv_cache_budget(model, args)
        if args.attention_implementation == "eager":
            set_move_attention_to_cpu(model, args)
            configure_accelerate_skip_attention(model)
    else:
        if torch.cuda.device_count() > 1:
            device_map = "auto"
        else:
            device_map = {"": "cuda:0"}
        
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
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=model_dtype, device_map=device_map,
                attn_implementation=args.attention_implementation,
            )
        set_attention_implementation(model, args)
        set_kv_cache_budget(model, args)
        if args.attention_implementation == "eager":
            set_move_attention_to_cpu(model, args)
            configure_accelerate_skip_attention(model)

    print("Load Success")

    trajectory_files = [f for f in os.listdir(args.agentnetbench_data) if f.endswith('.json') and not f.startswith('mapping')]

    if args.debug is not None:
        trajectory_files = trajectory_files[:args.debug]
        print(f"Debug mode: Limited to {len(trajectory_files)} trajectories")
    else:
        print(f"Total trajectories: {len(trajectory_files)}")
    
    all_results = []
    trajectory_scores = []
    trajectory_histories = {}
    action_type_scores = {
        'click': [],
        'doubleclick': [],
        'rightclick': [],
        'tripleclick': [],
        'moveto': [],
        'dragto': [],
        'write': [],
        'press': [],
        'hotkey': [],
        'scroll': [],
        'terminate': []
    }
    milestone_scores = []
    alternative_matches = 0
    total_steps = 0

    for traj_file in tqdm(trajectory_files, desc="Processing trajectories"):
        with open(os.path.join(args.agentnetbench_data, traj_file), 'r') as f:
            trajectory = json.load(f)
        
        task_id = trajectory['task_id']
        instruction = trajectory.get('high_level_task_description', trajectory.get('user_task_description', ''))
        steps = trajectory['steps']
        
        trajectory_results = []
        trajectory_score = 0
        trajectory_step_count = 0

        if task_id not in trajectory_histories:
            trajectory_histories[task_id] = {}

        for step_idx, step in enumerate(steps):
            total_steps += 1
            trajectory_step_count += 1

            img_filename = step['image']
            img_path = os.path.join(args.agentnetbench_imgs, img_filename)
            
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue
            
            image = Image.open(img_path)
            img_width, img_height = image.size

            messages = []
            print("step_idx: ", step_idx)

            max_history_length = 10
            max_detail_length = 0

            prev_indices = list(range(0, step_idx))
            previous_images_b64 = []

            image_slots = min(args.image_slots - 1, len(prev_indices))
            indices_with_images = prev_indices[-image_slots:] if image_slots > 0 else []

            for hist_idx in prev_indices:
                hist_step = steps[hist_idx]

                include_image = hist_idx in indices_with_images

                if include_image:
                    hist_img_path = os.path.join(args.agentnetbench_imgs, hist_step['image'])
                    if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
                        
                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "image", "image": hist_img_path},
                            ],
                        })
                        
                elif args.model_path == "xlangai/OpenCUA-7B":
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": hist_img_path},
                        ],
                    })

                if hist_idx >= max(0, step_idx - max_history_length):
                    if hist_idx in trajectory_histories[task_id]:
                        content = STEP_TEMPLATE.format(step_num=hist_idx + 1) + trajectory_histories[task_id][hist_idx]
                    else:
                        inner_monologue = hist_step.get('inner_monologue', {})
                        if hist_idx >= max(0, step_idx - max_detail_length):
                            content = STEP_TEMPLATE.format(step_num=hist_idx + 1) + RESPONSE_TEMPLATE.format(
                                observation=inner_monologue.get('observation', ''),
                                thought=inner_monologue.get('thought', ''),
                                action=inner_monologue.get('low_level_instruction', ''),
                                code=hist_step.get('action', '')
                            )
                        else:
                            content = STEP_TEMPLATE.format(step_num=hist_idx + 1) + HISTORY_TEMPLATE.format(
                                observation=inner_monologue.get('observation', ''),
                                thought=inner_monologue.get('thought', ''),
                                action=inner_monologue.get('low_level_instruction', '')
                            )
                    
                    messages.append({
                        "role": "assistant",
                        "content": content
                    })

            if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
                user_prompt = AGENTNETBENCH_PROMPT_UI_TARS.format(instruction=instruction)
            elif args.model_path == "xlangai/OpenCUA-7B":
                user_prompt = AGENTNETBENCH_PROMPT_OPENCUA.format(instruction=instruction)
            else:
                raise NotImplementedError(f"Model {args.model_path} not implemented")
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": user_prompt},
                ],
            })
            
            if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                print("messages: ", messages)
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(device)
            elif args.model_path == "xlangai/OpenCUA-7B":
                print("messages: ", messages)
                input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

                all_image_paths = []
                for msg in messages:
                    if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                        for content_item in msg["content"]:
                            if content_item.get("type") == "image" and "image" in content_item:
                                all_image_paths.append(content_item["image"])

                if all_image_paths:
                    images = []
                    for image_path in all_image_paths:
                        image = Image.open(image_path).convert('RGB')
                        images.append(image)

                    info = processor.preprocess(images=images)
                    pixel_values = torch.tensor(info['pixel_values']).to(dtype=torch.bfloat16, device=model.device)
                    grid_thws = torch.tensor(info['image_grid_thw'])
                else:
                    raise ValueError("No images found")

                input_ids = torch.tensor([input_ids]).to(model.device)
            else:
                raise NotImplementedError(f"Model {args.model_path} not implemented")

            # Analyze vision tokens for gui_kv and vl_cache
            if args.kv_cache in ["gui_kv", "vl_cache"]:
                num_images = sum(1 for msg in messages if msg.get("role") == "user" and isinstance(msg.get("content"), list) and any(content_item.get("type") == "image" for content_item in msg["content"]))
                if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
                    vision_analysis = analyze_vision_tokens_multi_images(processor, image_inputs, video_inputs, text, image_count=num_images)
                elif args.model_path == "xlangai/OpenCUA-7B":
                    vision_analysis = analyze_vision_tokens_opencua_multi_images(tokenizer, input_ids, image_grid_thw=grid_thws, merge_size=2, image_count=num_images)
                else:
                    raise NotImplementedError(f"Model {args.model_path} not implemented")
            set_window_size(model, args)
            if args.kv_cache == "vl_cache":
                if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
                    last_vision_indices = []
                    vision_end_idx = vision_analysis['vision_end_idx'][-1] if 'vision_analysis' in locals() else 0
                    last_vision_indices.append(vision_end_idx)
                elif args.model_path == "xlangai/OpenCUA-7B":
                    last_vision_indices = []
                    vision_end_idx = vision_analysis['vision_end_idx'][-1] if 'vision_analysis' in locals() else 0
                    last_vision_indices.append(vision_end_idx)
                else:
                    raise NotImplementedError(f"Model {args.model_path} not implemented")
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
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
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
            else:
                raise NotImplementedError(f"Model {args.model_path} not implemented")
                
            print("output_text: ", output_text)

            formatted_response = f"Step {step_idx + 1}:\n{output_text}"
            trajectory_histories[task_id][step_idx] = formatted_response

            gt_actions = step['ground_truth_actions']
            alternative_options = step.get('alternative_options', [])
            is_milestone = step.get('milestone', False)
            print("gt_actions: ", gt_actions)

            try:
                if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
                    parsed_actions = parse_action_to_structure_output(output_text, 
                        origin_resized_height=img_height,
                        origin_resized_width=img_width,
                        max_pixels=MAX_PIXELS,
                        min_pixels=MIN_PIXELS,
                        factor=IMAGE_FACTOR,
                        model_type="qwen25vl")
                    predicted_actions = convert_ui_tars_to_agentnetbench_actions(parsed_actions)
                elif args.model_path == "xlangai/OpenCUA-7B":
                    parsed_action = parse_response_actions_opencua(output_text)
                    if parsed_action:
                        predicted_actions = extract_actions(parsed_action,
                            origin_resized_height=img_height,
                            origin_resized_width=img_width,
                            max_pixels=MAX_PIXELS,
                            min_pixels=MIN_PIXELS,
                            factor=IMAGE_FACTOR,
                            model_type="qwen25vl")
                    else:
                        predicted_actions = []
                else:
                    raise NotImplementedError(f"Model {args.model_path} not implemented")
                    
                print("predicted_actions: ", predicted_actions)

                step_score = 0
                used_alternative = False
                if predicted_actions and gt_actions:
                    pred_action = predicted_actions[0] if predicted_actions else None
                    gt_action = gt_actions[0] if gt_actions else None
                    
                    if pred_action and gt_action:
                        score = evaluate_action(pred_action, gt_action, alternative_options)

                        if score == 0 and alternative_options:
                            for alt_actions in alternative_options:
                                if alt_actions:
                                    alt_score = evaluate_action(pred_action, alt_actions[0], None)
                                    if alt_score > score:
                                        score = alt_score
                                        used_alternative = True

                        step_score = score

                        action_type = gt_action['type'].lower()
                        if action_type in action_type_scores:
                            action_type_scores[action_type].append(score)

                        if is_milestone:
                            milestone_scores.append(score)

                        if used_alternative:
                            alternative_matches += 1

                trajectory_score += step_score
                print("step_score: ", step_score)

                step_result = {
                    'task_id': task_id,
                    'step_num': step_idx + 1,
                    'raw_response': output_text,
                    'predicted_actions': predicted_actions,
                    'ground_truth_actions': gt_actions,
                    'alternative_options': alternative_options,
                    'score': step_score,
                    'used_alternative': used_alternative,
                    'is_milestone': is_milestone
                }
                trajectory_results.append(step_result)
                
            except Exception as e:
                print(f"Error processing step {step_idx} of {task_id}: {e}")
                print(f"Output text: {output_text}")

                step_result = {
                    'task_id': task_id,
                    'step_num': step_idx + 1,
                    'raw_response': output_text,
                    'predicted_actions': [],
                    'ground_truth_actions': gt_actions,
                    'alternative_options': alternative_options,
                    'score': 0,
                    'error': str(e),
                    'is_milestone': is_milestone
                }
                trajectory_results.append(step_result)

        if trajectory_step_count > 0:
            trajectory_avg_score = trajectory_score / trajectory_step_count
            trajectory_scores.append(trajectory_avg_score)

        all_results.extend(trajectory_results)

    overall_score = sum(r['score'] for r in all_results) / len(all_results) if all_results else 0
    avg_trajectory_score = sum(trajectory_scores) / len(trajectory_scores) if trajectory_scores else 0

    action_type_avg_scores = {}
    for action_type, scores in action_type_scores.items():
        if scores:
            action_type_avg_scores[action_type] = sum(scores) / len(scores)
        else:
            action_type_avg_scores[action_type] = 0.0

    avg_milestone_score = sum(milestone_scores) / len(milestone_scores) if milestone_scores else 0

    print("\n" + "=" * 80)
    print("AGENTNETBENCH EVALUATION RESULTS")
    print("=" * 80)
    
    agentnetbench_metrics = {
        "overall_score": overall_score,
        "average_trajectory_score": avg_trajectory_score,
        "total_steps": total_steps,
        "total_trajectories": len(trajectory_files),
        "alternative_matches": alternative_matches,
        "alternative_match_percentage": (alternative_matches / total_steps * 100) if total_steps > 0 else 0,
        "milestone_score": avg_milestone_score,
        "milestone_steps": len(milestone_scores),
        "action_type_scores": action_type_avg_scores
    }
    
    print(f"Overall Score: {overall_score:.2%}")
    print(f"Average Trajectory Score: {avg_trajectory_score:.2%}")
    print(f"Total Steps Evaluated: {total_steps}")
    print(f"Alternative Matches: {alternative_matches} ({alternative_matches/total_steps*100:.1f}%)")
    print(f"Milestone Score: {avg_milestone_score:.2%} ({len(milestone_scores)} steps)")
    print("\nAction Type Scores:")
    for action_type, score in sorted(action_type_avg_scores.items()):
        count = len(action_type_scores[action_type])
        if count > 0:
            print(f"  {action_type}: {score:.2%} ({count} instances)")
    
    print("=" * 80)

    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)

        detailed_results = {
            "model_path": args.model_path,
            "kv_cache": args.kv_cache,
            "kv_cache_budget": args.kv_cache_budget,
            "attention_implementation": args.attention_implementation,
            "model_dtype": args.model_dtype,
            "max_new_tokens": args.max_new_tokens,
            "metrics": agentnetbench_metrics,
            "debug": args.debug is not None,
            "num_trajectories_evaluated": len(trajectory_files),
            "detailed_results": all_results
        }

        results_filename = f'agentnetbench_results_budget.json'
        results_path = os.path.join(args.results_dir, results_filename)
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved AgentNetBench results to {results_path}")