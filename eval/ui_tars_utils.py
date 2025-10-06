import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch.cuda
import pandas as pd
import json
import re
import ast

IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def convert_point_to_coordinates(text, is_answer=False):
    # 匹配 <bbox> 后面的四个数字
    pattern = r"<point>(\d+)\s+(\d+)</point>"

    def replace_match(match):
        x1, y1 = map(int, match.groups())
        x = (x1 + x1) // 2  # 使用截断取整
        y = (y1 + y1) // 2  # 使用截断取整
        if is_answer:
            return f"({x},{y})"  # 只返回 (x, y) 格式
        return f"({x},{y})"  # 返回带标签的格式

    # 去掉 [EOS] 并替换 <bbox> 坐标
    text = re.sub(r"\[EOS\]", "", text)
    return re.sub(pattern, replace_match, text).strip()


# 定义一个函数来解析每个 action
def parse_action(action_str):
    try:
        # 解析字符串为 AST 节点
        node = ast.parse(action_str, mode='eval')

        # 确保节点是一个表达式
        if not isinstance(node, ast.Expression):
            raise ValueError("Not an expression")

        # 获取表达式的主体
        call = node.body

        # 确保主体是一个函数调用
        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")

        # 获取函数名
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            func_name = None

        # 获取关键字参数
        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            # 处理不同类型的值，这里假设都是常量
            if isinstance(kw.value, ast.Constant):
                value = kw.value.value
            elif isinstance(kw.value, ast.Str):  # 兼容旧版本 Python
                value = kw.value.s
            else:
                value = None
            kwargs[key] = value

        return {'function': func_name, 'args': kwargs}

    except Exception as e:
        print(f"Failed to parse action '{action_str}': {e}")
        return None


def escape_single_quotes(text):
    # 匹配未转义的单引号（不匹配 \\'）
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def linear_resize(height: int,
                  width: int,
                  factor: int = IMAGE_FACTOR,
                  min_pixels: int = MIN_PIXELS,
                  max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    if width * height > max_pixels:
        """
        如果图片超过/低于像素限制，则计算一个缩放因子resize_factor，使图片的像素数缩小到等于或小于max_pixels。这个缩放因子是通过开平方根计算的，确保纵横比保持不变,这样原始的相对坐标可以不经转换直接复用
        """
        resize_factor = math.sqrt(max_pixels / (width * height))
        width, height = int(width * resize_factor), int(height * resize_factor)
    if width * height < min_pixels:
        resize_factor = math.sqrt(min_pixels / (width * height))
        width, height = math.ceil(width * resize_factor), math.ceil(
            height * resize_factor)

    return height, width


def smart_resize(height: int,
                 width: int,
                 factor: int = IMAGE_FACTOR,
                 min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def parse_action_to_structure_output(text,
                                     factor=IMAGE_FACTOR,
                                     origin_resized_height=None,
                                     origin_resized_width=None,
                                     model_type="qwen25vl",
                                     max_pixels=16384 * 28 * 28,
                                     min_pixels=100 * 28 * 28):
    text = text.strip()

    if "<point>" in text:
        text = convert_point_to_coordinates(text)
    if "start_point=" in text:
        text = text.replace("start_point=", "start_box=")
    if "end_point=" in text:
        text = text.replace("end_point=", "end_box=")
    if "point=" in text:
        text = text.replace("point=", "start_box=")

    if model_type == "qwen25vl":
        smart_resize_height, smart_resize_width = smart_resize(
            origin_resized_height,
            origin_resized_width,
            factor=IMAGE_FACTOR,
            min_pixels=min_pixels,
            max_pixels=max_pixels)

    # 正则表达式匹配 Action 字符串
    if text.startswith("Thought:"):
        thought_pattern = r"Thought: (.+?)(?=\s*Action: |$)"
        thought_hint = "Thought: "
    elif text.startswith("Reflection:"):
        thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action: |$)"
        thought_hint = "Reflection: "
    elif text.startswith("Action_Summary:"):
        thought_pattern = r"Action_Summary: (.+?)(?=\s*Action: |$)"
        thought_hint = "Action_Summary: "
    else:
        thought_pattern = r"Thought: (.+?)(?=\s*Action: |$)"
        thought_hint = "Thought: "
    reflection, thought = None, None
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    if thought_match:
        if len(thought_match.groups()) == 1:
            thought = thought_match.group(1).strip()
        elif len(thought_match.groups()) == 2:
            thought = thought_match.group(2).strip()
            reflection = thought_match.group(1).strip()
    if "Action:" not in text: return []
    action_str = text.split("Action: ")[-1]

    tmp_all_action = action_str.split(")\n\n")
    all_action = []
    for action_str in tmp_all_action:
        if "type(content" in action_str:
            if not action_str.strip().endswith(")"):
                action_str = action_str.strip() + ")"
            # 正则表达式匹配 content 中的字符串并转义单引号
            def escape_quotes(match):
                content = match.group(1)  # 获取 content 的值
                return content

            # 使用正则表达式进行替换
            pattern = r"type\(content='(.*?)'\)"  # 匹配 type(content='...')
            if re.search(pattern, action_str):  # 检查是否有匹配项
                content = re.sub(pattern, escape_quotes, action_str)
            else:
                raise ValueError("Pattern not found in the input string.")

            # 处理字符串
            action_str = escape_single_quotes(content)
            action_str = "type(content='" + action_str + "')"
        if not action_str.strip().endswith(")"):
            action_str = action_str.strip() + ")"
        all_action.append(action_str)

    parsed_actions = [
        parse_action(action.replace("\n", "\\n").lstrip())
        for action in all_action
    ]
    actions = []
    for action_instance, raw_str in zip(parsed_actions, all_action):
        if action_instance == None:
            print(f"Action can't parse: {raw_str}")
            raise ValueError(f"Action can't parse: {raw_str}")
        action_type = action_instance["function"]
        params = action_instance["args"]

        # import pdb; pdb.set_trace()
        action_inputs = {}
        for param_name, param in params.items():
            if param == "": continue
            param = param.lstrip()  # 去掉引号和多余的空格
            # 处理start_box或者end_box参数格式 '<bbox>x1 y1 x2 y2</bbox>'
            action_inputs[param_name.strip()] = param

            if "start_box" in param_name or "end_box" in param_name:
                ori_box = param
                # Remove parentheses and split the string by commas
                numbers = ori_box.replace("(", "").replace(")", "").split(",")

                # Convert to float and scale by 1000
                # Qwen2.5vl output absolute coordinates, qwen2vl output relative coordinates
                if model_type == "qwen25vl":
                    float_numbers = []
                    for num_idx, num in enumerate(numbers):
                        num = float(num)
                        if (num_idx + 1) % 2 == 0:
                            float_numbers.append(
                                float(num / smart_resize_height))
                        else:
                            float_numbers.append(
                                float(num / smart_resize_width))
                else:
                    float_numbers = [float(num) / factor for num in numbers]

                if len(float_numbers) == 2:
                    float_numbers = [
                        float_numbers[0], float_numbers[1], float_numbers[0],
                        float_numbers[1]
                    ]
                action_inputs[param_name.strip()] = str(float_numbers)

        # import pdb; pdb.set_trace()
        actions.append({
            "reflection": reflection,
            "thought": thought,
            "action_type": action_type,
            "action_inputs": action_inputs,
            "text": text
        })
    return actions


def parsing_response_to_pyautogui_code(responses,
                                       image_height: int,
                                       image_width: int,
                                       input_swap: bool = True) -> str:
    '''
    将M模型的输出解析为OSWorld中的action，生成pyautogui代码字符串
    参数:
        response: 包含模型输出的字典，结构类似于：
        {
            "action_type": "hotkey",
            "action_inputs": {
                "hotkey": "v ctrl",
                "start_box": None,
                "end_box": None
            }
        }
    返回:
        生成的pyautogui代码字符串
    '''

    pyautogui_code = f"import pyautogui\nimport time\n"
    if isinstance(responses, dict):
        responses = [responses]
    for response_id, response in enumerate(responses):
        if "observation" in response:
            observation = response["observation"]
        else:
            observation = ""

        if "thought" in response:
            thought = response["thought"]
        else:
            thought = ""

        if response_id == 0:
            pyautogui_code += f"'''\nObservation:\n{observation}\n\nThought:\n{thought}\n'''\n"
        else:
            pyautogui_code += f"\ntime.sleep(1)\n"

        action_dict = response
        action_type = action_dict.get("action_type")
        action_inputs = action_dict.get("action_inputs", {})

        if action_type == "hotkey":
            # Parsing hotkey action
            if "key" in action_inputs:
                hotkey = action_inputs.get("key", "")
            else:
                hotkey = action_inputs.get("hotkey", "")

            if hotkey == "arrowleft":
                hotkey = "left"

            elif hotkey == "arrowright":
                hotkey = "right"

            elif hotkey == "arrowup":
                hotkey = "up"

            elif hotkey == "arrowdown":
                hotkey = "down"

            if hotkey:
                # Handle other hotkeys
                keys = hotkey.split()  # Split the keys by space
                convert_keys = []
                for key in keys:
                    if key == "space":
                        key = ' '
                    convert_keys.append(key)
                pyautogui_code += f"\npyautogui.hotkey({', '.join([repr(k) for k in convert_keys])})"

        elif action_type in ["press", "keydown"]:
            # Parsing press action
            if "key" in action_inputs:
                key_to_press = action_inputs.get("key", "")
            else:
                key_to_press = action_inputs.get("press", "")

            if key_to_press == "arrowleft":
                key_to_press = "left"

            elif key_to_press == "arrowright":
                key_to_press = "right"

            elif key_to_press == "arrowup":
                key_to_press = "up"

            elif key_to_press == "arrowdown":
                key_to_press = "down"

            elif key_to_press == "space":
                key_to_press = " "

            if key_to_press:
                # Simulate pressing a single key
                pyautogui_code += f"\npyautogui.keyDown({repr(key_to_press)})"

        elif action_type in ["release", "keyup"]:
            # Parsing press action
            if "key" in action_inputs:
                key_to_press = action_inputs.get("key", "")
            else:
                key_to_press = action_inputs.get("press", "")

            if key_to_press == "arrowleft":
                key_to_press = "left"

            elif key_to_press == "arrowright":
                key_to_press = "right"

            elif key_to_press == "arrowup":
                key_to_press = "up"

            elif key_to_press == "arrowdown":
                key_to_press = "down"

            elif key_to_press == "space":
                key_to_press = " "

            if key_to_press:
                # Simulate pressing a single key
                pyautogui_code += f"\npyautogui.keyUp({repr(key_to_press)})"

        elif action_type == "type":
            # Parsing typing action using clipboard
            content = action_inputs.get("content", "")
            content = escape_single_quotes(content)
            stripped_content = content
            if content.endswith("\n") or content.endswith("\\n"):
                stripped_content = stripped_content.rstrip("\\n").rstrip("\n")
            if content:
                if input_swap:
                    pyautogui_code += f"\nimport pyperclip"
                    pyautogui_code += f"\npyperclip.copy('{stripped_content}')"
                    pyautogui_code += f"\npyautogui.hotkey('ctrl', 'v')"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"
                else:
                    pyautogui_code += f"\npyautogui.write('{stripped_content}', interval=0.1)"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"

        elif action_type in ["drag", "select"]:
            # Parsing drag or select action based on start and end_boxes
            start_box = action_inputs.get("start_box")
            end_box = action_inputs.get("end_box")
            if start_box and end_box:
                x1, y1, x2, y2 = eval(
                    start_box)  # Assuming box is in [x1, y1, x2, y2]
                sx = round(float((x1 + x2) / 2) * image_width, 3)
                sy = round(float((y1 + y2) / 2) * image_height, 3)
                x1, y1, x2, y2 = eval(
                    end_box)  # Assuming box is in [x1, y1, x2, y2]
                ex = round(float((x1 + x2) / 2) * image_width, 3)
                ey = round(float((y1 + y2) / 2) * image_height, 3)
                pyautogui_code += (
                    f"\npyautogui.moveTo({sx}, {sy})\n"
                    f"\npyautogui.dragTo({ex}, {ey}, duration=1.0)\n")

        elif action_type == "scroll":
            # Parsing scroll action
            start_box = action_inputs.get("start_box")
            if start_box:
                x1, y1, x2, y2 = eval(
                    start_box)  # Assuming box is in [x1, y1, x2, y2]
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)

                # # 先点对应区域，再滚动
                # pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
            else:
                x = None
                y = None
            direction = action_inputs.get("direction", "")

            if x == None:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5)"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5)"
            else:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5, x={x}, y={y})"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5, x={x}, y={y})"

        elif action_type in [
                "click", "left_single", "left_double", "right_single", "hover"
        ]:
            # Parsing mouse click actions
            start_box = action_inputs.get("start_box")
            start_box = str(start_box)
            if start_box:
                start_box = eval(start_box)
                if len(start_box) == 4:
                    x1, y1, x2, y2 = start_box  # Assuming box is in [x1, y1, x2, y2]
                elif len(start_box) == 2:
                    x1, y1 = start_box
                    x2 = x1
                    y2 = y1
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)
                if action_type == "left_single" or action_type == "click":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
                elif action_type == "left_double":
                    pyautogui_code += f"\npyautogui.doubleClick({x}, {y}, button='left')"
                elif action_type == "right_single":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='right')"
                elif action_type == "hover":
                    pyautogui_code += f"\npyautogui.moveTo({x}, {y})"

        elif action_type in ["finished"]:
            pyautogui_code = f"DONE"

        else:
            pyautogui_code += f"\n# Unrecognized action type: {action_type}"

    return pyautogui_code


def add_box_token(input_string):
    # Step 1: Split the string into individual actions
    if "Action: " in input_string and "start_box=" in input_string:
        suffix = input_string.split("Action: ")[0] + "Action: "
        actions = input_string.split("Action: ")[1:]
        processed_actions = []
        for action in actions:
            action = action.strip()
            # Step 2: Extract coordinates (start_box or end_box) using regex
            coordinates = re.findall(
                r"(start_box|end_box)='\((\d+),\s*(\d+)\)'", action)

            updated_action = action  # Start with the original action
            for coord_type, x, y in coordinates:
                # Convert x and y to integers
                updated_action = updated_action.replace(
                    f"{coord_type}='({x},{y})'",
                    f"{coord_type}='<|box_start|>({x},{y})<|box_end|>'")
            processed_actions.append(updated_action)

        # Step 5: Reconstruct the final string
        final_string = suffix + "\n\n".join(processed_actions)
    else:
        final_string = input_string
    return final_string





def plot_attention_heatmap(batched_attentions, head_ids=None, layer_ids=None, save_dir="", sample_id=0):
    """
    Visualizes attention weights as a heatmap.

    Args:
        batched_attentions (torch.Tensor): a list of attention weights tensor of shape (batch_size, num_heads, seq_len, seq_len).
        head_ids (list[int], optional): List of head indices to visualize. If None, all heads are averaged.
    """
    assert layer_ids is not None, "Please provide the layer_ids to visualize."
    
    attentions = [att[0]for att in batched_attentions]
    # (batch_size, num_heads, sequence_length, sequence_length)
    
    # torch.Size([1, 28, 5041, 5041])
    print(f"attentions shape: {[att.shape for att in attentions]}")
    for layer_id in layer_ids:
        attention = attentions[layer_id]
        if not head_ids:
            data = torch.mean(attention, dim=0).float().cpu().numpy()
            save_path = os.path.join(save_dir, f"sample-{sample_id}", f"layer{layer_id}.jpg") if save_dir else None
            plot_heatmap(data, f'Average Attention Map: Layer {layer_id}', save_path)
        else:
            for head_id in head_ids:
                data = attention[head_id].numpy()
                save_path = os.path.join(save_dir, f"sample-{sample_id}", f"layer{layer_id}_head{head_id}.jpg") if save_dir else None
                plot_heatmap(data, f'Attention Map: Layer {layer_id} Head {head_id}', save_path)
                
def plot_heatmap(
    data: torch.Tensor, 
    title: str, 
    save_path=None
) -> None:
    """
    Helper function to plot a heatmap for a tensor of shape (seq_len, seq_len).
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    seq_length = data.shape[0]

    # Set seaborn style for prettier plots
    sns.set_style("whitegrid")
    plt.style.use("seaborn-v0_8")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use seaborn's heatmap with a beautiful color palette
    sns.heatmap(
        data, 
        ax=ax,
        cmap='Blues',  # White for zero, red for max - perfect for attention!
        vmax=0.01,  # Set max value to 0.01 - values > 0.01 will render as most blue
        cbar_kws={
            'label': 'Attention Weight',
            'shrink': 0.8,
            'aspect': 20
        },
        square=True,  # Make cells square for better visual
        linewidths=0,  # Remove grid lines for cleaner look
        xticklabels=False,  # Hide tick labels for large matrices
        yticklabels=False,
        rasterized=True  # Better performance for large matrices
    )
    # Print diagnostic information
    print(f"Attention map shape: {data.shape}")
    print(f"Attention values - min: {data.min():.6f}, max: {data.max():.6f}")
    print(f"Attention statistics - mean: {data.mean():.6f}, std: {data.std():.6f}")
    print(f"Data type: {data.dtype}")
    print(f"Non-zero attention weights: {np.count_nonzero(data)}/{data.size}")
    print("--------------------------------")
    
    # Style the plot
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Key Positions', fontsize=14, fontweight='semibold')
    ax.set_ylabel('Query Positions', fontsize=14, fontweight='semibold')
    
    # Improve layout
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Heatmap saved to {save_path}")

def analyze_vision_tokens(processor, image_inputs, video_inputs, text):
    """
    Analyze vision token lengths and positions in the input sequence.
    
    Args:
        processor: The model processor (AutoProcessor)
        image_inputs: Output from process_vision_info
        video_inputs: Output from process_vision_info  
        text: The processed text prompt
        
    Returns:
        dict: Contains vision token analysis information
    """
    # Get inputs without vision
    text_only_inputs = processor(
        text=[text],
        images=None,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    text_only_length = text_only_inputs.input_ids.shape[1]
    
    # Get inputs with vision
    full_inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    full_length = full_inputs.input_ids.shape[1]
    
    # Calculate vision token information
    # vision_token_count = full_length - text_only_length
    
    # Get tokenizer for special token analysis
    tokenizer = processor.tokenizer
    
    # Decode the full sequence to understand structure
    full_tokens = tokenizer.convert_ids_to_tokens(full_inputs.input_ids[0])
    text_only_tokens = tokenizer.convert_ids_to_tokens(text_only_inputs.input_ids[0])
    
    # Find vision token positions (typically between special image tokens)
    vision_start_idx = None
    vision_end_idx = None
    vision_token_count = full_tokens.count('<|image_pad|>')
    # Look for image-related special tokens in Qwen2.5-VL
    for i, token in enumerate(full_tokens):
        if '<|vision_start|>' in token or '<|image_pad|>' in token or token.startswith('<|image'):
            if vision_start_idx is None:
                # we need to + 1 becuase image placeholders are <|vision_start|><image_pad><|vision_end|>
                vision_start_idx = i + 1
        elif '<|vision_end|>' in token or (vision_start_idx is not None and vision_end_idx is None):
            if not (token.startswith('<|image') or '<|image_pad|>' in token):
                vision_end_idx = i
                break
    
    # If we can't find specific markers, estimate based on length difference
    if vision_start_idx is None or vision_end_idx is None:
        # Heuristic: vision tokens are typically inserted after system tokens but before user text
        estimated_system_length = len([t for t in text_only_tokens if t in ['<|im_start|>', '<|im_end|>', 'system', 'user']])
        vision_start_idx = min(estimated_system_length, full_length - vision_token_count)
        vision_end_idx = vision_start_idx + vision_token_count
    
    analysis = {
        'total_sequence_length': full_length,
        'text_only_length': text_only_length, 
        'vision_token_count': vision_token_count,
        'vision_start_idx': vision_start_idx,
        'vision_end_idx': vision_end_idx,
        'vision_token_ratio': vision_token_count / full_length if full_length > 0 else 0,
        'has_vision_tokens': vision_token_count > 0,
        'sample_tokens': {
            'first_10_tokens': full_tokens[:10],
            'last_10_tokens': full_tokens[-10:],
            'vision_region_sample': full_tokens[max(0, vision_start_idx-2):min(len(full_tokens), vision_end_idx+2)] if vision_start_idx is not None else []
        }
    }
    
    return analysis


def analyze_vision_tokens_multi_images(processor, image_inputs, video_inputs, text, image_count=1):
    """
    Analyze vision token lengths and positions in the input sequence.
    
    Args:
        processor: The model processor (AutoProcessor)
        image_inputs: Output from process_vision_info
        video_inputs: Output from process_vision_info  
        text: The processed text prompt
        
    Returns:
        dict: Contains vision token analysis information
    """
    # Get inputs without vision
    text_only_inputs = processor(
        text=[text],
        images=None,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    text_only_length = text_only_inputs.input_ids.shape[1]
    
    # Get inputs with vision
    full_inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    full_length = full_inputs.input_ids.shape[1]
    
    # Calculate vision token information
    # vision_token_count = full_length - text_only_length
    
    # Get tokenizer for special token analysis
    tokenizer = processor.tokenizer
    
    # Decode the full sequence to understand structure
    full_tokens = tokenizer.convert_ids_to_tokens(full_inputs.input_ids[0])
    text_only_tokens = tokenizer.convert_ids_to_tokens(text_only_inputs.input_ids[0])
    
    # Find vision token positions (typically between special image tokens)
    vision_start_indices = []
    vision_end_indices = []
    vision_token_count = full_tokens.count('<|image_pad|>')
    
    # Look for image-related special tokens in Qwen2.5-VL
    current_vision_start = None
    for i, token in enumerate(full_tokens):
        if '<|vision_start|>' in token:
            # Found the start of a new vision token sequence
            # we need to + 1 because image placeholders are <|vision_start|><image_pad><|vision_end|>
            current_vision_start = i + 1
        elif '<|vision_end|>' in token and current_vision_start is not None:
            # Found the end of the current vision token sequence
            vision_start_indices.append(current_vision_start)
            vision_end_indices.append(i)
            current_vision_start = None
    
    # If we can't find specific markers, estimate based on length difference
    if len(vision_start_indices) == 0 and vision_token_count > 0:
        # Heuristic: vision tokens are typically inserted after system tokens but before user text
        estimated_system_length = len([t for t in text_only_tokens if t in ['<|im_start|>', '<|im_end|>', 'system', 'user']])
        vision_start_idx = min(estimated_system_length, full_length - vision_token_count)
        vision_end_idx = vision_start_idx + vision_token_count
        vision_start_indices = [vision_start_idx]
        vision_end_indices = [vision_end_idx]
    
    analysis = {
        'vision_start_idx': vision_start_indices,
        'vision_end_idx': vision_end_indices,
    }
    assert image_count == len(vision_start_indices) == len(vision_end_indices), f"Expected {image_count} images but found {len(vision_start_indices)} vision token pairs"
    return analysis

def plot_attention_heatmap_with_vision_analysis(batched_attentions, processor, image_inputs, video_inputs, text, head_ids=None, layer_ids=None, save_dir="", sample_id=0):
    """
    Enhanced version of plot_attention_heatmap that includes vision token analysis.
    """
    assert layer_ids is not None, "Please provide the layer_ids to visualize."
    
    # Analyze vision tokens
    vision_analysis = analyze_vision_tokens(processor, image_inputs, video_inputs, text)
    
    print(f"=== Vision Token Analysis ===")
    print(f"Total sequence length: {vision_analysis['total_sequence_length']}")
    print(f"Text-only length: {vision_analysis['text_only_length']}")
    print(f"Vision token count: {vision_analysis['vision_token_count']}")
    print(f"Vision token ratio: {vision_analysis['vision_token_ratio']:.2%}")
    print(f"Vision tokens span: [{vision_analysis['vision_start_idx']}:{vision_analysis['vision_end_idx']}]")
    
    print("=" * 50)
    
    attentions = [att[0] for att in batched_attentions]
    
    for layer_id in layer_ids:
        attention = attentions[layer_id]
        if not head_ids:
            data = torch.mean(attention, dim=0).float().cpu().numpy()
            # title = f'Average Attention Map: Layer {layer_id}\nVision Tokens: {vision_analysis["vision_token_count"]} ({vision_analysis["vision_token_ratio"]:.1%})'
            title = f'Average Attention Map: Layer {layer_id}'
            save_path = os.path.join(save_dir, f"sample-{sample_id}", f"layer{layer_id}_with_vision_analysis.jpg") if save_dir else None
            plot_heatmap_with_vision_regions(data, vision_analysis, title, save_path)
        else:
            for head_id in head_ids:
                data = attention[head_id].numpy()
                # title = f'Attention Map: Layer {layer_id} Head {head_id}\nVision Tokens: {vision_analysis["vision_token_count"]} ({vision_analysis["vision_token_ratio"]:.1%})'
                title = f'Attention Map: Layer {layer_id} Head {head_id}'
                save_path = os.path.join(save_dir, f"sample-{sample_id}", f"layer{layer_id}_head{head_id}_with_vision_analysis.jpg") if save_dir else None
                plot_heatmap_with_vision_regions(data, vision_analysis, title, save_path)

def plot_heatmap_with_vision_regions(
    data: torch.Tensor, 
    vision_analysis: dict,
    title: str, 
    save_path=None,
    generated_length=0
) -> None:
    """
    Enhanced heatmap plotting that highlights vision token regions.
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Set seaborn style for prettier plots
    sns.set_style("whitegrid")
    plt.style.use("seaborn-v0_8")
    
    fig, ax = plt.subplots(figsize=(14, 12))  # Slightly larger for annotations
    
    # Use seaborn's heatmap
    sns.heatmap(
        data, 
        ax=ax,
        cmap='Blues',  # White for zero, blue for max - perfect for attention!
        vmax=0.01,  # Set max value to 0.01 - values > 0.01 will render as most blue
        cbar_kws={
            'label': 'Attention Weight',
            'shrink': 0.8,
            'aspect': 20
        },
        square=True,
        linewidths=0,
        xticklabels=False,
        yticklabels=False,
        rasterized=True
    )
    
    from matplotlib.patches import Rectangle
    
    # Highlight vision token regions if available
    if vision_analysis['has_vision_tokens'] and vision_analysis['vision_start_idx'] is not None:
        vision_start = vision_analysis['vision_start_idx']
        vision_end = vision_analysis['vision_end_idx']
        
        # Add rectangles to highlight vision regions
        # Create a proper rectangle that forms a square region
        
        # Vision-to-vision attention region (square in top-left of vision tokens)
        vision_rect = Rectangle(
            (vision_start, vision_start), 
            vision_end - vision_start, 
            vision_end - vision_start,
            linewidth=2, 
            edgecolor='red', 
            facecolor='none', 
            linestyle='--', 
            alpha=0.7
        )
        ax.add_patch(vision_rect)
        
        # Optional: Add lines to show full vision token extent
        # Vertical lines for columns (keys - what we attend TO)
        ax.axvline(x=vision_start, color='red', linestyle=':', alpha=0.5, linewidth=1, label='Vision Tokens')
        ax.axvline(x=vision_end, color='red', linestyle=':', alpha=0.5, linewidth=1)
        
        # Horizontal lines for rows (queries - what ATTENDS)  
        ax.axhline(y=vision_start, color='red', linestyle=':', alpha=0.5, linewidth=1)
        ax.axhline(y=vision_end, color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    # Highlight generated token regions if available
    if generated_length > 0:
        # Use actual attention matrix dimensions instead of pre-computed values
        actual_seq_length = data.shape[0]  # or data.shape[1], they should be the same for square attention matrix
        
        # Generated tokens are the last `generated_length` tokens in the sequence
        gen_start = actual_seq_length - generated_length
        gen_end = actual_seq_length
        
        # Sanity check: make sure we have valid indices
        if gen_start >= 0 and gen_end <= actual_seq_length:
            # Vertical lines for generated token columns (keys - what we attend TO)
            ax.axvline(x=gen_start, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Generated Tokens')
            ax.axvline(x=gen_end, color='green', linestyle='--', alpha=0.7, linewidth=2)
            
            # Horizontal lines for generated token rows (queries - what ATTENDS)
            ax.axhline(y=gen_start, color='green', linestyle='--', alpha=0.7, linewidth=2)
            ax.axhline(y=gen_end, color='green', linestyle='--', alpha=0.7, linewidth=2)
            
            # Add rectangle to highlight generated-to-generated attention region
            gen_rect = Rectangle(
                (gen_start, gen_start), 
                gen_end - gen_start, 
                gen_end - gen_start,
                linewidth=2, 
                edgecolor='green', 
                facecolor='none', 
                linestyle='-', 
                alpha=0.7
            )
            ax.add_patch(gen_rect)
            
            print(f"Generated token region: [{gen_start}:{gen_end}] (length: {generated_length}) in attention matrix of size {data.shape}")
        else:
            print(f"Warning: Invalid generated token indices [{gen_start}:{gen_end}] for attention matrix of size {data.shape}")
    
    # Add legend to show what colors represent
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Print diagnostic information
    print(f"Attention map shape: {data.shape}")
    print(f"Attention values - min: {data.min():.6f}, max: {data.max():.6f}")
    print(f"Attention statistics - mean: {data.mean():.6f}, std: {data.std():.6f}")
    print(f"Data type: {data.dtype}")
    print(f"Non-zero attention weights: {np.count_nonzero(data)}/{data.size}")
    
    # Calculate attention statistics for vision regions
    if vision_analysis['has_vision_tokens'] and vision_analysis['vision_start_idx'] is not None:
        vision_start = vision_analysis['vision_start_idx']
        vision_end = vision_analysis['vision_end_idx']
        
        # Attention TO vision tokens (columns)
        vision_attention_received = data[:, vision_start:vision_end].mean()
        # Attention FROM vision tokens (rows)  
        vision_attention_given = data[vision_start:vision_end, :].mean()
    
        print(f"Average attention To vision tokens: {vision_attention_received:.6f}")
        print(f"Average attention From vision tokens: {vision_attention_given:.6f}")
    
    # Calculate attention statistics for generated token regions
    if generated_length > 0:
        original_input_length = vision_analysis['total_sequence_length']
        gen_start = original_input_length
        gen_end = original_input_length + generated_length
        
        if data.shape[1] >= gen_end and data.shape[0] >= gen_end:
            # Attention TO generated tokens (columns)
            generated_attention_received = data[:gen_start, gen_start:gen_end].mean() if gen_start > 0 else 0
            # Attention FROM generated tokens (rows)
            generated_attention_given = data[gen_start:gen_end, :gen_start].mean() if gen_start > 0 else 0
            # Self-attention among generated tokens
            generated_self_attention = data[gen_start:gen_end, gen_start:gen_end].mean()
            
            print(f"Average attention To generated tokens (from input): {generated_attention_received:.6f}")
            print(f"Average attention From generated tokens (to input): {generated_attention_given:.6f}")
            print(f"Average self-attention among generated tokens: {generated_self_attention:.6f}")
            
            # Cross-attention between vision and generated tokens
            if vision_analysis['has_vision_tokens'] and vision_analysis['vision_start_idx'] is not None:
                vision_start = vision_analysis['vision_start_idx']
                vision_end = vision_analysis['vision_end_idx']
                
                # Generated tokens attending to vision tokens
                gen_to_vision = data[gen_start:gen_end, vision_start:vision_end].mean()
                # Vision tokens attending to generated tokens (if this makes sense in the attention matrix)
                vision_to_gen = data[vision_start:vision_end, gen_start:gen_end].mean() if data.shape[1] >= gen_end else 0
                
                print(f"Average attention From generated TO vision tokens: {gen_to_vision:.6f}")
                print(f"Average attention From vision TO generated tokens: {vision_to_gen:.6f}")
    
    print("--------------------------------")
    
    # Style the plot
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Key Positions (What we attend TO)', fontsize=12, fontweight='semibold')
    ax.set_ylabel('Query Positions (What ATTENDs)', fontsize=12, fontweight='semibold')
    
    # Improve layout
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Enhanced heatmap with vision analysis saved to {save_path}")

def measure_generation_latency(model, inputs, max_new_tokens=400, pad_token_id=None, output_attentions=False):
    """
    Measure pre-filling and decoding latency for HuggingFace generation.
    Simple approach: Measure prefill separately, then total generation, then subtract.
    
    Args:
        model: The HuggingFace model
        inputs: Processed inputs from processor
        max_new_tokens: Maximum new tokens to generate
        pad_token_id: Padding token ID
        output_attentions: Whether to output attention weights
        
    Returns:
        dict: Contains timing analysis, and generation outputs
    """
    
    
    
    # Warm up GPU if using CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Handle both dictionary and object-like inputs
    if isinstance(inputs, dict):
        input_ids = inputs['input_ids']
        # Create attention mask if not provided
        attention_mask = inputs.get('attention_mask', torch.ones_like(inputs['input_ids']))
    else:
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
    
    input_length = input_ids.shape[1]
    
    
    prefill_start = time.time()
    
    with torch.no_grad():
        # Forward pass to measure prefill time
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'use_cache': False,
        }
        
        # Add vision-specific inputs if present
        if isinstance(inputs, dict):
            # For dictionary inputs, check for vision keys
            if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
                model_inputs['pixel_values'] = inputs['pixel_values']
            if 'image_grid_thw' in inputs and inputs['image_grid_thw'] is not None:
                model_inputs['image_grid_thw'] = inputs['image_grid_thw']
            if 'grid_thws' in inputs and inputs['grid_thws'] is not None:
                model_inputs['grid_thws'] = inputs['grid_thws']
        else:
            # For object-like inputs, check attributes
            if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
                model_inputs['pixel_values'] = inputs.pixel_values
            if hasattr(inputs, 'image_grid_thw') and inputs.image_grid_thw is not None:
                model_inputs['image_grid_thw'] = inputs.image_grid_thw
            
        # Single forward pass for prefill timing
        _ = model(**model_inputs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    prefill_end = time.time()
    prefill_time = prefill_end - prefill_start
    
    generation_start = time.time()
    
    # Prepare generation inputs
    if isinstance(inputs, dict):
        generation_inputs = inputs.copy()
    else:
        # Convert object-like inputs to dictionary for unpacking
        generation_inputs = dict(inputs)
    
    if output_attentions:
        outputs = model.generate(
            **generation_inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
            output_attentions=True,
            return_dict_in_generate=True,
            use_cache=True,
            do_sample=False,  # Deterministic for consistent timing
        )
    else:
        outputs = model.generate(
            **generation_inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
            do_sample=False,
            use_cache=True
        )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    generation_end = time.time()
    total_generation_time = generation_end - generation_start
    print(f"Total generation time: {total_generation_time} seconds")
    
    # Step 3: Calculate decoding time
    decode_time = total_generation_time - prefill_time
    
    # Handle edge case where decode time might be negative due to measurement noise
    decode_time = max(decode_time, 0.001)  # Minimum 1ms decode time
    
    # Calculate metrics
    if hasattr(outputs, 'sequences'):
        generated_length = outputs.sequences.shape[1] - input_length
    else:
        generated_length = outputs.shape[1] - input_length
    
    # Per-token metrics
    overall_tokens_per_second = generated_length / total_generation_time if total_generation_time > 0 else 0
    decode_time_per_token = decode_time / generated_length if generated_length > 0 else 0
    
    timing_analysis = {
        'total_generation_time': total_generation_time,
        'prefill_time': prefill_time,
        'decode_time': decode_time,
        'decode_time_per_token': decode_time_per_token,
        'input_length': input_length,
        'generated_length': generated_length,
        'total_tokens': input_length + generated_length,
        'overall_tokens_per_second': overall_tokens_per_second,
        'prefill_tokens_per_second': input_length / prefill_time if prefill_time > 0 else 0,
        'decode_tokens_per_second': generated_length / decode_time if decode_time > 0 else 0,
        'prefill_percentage': (prefill_time / total_generation_time) * 100 if total_generation_time > 0 else 0,
    }
    
    return {
        'outputs': outputs,
        'timing_analysis': timing_analysis,
    }



def print_timing_analysis(timing_analysis, vision_analysis=None):
    """
    Pretty print timing analysis results.
    
    Args:
        timing_analysis: Output from measure_generation_latency
        vision_analysis: Optional vision token analysis for context
    """
    print("\n" + "=" * 60)
    print("⏱️  GENERATION TIMING ANALYSIS")
    print("=" * 60)
    
    if vision_analysis:
        print(f"📊 Context Info:")
        print(f"   Total sequence length: {timing_analysis['input_length']:,} tokens")
        print(f"   Vision tokens: {vision_analysis['vision_token_count']:,} ({vision_analysis['vision_token_ratio']:.1%})")
        print(f"   Text tokens: {vision_analysis['text_only_length']:,}")
        print()
    
    print(f"⚡ Performance Metrics:")
    print(f"   Total generation time: {timing_analysis['total_generation_time']:.3f}s")
    print(f"   Pre-fill time: {timing_analysis['prefill_time']:.3f}s ({timing_analysis['prefill_percentage']:.1f}%)")
    print(f"   Decode time: {timing_analysis['decode_time']:.3f}s ({100-timing_analysis['prefill_percentage']:.1f}%)")
    print()
    
    print(f"🚀 Throughput:")
    print(f"   Overall tokens/sec: {timing_analysis['overall_tokens_per_second']:.2f}")
    print(f"   Pre-fill tokens/sec: {timing_analysis['prefill_tokens_per_second']:.2f}")
    print(f"   Decode tokens/sec: {timing_analysis['decode_tokens_per_second']:.2f}")
    print(f"   Time per output token: {timing_analysis['decode_time_per_token']*1000:.2f}ms")
    print()
    
    print(f"📝 Generation Stats:")
    print(f"   Input tokens: {timing_analysis['input_length']:,}")
    print(f"   Generated tokens: {timing_analysis['generated_length']:,}")
    print(f"   Total tokens processed: {timing_analysis['total_tokens']:,}")
    
    print("=" * 60)

def plot_attention_heatmap_with_precomputed_vision_analysis(batched_attentions, vision_analysis, head_ids=None, layer_ids=None, save_dir="", sample_id=0, generated_length=0):
    """
    Enhanced version of plot_attention_heatmap that uses pre-computed vision token analysis.
    More efficient when vision analysis is already computed.
    """
    assert layer_ids is not None, "Please provide the layer_ids to visualize."
    
    print(f"=== Vision Token Analysis ===")
    print(f"Total sequence length: {vision_analysis['total_sequence_length']}")
    print(f"Text-only length: {vision_analysis['text_only_length']}")
    print(f"Vision token count: {vision_analysis['vision_token_count']}")
    print(f"Vision token ratio: {vision_analysis['vision_token_ratio']:.2%}")
    print(f"Vision tokens span: [{vision_analysis['vision_start_idx']}:{vision_analysis['vision_end_idx']}]")
    
    print("=" * 50)
    
    attentions = [att[0] for att in batched_attentions]
    
    for layer_id in layer_ids:
        attention = attentions[layer_id]
        if not head_ids:
            data = torch.mean(attention, dim=0).float().cpu().numpy()
            title = f'Average Attention Map: Layer {layer_id}'
            save_path = os.path.join(save_dir, f"sample-{sample_id}", f"layer{layer_id}_with_vision_analysis.jpg") if save_dir else None
            plot_heatmap_with_vision_regions(data, vision_analysis, title, save_path, generated_length)
        else:
            for head_id in head_ids:
                data = attention[head_id].numpy()
                title = f'Attention Map: Layer {layer_id} Head {head_id}'
                save_path = os.path.join(save_dir, f"sample-{sample_id}", f"layer{layer_id}_head{head_id}_with_vision_analysis.jpg") if save_dir else None
                plot_heatmap_with_vision_regions(data, vision_analysis, title, save_path, generated_length)


# ---------------------------------------------------------
#   Better plots:  mean-curve + error band  and  heat-map
#   replace the whole block that starts with:
#   "for task in unique_tasks:"   until   "plt.close()"
# ---------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator   # integer x-axis ticks

def plot_sparsity_curves_and_heatmap(task_df, task_name, save_dir, model_dtype, logging):
    """
    task_df : dataframe containing columns [layer, stage, sparsity]
          task_name : 'mobile' | 'desktop' | ...
      save_dir : directory where plots are written
      model_dtype : 'float16' | 'float32' | ...
      logging : logging object
      """
    # ------------------------------------------------------------------
    # 1)  LINE PLOT  (mean ± std)  -------------------------------------
    # ------------------------------------------------------------------
    stats = (
        task_df
        .groupby(['layer', 'stage'])
        .agg(mean_sparsity=('sparsity', 'mean'),
             std_sparsity =('sparsity', 'std'))
        .reset_index()
    )

    # pivot so that stages are columns for easier plotting
    mean_pivot = stats.pivot(index='layer', columns='stage', values='mean_sparsity')
    std_pivot  = stats.pivot(index='layer', columns='stage', values='std_sparsity')

    
    plt.figure(figsize=(18, 8))
    colour_map = sns.color_palette("Set2", n_colors=len(mean_pivot.columns))

    for idx, stage in enumerate(mean_pivot.columns):
        x      = mean_pivot.index.to_numpy()
        y_mean = mean_pivot[stage].to_numpy()
        y_std  = std_pivot [stage].to_numpy()

        plt.plot(x, y_mean,
                 marker='o', label=stage.replace('_', ' ').title(),
                 color=colour_map[idx])
        # shaded error band
        plt.fill_between(x,
                         y_mean - y_std,
                         y_mean + y_std,
                         alpha=0.25, color=colour_map[idx])

    plt.title(f'Attention sparsity (mean ± std) across layers – {task_name.capitalize()}', fontsize=20)
    plt.xlabel('Layer', fontsize=16)
    plt.ylabel('Sparsity  (fraction ≤ 0.01)', fontsize=16)
    plt.legend(fontsize=13)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()

    curve_path = os.path.join(save_dir, f'sparsity_curves_{task_name}_{model_dtype}.png')
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved mean-curve sparsity figure: {curve_path}")

    # ------------------------------------------------------------------
    # 2)  HEAT-MAP  -----------------------------------------------------
    # ------------------------------------------------------------------
    # pivot with stage as rows, layer as cols
    heatmap_data = mean_pivot.T  # now rows=stage, cols=layer, values=mean sparsity

    plt.figure(figsize=(18, 4 + 0.5 * heatmap_data.shape[0]))
    sns.heatmap(
        heatmap_data,
        annot=True, fmt=".2f",
        cmap='YlGnBu',
        cbar_kws={'label': 'Mean sparsity'}
    )
    plt.title(f'Mean attention sparsity heat-map – {task_name.capitalize()}', fontsize=20)
    plt.xlabel('Layer', fontsize=16)
    plt.ylabel('Stage', fontsize=16)
    plt.tight_layout()

    heatmap_path = os.path.join(save_dir, f'sparsity_heatmap_{task_name}_{model_dtype}.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved heat-map sparsity figure: {heatmap_path}")


    # Store statistics in JSON format
    stats_dict = {}
    for stage in mean_pivot.columns:
        stats_dict[stage] = {}
        for layer in mean_pivot.index:
            stats_dict[stage][f'layer_{layer}'] = {
                'mean_sparsity': float(mean_pivot.loc[layer, stage]) if not pd.isna(mean_pivot.loc[layer, stage]) else None,
                'std_sparsity': float(std_pivot.loc[layer, stage]) if not pd.isna(std_pivot.loc[layer, stage]) else None
            }
    
    json_path = os.path.join(save_dir, f'sparsity_stats_{task_name}_{model_dtype}.json')
    with open(json_path, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    logging.info(f"Saved sparsity statistics JSON: {json_path}")

def visualize_pyramidkv_removed_patches(image, model, vision_analysis, save_dir="", sample_id=0, layer_id=0):
    """
    Visualize image patches that are removed by PyramidKV compression.
    Overlays removed patches with red color on the original image.
    
    Args:
        image: PIL Image - the original input image
        model: The model containing PyramidKV layers
        vision_analysis: Dict containing vision token analysis
        save_dir: Directory to save visualization
        sample_id: Sample identifier for filename
        layer_id: Which layer's kept_indices to use for visualization
    """
    try:
        import numpy as np
        from PIL import Image, ImageDraw
        import matplotlib.pyplot as plt
        import os
        
        # Get the vision token information
        vision_start_idx = vision_analysis.get('vision_start_idx', 0) 
        vision_end_idx = vision_analysis.get('vision_end_idx', 0)
        vision_token_count = vision_analysis.get('vision_token_count', 0)
        
        if vision_token_count == 0 or vision_start_idx is None or vision_end_idx is None:
            print("Warning: No vision tokens found or vision analysis incomplete. Skipping patch visualization.")
            return
            
        # Get kept_indices from the specified model layer
        kept_indices = None
        
        
        # Iterate through model layers to find the one with kv_cluster
        for id_,layer in enumerate(model.model.language_model.layers):
            
            
            if id_ == layer_id:
                kept_indices = layer.self_attn.kept_indices
                print(f"Found kept_indices in shape: {kept_indices.shape if kept_indices is not None else 'None'}")
                break
            
    
        if kept_indices is None:
            print("Warning: No kept_indices found in PyramidKV layers. Skipping patch visualization.")
            return
            
        # Convert kept_indices to CPU and flatten if needed
        if hasattr(kept_indices, 'cpu'):
            kept_indices = kept_indices.cpu()
        if len(kept_indices.shape) > 1:
            # Take the first batch and first head if multi-dimensional
            kept_indices = kept_indices[0, 0] if kept_indices.shape[0] > 0 else kept_indices.flatten()
            
        # Calculate image dimensions and patch layout
        # Qwen2.5-VL uses 28x28 pixel patches according to the documentation
        PATCH_SIZE = 28
        img_width, img_height = image.size
        resized_img_height, resized_img_width = smart_resize(img_height, img_width, factor=IMAGE_FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        # resize the image to img_width, img_height
        # Resize the image to the computed dimensions
        print("Before resize: ", img_width, img_height)
        print("After resize: ", resized_img_width, resized_img_height)
        image = image.resize((resized_img_width, resized_img_height), Image.LANCZOS)
        
        img_width, img_height = image.size
        
        # Calculate number of patches in each dimension
        patches_w = img_width // PATCH_SIZE
        patches_h = img_height // PATCH_SIZE
        total_patches = patches_w * patches_h
        
        print(f"Image size: {img_width}x{img_height}")
        print(f"Patch layout: {patches_h}h x {patches_w}w = {total_patches} patches")
        print(f"Vision tokens: {vision_token_count}, Kept indices count: {len(kept_indices)}")
        
        # Create a mask for removed patches
        # Initialize all patches as removed (True means removed)
        removed_patches_mask = np.ones((patches_h, patches_w), dtype=bool)
        
        # Mark kept patches as False (not removed)
        if len(kept_indices) > 0:
            # kept_indices contains token positions from the beginning of the entire prompt
            # We need to filter for vision tokens and adjust indices accordingly
            for global_token_idx in kept_indices:
                # Check if this token index falls within the vision token range
                if vision_start_idx <= global_token_idx < vision_end_idx:
                    # Convert to vision-relative index (0-based within vision tokens)
                    vision_token_idx = global_token_idx - vision_start_idx
                    
                    # Convert vision token index to patch coordinates
                    if 0 <= vision_token_idx < total_patches:
                        patch_y = vision_token_idx // patches_w
                        patch_x = vision_token_idx % patches_w
                        if 0 <= patch_y < patches_h and 0 <= patch_x < patches_w:
                            removed_patches_mask[patch_y, patch_x] = False
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Show original image
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=14)
        ax1.axis('off')
        
        # Create overlay image
        overlay_image = image.copy()
        overlay_array = np.array(overlay_image)
        
        # Add red overlay to removed patches
        for patch_y in range(patches_h):
            for patch_x in range(patches_w):
                if removed_patches_mask[patch_y, patch_x]:  # This patch was removed
                    # Calculate pixel coordinates
                    y_start = patch_y * PATCH_SIZE
                    y_end = min(y_start + PATCH_SIZE, img_height)
                    x_start = patch_x * PATCH_SIZE  
                    x_end = min(x_start + PATCH_SIZE, img_width)
                    
                    # Apply completely opaque red overlay
                    overlay_array[y_start:y_end, x_start:x_end, 0] = 0  # Full black
                    overlay_array[y_start:y_end, x_start:x_end, 1] = 0    # No green
                    overlay_array[y_start:y_end, x_start:x_end, 2] = 0    # No blue
        
        # Show image with removed patches highlighted
        ax2.imshow(overlay_array)
        ax2.set_title(f'PyramidKV Removed Patches (Layer {layer_id})\nBlack = Removed, Original = Kept', fontsize=14)
        ax2.axis('off')
        
        # Add statistics
        removed_count = np.sum(removed_patches_mask)
        kept_count = total_patches - removed_count
        compression_ratio = (removed_count / total_patches) * 100 if total_patches > 0 else 0
        
        fig.suptitle(f'PyramidKV Patch Compression Analysis\n'
                    f'Total: {total_patches} patches, Kept: {kept_count}, Removed: {removed_count}\n'
                    f'Compression: {compression_ratio:.1f}% removed', 
                    fontsize=16)
        
        plt.tight_layout()
        
        # Save the visualization
        if save_dir:
            os.makedirs(os.path.join(save_dir, f"sample-{sample_id}"), exist_ok=True)
            save_path = os.path.join(save_dir, f"sample-{sample_id}", f"pyramidkv_removed_patches_layer{layer_id}.jpg")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"PyramidKV patch visualization saved to {save_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"Error in visualize_pyramidkv_removed_patches: {e}")
        import traceback
        traceback.print_exc()

def visualize_vlcache_removed_patches(image, inputs, model, vision_analysis, save_dir="", sample_id=0, layer_id=0):
    """
    Visualize image patches that are removed by VLCache compression.
    Overlays removed patches with blue color on the original image.
    
    Args:
        image: PIL Image - the original input image
        model: The model containing VLCache layers
        vision_analysis: Dict containing vision token analysis
        save_dir: Directory to save visualization
        sample_id: Sample identifier for filename
        layer_id: Which layer's kept_indices to use for visualization
    """
    try:
        import numpy as np
        from PIL import Image, ImageDraw
        import matplotlib.pyplot as plt
        import os
        
        # Get the vision token information
        vision_start_idx = vision_analysis.get('vision_start_idx', 0) 
        vision_end_idx = vision_analysis.get('vision_end_idx', 0)
        vision_token_count = vision_analysis.get('vision_token_count', 0)
        
        
        if vision_token_count == 0 or vision_start_idx is None or vision_end_idx is None:
            print("Warning: No vision tokens found or vision analysis incomplete. Skipping VLCache patch visualization.")
            return
            
        # Get kept_indices from the specified model layer
        kept_indices = None
        layer_count = 0
        layer_budget = None
        last_vision_indices = None
        
        # Get kept_indices from the specified model layer
        kept_indices = None
        
        
        # Iterate through model layers to find the one with kv_cluster
        for id_,layer in enumerate(model.model.language_model.layers):
            
            
            if id_ == layer_id:
                kept_indices = layer.self_attn.kept_indices
                print(f"Found kept_indices in shape: {kept_indices.shape if kept_indices is not None else 'None'}")
                break
            
    
        if kept_indices is None:
            print("Warning: No kept_indices found in VLCache layers. Skipping patch visualization.")
            return
            
        # Convert kept_indices to CPU and flatten if needed
        if hasattr(kept_indices, 'cpu'):
            kept_indices = kept_indices.cpu()
        if len(kept_indices.shape) > 1:
            # Take the first batch and first head if multi-dimensional
            kept_indices = kept_indices[0, 0] if kept_indices.shape[0] > 0 else kept_indices.flatten()
            
        # Calculate image dimensions and patch layout
        # Qwen2.5-VL uses 28x28 pixel patches according to the documentation
        PATCH_SIZE = 28
        img_width, img_height = image.size
        
        resized_img_height, resized_img_width = smart_resize(img_height, img_width, factor=IMAGE_FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        # resize the image to img_width, img_height
        # Resize the image to the computed dimensions
        print("Before resize: ", img_width, img_height)
        print("After resize: ", resized_img_width, resized_img_height)
        image = image.resize((resized_img_width, resized_img_height), Image.LANCZOS)
        
        img_width, img_height = image.size
        
        # Calculate number of patches in each dimension
        patches_w = img_width // PATCH_SIZE
        patches_h = img_height // PATCH_SIZE
        total_patches = patches_w * patches_h
        
        print(f"Image size: {img_width}x{img_height}")
        print(f"Patch layout: {patches_h}h x {patches_w}w = {total_patches} patches")
        print(f"Vision tokens: {vision_token_count}, Kept indices count: {len(kept_indices)}")
        
        # Create a mask for removed patches
        # Initialize all patches as removed (True means removed)
        removed_patches_mask = np.ones((patches_h, patches_w), dtype=bool)
        
        # Mark kept patches as False (not removed)
        if len(kept_indices) > 0:
            # kept_indices contains token positions from the beginning of the entire prompt
            # We need to filter for vision tokens and adjust indices accordingly
            for global_token_idx in kept_indices:
                # Check if this token index falls within the vision token range
                if vision_start_idx <= global_token_idx < vision_end_idx:
                    # Convert to vision-relative index (0-based within vision tokens)
                    vision_token_idx = global_token_idx - vision_start_idx
                    
                    # Convert vision token index to patch coordinates
                    if 0 <= vision_token_idx < total_patches:
                        patch_y = vision_token_idx // patches_w
                        patch_x = vision_token_idx % patches_w
                        if 0 <= patch_y < patches_h and 0 <= patch_x < patches_w:
                            removed_patches_mask[patch_y, patch_x] = False
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Show original image
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=14)
        ax1.axis('off')
        
        # Create overlay image
        overlay_image = image.copy()
        overlay_array = np.array(overlay_image)
        
        # Add blue overlay to removed patches (different color from PyramidKV)
        for patch_y in range(patches_h):
            for patch_x in range(patches_w):
                if removed_patches_mask[patch_y, patch_x]:  # This patch was removed
                    # Calculate pixel coordinates
                    y_start = patch_y * PATCH_SIZE
                    y_end = min(y_start + PATCH_SIZE, img_height)
                    x_start = patch_x * PATCH_SIZE  
                    x_end = min(x_start + PATCH_SIZE, img_width)
                    
                    # Apply completely opaque red overlay
                    overlay_array[y_start:y_end, x_start:x_end, 0] = 0  # Full black
                    overlay_array[y_start:y_end, x_start:x_end, 1] = 0    # No green
                    overlay_array[y_start:y_end, x_start:x_end, 2] = 0    # No blue
        
        # Show image with removed patches highlighted
        ax2.imshow(overlay_array)
        ax2.set_title(f'VLCache Removed Patches (Layer {layer_id})\Black = Removed, Original = Kept', fontsize=14)
        ax2.axis('off')
        
        # Add statistics
        removed_count = np.sum(removed_patches_mask)
        kept_count = total_patches - removed_count
        compression_ratio = (removed_count / total_patches) * 100 if total_patches > 0 else 0
        
        budget_info = f"Budget: {layer_budget}" if layer_budget is not None else "Budget: Unknown"
        vision_info = f"Last Vision Idx: {last_vision_indices}" if last_vision_indices is not None else ""
        
        fig.suptitle(f'VLCache Patch Compression Analysis\n'
                    f'Total: {total_patches} patches, Kept: {kept_count}, Removed: {removed_count}\n'
                    f'Compression: {compression_ratio:.1f}% removed | {budget_info} | {vision_info}', 
                    fontsize=16)
        
        plt.tight_layout()
        
        # Save the visualization
        if save_dir:
            os.makedirs(os.path.join(save_dir, f"sample-{sample_id}"), exist_ok=True)
            save_path = os.path.join(save_dir, f"sample-{sample_id}", f"vlcache_removed_patches_layer{layer_id}.jpg")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"VLCache patch visualization saved to {save_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"Error in visualize_vlcache_removed_patches: {e}")
        import traceback
        traceback.print_exc()

def visualize_snapkv_removed_patches(image, model, vision_analysis, save_dir="", sample_id=0, layer_id=0):
    """
    Visualize image patches that are removed by SnapKV compression.
    Overlays removed patches with green color on the original image.
    
    Args:
        image: PIL Image - the original input image
        model: The model containing SnapKV layers
        vision_analysis: Dict containing vision token analysis
        save_dir: Directory to save visualization
        sample_id: Sample identifier for filename
        layer_id: Which layer's kept_indices to use for visualization
    """
    try:
        import numpy as np
        from PIL import Image, ImageDraw
        import matplotlib.pyplot as plt
        import os
        
        # Get the vision token information
        vision_start_idx = vision_analysis.get('vision_start_idx', 0) 
        vision_end_idx = vision_analysis.get('vision_end_idx', 0)
        vision_token_count = vision_analysis.get('vision_token_count', 0)
        
        if vision_token_count == 0 or vision_start_idx is None or vision_end_idx is None:
            print("Warning: No vision tokens found or vision analysis incomplete. Skipping SnapKV patch visualization.")
            return
            
        # Get kept_indices from the specified model layer
        kept_indices = None
        
        
        
        # Iterate through model layers to find the one with kv_cluster
        for id_,layer in enumerate(model.model.language_model.layers):
            
            
            if id_ == layer_id:
                kept_indices = layer.self_attn.kept_indices
                print(f"Found kept_indices in shape: {kept_indices.shape if kept_indices is not None else 'None'}")
                break
        
        if kept_indices is None:
            print("Warning: No kept_indices found in SnapKV layers. Skipping patch visualization.")
            return
            
        # Convert kept_indices to CPU and flatten if needed
        if hasattr(kept_indices, 'cpu'):
            kept_indices = kept_indices.cpu()
        if len(kept_indices.shape) > 1:
            # Take the first batch and first head if multi-dimensional
            kept_indices = kept_indices[0, 0] if kept_indices.shape[0] > 0 else kept_indices.flatten()
            
        # Calculate image dimensions and patch layout
        # Qwen2.5-VL uses 28x28 pixel patches according to the documentation
        PATCH_SIZE = 28
        
        img_width, img_height = image.size
        resized_img_height, resized_img_width = smart_resize(img_height, img_width, factor=IMAGE_FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        # resize the image to img_width, img_height
        # Resize the image to the computed dimensions
        print("Before resize: ", img_width, img_height)
        print("After resize: ", resized_img_width, resized_img_height)
        image = image.resize((resized_img_width, resized_img_height), Image.LANCZOS)
        
        img_width, img_height = image.size
        # Calculate number of patches in each dimension
        patches_w = img_width // PATCH_SIZE
        patches_h = img_height // PATCH_SIZE
        total_patches = patches_w * patches_h
        
        print(f"Image size: {img_width}x{img_height}")
        print(f"Patch layout: {patches_h}h x {patches_w}w = {total_patches} patches")
        print(f"Vision tokens: {vision_token_count}, Kept indices count: {len(kept_indices)}")
        
        # Create a mask for removed patches
        # Initialize all patches as removed (True means removed)
        removed_patches_mask = np.ones((patches_h, patches_w), dtype=bool)
        
        # Mark kept patches as False (not removed)
        if len(kept_indices) > 0:
            # kept_indices contains token positions from the beginning of the entire prompt
            # We need to filter for vision tokens and adjust indices accordingly
            for global_token_idx in kept_indices:
                # Check if this token index falls within the vision token range
                if vision_start_idx <= global_token_idx < vision_end_idx:
                    # Convert to vision-relative index (0-based within vision tokens)
                    vision_token_idx = global_token_idx - vision_start_idx
                    
                    # Convert vision token index to patch coordinates
                    if 0 <= vision_token_idx < total_patches:
                        patch_y = vision_token_idx // patches_w
                        patch_x = vision_token_idx % patches_w
                        if 0 <= patch_y < patches_h and 0 <= patch_x < patches_w:
                            removed_patches_mask[patch_y, patch_x] = False
                        else:
                            print(f"Token index {vision_token_idx} is out of range for image size {img_width}x{img_height} with {total_patches} patches")
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Show original image
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=14)
        ax1.axis('off')
        
        # Create overlay image
        overlay_image = image.copy()
        overlay_array = np.array(overlay_image)
        
        # Add green overlay to removed patches (different color from PyramidKV and VLCache)
        for patch_y in range(patches_h):
            for patch_x in range(patches_w):
                if removed_patches_mask[patch_y, patch_x]:  # This patch was removed
                    # Calculate pixel coordinates
                    y_start = patch_y * PATCH_SIZE
                    y_end = min(y_start + PATCH_SIZE, img_height)
                    x_start = patch_x * PATCH_SIZE  
                    x_end = min(x_start + PATCH_SIZE, img_width)
                    
                    # Apply completely opaque red overlay
                    overlay_array[y_start:y_end, x_start:x_end, 0] = 0  # Full black
                    overlay_array[y_start:y_end, x_start:x_end, 1] = 0    # No green
                    overlay_array[y_start:y_end, x_start:x_end, 2] = 0    # No blue
        
        # Show image with removed patches highlighted
        ax2.imshow(overlay_array)
        ax2.set_title(f'SnapKV Removed Patches (Layer {layer_id})\Black = Removed, Original = Kept', fontsize=14)
        ax2.axis('off')
        
        # Add statistics
        removed_count = np.sum(removed_patches_mask)
        kept_count = total_patches - removed_count
        compression_ratio = (removed_count / total_patches) * 100 if total_patches > 0 else 0
        
        
        
        fig.suptitle(f'SnapKV Patch Compression Analysis\n'
                    f'Total: {total_patches} patches, Kept: {kept_count}, Removed: {removed_count}\n'
                    f'Compression: {compression_ratio:.1f}% removed', 
                    fontsize=16)
        
        plt.tight_layout()
        
        # Save the visualization
        if save_dir:
            os.makedirs(os.path.join(save_dir, f"sample-{sample_id}"), exist_ok=True)
            save_path = os.path.join(save_dir, f"sample-{sample_id}", f"snapkv_removed_patches_layer{layer_id}.jpg")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"SnapKV patch visualization saved to {save_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"Error in visualize_snapkv_removed_patches: {e}")
        import traceback
        traceback.print_exc()


def visualize_gui_kv_removed_patches(image, model, vision_analysis, save_dir="", sample_id=0, layer_id=0):
    """
    Visualize image patches that are removed by SnapKV compression.
    Overlays removed patches with green color on the original image.
    
    Args:
        image: PIL Image - the original input image
        model: The model containing SnapKV layers
        vision_analysis: Dict containing vision token analysis
        save_dir: Directory to save visualization
        sample_id: Sample identifier for filename
        layer_id: Which layer's kept_indices to use for visualization
    """
    try:
        import numpy as np
        from PIL import Image, ImageDraw
        import matplotlib.pyplot as plt
        import os
        
        # Get the vision token information
        vision_start_idx = vision_analysis.get('vision_start_idx', 0) 
        vision_end_idx = vision_analysis.get('vision_end_idx', 0)
        vision_token_count = vision_analysis.get('vision_token_count', 0)
        
        if vision_token_count == 0 or vision_start_idx is None or vision_end_idx is None:
            print("Warning: No vision tokens found or vision analysis incomplete. Skipping SnapKV patch visualization.")
            return
            
        # Get kept_indices from the specified model layer
        kept_indices = None
        
        
        
        # Iterate through model layers to find the one with kv_cluster
        for id_,layer in enumerate(model.model.language_model.layers):
            
            
            if id_ == layer_id:
                kept_indices = layer.self_attn.kept_indices
                print(f"Found kept_indices in shape: {kept_indices.shape if kept_indices is not None else 'None'}")
                break
        
        if kept_indices is None:
            print("Warning: No kept_indices found in SnapKV layers. Skipping patch visualization.")
            return
            
        # Convert kept_indices to CPU and flatten if needed
        if hasattr(kept_indices, 'cpu'):
            kept_indices = kept_indices.cpu()
        if len(kept_indices.shape) > 1:
            # Take the first batch and first head if multi-dimensional
            kept_indices = kept_indices[0, 0] if kept_indices.shape[0] > 0 else kept_indices.flatten()
            
        # Calculate image dimensions and patch layout
        # Qwen2.5-VL uses 28x28 pixel patches according to the documentation
        PATCH_SIZE = 28
        
        img_width, img_height = image.size
        resized_img_height, resized_img_width = smart_resize(img_height, img_width, factor=IMAGE_FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        # resize the image to img_width, img_height
        # Resize the image to the computed dimensions
        print("Before resize: ", img_width, img_height)
        print("After resize: ", resized_img_width, resized_img_height)
        image = image.resize((resized_img_width, resized_img_height), Image.LANCZOS)
        
        img_width, img_height = image.size
        # Calculate number of patches in each dimension
        patches_w = img_width // PATCH_SIZE
        patches_h = img_height // PATCH_SIZE
        total_patches = patches_w * patches_h
        
        print(f"Image size: {img_width}x{img_height}")
        print(f"Patch layout: {patches_h}h x {patches_w}w = {total_patches} patches")
        print(f"Vision tokens: {vision_token_count}, Kept indices count: {len(kept_indices)}")
        
        # Create a mask for removed patches
        # Initialize all patches as removed (True means removed)
        removed_patches_mask = np.ones((patches_h, patches_w), dtype=bool)
        
        # Mark kept patches as False (not removed)
        if len(kept_indices) > 0:
            # kept_indices contains token positions from the beginning of the entire prompt
            # We need to filter for vision tokens and adjust indices accordingly
            for global_token_idx in kept_indices:
                # Check if this token index falls within the vision token range
                if vision_start_idx <= global_token_idx < vision_end_idx:
                    # Convert to vision-relative index (0-based within vision tokens)
                    vision_token_idx = global_token_idx - vision_start_idx
                    
                    # Convert vision token index to patch coordinates
                    if 0 <= vision_token_idx < total_patches:
                        patch_y = vision_token_idx // patches_w
                        patch_x = vision_token_idx % patches_w
                        if 0 <= patch_y < patches_h and 0 <= patch_x < patches_w:
                            removed_patches_mask[patch_y, patch_x] = False
                        else:
                            print(f"Token index {vision_token_idx} is out of range for image size {img_width}x{img_height} with {total_patches} patches")
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Show original image
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=14)
        ax1.axis('off')
        
        # Create overlay image
        overlay_image = image.copy()
        overlay_array = np.array(overlay_image)
        
        # Add green overlay to removed patches (different color from PyramidKV and VLCache)
        for patch_y in range(patches_h):
            for patch_x in range(patches_w):
                if removed_patches_mask[patch_y, patch_x]:  # This patch was removed
                    # Calculate pixel coordinates
                    y_start = patch_y * PATCH_SIZE
                    y_end = min(y_start + PATCH_SIZE, img_height)
                    x_start = patch_x * PATCH_SIZE  
                    x_end = min(x_start + PATCH_SIZE, img_width)
                    
                    # Apply completely opaque red overlay
                    overlay_array[y_start:y_end, x_start:x_end, 0] = 0  # Full black
                    overlay_array[y_start:y_end, x_start:x_end, 1] = 0    # No green
                    overlay_array[y_start:y_end, x_start:x_end, 2] = 0    # No blue
        
        # Show image with removed patches highlighted
        ax2.imshow(overlay_array)
        ax2.set_title(f'SnapKV Removed Patches (Layer {layer_id})\Black = Removed, Original = Kept', fontsize=14)
        ax2.axis('off')
        
        # Add statistics
        removed_count = np.sum(removed_patches_mask)
        kept_count = total_patches - removed_count
        compression_ratio = (removed_count / total_patches) * 100 if total_patches > 0 else 0
        
        
        
        fig.suptitle(f'SnapKV Patch Compression Analysis\n'
                    f'Total: {total_patches} patches, Kept: {kept_count}, Removed: {removed_count}\n'
                    f'Compression: {compression_ratio:.1f}% removed', 
                    fontsize=16)
        
        plt.tight_layout()
        
        # Save the visualization
        if save_dir:
            os.makedirs(os.path.join(save_dir, f"sample-{sample_id}"), exist_ok=True)
            save_path = os.path.join(save_dir, f"sample-{sample_id}", f"guikv_removed_patches_layer{layer_id}.jpg")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"SnapKV patch visualization saved to {save_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"Error in visualize_snapkv_removed_patches: {e}")
        import traceback
        traceback.print_exc()
        
        
def visualize_token_information_scores(image, model, save_dir="", sample_id=0):
    """
    Visualize token information scores for image patches.
    Shows a heatmap of information scores overlaid on the original image.
    
    Args:
        image: PIL Image - the original input image
        model: The model containing token information scores
        vision_analysis: Dict containing vision token analysis
        save_dir: Directory to save visualization
        sample_id: Sample identifier for filename
        layer_id: Layer identifier for filename (not used for scores but kept for consistency)
    """
    try:
        import numpy as np
        from PIL import Image, ImageDraw
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import os
        
      
            
        # token_information_scores is a 1-d tensor, each value correspond to each image patch
        token_information_scores = model.model.language_model.layers[0].self_attn.config.token_information_scores
        
        if token_information_scores is None:
            print("Warning: No token_information_scores found. Skipping visualization.")
            return
            
        # Convert to CPU numpy array
        if hasattr(token_information_scores, 'cpu'):
            scores_array = token_information_scores.cpu().numpy()
        else:
            scores_array = np.array(token_information_scores)
            
        # Calculate image dimensions and patch layout
        # Qwen2.5-VL uses 28x28 pixel patches according to the documentation
        PATCH_SIZE = 28
        
        img_width, img_height = image.size
        resized_img_height, resized_img_width = smart_resize(img_height, img_width, factor=IMAGE_FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        
        print("Before resize: ", img_width, img_height)
        print("After resize: ", resized_img_width, resized_img_height)
        image = image.resize((resized_img_width, resized_img_height), Image.LANCZOS)
        
        img_width, img_height = image.size
        # Calculate number of patches in each dimension
        patches_w = img_width // PATCH_SIZE
        patches_h = img_height // PATCH_SIZE
        total_patches = patches_w * patches_h
        
        print(f"Image size: {img_width}x{img_height}")
        print(f"Patch layout: {patches_h}h x {patches_w}w = {total_patches} patches")
        
        
        # Create a 2D array to hold information scores for visualization
        scores_2d = np.zeros((patches_h, patches_w))
        
        # Fill the 2D array with information scores
        if len(scores_array) > 0:
            
            
            for i, score in enumerate(scores_array):
                if i < total_patches:
                    patch_y = i // patches_w
                    patch_x = i % patches_w
                    if 0 <= patch_y < patches_h and 0 <= patch_x < patches_w:
                        scores_2d[patch_y, patch_x] = score
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Show original image
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=14)
        ax1.axis('off')
        
        # Show heatmap of information scores
        im = ax2.imshow(scores_2d, cmap='viridis', aspect='equal')
        ax2.set_title('Token Information Scores Heatmap', fontsize=14)
        ax2.set_xlabel('Patch X')
        ax2.set_ylabel('Patch Y')
        
        # Add colorbar without affecting subplot size
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Information Score', rotation=270, labelpad=15)
        
        # Add statistics
        min_score = scores_2d.min() if scores_2d.size > 0 else 0
        max_score = scores_2d.max() if scores_2d.size > 0 else 0
        mean_score = scores_2d.mean() if scores_2d.size > 0 else 0
        
        fig.suptitle(f'Token Information Scores Visualization\n'
                    f'Total: {total_patches} patches, Score range: [{min_score:.3f}, {max_score:.3f}], Mean: {mean_score:.3f}', 
                    fontsize=16)
        
        plt.tight_layout()
        
        # Save the visualization
        if save_dir:
            os.makedirs(os.path.join(save_dir, f"sample-{sample_id}"), exist_ok=True)
            save_path = os.path.join(save_dir, f"sample-{sample_id}", f"token_information_scores.jpg")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Token information scores visualization saved to {save_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"Error in visualize_token_information_scores: {e}")
        import traceback
        traceback.print_exc()