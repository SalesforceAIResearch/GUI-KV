import re
import os
import math
import base64
from io import BytesIO
from PIL import Image, ImageDraw
from typing import Optional, Dict, Any, List, Tuple


def analyze_vision_tokens_opencua_multi_images(tokenizer, input_ids, image_grid_thw, merge_size=2, image_count=1):
    """
    Analyze vision token lengths and positions for multiple images in the input sequence.

    Args:
        tokenizer: The tokenizer
        input_ids: Token IDs
        image_grid_thw: Tensor of shape [num_images, 3] where 3 = (T, H, W)
        merge_size: Merge size for visual tokens (default: 2)
        image_count: Expected number of images

    Returns:
        dict: Contains vision token analysis information with lists of start/end indices
    """
    full_tokens = tokenizer.convert_ids_to_tokens(input_ids[0] if len(input_ids.shape) > 1 else input_ids)

    if len(image_grid_thw.shape) == 3:
        image_grid_thw = image_grid_thw[0]
    elif len(image_grid_thw.shape) == 1:
        image_grid_thw = image_grid_thw.unsqueeze(0)

    vision_start_indices = []
    vision_end_indices = []

    media_begin_positions = []
    media_end_positions = []
    
    for i, token in enumerate(full_tokens):
        if '<|media_begin|>' in token:
            media_begin_positions.append(i)
        elif '<|media_end|>' in token:
            media_end_positions.append(i)

    for img_idx in range(min(len(media_begin_positions), len(image_grid_thw))):
        _, patch_h, patch_w = image_grid_thw[img_idx]

        num_visual_tokens = int(patch_h * patch_w / (merge_size ** 2))

        if img_idx == 0:
            vision_start_idx = media_begin_positions[0] + 1
        else:
            prev_vision_end = vision_end_indices[-1]
            prev_media_end = media_end_positions[img_idx - 1]
            curr_media_begin = media_begin_positions[img_idx]

            text_tokens_between = curr_media_begin - prev_media_end - 1

            vision_start_idx = prev_vision_end + text_tokens_between + 1
        
        vision_end_idx = vision_start_idx + num_visual_tokens

        vision_start_indices.append(vision_start_idx)
        vision_end_indices.append(vision_end_idx)

    assert len(vision_start_indices) == image_count, f"Expected {image_count} images but found {len(vision_start_indices)} <|media_begin|> markers"

    analysis = {
        'vision_start_idx': vision_start_indices,
        'vision_end_idx': vision_end_indices,
    }

    return analysis

def opencua_parse_action(code, origin_resized_height, origin_resized_width,
                    max_pixels, min_pixels, factor, model_type):
    """
    Convert pyautogui code to normalized coordinates.
    Example: pyautogui.click(x=1424, y=264)
    """
    if model_type == "qwen25vl":
        smart_resize_height, smart_resize_width = smart_resize(
                origin_resized_height,
                origin_resized_width,
                factor=factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels)

    coordinates = parse_coordinates_from_code(code)
    actions = []
    for coordinate in coordinates:
        x, y = coordinate
        if  model_type == "qwen25vl":
            x = float(x / smart_resize_width)
            y = float(y / smart_resize_height)
        else:
            x = float(x / factor)
            y = float(y / factor)
        actions.append({
            "action_type": "click",
            "coordinate": [x, y],
            "text": code
        })
    return actions
    

def clean_invalid_json_escapes(s: str) -> str:
    if s is None:
        return None
    s = re.sub(r"^```json\s*|\s*```$", "", s.strip())
    s = s.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
    s = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', s)
    return s

def draw_coords(image, x, y):
    width, height = image.size
    absolute_x = int(x * width)
    absolute_y = int(y * height)

    draw = ImageDraw.Draw(image)
    circle_size = 40

    draw.ellipse(
        [
            absolute_x - circle_size // 2,
            absolute_y - circle_size // 2,
            absolute_x + circle_size // 2,
            absolute_y + circle_size // 2
        ],
        outline="red",
        width=3
    )

    point_radius = 2
    draw.ellipse(
        [
            absolute_x - point_radius,
            absolute_y - point_radius,
            absolute_x + point_radius,
            absolute_y + point_radius
        ],
        fill="red"  
    )
    return image

def parse_coordinates_from_line(line, max_num = 2):
    if not line:
        return None

    if line.startswith((
        "pyautogui.click", 
        "pyautogui.moveTo", 
        "pyautogui.dragTo",
        "pyautogui.doubleClick", 
        "pyautogui.rightClick", 
        "pyautogui.middleClick", 
        "pyautogui.tripleClick",
        "computer.tripleClick",
    )):
        numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
        floats = [float(n) for n in numbers][:max_num]
        return tuple(floats)

    return None

def parse_coordinates_from_code(code, max_num = 2):
    if not code:
        return None

    all_coords = []
    code_lines = code.split("\n")
    for line in code_lines:
        line = line.strip()
        if not line:
            continue
        
        coords = parse_coordinates_from_line(line)
        if coords:
            x, y = coords
            all_coords.append((x, y))
            
    return all_coords
    

def draw_coords_from_code(image, code):
    coords = parse_coordinates_from_code(code)
    print(coords)
    if coords:
        for x, y in coords:
            image = draw_coords(image, x, y)
    return image




def load_image(image_name, image_folder=None):
    if image_folder is None:
        image_path = image_name
    else:
        image_path = os.path.join(image_folder, image_name)
    with open(image_path, 'rb') as f:
        image_data = f.read()
    image = Image.open(BytesIO(image_data))
    return image

def image_to_base64(pil_image):
    image_bytes = BytesIO()
    pil_image.save(image_bytes, format="JPEG")
    image_bytes = image_bytes.getvalue()  # 获取字节数据

    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    return image_base64

def crop_image_patch(image: Image.Image, x: float, y: float, patch_size: int = 300, target_size: int = 300) -> Image.Image:
    """
    Crop a square patch centered at coordinates (x,y) and resize based on patch_size and target_size ratio.

    Args:
        image: PIL Image
        x: Pixel x coordinate
        y: Pixel y coordinate
        patch_size: Size of square patch to crop from original image (default 200)
        target_size: Maximum dimension for resize (default 500)

    Returns:
        PIL Image of cropped and resized patch with original aspect ratio
    """
    width, height = image.size

    center_x = int(x * width)
    center_y = int(y * height)

    half_size = patch_size // 2
    left = max(0, center_x - half_size)
    top = max(0, center_y - half_size)
    right = min(image.width, center_x + half_size)
    bottom = min(image.height, center_y + half_size)

    patch = image.crop((left, top, right, bottom))

    scaling_factor = target_size / patch_size

    w, h = patch.size
    new_w = int(w * scaling_factor)
    new_h = int(h * scaling_factor)
    patch = patch.resize((new_w, new_h))

    return patch

def draw_bounding_box(image, center_x, center_y):
    width, height = image.size
    absolute_x = int(center_x * width)
    absolute_y = int(center_y * height)

    draw = ImageDraw.Draw(image)
    circle_size = 40

    draw.ellipse(
        [
            absolute_x - circle_size // 2,
            absolute_y - circle_size // 2,
            absolute_x + circle_size // 2,
            absolute_y + circle_size // 2
        ],
        outline="red",
        width=3
    )

    point_radius = 2
    draw.ellipse(
        [
            absolute_x - point_radius,
            absolute_y - point_radius,
            absolute_x + point_radius,
            absolute_y + point_radius
        ],
        fill="red" 
    )

    return image

def draw_bounding_box_and_crop_patch(image, code):
    image_patch = None
    code_lines = code.split("\n")
    for line in code_lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith((
            "pyautogui.click", 
            "pyautogui.moveTo", 
            "pyautogui.dragTo",
            "pyautogui.doubleClick", 
            "pyautogui.rightClick", 
            "pyautogui.middleClick", 
            "pyautogui.tripleClick",
            "computer.tripleClick",
            )):

            match = re.search(r"x\s*=\s*([0-9.]+).*?y\s*=\s*([0-9.]+)", line)
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
                if 0 <= x <= 1 and 0 <= y <= 1:
                    image = draw_bounding_box(image, x, y)
                    if not image_patch:
                        image_patch = crop_image_patch(image, x, y)

    return image, image_patch

def smart_resize(
    height: int,
    width: int,
    factor=28, 
    min_pixels=3136, 
    max_pixels=12845056,
    max_aspect_ratio_allowed: float | None = None,
    size_can_be_smaller_than_factor: bool = False,
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if not size_can_be_smaller_than_factor and (height < factor or width < factor):
        raise ValueError(
            f"height:{height} or width:{width} must be larger than factor:{factor} "
            f"(when size_can_be_smaller_than_factor is False)"
        )
    elif max_aspect_ratio_allowed is not None and max(height, width) / min(height, width) > max_aspect_ratio_allowed:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {max_aspect_ratio_allowed}, "
            f"got {max(height, width) / min(height, width)}"
            f"(when max_aspect_ratio_allowed is not None)"
        )
    h_bar = max(1, round(height / factor)) * factor
    w_bar = max(1, round(width / factor)) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(1, math.floor(height / beta / factor)) * factor
        w_bar = max(1, math.floor(width / beta / factor)) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def convert_code_relative_to_absolute(code: str, image: Image.Image) -> tuple[str, tuple[int, int], list[tuple[float, float]]]:
    """
        Convert relative coordinates to absolute coordinates.
    """
    original_w, original_h = image.size
    resized_h, resized_w = smart_resize(original_h, original_w)

    lines = code.split("\n")
    new_lines = []
    all_coords = []

    for line in lines:
        line = line.strip()
        if not line:
            new_lines.append(line)
            continue

        coords = parse_coordinates_from_line(line)
        if coords and len(coords) == 2:
            x_rel, y_rel = coords
            all_coords.append((x_rel, y_rel))
            x_abs = int(x_rel * resized_w)
            y_abs = int(y_rel * resized_h)

            if "x=" in line and "y=" in line:
                line = re.sub(r"x\s*=\s*([-+]?\d*\.\d+|[-+]?\d+)", f"x={x_abs}", line)
                line = re.sub(r"y\s*=\s*([-+]?\d*\.\d+|[-+]?\d+)", f"y={y_abs}", line)
            else:
                line = re.sub(
                    r"([-+]?\d*\.\d+|[-+]?\d+)", 
                    lambda m, c=[x_abs, y_abs]: str(c.pop(0)) if c else m.group(0), 
                    line, 
                    count=2
                )

        new_lines.append(line)

    return "\n".join(new_lines), (resized_w, resized_h), all_coords


def convert_code_absolute_to_relative(
    code: str,
    image: Image.Image,
    coord_type: str = "qwen25"
) -> tuple[str, tuple[int, int], list[tuple[float, float]]]:
    """
    Convert different types of coordinate to relative coordinates.
    """
    orig_w, orig_h = image.size
    if coord_type == "qwen25":
        resized_h, resized_w = smart_resize(orig_h, orig_w)
    elif coord_type == "absolute":
        resized_h, resized_w = orig_h, orig_w
    else:
        raise ValueError(f"Unsupported coord_type: {coord_type}. Use 'qwen25' or 'absolute'.")

    def to_rel(val: int, denom: int) -> str:
        return f"{val / denom:.4f}".rstrip("0").rstrip(".")

    lines, new_lines, all_coords = code.split("\n"), [], []

    for line in lines:
        l_strip = line.strip()
        if not l_strip:
            new_lines.append(line)
            continue

        coords = parse_coordinates_from_line(l_strip)
        if coords and len(coords) == 2:
            x_abs, y_abs = coords
            x_rel, y_rel = x_abs / resized_w, y_abs / resized_h
            all_coords.append((x_rel, y_rel))

            if "x=" in l_strip and "y=" in l_strip:
                line = re.sub(r"x\s*=\s*[-+]?\d+(\.\d+)?", f"x={to_rel(x_abs, resized_w)}", line)
                line = re.sub(r"y\s*=\s*[-+]?\d+(\.\d+)?", f"y={to_rel(y_abs, resized_h)}", line)
            else:
                repl_vals = [to_rel(x_abs, resized_w), to_rel(y_abs, resized_h)]
                line = re.sub(
                    r"[-+]?\d+(\.\d+)?",
                    lambda m, c=repl_vals: (c.pop(0) if c else m.group(0)),
                    line,
                    count=2,
                )
        new_lines.append(line)

    return "\n".join(new_lines), (resized_w, resized_h), all_coords



def parse_response_actions_opencua(response: str, trajectory: Optional[Dict[str, Any]] = None, step_idx: Optional[int] = None) -> Optional[str]:
    """Parse model output, extracting pyautogui/computer lines.

    Also normalizes absolute pixel coordinates to relative if coord_type is 'qwen25'.
    """
    if response is None:
        return None

    lines = response.split("\n")
    action_lines: List[str] = []

    # First pass: lines that start with commands
    for raw in lines:
        line = raw.strip()
        if line.startswith("pyautogui.") or line.startswith("computer."):
            action_lines.append(line)

    # If we already have extracted lines, optionally normalize coordinates
    if action_lines:
        
        new_action_lines = []
        # for action in action_lines:
        return "\n".join(action_lines)

    # Second pass: find commands anywhere within lines
    
    return action_lines

def _maybe_normalize_coordinates(
    action_lines: List[str],
    trajectory: Optional[Dict[str, Any]],
    step_idx: Optional[int],
) -> List[str]:
    """Normalize x,y coordinates for qwen25 if needed, using current step image size.

    Converts pixel coordinates (from resized inputs) to relative [0,1] based on
    resized width/height computed with smart-resize.
    """
    if not action_lines or not trajectory or step_idx is None:
        return action_lines

    steps = trajectory.get("steps", [])
    if step_idx < 0 or step_idx >= len(steps):
        return action_lines

    image_file = steps[step_idx].get("image")
    if not image_file or not self.image_dir:
        return action_lines

    try:
        img_bytes = self.load_image(image_file, self.image_dir)
        img = Image.open(BytesIO(img_bytes))
        width, height = img.size
        new_w, new_h = self._smart_resize_qwen25(width, height)
    except Exception:
        return action_lines

    normalized: List[str] = []
    for line in action_lines:
        if not line.startswith("pyautogui."):
            normalized.append(line)
            continue

        # Find x and y
        m = re.search(r"x=([\d.]+),\s*y=([\d.]+)", line)
        if not m:
            normalized.append(line)
            continue

        try:
            x_val = float(m.group(1))
            y_val = float(m.group(2))
        except ValueError:
            normalized.append(line)
            continue

        # Only normalize if looks like pixel coordinates
        if x_val > 1.0 and y_val > 1.0:
            rel_x = x_val / float(new_w)
            rel_y = y_val / float(new_h)
            # Replace the first occurrence of the coordinate pair
            new_line = re.sub(r"x=([\d.]+),\s*y=([\d.]+)", f"x={rel_x}, y={rel_y}", line, count=1)
            normalized.append(new_line)
        else:
            normalized.append(line)

    return normalized

# ----------------------- Action extraction --------------------------
def extract_actions(action: str, origin_resized_height, origin_resized_width, 
                    max_pixels, min_pixels, factor, model_type) -> List[Tuple[str, Any]]:
    """Extract (type, value) tuples from parsed action string.

    Follows the logic of extract_actions() in opencua_eval_all_in_one.py.
    """
    
    if model_type == "qwen25vl":

        smart_resize_height, smart_resize_width = smart_resize(
                origin_resized_height,
                origin_resized_width,
                factor=factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels)
        
    if not action:
        return []

    actions: List[Tuple[str, Any]] = []

    action_lines = action.strip().split("\n")
    for raw in action_lines:
        line = raw.strip()

        # computer.terminate
        if line.startswith("computer.terminate"):
            status_match = re.search(r"status=['\"](\w+)['\"]", line)
            if status_match:
                actions.append(("terminate", status_match.group(1)))
                continue

        # computer.triple_click
        if line.startswith("computer.triple_click"):
            coord_match = re.search(r"x=([\d.]+),\s*y=([\d.]+)", line)
            if coord_match:
                x, y = map(float, coord_match.groups())
                if model_type == "qwen25vl":
                    x = float(x / smart_resize_width)
                    y = float(y / smart_resize_height)
                else:
                    x = float(x / factor)
                    y = float(y / factor)
                actions.append(("triple_click", (x, y)))
                continue

        # pyautogui.*
        if line.startswith("pyautogui."):
            coord_match = re.search(r"x=([\d.]+),\s*y=([\d.]+)", line)
            if coord_match:
                x, y = map(float, coord_match.groups())
                if model_type == "qwen25vl":
                    x = float(x / smart_resize_width)
                    y = float(y / smart_resize_height)
                else:
                    x = float(x / factor)
                    y = float(y / factor)
                if "click" in line and "doubleClick" not in line and "rightClick" not in line:
                    actions.append(("click", (x, y)))
                elif "moveTo" in line:
                    actions.append(("moveTo", (x, y)))
                elif "doubleClick" in line:
                    actions.append(("doubleClick", (x, y)))
                elif "rightClick" in line:
                    actions.append(("rightClick", (x, y)))
                elif "dragTo" in line:
                    actions.append(("dragTo", (x, y)))

            # write(message=...)
            write_match = re.search(r"message=['\"](.+?)['\"]", line)
            if write_match:
                text = write_match.group(1)
                actions.append(("write", text))

            # write('...') positional
            if not write_match:
                write_positional = re.search(r"pyautogui\.write\((['\"])(.*?)\1\)", line)
                if write_positional:
                    actions.append(("write", write_positional.group(2)))

            # press/hotkey keys=[...]
            keys_match = re.findall(r"keys=\[(.*?)\]", line)
            if keys_match:
                key_string = keys_match[0]
                key_list = re.findall(r"['\"]([^'\"]*)['\"]|(\w+)", key_string)
                keys = [m[0] or m[1] for m in key_list if m[0] or m[1]]
                normalized_keys: List[str] = []
                for k in keys:
                    k = k.strip()
                    normalized_keys.append("ctrl" if k.lower() in ("cmd", "command") else k)
                if "hotkey" in line:
                    actions.append(("hotkey", normalized_keys))
                else:
                    actions.append(("press", normalized_keys))

            # hotkey positional: pyautogui.hotkey('ctrl', 'v')
            if "hotkey(" in line and "keys=" not in line:
                inside = re.search(r"pyautogui\.hotkey\((.*)\)", line)
                if inside:
                    arg_str = inside.group(1)
                    parts = re.findall(r"['\"]([^'\"]+)['\"]", arg_str)
                    if parts:
                        normalized_keys = [
                            ("ctrl" if p.strip().lower() in ("cmd", "command") else p.strip()) for p in parts
                        ]
                        actions.append(("hotkey", normalized_keys))

            # press positional: pyautogui.press('enter') or press(['ctrl','v'])
            if "press(" in line and "keys=" not in line:
                inside = re.search(r"pyautogui\.press\((.*)\)", line)
                if inside:
                    arg_str = inside.group(1).strip()
                    keys: List[str] = []
                    if arg_str.startswith("["):
                        parts = re.findall(r"['\"]([^'\"]+)['\"]", arg_str)
                        keys = [p.strip() for p in parts]
                    else:
                        one = re.search(r"['\"]([^'\"]+)['\"]", arg_str)
                        if one:
                            keys = [one.group(1).strip()]
                    if keys:
                        normalized_keys = [
                            ("ctrl" if k.lower() in ("cmd", "command") else k) for k in keys
                        ]
                        if len(normalized_keys) > 1:
                            actions.append(("hotkey", normalized_keys))
                        else:
                            actions.append(("press", normalized_keys))

            # scroll
            scroll_match = re.search(r"pyautogui\.scroll\(([-\d]+)\)", line)
            if scroll_match:
                actions.append(("scroll", int(scroll_match.group(1))))

    return actions