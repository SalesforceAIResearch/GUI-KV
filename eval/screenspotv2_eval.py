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
# load this for vllm
from multiprocessing import freeze_support



from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModel, AutoImageProcessor
from qwen_vl_utils import process_vision_info
from opencua_utils import opencua_parse_action, analyze_vision_tokens_opencua_multi_images

# UI-TARS utilities
from ui_tars_utils import (
    parse_action_to_structure_output, MIN_PIXELS, MAX_PIXELS, IMAGE_FACTOR,
    analyze_vision_tokens_multi_images
)

# Attention mechanism helpers and KV cache implementations
from attention_helpers import (
    replace_qwen2_5_vl,
    replace_opencua,
    set_attention_implementation,
    configure_accelerate_skip_attention,
    set_kv_cache_budget,
    set_move_attention_to_cpu,
    set_last_vision_indices,
    set_vision_start_idx,
    set_vision_end_idx,
    set_temperature,
    set_alpha,
    set_window_size,
)

GROUNDING_DOUBAO = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. \n\n## Output Format\n\nAction: ...\n\n\n## Action Space\nclick(point='<point>x1 y1</point>'')\n\n## User Instruction
{instruction}"""

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

if __name__ == '__main__':
    freeze_support()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--screenspot_imgs', type=str, required=True)
    parser.add_argument('--screenspot_test', type=str, required=True)
    parser.add_argument('--task', type=str, required=True, choices=["mobile", "desktop", "web", "all"])
    parser.add_argument('--debug', default=None, type=int)
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/mps/cpu). If not specified, will auto-detect best available device.')
    parser.add_argument('--model_dtype', type=str, default="float32", choices=["auto", "bfloat16", "float16", "float32"], help='Data type to use (auto/bfloat16/float16/float32).')
    parser.add_argument('--attention_implementation', type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"], help='Attention implementation to use (eager/flash_attention_2).')
    parser.add_argument('--kv_cache', type=str, default="original", choices=["original", "pyramid_kv", "vl_cache", "snap_kv", "gui_kv"], help='KV cache method to use (original/pyramid_kv/vl_cache/snap_kv/gui_kv).')
    parser.add_argument('--kv_cache_budget', type=int, default=100, help='KV cache budget in tokens.')
    parser.add_argument('--alpha', type=float, default=None, help='Alpha for GUIKV.')
    parser.add_argument('--window_size', type=int, default=None, help='Window size for GUIKV.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for token information scores.')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory to store evaluation results in JSON format.')
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
    # min_pixels = 2562828
    # max_pixels = 13402828
    if args.model_path == "xlangai/OpenCUA-7B":
        processor = AutoImageProcessor.from_pretrained("xlangai/OpenCUA-7B", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("xlangai/OpenCUA-7B", trust_remote_code=True)
    else:
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    if args.model_dtype == "float32":
        model_dtype = torch.float32
    elif args.model_dtype == "bfloat16":
        model_dtype = torch.bfloat16
    elif args.model_dtype == "float16":
        model_dtype = torch.float16
    else:
        raise ValueError(f"Invalid model dtype: {args.model_dtype}")
    
    # replace the attention forward function
    # if args.attention_implementation == "eager":
    if args.model_path == "xlangai/OpenCUA-7B":
        replace_opencua(kv_cache_mode=args.kv_cache)
        print("Replaced OpenCUA attention with custom implementation for CPU memory management")
    else:
        replace_qwen2_5_vl(kv_cache_mode=args.kv_cache)
        print("Replaced Qwen2.5-VL attention with custom implementation for CPU memory management")
    
    # Verify the replacement worked
    # import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen_module
        
    # Load model with dynamic device selection
    if device == "cpu":
        if args.model_path == "xlangai/OpenCUA-7B":
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
            # Configure accelerate to skip moving attention tensors back to GPU
            configure_accelerate_skip_attention(model)
    else:
        # Check if we have multiple GPUs
        if torch.cuda.device_count() > 1:
            device_map = "auto"
        else:
            # For single GPU, use explicit device mapping
            device_map = {"": "cuda:0"}  # Map entire model to GPU 0
        
        if args.model_path == "xlangai/OpenCUA-7B":
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
            # Configure accelerate to skip moving attention tensors back to GPU
            configure_accelerate_skip_attention(model)
    
    
    


    print("Load Success")

    if args.task == "all":
        tasks = ["mobile", "desktop", "web"]
    else:
        tasks = [args.task]
    tasks_result = []
    result = []
    for task in tasks:
        dataset = "screenspot_" + task + "_v2.json"
        screenspot_data = json.load(open(os.path.join(args.screenspot_test, dataset), 'r'))
        if args.debug is not None:
            # Limit to 100 examples for quick evaluation
            screenspot_data = screenspot_data[:args.debug]
            print("Num of sample: " + str(len(screenspot_data)) + " (limited to 100 for quick evaluation)")
        else:
            print("Num of sample: " + str(len(screenspot_data)))
        
        num_action = 0
        corr_action = 0
        text_correct = []
        icon_correct = []
        num_wrong_format = 0
        for j, item in tqdm(enumerate(screenspot_data), desc=f"Processing {task} data", total=len(screenspot_data)):
            num_action += 1
            filename = item["img_filename"]
            img_path = os.path.join(args.screenspot_imgs, filename)
            if not os.path.exists(img_path):
                print("img not found")
                input()
            image = Image.open(img_path)
            # img_width, img_height = image.size  # PIL Image.size returns (width, height)
            instruction = item["instruction"]
            
            bbox = item["bbox"]
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            img_size = image.size
            img_width, img_height = img_size
            bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]

            # resized_height, resized_width = smart_resize(img_height, img_width)
            if args.model_path == "xlangai/OpenCUA-7B":
                SYSTEM_PROMPT = (
                    "You are a GUI agent. You are given a task and a screenshot of the screen. "
                    "You need to perform a series of pyautogui actions to complete the task."
                )
       
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_path},
                            {"type": "text", "text": SYSTEM_PROMPT + "\n" + instruction},
                        ],
                    },
                ]
                input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        
                image = Image.open(img_path).convert('RGB')
                info = processor.preprocess(images=[image])
                pixel_values = torch.tensor(info['pixel_values']).to(dtype=torch.bfloat16, device=model.device)
                grid_thws = torch.tensor(info['image_grid_thw'])
                input_ids = torch.tensor([input_ids]).to(model.device)
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": img_path, 
                            },
                            {"type": "text", "text": GROUNDING_DOUBAO.format(instruction=instruction)},
                        ],
                    }
                ]
               
                
                
                # Preparation for inference
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

              
                
                # print(f'Original resolution: {img_width}, {img_height}')
                
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

            # Conditionally analyze vision tokens for gui_kv and vl_cache
            if args.kv_cache == "gui_kv" or args.kv_cache == "vl_cache":
                if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
                    # vision_analysis = analyze_vision_tokens(processor, image_inputs, video_inputs, text)
                    vision_analysis = analyze_vision_tokens_multi_images(processor, image_inputs, video_inputs, text, image_count=1)
                elif args.model_path == "xlangai/OpenCUA-7B":
                    # vision_analysis = analyze_vision_tokens_opencua(tokenizer, input_ids, image_grid_thw=info["image_grid_thw"], merge_size=2)
                    vision_analysis = analyze_vision_tokens_opencua_multi_images(tokenizer, input_ids, image_grid_thw=info["image_grid_thw"], merge_size=2, image_count=1)

            # Standard generation path
            set_window_size(model, args)
            if args.kv_cache == "vl_cache":
                
                last_vision_indices = []
                vision_end_idx = vision_analysis.get('vision_end_idx', 0)
                last_vision_indices.append(vision_end_idx)

                set_last_vision_indices(model, last_vision_indices, args)
            elif args.kv_cache == "gui_kv":
               
               
                set_vision_start_idx(model, vision_analysis['vision_start_idx'], args)
                set_vision_end_idx(model, vision_analysis['vision_end_idx'], args)
                set_alpha(model, args)
                set_temperature(model, args)

            try:
                # Standard generation without timing measurement for maximum performance
                if args.model_path == "xlangai/OpenCUA-7B":
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
            except Exception as e:
                print(f"Error in generation: {e}")
                print("Using outputs from the previous iteration")
                if args.model_path != "xlangai/OpenCUA-7B":
                    generated_ids = outputs if not hasattr(outputs, 'sequences') else outputs.sequences

            print("output_text: ", output_text)

            try:
                if args.model_path == "xlangai/OpenCUA-7B":
                    parsed_actions = opencua_parse_action(output_text,
                                origin_resized_height=img_height,
                                origin_resized_width=img_width,
                                max_pixels=MAX_PIXELS,
                                min_pixels=MIN_PIXELS,
                                factor=IMAGE_FACTOR,
                                model_type="qwen25vl")
                    click_point = parsed_actions[0]["coordinate"]
                else:
                    parsed_actions = parse_action_to_structure_output(output_text, \
                        origin_resized_height=img_height, \
                        origin_resized_width=img_width, \
                        max_pixels=MAX_PIXELS, \
                        min_pixels=MIN_PIXELS, \
                        factor=IMAGE_FACTOR, \
                        model_type="qwen25vl")[0]
                    

                    
                    click_point = list(parsed_actions["action_inputs"].values())[0]
                    
                    # click point is string repr of [0.44982993197278914, 0.08716707021791767, 0.44982993197278914, 0.08716707021791767]
                    
                    # Convert click_point from string representation to list
                    click_point = ast.literal_eval(click_point)
                
                if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
                    corr_action += 1
                    if item["data_type"] == 'text':
                        text_correct.append(1)
                    else:
                        icon_correct.append(1)
                    
                else:
                    if item["data_type"] == 'text':
                        text_correct.append(0)
                    else:
                        icon_correct.append(0)
                
                result.append({"img_path": img_path, "text": instruction, "bbox": bbox, "pred": click_point,
                               "type": item["data_type"], "source": item["data_source"]})
            except Exception as e:
                print(output_text)
                print(e)
                
                num_wrong_format += 1
                if item["data_type"] == 'text':
                    text_correct.append(0)
                else:
                    icon_correct.append(0)
                logging.info("Step: " + str(j) + " wrong format")
        # Calculate metrics
        action_acc = corr_action / num_action
        text_acc = sum(text_correct) / len(text_correct) if len(text_correct) != 0 else 0
        icon_acc = sum(icon_correct) / len(icon_correct) if len(icon_correct) != 0 else 0
        
        # Log current task results
        logging.info("=" * 60)
        logging.info(f"Task Results:")
        logging.info(f"  Action Accuracy: {action_acc:.4f} ({corr_action}/{num_action})")
        logging.info(f"  Text Accuracy:   {text_acc:.4f} ({sum(text_correct)}/{len(text_correct)})")
        logging.info(f"  Icon Accuracy:   {icon_acc:.4f} ({sum(icon_correct)}/{len(icon_correct)})")
        logging.info(f"  Wrong Format:    {num_wrong_format}")
        logging.info("=" * 60)
        
        tasks_result.append([text_acc, icon_acc])

    # Print final results table
    logging.info("\n" + "=" * 80)
    logging.info("FINAL EVALUATION RESULTS")
    logging.info("=" * 80)
    
    # Print detailed task breakdown first
    if len(tasks_result) > 0:
        logging.info(f"\nResults by Task:")
        logging.info(f"{'Task':<15} {'Text Acc':<12} {'Icon Acc':<12} {'Combined':<12}")
        logging.info("-" * 51)
        for i, (text_acc, icon_acc) in enumerate(tasks_result):
            combined_acc = (text_acc + icon_acc) / 2
            logging.info(f"{'Task ' + str(i+1):<15} {text_acc*100:<12.2f}% {icon_acc*100:<12.2f}% {combined_acc*100:<12.2f}%")
        logging.info("-" * 51)
    
    # Calculate overall statistics
    all_text_accs = [result[0] for result in tasks_result]
    all_icon_accs = [result[1] for result in tasks_result]
    
    avg_text_acc = sum(all_text_accs) / len(all_text_accs) if all_text_accs else 0
    avg_icon_acc = sum(all_icon_accs) / len(all_icon_accs) if all_icon_accs else 0
    overall_acc = (avg_text_acc + avg_icon_acc) / 2
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Save detailed results to JSON
    detailed_results_path = os.path.join(args.results_dir, 'detailed_results.json')
    with open(detailed_results_path, 'w') as f:
        json.dump(result, f, indent=2)
    logging.info(f"Saved detailed results to {detailed_results_path}")
    
    # Create summary results
    summary_results = {
        "model_path": args.model_path,
        "task": args.task,
        "total_samples": len(result),
        "overall_accuracy": overall_acc,
        "text_accuracy": avg_text_acc,
        "icon_accuracy": avg_icon_acc,
        "task_breakdown": []
    }
    
    # Add task-specific results to summary
    task_names = tasks if args.task == "all" else [args.task]
    for i, (text_acc, icon_acc) in enumerate(tasks_result):
        task_name = task_names[i] if i < len(task_names) else f"task_{i+1}"
        combined_acc = (text_acc + icon_acc) / 2
        summary_results["task_breakdown"].append({
            "task": task_name,
            "text_accuracy": text_acc,
            "icon_accuracy": icon_acc,
            "combined_accuracy": combined_acc
        })
    
    # Save summary results to JSON
    summary_results_path = os.path.join(args.results_dir, 'summary_results.json')
    with open(summary_results_path, 'w') as f:
        json.dump(summary_results, f, indent=2)
    logging.info(f"Saved summary results to {summary_results_path}")
    
    # Print summary table
    logging.info(f"\nOverall Summary:")
    logging.info(f"{'Metric':<20} {'Accuracy':<10}")
    logging.info("-" * 30)
    logging.info(f"{'Text Average':<20} {avg_text_acc*100:<10.2f}%")
    logging.info(f"{'Icon Average':<20} {avg_icon_acc*100:<10.2f}%")
    logging.info(f"{'Overall Average':<20} {overall_acc*100:<10.2f}%")
    logging.info("-" * 30)
    
    logging.info("=" * 80)