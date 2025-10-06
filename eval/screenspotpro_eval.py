import torch
import ast
import json
import re
import argparse
import os
from PIL import Image
import logging
from tqdm import tqdm
import copy
from multiprocessing import freeze_support
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModel, AutoImageProcessor
from qwen_vl_utils import process_vision_info
from screenspotpro_utils import screenspotpro_evaluate, eval_sample_positive_gt
from opencua_utils import opencua_parse_action, analyze_vision_tokens_opencua_multi_images
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



logging.basicConfig(level=logging.INFO)
torch.manual_seed(1234)


GROUNDING_DOUBAO = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. \n\n## Output Format\n\nAction: ...\n\n\n## Action Space\nclick(point='<point>x1 y1</point>'')\n\n## User Instruction
{instruction}"""

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
    if "Action: " in input_string and "start_box=" in input_string:
        suffix = input_string.split("Action: ")[0] + "Action: "
        actions = input_string.split("Action: ")[1:]
        processed_actions = []
        for action in actions:
            action = action.strip()
            coordinates = re.findall(r"(start_box|end_box)='\((\d+),\s*(\d+)\)'", action)

            updated_action = action
            for coord_type, x, y in coordinates:
                updated_action = updated_action.replace(f"{coord_type}='({x},{y})'", f"{coord_type}='<|box_start|>({x},{y})<|box_end|>'")
            processed_actions.append(updated_action)

        final_string = suffix + "\n\n".join(processed_actions)
    else:
        final_string = input_string
    return final_string

if __name__ == '__main__':
    freeze_support()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--screenspot_imgs', type=str, required=True)
    parser.add_argument('--screenspot_test', type=str, required=True)
    parser.add_argument('--task', type=str, required=True, choices=["all"])
    parser.add_argument('--debug', default=None, type=int)
    parser.add_argument('--max_new_tokens', type=int, default=400)
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
    elif args.model_path == "xlangai/OpenCUA-7B":
        processor = AutoImageProcessor.from_pretrained("xlangai/OpenCUA-7B", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("xlangai/OpenCUA-7B", trust_remote_code=True)
    else:
        raise NotImplementedError("Model not supported")
    
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
        raise NotImplementedError("Model not supported")

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
        set_attention_implementation(model, args)
        set_kv_cache_budget(model, args)
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
            raise NotImplementedError("Model not supported")
        set_attention_implementation(model, args)
        set_kv_cache_budget(model, args)
        if args.attention_implementation == "eager":
            set_move_attention_to_cpu(model, args)
            configure_accelerate_skip_attention(model)

    print("Load Success")

    if args.task == "all":
        task_filenames = [
            os.path.splitext(f)[0]
            for f in os.listdir(args.screenspot_test)
            if f.endswith(".json")
        ]
        print("task_filenames: ", len(task_filenames))
    else:
        raise NotImplementedError("Task not implemented")
    
    
    tasks_to_run = []
    for task_filename in task_filenames:
        dataset = task_filename + ".json"
        with open(os.path.join(args.screenspot_test, dataset), 'r') as f:
            task_data = json.load(f)
        gt_types = ["positive"]
        inst_styles = ["instruction"]
        languages = ["en"]
        # Create the list of tasks to run, one item as an instance. Tasks may be reused.
        for inst_style in inst_styles:  # Expand tasks based on user configurations
            for gt_type in gt_types:
                for lang in languages:
                    for task_instance in task_data:
                        task_instance = copy.deepcopy(task_instance)
                        task_instance["task_filename"] = task_filename
                        task_instance["gt_type"] = gt_type
                        task_instance["instruction_style"] = inst_style
                        task_instance["language"] = lang
                        if lang == "cn":
                            if inst_style!= 'instruction' or gt_type != 'positive':
                                # TODO: Translate the data
                                raise AttributeError("Only positive samples and 'instruction' style are supported for Chinese instructions.")
                            task_instance["prompt_to_evaluate"] = task_instance["instruction_cn"]
                        elif lang == "en":
                            task_instance["prompt_to_evaluate"] = task_instance["instruction"]

                        tasks_to_run.append(task_instance)
        print(f"Num of sample in {task_filename}: {len(task_data)} * {len(inst_styles)} * {len(gt_types)} * {len(languages)} = {len(task_data) * len(inst_styles) * len(gt_types) * len(languages)}")
    print(f"Total tasks: {len(tasks_to_run)}")

    results = []

    if args.debug is not None:
        tasks_to_run = tasks_to_run[:args.debug]
        print("Num of sample: " + str(len(tasks_to_run)) + f" (limited to {args.debug} for quick evaluation)")
    else:
        print("Num of sample: " + str(len(tasks_to_run)))
    
    
    for sample in tqdm(tasks_to_run, desc="Processing samples", total=len(tasks_to_run)):
        filename = sample["img_filename"]
        img_path = os.path.join(args.screenspot_imgs, filename)

       
        
        image = Image.open(img_path)
        # img_width, img_height = image.size  # PIL Image.size returns (width, height)
        instruction = sample["prompt_to_evaluate"]
            
        bbox = sample["bbox"]
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        img_size = image.size
        img_width, img_height = img_size
        bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]

        # resized_height, resized_width = smart_resize(img_height, img_width)
        if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
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
            SYSTEM_PROMPT = (
                "You are a GUI agent. You are given a task and a screenshot of the screen. "
                "You need to perform a series of pyautogui actions to complete the task."
            )
            # messages = [
            #     {"role": "system", "content": SYSTEM_PROMPT},
            #     {
            #         "role": "user",
            #         "content": [
            #             {"type": "image", "image": img_path},
            #             {"type": "text", "text": instruction},
            #         ],
            #     },
            # ]
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
           
            
            
        

        # Analyze vision tokens for KV cache methods that require it
        if args.kv_cache == "gui_kv" or args.kv_cache == "vl_cache":
            if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
                vision_analysis = analyze_vision_tokens_multi_images(processor, image_inputs, video_inputs, text, image_count=1)
            elif args.model_path == "xlangai/OpenCUA-7B":
                vision_analysis = analyze_vision_tokens_opencua_multi_images(tokenizer, input_ids, image_grid_thw=info["image_grid_thw"], merge_size=2, image_count=1)
            else:
                raise NotImplementedError("Model not supported")

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
        except Exception as e:
            print(f"Error in generation: {e}")
            print("Using outputs from the previous iteration")
            print(f"Image dimensions: {image.size}")

        sample_result = {
            "id": sample["id"],
            "img_path": img_path,
            "group": sample["group"] if "group" in sample else None,
            "platform": sample["platform"],
            "application": sample["application"],
            "lang": sample["language"],
            "instruction_style": sample["instruction_style"],
            "prompt_to_evaluate": sample["prompt_to_evaluate"],
            "gt_type": sample["gt_type"],
            "ui_type": sample["ui_type"],
            "task_filename": sample["task_filename"],

            "raw_response": output_text
        }

        try:
            if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
                parsed_actions = parse_action_to_structure_output(output_text,
                    origin_resized_height=img_height,
                    origin_resized_width=img_width,
                    max_pixels=MAX_PIXELS,
                    min_pixels=MIN_PIXELS,
                    factor=IMAGE_FACTOR,
                    model_type="qwen25vl")[0]

                click_point = list(parsed_actions["action_inputs"].values())[0]

                click_point = ast.literal_eval(click_point)
            elif args.model_path == "xlangai/OpenCUA-7B":
                parsed_actions = opencua_parse_action(output_text,
                            origin_resized_height=img_height,
                            origin_resized_width=img_width,
                            max_pixels=MAX_PIXELS,
                            min_pixels=MIN_PIXELS,
                            factor=IMAGE_FACTOR,
                            model_type="qwen25vl")
                click_point =  parsed_actions[0]["coordinate"]

            else:
                raise NotImplementedError("Model not supported")

            response = {
                "point": click_point,
            }
            if sample["gt_type"] == "positive":
                correctness = eval_sample_positive_gt(sample, response)
                sample_result.update({
                    "bbox": sample["bbox"],
                })
            else:
                raise NotImplementedError("Negative samples are not supported")

        except Exception as e:
            print(output_text)
            print(e)
            click_point = None
            correctness = "wrong"

        sample_result.update({
            "pred": click_point,
            "correctness": correctness,
        })

        results.append(sample_result)


    result_report = screenspotpro_evaluate(results)

    print("\n" + "=" * 80)
    print("SCREENSPOT PRO EVALUATION RESULTS")
    print("=" * 80)
    print(json.dumps(result_report["metrics"], indent=2, ensure_ascii=False))

    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)

        detailed_results_path = os.path.join(args.results_dir, 'screenspotpro_detailed_results.json')
        with open(detailed_results_path, 'w') as f:
            json.dump(result_report, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved ScreenSpot Pro detailed results to {detailed_results_path}")

        overall_metrics = result_report["metrics"]["overall"]
        summary_results = {
            "model_path": args.model_path,
            "task": args.task,
            "total_samples": overall_metrics["num_total"],
            "overall_accuracy": overall_metrics["action_acc"],
            "text_accuracy": overall_metrics["text_acc"],
            "icon_accuracy": overall_metrics["icon_acc"],
            "wrong_format_samples": overall_metrics["wrong_format_num"],
            "kv_cache": args.kv_cache,
            "kv_cache_budget": args.kv_cache_budget,
            "attention_implementation": args.attention_implementation,
            "model_dtype": args.model_dtype,
            "max_new_tokens": args.max_new_tokens
        }

        summary_results_path = os.path.join(args.results_dir, 'screenspotpro_summary_results.json')
        with open(summary_results_path, 'w') as f:
            json.dump(summary_results, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved ScreenSpot Pro summary results to {summary_results_path}")

        print(f"\nScreenSpot Pro Summary:")
        print(f"{'Metric':<25} {'Value':<15}")
        print("-" * 40)
        print(f"{'Overall Accuracy':<25} {overall_metrics['action_acc']*100:<15.2f}%")
        print(f"{'Text Accuracy':<25} {overall_metrics['text_acc']*100:<15.2f}%")
        print(f"{'Icon Accuracy':<25} {overall_metrics['icon_acc']*100:<15.2f}%")
        print(f"{'Total Samples':<25} {overall_metrics['num_total']:<15}")
        print(f"{'Wrong Format':<25} {overall_metrics['wrong_format_num']:<15}")
        print("-" * 40)

    print("=" * 80)
