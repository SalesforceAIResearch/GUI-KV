# Configuration
kv_cache_budgets=(1 3 5 10 15 20 40 80)

num_gpus=8
model_path=ByteDance-Seed/UI-TARS-1.5-7B
# model_path=xlangai/OpenCUA-7B
kv_cache=gui_kv
temperature=3.5
alpha=2
window_size=8
debug=""

attention_implementation=flash_attention_2
model_dtype=bfloat16

# Dataset paths - configure via environment variables or edit defaults below
AGENTNETBENCH_IMGS="${AGENTNETBENCH_IMGS:-/path/to/AgentNetBench/test_data/images}"
AGENTNETBENCH_DATA="${AGENTNETBENCH_DATA:-/path/to/AgentNetBench/test_data}"


mkdir -p logs/agentnetbench_eval
mkdir -p results


for i in "${!kv_cache_budgets[@]}"; do
    kv_cache_budget=${kv_cache_budgets[$i]}
    gpu_id=$(((i) % num_gpus))
    
    echo "Starting AgentNetBench experiment with kv_cache_budget=$kv_cache_budget on GPU $gpu_id"
    
    
    if [ $kv_cache == "gui_kv" ]; then
        log_path=logs/agentnetbench_eval/${model_path//\//_}/${kv_cache}_alpha-${alpha}_temperature-${temperature}_window-${window_size}_residual-stream_softmax_zero-out-previous-position-dynamic-key-qr-unweighted-r32-global-uncentered/budget-${kv_cache_budget}.log
        results_dir=results/agentnetbench/${model_path//\//_}/${kv_cache}_alpha-${alpha}_temperature-${temperature}_window-${window_size}_residual-stream_softmax_zero-out-previous-position-dynamic-key-qr-unweighted-r32-global-uncentered_budget-${kv_cache_budget}
    else
        log_path=logs/agentnetbench_eval/${model_path//\//_}/${kv_cache}-window-${window_size}/budget-${kv_cache_budget}.log
        results_dir=results/agentnetbench/${model_path//\//_}/${kv_cache}-window-${window_size}_budget-${kv_cache_budget}
    fi

    if [ ! -z "$debug" ]; then
        log_path=${log_path%.log}_debug-${debug}.log
        results_dir=${results_dir}_debug-${debug}
    fi

    echo "  log_path: $log_path"
    echo "  results_dir: $results_dir"

    
    mkdir -p $results_dir
    mkdir -p $(dirname $log_path)

    
    CUDA_VISIBLE_DEVICES="$gpu_id" nohup python -u agentnetbench_eval.py --model_path $model_path \
                    --agentnetbench_imgs $AGENTNETBENCH_IMGS \
                    --agentnetbench_data $AGENTNETBENCH_DATA \
                    --model_dtype $model_dtype \
                    --attention_implementation $attention_implementation \
                    --kv_cache $kv_cache \
                    --kv_cache_budget $kv_cache_budget \
                    --temperature $temperature \
                    $([ ! -z "$debug" ] && echo "--debug $debug") \
                    --alpha $alpha \
                    --window_size $window_size \
                    --results_dir $results_dir \
                    --task all > $log_path 2>&1 &
    
    echo "  Process started with PID $!"
    echo ""
    
    
done

