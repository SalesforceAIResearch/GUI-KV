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
SCREENSPOT_IMGS="${SCREENSPOT_IMGS:-/path/to/ScreenSpot-v2/screenspotv2_image}"
SCREENSPOT_TEST="${SCREENSPOT_TEST:-/path/to/ScreenSpot-v2}"

mkdir -p logs
mkdir -p results

for i in "${!kv_cache_budgets[@]}"; do
    kv_cache_budget=${kv_cache_budgets[$i]}
    gpu_id=$((i % num_gpus))

    echo "Starting experiment with kv_cache_budget=$kv_cache_budget on GPU $gpu_id"

    if [ $kv_cache == "gui_kv" ]; then
        log_path=logs/screenspotv2_eval/${model_path//\//_}/${kv_cache}_alpha-${alpha}_temperature-${temperature}_window-${window_size}_residual-stream_softmax/budget-${kv_cache_budget}.log
        results_dir=results/screenspotv2/${model_path//\//_}/${kv_cache}_alpha-${alpha}_temperature-${temperature}_window-${window_size}_residual-stream_softmax_budget-${kv_cache_budget}
    else
        log_path=logs/screenspotv2_eval/${model_path//\//_}/${kv_cache}-window-${window_size}/budget-${kv_cache_budget}.log
        results_dir=results/screenspotv2/${model_path//\//_}/${kv_cache}-window-${window_size}_budget-${kv_cache_budget}
    fi

    if [ ! -z "$debug" ]; then
        log_path=${log_path%.log}_debug-${debug}.log
        results_dir=${results_dir}_debug-${debug}
    fi

    echo "  log_path: $log_path"
    echo "  results_dir: $results_dir"

    mkdir -p $results_dir
    mkdir -p $(dirname $log_path)

    CUDA_VISIBLE_DEVICES="$gpu_id" nohup python -u screenspotv2_eval.py --model_path $model_path \
                    --screenspot_imgs $SCREENSPOT_IMGS \
                    --screenspot_test $SCREENSPOT_TEST \
                    --model_dtype $model_dtype \
                    --attention_implementation $attention_implementation \
                    $([ ! -z "$debug" ] && echo "--debug $debug") \
                    --kv_cache $kv_cache \
                    --kv_cache_budget $kv_cache_budget \
                    --temperature $temperature \
                    --alpha $alpha \
                    --window_size $window_size \
                    --results_dir $results_dir \
                    --task all > $log_path 2>&1 &

    echo "  Process started with PID $!"
    echo ""
done
