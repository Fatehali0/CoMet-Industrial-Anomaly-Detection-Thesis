#!/bin/bash
# =============================================================================
# CoMet: Training Script for MVTec AD
# Paper: Towards Real Unsupervised Anomaly Detection Via Confident Meta-Learning
# ICCV 2025 | https://openaccess.thecvf.com/content/ICCV2025/html/Aqeel_Towards_Real_Unsupervised_Anomaly_Detection_Via_Confident_Meta-Learning_ICCV_2025_paper.html
# =============================================================================

# -------------------------------
# USER CONFIGURATION (EDIT THESE)
# -------------------------------

# Path to dataset in this workspace
DATAPATH="./mvtec_ad_2"

# Thesis classes currently used on MVTec AD2
DATASETS=('can' 'fabric' 'rice')

# Results directory
RESULTS_PATH="results"

# GPU ID (set to -1 for CPU)
GPU_ID=0

# Python executable
PYTHON_BIN="python"

# full               = Full CoMet
# no_confidence      = Remove confidence learning only
# no_model_confidence = Remove adaptive model-confidence regularization only
# no_data_confidence = Remove sample-weight-based data confidence only
# no_meta            = Remove inner/outer meta-learning loop
# no_conf_no_meta    = Simple baseline for MVTec AD2
EXPERIMENT="no_model confidene"



# -------------------------------
# DO NOT EDIT BELOW THIS LINE
# -------------------------------

# Build dataset flags: -d screw -d pill ...
DATASET_FLAGS=()
for dataset in "${DATASETS[@]}"; do
    DATASET_FLAGS+=("-d" "$dataset")
done

# Create results directory
mkdir -p "$RESULTS_PATH"

case "$EXPERIMENT" in
    full)
        RUN_NAME_PREFIX="full_comet"
        NET_ABLATION_FLAGS=()
        ;;
    no_confidence)
        RUN_NAME_PREFIX="no_confidence"
        NET_ABLATION_FLAGS=(--no_confidence_learning --fixed_lambda 0.01)
        ;;
    no_meta)
        RUN_NAME_PREFIX="no_meta"
        NET_ABLATION_FLAGS=(--no_meta_learning --fixed_lambda 0.01)
        ;;
            no_model_confidence)
                    # Disable adaptive model-confidence regularization only.

        RUN_NAME_PREFIX="no_model_confidence"
        NET_ABLATION_FLAGS=(--no_model_confidence --fixed_lambda 0.0)
        ;;
    no_data_confidence)
            # Disable sample-weight-based data confidence only.

        RUN_NAME_PREFIX="no_data_confidence"
        NET_ABLATION_FLAGS=(--no_data_confidence --fixed_lambda 0.01)
        ;;
    no_conf_no_meta)
        RUN_NAME_PREFIX="baseline"
        NET_ABLATION_FLAGS=(--no_confidence_learning --no_meta_learning --fixed_lambda 0.0)
        ;;
    *)
        echo "Unknown EXPERIMENT: $EXPERIMENT"
        echo "Use one of: full, no_confidence, no_meta, no_conf_no_meta"
        exit 1
        ;;
esac
echo "Use one of: full, no_confidence, no_meta, no_conf_no_meta, no_model_confidence, no_data_confidence"
echo "Starting CoMet training on MVTec AD..."
echo "Dataset path: $DATAPATH"
echo "Categories: ${DATASETS[*]}"
echo "Saving results to: $RESULTS_PATH"
echo "Experiment: $EXPERIMENT"

"$PYTHON_BIN" main.py \
    --gpu "$GPU_ID" \
    --seed 0 \
    --log_group "comet_mvtec" \
    --log_project "MVTecAD_Results" \
    --results_path "$RESULTS_PATH" \
    --run_name "${RUN_NAME_PREFIX}_$(date +%Y%m%d_%H%M%S)" \
    net \
    -b wideresnet50 \
    -le layer2 \
    -le layer3 \
    --pretrain_embed_dimension 1536 \
    --target_embed_dimension 1536 \
    --patchsize 3 \
    --meta_epochs 40 \
    --embedding_size 256 \
    --gan_epochs 4 \
    --noise_std 0.015 \
    --dsc_hidden 1024 \
    --dsc_layers 2 \
    --dsc_margin .5 \
    --pre_proj 1 \
    "${NET_ABLATION_FLAGS[@]}" \
    dataset \
    --batch_size 8 \
    --resize 329 \
    --imagesize 288 \
    "${DATASET_FLAGS[@]}" \
    mvtec "$DATAPATH"

echo "Training completed. Results saved in: $RESULTS_PATH"
