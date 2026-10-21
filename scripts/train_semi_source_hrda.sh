#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
TRANSFER=${TRANSFER:?Set one of the five transfer names}
SPLIT_SOURCE=${SPLIT_SOURCE:-random}
CUDA_DEVICES=${CUDA_DEVICES:?Set the GPU explicitly}
if [[ "$CUDA_DEVICES" == *,* ]]; then
    echo "The controlled protocol uses one GPU" >&2
    exit 2
fi
case "$TRANSFER" in
    gta2cityscapes) SOURCE=gta; TARGET=cityscapes; RATIO=${RATIO:-1_64} ;;
    synthia2cityscapes) SOURCE=synthia; TARGET=cityscapes16; RATIO=${RATIO:-1_64} ;;
    cityscapes2acdc) SOURCE=cityscapes; TARGET=acdc; RATIO=${RATIO:-1_64} ;;
    cityscapes2muses) SOURCE=cityscapes; TARGET=muses; RATIO=${RATIO:-1_64} ;;
    cityscapes2mapillary) SOURCE=cityscapes; TARGET=mapillary; RATIO=${RATIO:-1_128} ;;
    *) echo "Unsupported TRANSFER: $TRANSFER" >&2; exit 2 ;;
esac
CONFIG_PATH=${CONFIG_PATH:-configs/road_hrda_source/${TRANSFER}.yaml}
TARGET_SPLIT_DIR=${TARGET_SPLIT_DIR:-splits/${TARGET}/${SPLIT_SOURCE}/${RATIO}}
LABELED_ID_PATH=${LABELED_ID_PATH:-${TARGET_SPLIT_DIR}/labeled.txt}
UNLABELED_ID_PATH=${UNLABELED_ID_PATH:-${TARGET_SPLIT_DIR}/unlabeled.txt}
SOURCE_DATA_ROOT=${SOURCE_DATA_ROOT:-data/${SOURCE}}
SOURCE_ID_PATH=${SOURCE_ID_PATH:-${SOURCE_DATA_ROOT}/train.txt}
SAVE_PATH=${SAVE_PATH:-exp/transfer/${TRANSFER}/mccl_source/dinov3_base_rein_hrda/${SPLIT_SOURCE}/${RATIO}}
for path in "$CONFIG_PATH" "$LABELED_ID_PATH" "$UNLABELED_ID_PATH" "$SOURCE_ID_PATH" "splits/${TARGET}/val.txt" "${PRETRAINED_PATH:-pretrained/dinov3_base.pth}"; do
    [[ -f "$path" ]] || { echo "Missing: $path; run scripts/prepare_road_data.py" >&2; exit 1; }
done
cmd=(python mccl_source.py --config "$CONFIG_PATH" --labeled-id-path "$LABELED_ID_PATH"
    --unlabeled-id-path "$UNLABELED_ID_PATH" --source-data-root "$SOURCE_DATA_ROOT"
    --source-id-path "$SOURCE_ID_PATH" --source-batch-size "${SOURCE_BATCH_SIZE:-2}"
    --source-loss-weight "${SOURCE_LOSS_WEIGHT:-1.0}" --save-path "$SAVE_PATH")
[[ -z "${PRETRAINED_PATH:-}" ]] || cmd+=(--pretrained-path "$PRETRAINED_PATH")
[[ "${RESUME:-0}" != 1 ]] || cmd+=(--resume)
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    read -r -a extra <<< "$EXTRA_ARGS"
    cmd+=("${extra[@]}")
fi
echo "MCCL+Source: $TRANSFER, selection=$SPLIT_SOURCE, budget=$RATIO, target/source batch=2/${SOURCE_BATCH_SIZE:-2}"
printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_DEVICES"
printf '%q ' "${cmd[@]}"
printf '\n'
[[ "${DRY_RUN:-0}" != 1 ]] || exit 0
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "${cmd[@]}"
