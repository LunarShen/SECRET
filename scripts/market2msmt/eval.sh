python main.py --config-file configs/mutualrefine.yml \
  DATASETS.DIR "Data" \
  DATASETS.TARGET "msmt17" \
  CHECKPOING.EVAL "log/market2msmt/mutualrefine/model_mAP_best.pth.tar" \
  OUTPUT_DIR "log/market2msmt/eval" \
  GPU_Device [0,1,2,3] \
  MODE 'mutualrefine' \
  MODEL.ARCH "resnet50"
