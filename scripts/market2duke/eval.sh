python main.py --config-file configs/mutualrefine.yml \
  DATASETS.DIR "Data" \
  DATASETS.TARGET "dukemtmc" \
  CHECKPOING.EVAL "log/market2duke/mutualrefine/model_mAP_best.pth.tar" \
  OUTPUT_DIR "log/market2duke/eval" \
  GPU_Device [0,1,2,3] \
  MODE 'mutualrefine' \
  MODEL.ARCH "resnet50"
