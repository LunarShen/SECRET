python main.py --config-file configs/mutualrefine.yml \
  DATASETS.DIR "Data" \
  DATASETS.TARGET "market1501" \
  CHECKPOING.EVAL "log/duke2market/mutualrefine/model_mAP_best.pth.tar" \
  OUTPUT_DIR "log/duke2market/eval" \
  GPU_Device [0,1,2,3] \
  MODE 'mutualrefine' \
  MODEL.ARCH "resnet50"
