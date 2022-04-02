python main.py --config-file configs/pretrain.yml \
  DATASETS.DIR "Data" \
  DATASETS.SOURCE "market1501" \
  DATASETS.TARGET "dukemtmc" \
  OUTPUT_DIR "log/market2duke/pretrain" \
  GPU_Device [0,1,2,3] \
  MODE 'pretrain' \
  MODEL.ARCH "resnet50"
