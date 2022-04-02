python main.py --config-file configs/pretrain.yml \
  DATASETS.DIR "Data" \
  DATASETS.SOURCE "dukemtmc" \
  DATASETS.TARGET "market1501" \
  OUTPUT_DIR "log/duke2market/pretrain" \
  GPU_Device [0,1,2,3] \
  MODE 'pretrain' \
  MODEL.ARCH "resnet50"
