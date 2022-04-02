python main.py --config-file configs/mutualrefine.yml \
  DATASETS.DIR "Data" \
  DATASETS.SOURCE "market1501" \
  DATASETS.TARGET "dukemtmc" \
  CHECKPOING.PRETRAIN_PATH "log/market2duke/pretrain/checkpoint_new.pth.tar" \
  OUTPUT_DIR "log/market2duke/mutualrefine" \
  GPU_Device [0,1,2,3] OPTIM.EPOCHS 50 \
  MODE 'mutualrefine' \
  MODEL.ARCH "resnet50"
