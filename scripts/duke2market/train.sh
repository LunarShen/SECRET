python main.py --config-file configs/mutualrefine.yml \
  DATASETS.DIR "Data" \
  DATASETS.SOURCE "dukemtmc" \
  DATASETS.TARGET "market1501" \
  CHECKPOING.PRETRAIN_PATH "log/duke2market/pretrain/checkpoint_new.pth.tar" \
  OUTPUT_DIR "log/duke2market/mutualrefine" \
  GPU_Device [0,1,2,3] OPTIM.EPOCHS 50 \
  MODE 'mutualrefine' \
  MODEL.ARCH "resnet50"
