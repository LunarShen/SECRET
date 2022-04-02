conda create -n secret python=3.8
conda activate secret
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install tqdm numpy six h5py Pillow scipy scikit-learn metric-learn pyyaml yacs termcolor faiss-gpu==1.6.3 opencv-python Cython
python setup.py develop
