pip --no-cache-dir install numpy==1.20.3
pip --no-cache-dir install scikit-learn==0.24.2 
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=11.0 -c pytorch -y && conda clean --all -y
pip --no-cache-dir install opencv-python -U
pip --no-cache-dir install tensorboard -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
pip --no-cache-dir install pycocotools
pip --no-cache-dir install pyyaml==3.12 --ignore-installed 
pip --no-cache-dir install tensorboardX 
pip --no-cache-dir install cython==0.27.3 
pip --no-cache-dir install mmcv-full==1.3.5+torch1.7.0+cu110 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html

cd lib && sh make.sh && cd ..