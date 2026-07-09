pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
conda install ffmpeg=7.1.1 -c conda-forge
pip install torchcodec==0.5 --index-url=https://download.pytorch.org/whl/cu129
# pip install num2words==0.5.14   --index-url https://pypi.org/simple   --no-cache-dir -v
pip install -e ".[smolvla,dataset]"