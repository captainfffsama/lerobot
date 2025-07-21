We use [gello](https://github.com/wuphilipp/gello_software) to get leader arm info.
Here is a simple tutorial for gello install:
```bash
git clone https://github.com/wuphilipp/gello_software.git
cd gello_software
git submodule init
git submodule update
pip install -r requirements.txt
pip install -e .
pip install -e third_party/DynamixelSDK/python
```