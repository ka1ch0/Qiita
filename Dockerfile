FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt upgrade -y
RUN apt install -y tzdata
RUN apt install -y libgl1-mesa-dev
RUN apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget git

RUN apt install -y python3 python3-pip
RUN pip3 install --no-cache --upgrade pip setuptools wheel
RUN pip3 install numpy pandas scikit-learn scipy requests optuna lightgbm xgboost catboost tensorflow-gpu==2.8.0rc0 opencv-python jupyterlab jupyterlab-lsp plotly==5.3.1 'python-lsp-server[all]' tqdm pillow ipywidgets jupyter-dash lckr-jupyterlab-variableinspector jupyterlab_nvdashboard matplotlib seaborn	
RUN pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip3 install transformers

RUN apt install -y nodejs

RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py

ADD . /root
