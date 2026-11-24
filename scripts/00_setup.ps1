# Create the project virtualenv and install dependencies (CUDA 12.1 PyTorch wheels).
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install opencv-python scikit-image scikit-learn numpy pandas matplotlib tqdm pyyaml jupyter nbconvert ipykernel requests pillow
# facenet-pytorch pins an old torch; install without deps (only MTCNN + the reference model are used)
.\.venv\Scripts\python.exe -m pip install facenet-pytorch --no-deps
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
