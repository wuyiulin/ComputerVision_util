# Two implements of SSIM algorithm 
##  Usage

### Step1
Replace img1_path, img2_path to yours.

### Step2
```shell
python SSIM.py
```

Under RGB image size of 2592 × 1520,
PyTorch(CPU) implement will fast two times of numpy implement.

If you use PyTorch method and you turned image tensor to tensor.Tensor.cuda, accuracy will be decrease by turn loss.

Ref.

https://blog.csdn.net/a2824256/article/details/115013851

https://stackoverflow.com/questions/60534909/gaussian-filter-in-pytorch
