# CycleGAN and pix2pix in PyTorch

Forked from [Auto Painter] (https://github.com/irfanICMLL/Auto_painter/blob/master/preprocessing/gen_sketch/sketch.py)
[Paper] (https://www.sciencedirect.com/science/article/pii/S0925231218306209?via%3Dihub)

Sketch-to-image synthesis using Conditional Generative Adversarial Networks (cGAN)

# Usage

python3 sketch.py --input INPUT_FOLDER_PATH --gen PATH_FOR_SKETCH [--gentoorg PATH_FOR_GENTOORG] [--orgtogen PATH_FOR_ORGTOGEN]

To save only the generated sketch, run it with --gen option with the directory you want to store the generated images

```bash
                python sketch.py --input INPUT_FOLDER_PATH --gen SKETCH_PATH
```

To store a concatenated image with the original and generated pictures

```bash
                python sketch.py --input INPUT_FOLDER_PATH --orgtogen SKETCH_PATH
```

To store a concatenated image with the generated and original picture

```bash
                python sketch.py --input INPUT_FOLDER_PATH --gentoorg SKETCH_PATH
```

# TODO

- Take specific inputs in addition to a folder
