# seam_carving


# dependencies

`pip install numpy opencv-python`

# usage

```bash
# python src/seam_carving.py -h

usage: seam_carving.py [-h] --image_path IMAGE_PATH --output OUTPUT --mode {enlarge,removal} --num_seams NUM_SEAMS

optional arguments:
  -h, --help            show this help message and exit
  --image_path IMAGE_PATH, -i IMAGE_PATH
                        input image path
  --output OUTPUT, -o OUTPUT
                        output image path
  --mode {enlarge,removal}, -m {enlarge,removal}
                        enlarge or removal
  --num_seams NUM_SEAMS, -n NUM_SEAMS
                        number of seams used in algorithm
```

# examples

```bash
# seam removal
python src/seam_carving.py -i ./examples/example01.jpg -m enlarge -n 10 -o ./examples/example01.enlarge.png
# seam enlarge
python src/seam_carving.py -i ./examples/example01.jpg -m removal -n 10 -o ./examples//example01.removal.png
```
