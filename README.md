# Layout Recognition and Line Segmentation

### Install dependencies

#### Recomended:

```bash
conda env create -f environment.yml
```

```bash
conda activate script_env
```

### How to use

#### Script to test a specific solution

```bash
python3 test.py --help

    usage: test.py [-h] [--image path/to/image.jpg] [--layout yolo / kraken] [--seg maskrcnn / kraken] [--maskweights /path/to/mask-rcnn-weights.h5]
                [--yoloweights /path/to/yolo-weights.pt]

    Layout Recognition and Line Segmentation in an historical document image.

    options:
    -h, --help            show this help message and exit
    --image path/to/image.jpg
                            Image to apply testing
    --layout yolo / kraken
                            Model to perform layout recognition (yolo or kraken)
    --seg maskrcnn / kraken
                            Model to perform segmentation (maskrcnn or kraken)
    --maskweights /path/to/mask-rcnn-weights.h5
                            Path to mask r-cnn weights .h5 file
    --yoloweights /path/to/yolo-weights.pt
                            Path to yolo weights .pt file
```


#### Script to test coverages of all available models


```bash
usage: test_subset.py [-h] [--maskweights /path/to/mask-rcnn-weights.h5] [--yoloweights /path/to/yolo-weights.pt] [--krakenweights /path/to/kraken-weights.pt]

Layout Recognition and Line Segmentation in an historical document image.

options:
  -h, --help            show this help message and exit
  --maskweights /path/to/mask-rcnn-weights.h5
                        Path to mask r-cnn weights .h5 file
  --yoloweights /path/to/yolo-weights.pt
                        Path to yolo weights .pt file
  --krakenweights /path/to/kraken-weights.pt
                        Path to kraken weights .mlmodel file
```