# main
import argparse
import click
import numpy as np
from matplotlib.collections import PatchCollection
import time

# yolov8
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

# mask rcnn
import tensorflow as tf
import os
import sys
# Root directory of the project
ROOT_DIR = os.path.abspath("./")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import visualize
import mrcnn.model as modellib
import mrcnn.config
from mrcnn.visualize import display_images
from samples.segmentation import segmentation
from matplotlib.patches import Polygon


# kraken
from PIL import Image, ImageDraw
from kraken import blla
from kraken.lib import vgsl
from collections import defaultdict
from itertools import cycle

CMAP_LIST = cycle([(230, 25, 75, 127),
              (60, 180, 75, 127)])

BMAP_LIST = (0, 130, 200, 255)


def draw_boxes_on_image(image_path, annotations):
    """
    Draw text boxes on image.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for annotation in annotations:
        x_min = annotation[0]
        y_min = annotation[1]
        width = annotation[2] - x_min
        height = annotation[3] - y_min

        # Create a Rectangle patch
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()


def yolov8_detection(image_path, weights_path):
    """
    Detect text region boxes using YOLOv8.

    Returns boxes coordinates.
    """

    click.echo(f'Processing layout recognition with YOLOv8 on {image_path} ', nl=True)

    # Load the pre-trained YOLOv8 model
    model = YOLO(weights_path)

    # Run inference with the YOLOv8n model on the intended image
    results = model.predict(image_path, imgsz=640, conf=0.5, classes=[0])

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        result.save(filename="results/yolo_layout.jpg")  # save to disk
    
    click.secho('\u2713', fg='green')

    # return box coordinates
    return boxes.xyxy.tolist()


def maskrcnn_detection(image_path, weights_path):
    """
    Segment text by lines using Mask R-CNN.

    Returns line boxes coordinates.
    """

    click.echo(f'Processing line segmentation with Mask r-cnn on {image_path} ', nl=True)

    # class names list (DO NOT CHANGE)
    CLASS_NAMES = ['BG', 'line']

    # Inference class
    class InferenceConfig(mrcnn.config.Config):
        # Name
        NAME = "segmentation_inference"

        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

        # make sure all lines are detected
        DETECTION_MIN_CONFIDENCE = 0.3
        DETECTION_MAX_INSTANCES = 1000

        # Number of classes
        NUM_CLASSES = len(CLASS_NAMES)


    # create config instance and set device
    config = InferenceConfig()
    DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

    # Create model in inference mode
    with tf.device(DEVICE): model = modellib.MaskRCNN(mode="inference", model_dir=os.getcwd(), config=config)

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(filepath=weights_path, by_name=True)

    # load the input image, convert it from BGR to RGB channel
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform a forward pass of the network to obtain the results
    r = model.detect([image], verbose=1)

    # Get the results for the first image.
    r = r[0]

    click.secho('\u2713', fg='green')

    # return line box coordinates
    return r['rois'].tolist()


def kraken_segmentation(image_path, layout_mode, seg_mode, weights_path):
    """
    Detect text region boxes and segment by lines using Kraken.

    Returns text boxes and line boxes coordinates.
    """

    # load image
    image = Image.open(image_path)

    # load model
    model = vgsl.TorchVGSLModel.load_model(weights_path)

    # apply kraken segmentation in image
    res = blla.segment(image, model=model, text_direction='horizontal-lr')

    # reorder lines by type
    lines = defaultdict(list)
    for line in res.lines:
        lines[line.tags['type']].append(line)
    
    # convert image
    image = image.convert('RGBA')

    # if line segmentation
    if seg_mode == 'kraken':

        click.echo(f'Processing line segmentation with Kraken on {image_path} ', nl=True)

        for t, ls in lines.items():

            tmp = Image.new('RGBA', image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(tmp)

            # draw each line
            for idx, line in enumerate(ls):
                c = next(CMAP_LIST)
                draw.polygon([tuple(x) for x in line.boundary], fill=c, outline=c[:3])
                draw.line([tuple(x) for x in line.baseline], fill=BMAP_LIST, width=2, joint='curve')
                draw.text(line.baseline[0], str(idx), fill=(0, 0, 0, 255))

            # save as image
            base_image = Image.alpha_composite(image, tmp)
            base_image.save(f'{"results/kraken_segmentation"}.png')

        click.secho('\u2713', fg='green')
    
    # if layout recognition
    if layout_mode == 'kraken':

        click.echo(f'Processing layout recognition with Kraken on {image_path} ', nl=True)

        for t, regs in res.regions.items():
            
            tmp = Image.new('RGBA', image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(tmp)

            for reg in regs:
                
                # draw text regions
                c = next(CMAP_LIST)
                try:
                    draw.polygon([tuple(x) for x in reg.boundary], fill=c, outline=c[:3])
                except Exception:
                    pass

            # save as image
            base_image = Image.alpha_composite(image, tmp)
            base_image.save(f'{"results/kraken_layout"}.png')

        click.secho('\u2713', fg='green')

    
    if layout_mode == 'kraken' and seg_mode == 'kraken':
        results = dict()
        results['layout'] = [reg.boundary for reg in regs]
        results['segmentation'] = [line.boundary for line in res.lines]
        # return text regions and line box coordinates
        return results
    elif layout_mode == 'kraken':
        results = dict()
        results['layout'] = [reg.boundary for reg in regs]
        # return box coordinates of text regions
        return results
    else:
        results = dict()
        results['segmentation'] = [line.boundary for line in res.lines]
        # return line box coordinates
        return results
    
def get_min_max_box(box):
    """Helper to get min and max coordinates from box"""
    x_min = min(box, key=lambda x: x[0])[0]
    y_min = min(box, key=lambda x: x[1])[1]
    x_max = max(box, key=lambda x: x[0])[0]
    y_max = max(box, key=lambda x: x[1])[1]
    return x_min, y_min, x_max, y_max

def get_box_area(box):
    """Calculate the area of a box given min and max coordinates"""
    x_min, y_min, x_max, y_max = box
    return (x_max - x_min) * (y_max - y_min)

def get_intersection_area(box1, box2):
    """Calculate the intersection area of two boxes"""
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)

    if x_min_inter < x_max_inter and y_min_inter < y_max_inter:
        return (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
    else:
        return 0
    
def merge_boxes(boxes):
    """Merge overlapping boxes into a single bounding box"""
    if not boxes:
        return []
    x_min = min(box[0] for box in boxes)
    y_min = min(box[1] for box in boxes)
    x_max = max(box[2] for box in boxes)
    y_max = max(box[3] for box in boxes)
    return [x_min, y_min, x_max, y_max]


def compare_boxes(text_regions, line_boxes, image_path, layout_mode, seg_mode):
    """
    Compare text region boxes with line boxes and alert if lines are incomplete.
    """

    text_coverages = []

    # Load the input image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    for text_region in text_regions:

        if layout_mode == 'yolo':
            text_x_min, text_y_min = int(text_region[0]), int(text_region[1])
            text_x_max, text_y_max = int(text_region[2]), int(text_region[3])
        elif layout_mode == 'kraken':
            text_x_min, text_y_min = min(text_region, key=lambda x: x[0])[0], min(text_region, key=lambda x: x[1])[1]
            text_x_max, text_y_max = max(text_region, key=lambda x: x[0])[0], max(text_region, key=lambda x: x[1])[1]

        text_box = (text_x_min, text_y_min, text_x_max, text_y_max)
        text_area = get_box_area(text_box)


        # Merge overlapping line boxes
        merged_line_boxes = []

        for line in line_boxes:
            if seg_mode == 'kraken':
                line_x_min, line_y_min, line_x_max, line_y_max = get_min_max_box(line)
            elif seg_mode == 'maskrcnn':
                line_x_min, line_y_min, line_x_max, line_y_max = line[1], line[0], line[3], line[2]

            line_box = [line_x_min, line_y_min, line_x_max, line_y_max]
            merged_line_boxes.append(line_box)
        
        merged_line_box = merge_boxes(merged_line_boxes)

        # Calculate the intersection area
        intersection_area = get_intersection_area(text_box, merged_line_box)

        coverage_percentage = (intersection_area / text_area) * 100

        print(f"Text region ({text_x_min}, {text_y_min}, {text_x_max}, {text_y_max}) coverage: {coverage_percentage:.2f}%")

        if coverage_percentage < 95: 
            print(f"Alert: Low coverage ({coverage_percentage:.2f}%) in text region ({text_x_min}, {text_y_min}, {text_x_max}, {text_y_max})")
        else:
            print(f"Lines found with sufficient coverage in text region ({text_x_min}, {text_y_min}, {text_x_max}, {text_y_max})")

        if text_area > 10000:
            text_coverages.append(coverage_percentage)
    
    return text_coverages
        
        


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Layout Recognition and Line Segmentation in an historical document image.')

    parser.add_argument('--maskweights', required=False, metavar="/path/to/mask-rcnn-weights.h5", 
                        help="Path to mask r-cnn weights .h5 file", default="models/maskrcnn/75_model.h5")

    parser.add_argument('--yoloweights', required=False, metavar="/path/to/yolo-weights.pt", 
                        help="Path to yolo weights .pt file", default="models/yolov8/best.pt")
    
    parser.add_argument('--krakenweights', required=False, metavar="/path/to/kraken-weights.pt", 
                        help="Path to kraken weights .mlmodel file", default="models/kraken/transfer.mlmodel")
    
    args = parser.parse_args()

    # assert if paths exist
    #assert os.path.exists(args.image), f"Image '{args.image}' does not exist."
    assert os.path.exists(args.maskweights), f"Mask r-cnn weights file '{args.maskweights}' does not exist."
    assert os.path.exists(args.yoloweights), f"YOLO weights file '{args.yoloweights}' does not exist."
    assert os.path.exists(args.krakenweights), f"Kraken weights file '{args.krakenweights}' does not exist."

    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir("../../DATASET_JUNTO/test/images") if isfile(join("../../DATASET_JUNTO/test/images", f)) and f.split(".")[1] != "DS_Store"]

    final_coverages = []
    yolo_mask_coverages = []
    kraken_coverages = []

    for image in onlyfiles:
        
        image = "../../DATASET_JUNTO/test/images/" + image

        #if yolo_mask_avg_coverage > kraken_kraken_avg_coverage:
        print("Using YOLO for layout recognition and Mask R-CNN for line segmentation.")
        text_regions_yolo = yolov8_detection(image, args.yoloweights)
        line_boxes_mask = maskrcnn_detection(image, args.maskweights)
        #else:
        print("Using Kraken for both layout recognition and line segmentation.")
        kraken_results = kraken_segmentation(image, "kraken", "kraken", args.krakenweights)
        text_regions_kraken = kraken_results['layout']
        line_boxes_kraken = kraken_results['segmentation']
        
        print("YOLO + MASKRCNN results: ")
        cov_yolo_mask = compare_boxes(text_regions_yolo, line_boxes_mask, image, "yolo", "maskrcnn")

        print("KRAKEN + KRAKEN results: ")
        cov_kraken = compare_boxes(text_regions_kraken, line_boxes_kraken, image, "kraken", "kraken")

        if (sum(cov_yolo_mask) / len(cov_yolo_mask)) > (sum(cov_kraken) / len(cov_kraken)):
            for cov in cov_yolo_mask:
                final_coverages.append(cov)
        else:
            for cov in cov_kraken:
                final_coverages.append(cov)
        
        for cov in cov_yolo_mask:
            yolo_mask_coverages.append(cov)
        for cov in cov_kraken:
                kraken_coverages.append(cov)

    print("BEST COVERAGES: " + str(sum(final_coverages) / len(final_coverages)))
    print("YOLO + MASK COVERAGES: " + str(sum(yolo_mask_coverages) / len(yolo_mask_coverages)))
    print("KRAKEN COVERAGES: " + str(sum(kraken_coverages) / len(kraken_coverages)))

    exit()