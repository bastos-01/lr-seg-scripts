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

    # visualize results and save them
    visualize.display_instances(image=image, boxes=r['rois'], masks=r['masks'], class_ids=r['class_ids'], class_names=CLASS_NAMES, scores=r['scores'], image_id="results/maskrcnn_segmentation.jpg")

    # save image splash
    splash = segmentation.color_splash(image, r['masks'])
    display_images([splash], cols=1, image_id="results/maskrcnn_segmentation_splash")

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

            # Visualize the problematic region
            fig, ax = plt.subplots(1)
            ax.imshow(image)

            rect = patches.Rectangle((text_x_min, text_y_min), text_x_max - text_x_min, text_y_max - text_y_min,
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            merged_rect = patches.Rectangle((merged_line_box[0], merged_line_box[1]),
                                            merged_line_box[2] - merged_line_box[0],
                                            merged_line_box[3] - merged_line_box[1],
                                            linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(merged_rect)

            plt.show()
        else:
            print(f"Lines found with sufficient coverage in text region ({text_x_min}, {text_y_min}, {text_x_max}, {text_y_max})")


def evaluate_model_performance(image_path, layout_fn, layout_weights, seg_fn, seg_weights, model_flag):
    start_time = time.time()
    if model_flag == "yolo_mask":
        text_regions = layout_fn(image_path, layout_weights)
        line_boxes = seg_fn(image_path, seg_weights)
    else:
        results = layout_fn(image_path, "kraken", "kraken", layout_weights)
        text_regions = results['layout']
        line_boxes = results['segmentation']
    end_time = time.time()

    covered_areas = []
    for text_region in text_regions:
        if model_flag == "yolo_mask":
            text_x_min, text_y_min, text_x_max, text_y_max = int(text_region[0]), int(text_region[1]), int(text_region[2]), int(text_region[3])
        else:
            text_x_min, text_y_min = min(text_region, key=lambda x: x[0])[0], min(text_region, key=lambda x: x[1])[1]
            text_x_max, text_y_max = max(text_region, key=lambda x: x[0])[0], max(text_region, key=lambda x: x[1])[1]
        text_box = (text_x_min, text_y_min, text_x_max, text_y_max)
        text_area = get_box_area(text_box)

        merged_line_boxes = []
        for line in line_boxes:
            if model_flag == "yolo_mask":
                line_x_min, line_y_min, line_x_max, line_y_max = line[1], line[0], line[3], line[2]
            else:
                line_x_min, line_y_min, line_x_max, line_y_max = get_min_max_box(line)
            line_box = [line_x_min, line_y_min, line_x_max, line_y_max]
            merged_line_boxes.append(line_box)
        merged_line_box = merge_boxes(merged_line_boxes)

        intersection_area = get_intersection_area(text_box, merged_line_box)
        coverage_percentage = min((intersection_area / text_area) * 100, 100)
        covered_areas.append(coverage_percentage)

    average_coverage = sum(covered_areas) / len(covered_areas) if covered_areas else 0
    execution_time = end_time - start_time

    return average_coverage, execution_time
        


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Layout Recognition and Line Segmentation in an historical document image.')

    parser.add_argument('--image', required=False, metavar="path/to/image.jpg",
                        help='Image to apply testing', default="example_1.jpg")

    parser.add_argument('--layout', required=False, metavar="yolo / kraken", 
                        help="Model to perform layout recognition (yolo or kraken)", default="yolo")
    
    parser.add_argument('--seg', required=False, metavar="maskrcnn / kraken", 
                        help="Model to perform segmentation (maskrcnn or kraken)", default="maskrcnn")

    parser.add_argument('--maskweights', required=False, metavar="/path/to/mask-rcnn-weights.h5", 
                        help="Path to mask r-cnn weights .h5 file", default="models/maskrcnn/75_model.h5")

    parser.add_argument('--yoloweights', required=False, metavar="/path/to/yolo-weights.pt", 
                        help="Path to yolo weights .pt file", default="models/yolov8/best.pt")
    
    parser.add_argument('--krakenweights', required=False, metavar="/path/to/kraken-weights.pt", 
                        help="Path to kraken weights .mlmodel file", default="models/kraken/best.mlmodel")
    
    args = parser.parse_args()

    # assert if paths exist
    assert os.path.exists(args.image), f"Image '{args.image}' does not exist."
    assert os.path.exists(args.maskweights), f"Mask r-cnn weights file '{args.maskweights}' does not exist."
    assert os.path.exists(args.yoloweights), f"YOLO weights file '{args.yoloweights}' does not exist."
    assert os.path.exists(args.krakenweights), f"Kraken weights file '{args.krakenweights}' does not exist."

    if args.layout == "yolo":
        text_regions = yolov8_detection(args.image, args.yoloweights)

    if args.layout == "kraken" or args.seg == "kraken":
        kraken_results = kraken_segmentation(args.image, args.layout, args.seg, args.krakenweights)
        if 'layout' in kraken_results.keys():
            text_regions = kraken_results['layout']
        if 'segmentation' in kraken_results.keys():
            line_boxes = kraken_results['segmentation']
    else:
        Exception("Layout option needs to be either 'yolo' or 'kraken'.")
    
    if args.seg == "maskrcnn":
        line_boxes = maskrcnn_detection(args.image, args.maskweights)
    else:
        Exception("Segmentation option needs to be either 'maskrcnn' or 'kraken'.")

    compare_boxes(text_regions, line_boxes, args.image, args.layout, args.seg)