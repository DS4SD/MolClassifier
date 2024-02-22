import json

import numpy as np
from pycocotools import mask as maskUtils


def load_masks(coco, img_id):
    """
    https://github.com/multimodallearning/pytorch-mask-rcnn/blob/809abba590db89779ac02c42286135f18ea08b53/coco.py
    Load instance masks for the given image.
    Different datasets use different ways to store masks. This
    function converts the different mask format to one format
    in the form of a bitmap [height, width, instances].
    Returns:
    masks: A bool array of shape [height, width, instance count] with
        one mask per instance.
    class_ids: a 1D array of class IDs of the instance masks.
    """
    # If not a COCO image, delegate to parent class.
    # image_info = self.image_info[image_id]
    # if image_info["source"] != "coco":
    #    return super(CocoDataset, self).load_mask(image_id)
    #

    ann_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(ann_ids)

    instance_masks = []
    class_ids = []
    bboxes = []
    # annotations = self.image_info[image_id]["annotations"]
    # Build mask of shape [height, width, instance_count] and list
    # of class IDs that correspond to each channel of the mask.
    for annotation in annotations:
        class_id = annotation["category_id"]

        # class_id = self.map_source_class_id("coco.{}".format(annotation['category_id']))
        if class_id:
            # bbox
            x, y, w, h = annotation["bbox"]
            bbox = [x, y, x + w, y + h, class_id]

            # mask
            m = _annToMask(annotation)
            # Some objects are so small that they're less than 1 pixel area
            # and end up rounded out. Skip those objects.
            if m.max() < 1:
                continue
            # Is it a crowd? If so, use a negative class ID.
            if annotation["iscrowd"]:
                assert False
                # Use negative class ID for crowds
                class_id *= -1
                # For crowd masks, annToMask() sometimes returns a mask
                # smaller than the given dimensions. If so, resize it.
                if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                    m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                #
            #
            instance_masks.append(m)
            bboxes.append(bbox)
            class_ids.append(class_id)
        #
    #

    # Pack instance masks into an array
    _check_mask_shapes(instance_masks, img_id, coco)

    masks = instance_masks
    # should return list of ndarrays
    # masks = np.stack(instance_masks, axis=0).astype('float')

    return masks, bboxes, class_ids


#


def _check_mask_shapes(masks, img_id, coco):
    ## Check all mask have the same dim
    if isinstance(masks, list) and len(masks) > 0:
        shape_ref = masks[0].shape
        for mask in masks:
            shape = mask.shape
            if shape_ref != shape:
                print("mask shapes different img_id", img_id)
                print("shape_ref", shape_ref)
                print("shape", shape)
                img_info = coco.loadImgs(img_id)
                print(json.dumps(img_info, indent=3))

                ann_ids = coco.getAnnIds(imgIds=img_id)
                annotations = coco.loadAnns(ann_ids)
                for annotation in annotations:
                    segm = annotation["segmentation"]
                    print(segm["size"])
                #
                assert False
                break
            #
        #
    #


#


def binary_mask_to_rle(binary_mask):
    rle = {"counts": [], "size": list(binary_mask.shape)}
    counts = rle.get("counts")

    last_elem = 0
    running_length = 0

    # fastest
    for elem in binary_mask.ravel(order="F").tolist():
        if elem != last_elem:
            counts.append(running_length)
            running_length = 0
            last_elem = elem
        #
        running_length += 1
    #

    # orginal
    """
    for i, elem in enumerate(binary_mask.ravel(order='F')):
        if elem == last_elem:
            pass
        else:
            counts.append(running_length)
            running_length = 0
            last_elem = elem
        running_length += 1
    """

    counts.append(running_length)

    return rle


#


def _annToRLE(ann):  # , height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann["segmentation"]
    height, width = segm["size"]
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(segm["counts"], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, height, width)
    else:
        # rle
        rle = ann["segmentation"]
    return rle


#


def _annToMask(ann):  # , height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = _annToRLE(ann)  # , height, width)
    m = maskUtils.decode(rle)
    return m


#
