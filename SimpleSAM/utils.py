import datetime

import numpy as np


def create_coco_info(year = 3000, version= "1.0.0", description = "", contributor = "", url = "", date_created=None):
    """
    Convenience method for create coco info section
    info{
        "year": int,
        "version": str,
        "description": str,
        "contributor": str,
        "url": str,
        "date_created": datetime,
    }
    """
    return {
        "year": year,
        "version": version,
        "description": description,
        "contributor": contributor,
        "url": url,
        "date_created": date_created,
    }


def create_coco_image(image_id, width, height, file_name, license = 0, date_captured = None):
    """
    image{
        "id": int,
        "width": int,
        "height": int,
        "file_name": str,
        "license": int,
        "flickr_url": str, OMITTED
        "coco_url": str, OMITED
        "date_captured": datetime, OMITTED
    }
    """
    image =  {
        "id": image_id,
        "width": width,
        "height": height,
        "file_name": file_name,
        "license": license,
    }
    if date_captured != None:
        image["date_captured"] = date_captured
    return image

def create_coco_detection_annotation(anno_id, image_id, category_id, segmentation, area, bbox, iscrowd, score=None):
    """
    Convenience method for creating coco annotations.
    annotation{
        "id": int,
        "image_id": int,
        "category_id": int,
        "segmentation": RLE or [polygon],
        "area": float,
        "bbox": [x,y,width,height],
        "iscrowd": 0 or 1,
    }
    """
    annotation =  {
        "id": anno_id,
        "image_id": image_id,
        "category_id" : category_id,
        "segmentation" : segmentation,
        "area" : area,
        "bbox" : bbox,
        "iscrowd" : iscrowd
    }
    if score != None:
        annotation["score"] = score
    return annotation