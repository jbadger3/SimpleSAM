import json

class SAMDataset():
    """


    SA-1B Datset format.  Saved as one json file per image.
    https://github.com/facebookresearch/segment-anything
    {
    "image"                 : image_info,
    "annotations"           : [annotation],
    }

    image_info {
        "image_id"              : int,              # Image id
        "width"                 : int,              # Image width
        "height"                : int,              # Image height
        "file_name"             : str,              # Image filename
    }

    annotation {
        "id"                    : int,              # Annotation id
        "segmentation"          : dict,             # Mask saved in COCO RLE format.
        "bbox"                  : [x, y, w, h],     # The box around the mask, in XYWH format
        "area"                  : int,              # The area in pixels of the mask
        "predicted_iou"         : float,            # The model's own prediction of the mask's quality
        "stability_score"       : float,            # A measure of the mask's quality
        "crop_box"              : [x, y, w, h],     # The crop of the image used to generate the mask, in XYWH format
        "point_coords"          : [[x, y]],         # The point coordinates input to the model to generate the mask
    """

    def __init__(self, file_path = None):
        if isinstance(file_path, str):
            with open(file_path, 'r') as fh:
                dataset = json.load(fh)
                self.image_info = dataset['image']
                self.annotations = dataset['annotations']

        else:
            self.image_info = {}
            self.annotations = []

    def json(self):
        return {
            "image": self.image_info,
            "annotations": self.annotations
        }
    


