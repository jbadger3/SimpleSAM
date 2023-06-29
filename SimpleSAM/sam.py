import os
import re

import numpy as np
import cv2 as cv
from segment_anything import sam_model_registry
from segment_anything.modeling import Sam

from SimpleSAM.prompt import Prompts
from SimpleSAM.predictor import SimpleSamPredictor
from SimpleSAM.imagemask import ImageMask

class SAMController():
    def __init__(self):
        self.model_name = None
        self.coco_dir = None
        self.prompts = Prompts()
        self.predictor = None

        self.masks = [] # [ImageMask]
        self.mask_qualities = []
        self.low_res_masks = [] #can be used as input to SamPredictor.predict

        self.image_shape = (0,0,3) #shape of current image
        self.prompts_image = None #opencv image used to store points and bbox

    @property
    def model_loaded(self):
        return isinstance(self.predictor, SimpleSamPredictor)
    

    def load_checkpoint(self, checkpoint):
        search_results = re.findall('sam_(vit_[a-z])', checkpoint)
        if not len(search_results) > 0:
            return "Failed to load checkpoint!  Ensure checkpoint file has not been renamed and was downloaded from https://github.com/facebookresearch/segment-anything."
        if checkpoint.endswith('.onnx'):
            return "ONNX models are not supported!  Please use one of the .pth checkpoint files from https://github.com/facebookresearch/segment-anything."
        model_type = search_results[0]
        self.model_name = f'sam_{model_type}'
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device='cuda')
        self.predictor = SimpleSamPredictor(sam)
        return True

    def set_coco_dir(self, coco_dir):
        self.coco_dir = coco_dir


    def reset_prompts(self):
        self.prompts.reset()

    def clear_output(self):
        self.masks = []
        self.mask_qualities = []
        self.low_res_masks = []

    def reset_sam(self):
        self.reset_prompts()
        self.clear_output()
        self.prompts_image = np.zeros(self.image_shape, dtype='uint8')

    def set_image(self, image_file, image):
        """
        Setting the image file prepares the controller and underlying SamPredictor for prediction.
        Parameters
        ----------
        image_file : str
        image : np.ndarray image format should be HWC where C is RGB
        """
        self.image_shape = image.shape
        self.reset_sam()
        if not isinstance(self.predictor, SimpleSamPredictor):
            return

        embedding_file_path = self.embedding_file_for_image_file(image_file)
        if os.path.exists(embedding_file_path):
            image_embedding = self.load_embedding(embedding_file_path)
            image_height = image.shape[0]
            image_width = image.shape[1]
            self.predictor.set_embedding(image_embedding, image_height, image_width)
        else:
            self.predictor.set_image(image)
            image_embedding = self.predictor.get_image_embedding().cpu().numpy()
            self.save_embedding(image_embedding, embedding_file_path)

    def save_embedding(self, image_embedding, file_path):
        """
        Saves an image embedding to file_path 
        """
        with open(file_path, 'wb') as fh:
            np.save(fh, image_embedding)

    def load_embedding(self, file_path):
        with open(file_path, 'rb') as fh:
            image_embedding = np.load(fh)
        return image_embedding

    def embedding_file_for_image_file(self, image_file):
        """
        Given a path/to/data/image.jpg returns the corresponding path/to/sam/image.npy
        """
        directory_parts = image_file.split('/')
        directory_parts[-2] = 'sam'
        image_name = directory_parts[-1].split('.')[0]
        directory_parts[-1] = f'{image_name}.npy'
        return '/'.join(directory_parts)

    def add_point(self, point, point_label):
        self.prompts.add_point(point, point_label)
        color = (255, 0, 0) if point_label == 0 else (0, 255, 0)
        self.prompts_image = cv.circle(self.prompts_image, point, 6, color, thickness=-1, lineType=16)
        self.prompts_image = cv.circle(self.prompts_image, point, 6, (255,255,255), thickness=1, lineType=16)
        self.predict()

    def set_box(self, box):
        self.prompts.box = box
        self.prompts_image = cv.rectangle(self.prompts_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0))
        self.predict()

    def redraw_prompts_image(self):
        self.prompts_image = np.zeros(self.image_shape, dtype='uint8')
        for point, point_label in zip(self.prompts.point_coords, self.prompts.point_labels):
            color = (255, 0, 0) if point_label == 0 else (0, 255, 0)
            self.prompts_image = cv.circle(self.prompts_image, point, 6, color, thickness=-1, lineType=16)
            self.prompts_image = cv.circle(self.prompts_image, point, 6, (255,255,255), thickness=1, lineType=16)
        if isinstance(self.prompts.box, np.ndarray):
            box = self.prompts.box
            self.prompts_image = cv.rectangle(self.prompts_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0))

    def predict(self):
        point_coords = None
        if len(self.prompts.point_coords) > 0:
            point_coords = self.prompts.point_coords
        point_labels = None
        if len(self.prompts.point_labels) > 0:
            point_labels = self.prompts.point_labels
        box = self.prompts.box
        masks, mask_qualities, low_res_masks = self.predictor.predict(point_coords = point_coords, point_labels = point_labels, box = box)
        self.masks = [ImageMask(mask) for mask in masks]
        self.mask_qualities = mask_qualities
        self.low_res_masks = low_res_masks