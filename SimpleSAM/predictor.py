
import torch
from segment_anything.modeling import Sam
from segment_anything import SamPredictor
import numpy as np

from SimpleSAM.prompt import Prompts

class SimpleSamPredictor(SamPredictor):
    def __init__(
        self,
        sam_model: Sam,
    ) -> None:
        """
        A wrapper class around SamPredictor that adds set_embedding to set embeddings directly for inference.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__(sam_model)
        

    def set_embedding(self, image_embedding, image_height, image_width):
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image_height: int
            height of the original image in pixels
          image_width: int
            width of the original image in pixels
          image_embedding: np.ndarray
            numpy array from embedding previously calculated using SamPredictor
        """
        self.reset_image()
        target_length = self.model.image_encoder.img_size
        new_height, new_width = self.transform.get_preprocess_shape(image_height, image_width, self.model.image_encoder.img_size)
        self.original_size = (image_height, image_width)
        self.input_size = (new_height, new_width)
        self.features = torch.as_tensor(image_embedding, device = self.device)
        self.is_image_set = True


