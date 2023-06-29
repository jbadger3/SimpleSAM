import colorsys

import numpy as np
import cv2 as cv

import pycocotools.mask as cocotoolsmask


class ImageMask:
    def __init__(self, mask, color=None):
        """
        Parameters
        ----------
        mask : np.ndarray
            bool array for image mask
        color : (int, int, int)
            tuple in RGB for color of mask

        Notes
        -----
        Random colors are generated in HSL format keeping the saturation high (>= 0.5) and lightness fixed
        in a range [0.3, 0.6] for nice vibrant colors. 
        See this post: https://stackoverflow.com/questions/43437309/get-a-bright-random-colour-python for
        a discussion and
        this link: https://hslpicker.com/ for a nice HSL color picker.

        """
        self.bool_mask = mask
        if color == None:
            h, s, l = np.random.uniform(), np.random.uniform(low=0.5, high=1.0), np.random.uniform(low=0.3, high=0.6)
            self.color = np.array([int(256*i) for i in colorsys.hls_to_rgb(h,l,s)])
        else:
            self.color = color

    @property
    def image(self):
        """
        Returns an RGB based image for the mask.
        """
        h, w = self.bool_mask.shape[-2:]
        mask_image = self.bool_mask.reshape(h, w, 1) * self.color.reshape(1, 1, -1)
        return np.ascontiguousarray(mask_image, dtype=np.uint8)

    @property
    def rle(self):
        """
        The rle segmentation object for the image mask 
        """
        rle = cocotoolsmask.encode(np.asfortranarray(self.bool_mask))
        rle['counts'] = rle['counts'].decode('ascii')
        return rle


    