import numpy as np

class Prompts:
    """
    Stores the state of input prompts used by SamPredictor.predict method.
    
    Attributes
    ----------
    point_coords : np.ndarray or None 
        A Nx2 array of point prompts to the model. Each point is in (X,Y) in pixels.
    point_labels : np.ndarray or None
        A length N array of labels for the point prompts. 1 indicates a foreground point and 0 indicates abackground point.
    box : np.ndarray or None
        A length 4 array given a box prompt to the
        model, in XYXY format.
    mask_input: np.ndarray
        A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where for SAM, H=W=256.
    multimask_output: bool
        If true, the model will return three masks. For ambiguous input prompts (such as a single click), this will often produce better masks than a single prediction. If only a single mask is needed, the model's predicted quality score can be used to select the best mask. For non-ambiguous prompts, such as multiple input prompts, multimask_output=False can give better results.
    return_logits: bool 
        If true, returns un-thresholded masks logits instead of a binary mask.

    Notes
    -----
    * The Attributes section of this doc is verbatim from the SAM project from the SamPredictor.predict method.
    """
    def __init__(
        self, 
        point_coords = None, 
        point_labels = None, 
        box = None, 
        mask_input = None,
        multimask_output = True, 
        return_logits = False):
        
        self._point_coords = point_coords
        self._point_labels = point_labels
        self._box = box
        self._mask_input = mask_input
        self.multimask_output = multimask_output
        self.return_logits = return_logits

    @property
    def point_coords(self):
        if not isinstance(self._point_coords, np.ndarray):
            return []
        return self._point_coords

    @property
    def point_labels(self):
        if not isinstance(self._point_labels, np.ndarray):
            return []
        return self._point_labels

    @property
    def box(self):
        return self._box

    @box.setter
    def box(self, box):
        if not isinstance(box, np.ndarray) and box != None:
            self._box = np.array(box)
        else:
            self._box = box

    @property
    def mask_input(self):
        return self._mask_input

    @mask_input.setter
    def mask_input(self, mask_input):
        self._mask_input = mask_input
    
    def add_point(self, point, label):
        """
        Parameters
        ----------
        point : [x, y] or equivalent np.ndarray
        """
        if not isinstance(self._point_coords, np.ndarray):
            self._point_coords = np.array([point])
        else:
            self._point_coords = np.append(self._point_coords, [point], axis=0)
        if not isinstance(self._point_labels, np.ndarray):
            self._point_labels = np.array([label])
        else:
            self._point_labels = np.append(self._point_labels, [label], axis=0)

    def remove_point(self, point):
        """
        Removes a point and corresponding label from the prompt inputs

        Parameters
        ----------
        point : [x, y] or equivalent np.ndarray
        """
        matching_indices = np.argwhere((self._point_coords == point).all(axis=1)).flatten()
        if len(matching_indices) > 0:
            point_index = matching_indices[0]
        else:
            return
        self._point_coords = np.delete(self._point_coords, point_index, axis=0)
        self._point_labels = np.delete(self._point_labels, point_index)

    
    def reset(self):
        """Resets all promt input attributes.
        """
        self._point_coords = None
        self._point_labels = None
        self._box = None
        self._multimask_output = True
        self._return_logits = False