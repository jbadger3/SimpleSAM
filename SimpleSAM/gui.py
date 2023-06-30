import os
import copy
import json
import sys
import datetime
import traceback

import PySimpleGUI as sg
import tkinter as tk
import cv2 as cv
import numpy as np
import pycocotools.mask as cocotoolsmask
from pycocotools.coco import COCO
from skimage import draw

from SimpleSAM.sam import SAMController 
from SimpleSAM.imagemask import ImageMask
from SimpleSAM.utils import create_coco_info, create_coco_detection_annotation, create_coco_image

class GUI():
    def __init__(self):
        self.dataset = None
        self.coco_dir = None
        self.coco_data = None

        self.image_id = None
        self.image = None # an opencv image aka numpy.ndarray
        self.masks = [] # annotation masks
        self.mask_color = (0, 255, 0)
        
        self.graph_size = (1024, 1024) 
        self.zoom = 1
        self.lastxy = None # used to store last (x, y) when panning
        self.xoffset = 0
        self.yoffset = 0
        
        self.graph_elem_image_id = None
        self.anno_id = None
        
        self.pen_size = 1
        self.alpha = 0.6

        #SAM
        self.sam_controller = SAMController()
        self.running_bg_thread = False

        self.plot_box_start = None # (X, Y) for box start in graph coordinates
        self.box_start = None # (X, Y) tuple of box start in og image coordinates 
        self.composite_buffer = None #stores a buffered image for bbox drawing func
      
        self.window = sg.Window('SimpleSAM', self._layout(), return_keyboard_events=True, location=(0,0), resizable=True)
    
    @property
    def annotations_visible(self):
        annos_visible_elem = self.window['-ANNOTATIONS-VISIBLE-']
        return annos_visible_elem.get()

    @property
    def sam_masks_visible(self):
        sam_masks_vis_elem = self.window['-SAM-MASKS-VISIBLE-']
        return sam_masks_vis_elem.get()

    @property
    def sam_prompts_visible(self):
        sam_prompts_vis_elem = self.window['-SAM-PROMPTS-VISIBLE-']
        return sam_prompts_vis_elem.get()

    def _layout(self):
        sg.theme('DarkBlue13')
        header_font = ("Arial Bold", 16)
        subheader_font = ("Arial Bold", 10)

        self.graph_elem = sg.Graph((1024, 1024), (0, 1024), (1024, 0), background_color='white', key='-GRAPH-', enable_events=True, drag_submits=True, expand_x=True, expand_y=True)
        self.dataset_display_elem = sg.Text(f"Name: {self.dataset}", size=(80, 1), key ='-DATASET-NAME-ELEM-')
        img_col = [
        [sg.Text(f'Zoom: {self.zoom}', size=(10,1), key='-ZOOM-TEXT-'),sg.Text('', size=(80,1), key='-IMAGE-NAME-')],
        [self.graph_elem]]

        sel_img_col = sg.Column([[sg.Text('Images')],[sg.Listbox([], key='-SELECT-IMAGE-', size=(20, 20), enable_events=True)]], vertical_alignment='top')
        anno_col = sg.Column([[sg.Text('Annotations'), sg.Checkbox('Visible', default=True, key='-ANNOTATIONS-VISIBLE-', enable_events=True)], [sg.Listbox([], key='-SELECT-ANNOTATION-', size=(20, 20), enable_events=True)],[sg.Button('Remove', key='-REMOVE-ANNOTATION-')]], vertical_alignment='top')
        category_col = sg.Column([[sg.Text('Category')], [sg.Listbox([], key='-SELECT-CATEGORY-', size=(20, 20), enable_events=True)], [sg.Button('Add', key='-ADD-CATEGORY-'), sg.Button('Remove', key='-REMOVE-CATEGORY-')]], vertical_alignment='top')
        drawing_tools_col = sg.Column([
            [sg.Radio('Pan', 1,  key='-PAN-', default=True, enable_events=True), sg.Radio('Draw', 1,  key='-DRAW-', enable_events=True), sg.Radio('Erase', 1, key='-ERASE-', enable_events=True)],
            [sg.vbottom(sg.Text('Pen Size')), sg.Slider(range=(1, 50), default_value=self.pen_size, resolution=1, orientation='horizontal', key='-PEN-SIZE-', enable_events=True)],
            [sg.vbottom(sg.Text('Mask Alpha')),sg.Slider(range=(0, 1), default_value=self.alpha, resolution=0.1, orientation='horizontal', key='-ALPHA-', enable_events=True)]])
        sam_col = sg.Column([
            [sg.Text("SAM", font=header_font), sg.Text(f'Model: {self.sam_controller.model_name}', text_color='red', key='-MODEL-NAME-'), sg.Image(data=None, size=(10, 10), enable_events=True, background_color=None, key='-IMAGE-')],
            [sg.Button('Load Checkpoint', key='-LOAD-CHECKPOINT-', enable_events=True)],
            [sg.Text("Prompt Tools", font= subheader_font), sg.Checkbox('Visible', default=True, key='-SAM-PROMPTS-VISIBLE-', enable_events=True)],
            [sg.Radio('+ Add Area', 1, key='-ADD-AREA-', enable_events=True), sg.Radio('- Subtract Area', 1, key='-SUBTRACT-AREA-', enable_events=True), sg.Radio("BBox", 1, key='-BBOX-', enable_events=True),  sg.Button('Reset', key='-RESET-SAM-', enable_events=True)],
            [sg.Column([[sg.Text("Masks (predicted quality)", font= subheader_font), sg.Checkbox('Visible', default=True, key='-SAM-MASKS-VISIBLE-', enable_events=True)], [sg.Listbox([], key='-SELECT-SAM-MASK-', size=(12, 3), enable_events=True), sg.Button('Add to dataset', key='-ADD-ANNOTATION-', enable_events=True)]])],
            ])      
        tools_col = [
            [sg.Text("Dataset",font=header_font)], 
            [sg.Button('Load'), sg.Button('Save'), sg.Button('Exit')],
            [self.dataset_display_elem],
            [sg.Text("_" * 80)],
            [sel_img_col, anno_col, category_col],
            [sg.Text("_" * 80)],
            [sam_col],
            [sg.Text("_" * 80)],
            [sg.Text('Tools', font=header_font)],
            [drawing_tools_col]
            ]
        layout = [[sg.Column(img_col, justification='center'), sg.Column(tools_col, vertical_alignment='top')]]
        return layout
    

    def run(self):
        functions = {
            'Load': self.load_annotations, 
            'Save': self.save_annotations_file,
            '-ADD-CATEGORY-': self.add_category,
            '-REMOVE-CATEGORY-': self.remove_category,
            '-SELECT-IMAGE-':self.select_image,
            '-SELECT-ANNOTATION-': self.select_annotation,
            '-UPDATE-ANNOTATION-': self.update_annotation,
            '-REMOVE-ANNOTATION-': self.remove_annotation,
            '-ANNOTATIONS-VISIBLE-': self.draw_composit_image_to_graph,
            '-SAM-MASKS-VISIBLE-': self.draw_composit_image_to_graph,
            '-SAM-PROMPTS-VISIBLE-': self.draw_composit_image_to_graph,
            '-RESET-SAM-': self.reset_sam,
            '-SELECT-SAM-MASK-': self.select_sam_mask,
            '-ADD-ANNOTATION-': self.add_annotation,
            'MouseWheel:Down': self.zoom_out, 
            'MouseWheel:Up': self.zoom_in,
            '-PEN-SIZE-': self.update_pen_size,
            '-ALPHA-': self.update_alpha,
            '-LOAD-CHECKPOINT-FINISHED-': self.handle_load_checkpoint_results
            }
        while True:
            try:
                event, values = self.window.read(timeout=100)
                if event in (sg.WINDOW_CLOSED, 'Exit'):
                    self.window.close()
                    break
                elif event.startswith('-GRAPH-'):
                    self.handle_graph_event(event, values)
                elif event in functions:
                    functions[event](values)
                elif event == '-LOAD-CHECKPOINT-':
                    checkpoint = sg.popup_get_file('Select a SAM checkpoint file (e.g. ).', title="Select Checkpoint")
                    if checkpoint == None:
                        continue
                    self.window.start_thread(lambda: self.load_checkpoint(checkpoint, self.window), '-THREAD-FINISHED-')
                    self.set_thread_status(True)
                elif event == '-THREAD-FINISHED-':
                    self.set_thread_status(False)
                if self.running_bg_thread:
                    self.window['-IMAGE-'].update_animation(sg.DEFAULT_BASE64_LOADING_GIF,  time_between_frames=20)
                    
                #print(event, values) #useful for debugging/sanity checking
            except:
                traceback.print_exc(file=sys.stdout)

    def set_thread_status(self, running):
        image_elem = self.window['-IMAGE-']
        if running:
            self.running_bg_thread = True
            image_elem.update(sg.DEFAULT_BASE64_LOADING_GIF)
        else:
            self.running_bg_thread = False
            image_elem.update(None)

    def handle_graph_event(self, event, values):
        if values['-PAN-']:
            current_xy = values['-GRAPH-']
            if self.lastxy != None:
                detla_xy = (self.lastxy[0] - current_xy[0] ,self.lastxy[1] - current_xy[1])
                self.pan(detla_xy)
            if event.endswith('+UP'):
                self.lastxy = None
            else:
                self.lastxy = current_xy

        elif values['-DRAW-']:
            point = values['-GRAPH-']
            self.draw(point, 1)
        elif values['-ERASE-']:
            point = values['-GRAPH-']
            self.draw(point, 0)
        elif values['-BBOX-']:
            if event == '-GRAPH-':
                graph_point = values['-GRAPH-']
                image_point  = self.image_point_for_graph_point(graph_point)
                if self.box_start == None:
                    self.box_start = image_point
                    self.plot_box_start = graph_point
                    self.sam_controller.prompts.box = None
                    self.sam_controller.redraw_prompts_image()
                    self.set_annotation_mask_visibility(False)
                    self.set_sam_mask_visibility(True)
                    self.set_sam_prompts_visibility(True)
                    self.draw_composit_image_to_graph()
                self.draw_box(self.plot_box_start, graph_point)

            if event == '-GRAPH-+UP':
                graph_point = values['-GRAPH-']
                box_end = self.image_point_for_graph_point(graph_point)
                box = [self.box_start[0], self.box_start[1], box_end[0], box_end[1]]
                self.sam_controller.set_box(box)
                self.box_start = None
                self.plot_box_start = None
                self.set_sam_masks_list()
                self.draw_composit_image_to_graph()

        elif event == '-GRAPH-+UP':
            graph_point = values['-GRAPH-']
            image_point = self.image_point_for_graph_point(graph_point)
            if values['-ADD-AREA-']:
                self.sam_controller.add_point(image_point, 1)

            elif values['-SUBTRACT-AREA-']:
                self.sam_controller.add_point(image_point, 0)

            self.set_sam_masks_list()
            self.set_annotation_mask_visibility(False)
            self.set_sam_mask_visibility(True)
            self.set_sam_prompts_visibility(True)
            self.draw_composit_image_to_graph()

    # Loading, saving, add category****************************************************************

    def load_annotations(self, values):
        coco_dir = sg.popup_get_folder('Select a folder containing a coco dataset or a folder containing a subfolder called \'data\' with images.', title="Load COCO dataset", keep_on_top=True)
        if coco_dir == None:
            return
        self.coco_dir = coco_dir
        #validate coco dataset exists and fits expected format
        images_dir = os.path.join(coco_dir, 'data')
        assert os.path.exists(images_dir), '\'data\' folder not found in supplied directory.'
        annotation_file = os.path.join(coco_dir, 'labels.json')
        if not os.path.exists(annotation_file):
            create_coco_dataset = sg.popup_ok_cancel('No file named labels.json found','Create a new COCO dataset from images in /data folder?.', title='Create new COCO dataset?')
            if create_coco_dataset == 'Cancel':
                return
            self.create_coco_dataset()
            
        assert os.path.exists(annotation_file), '\'labels.json\' file not found in supplied directory.'
        sam_dir = os.path.join(coco_dir, 'sam')
        os.makedirs(sam_dir, exist_ok=True)
        self.sam_controller.coco_dir = self.coco_dir

        file_elem = self.window['-DATASET-NAME-ELEM-']
        file_elem.update(f"Name: {annotation_file.split('/')[-2]}")
        self.coco_data = COCO(annotation_file=annotation_file)
        category_list_elem = self.window['-SELECT-CATEGORY-']
        category_list_elem.update(list(map(lambda category: category['name'], self.coco_data.dataset['categories'])))
        
        image_ids = list(self.coco_data.imgs.keys())
        image_ids.sort()
        first_id = image_ids[0]
        listbox = self.window['-SELECT-IMAGE-']
        listbox.update(image_ids, set_to_index=[0])
        self.select_image({'-SELECT-IMAGE-':[first_id]})

    def create_coco_dataset(self):
        coco = COCO()
        today = datetime.date.today()
        coco.dataset['info'] = create_coco_info(year = today.year, version= "1.0.0", description = "Created using SimpleSAM. https://github.com/jbadger3/SimpleSAM", contributor = "", url = "", date_created=today.strftime('%Y-%m-%d'))
        coco.dataset['categories'] = []
        coco.dataset['annotations'] = []
        coco.dataset['images'] = []
        default_license_id = 0
        coco.dataset['licenses'] = [{"id": 0, "name": "default license", "url": ""}]
        image_extensions = ['bmp', 'jpeg', 'jpg', 'jpe', 
            'jp2', 'png', 'pbm', 'ppm','pxm','pnm',
            'tiff','tif', 'exr', 'hdr', 'pic']
        images_dir = os.path.join(self.coco_dir, 'data')
        images = [image for image in os.listdir(images_dir) if image.split('.')[-1] in image_extensions]
        image_id = 0
        for file_name in images:
            file_path = os.path.join(images_dir, file_name)
            image = cv.imread(file_path)
            height = image.shape[0]
            width = image.shape[1]
            image_coco = create_coco_image(image_id, width, height, file_name, default_license_id, date_captured = None)
            coco.dataset['images'].append(image_coco)
            image_id += 1
        self.coco_data = coco
        self.save_annotations_file(None)


    def save_annotations_file(self, values):
        self.update_annotation()
        coco_path = os.path.join(self.coco_dir, 'labels.json')
        with open(coco_path, 'w') as fh:
            json.dump(self.coco_data.dataset, fh)

    def add_category(self, values):
        new_category = sg.popup_get_text('Category name', title='Add Category')
        if new_category == None or new_category == '' or not isinstance(self.coco_data, COCO):
            return
        new_supercategory = sg.popup_get_text(f'Add supercategory name for new {new_category} category?\n Leave blank and click OK to skip.', title='Add Supercategory?')
        category_list_elem = self.window['-SELECT-CATEGORY-']
        if new_supercategory == None:
            return

        current_categories = self.coco_data.dataset['categories']
        if len(current_categories) == 0:
            next_id = 0
        else:
            next_id = max(list(map(lambda category: category['id'], current_categories))) + 1

        current_categories.append({'id': next_id, 'name':new_category, 'supercateogry':new_supercategory})
        self.coco_data.dataset['categories'] = current_categories
        self.coco_data.createIndex()
        category_list_elem = self.window['-SELECT-CATEGORY-']
        category_list_elem.update(list(map(lambda category: category['name'], self.coco_data.dataset['categories'])))
    
    def remove_category(self, values):
        category_list_elem = self.window['-SELECT-CATEGORY-']
        cat_id_to_remove = self.selected_category_id()
        
        if cat_id_to_remove == None:
            return
        category_name = category_list_elem.get()[0]
        result = sg.popup_yes_no(f"Are you sure you want to remove {category_name} from categories? This cannot be undone.",  title="Remove Category?")
        if result == 'No':
            return
        else:
            category_idx = next((idx for idx, cat in enumerate(self.coco_data.dataset['categories']) if cat['id'] == cat_id_to_remove), None)
            current_categories = self.coco_data.dataset['categories']
            category_to_remove = current_categories[category_idx]
            current_categories.remove(category_to_remove)
            self.coco_data.dataset['categories'] = current_categories
            self.coco_data.createIndex()
            category_list_elem.update(list(map(lambda category: category['name'], self.coco_data.dataset['categories'])))


    # Graph drawing **********************************************************************

    def draw_composit_image_to_graph(self, *args):
        composit_img = self.image_for_viewer(self.image)
        #draw annotation masks
        if self.annotations_visible:
            select_annotations_elem = self.window['-SELECT-ANNOTATION-']
            selected_mask_indexes = select_annotations_elem.get_indexes()
            for mask_index in selected_mask_indexes:
                anno_mask = self.masks[mask_index].image
                mask_img = self.image_for_viewer(anno_mask)
                composit_img = cv.addWeighted(composit_img, 1.0 - self.alpha * 0.1, mask_img, self.alpha, 0)

        #draw SAM masks
        if self.sam_controller.model_loaded:
            if self.sam_masks_visible:
                sam_mask_composite = np.zeros(composit_img.shape, dtype='uint8')
                selected_mask_index = self.selected_mask_index()
                if selected_mask_index != None:
                    mask_image = self.sam_controller.masks[selected_mask_index].image
                    mask_image  = self.image_for_viewer(mask_image)
                    composit_img = cv.addWeighted(composit_img, 1.0, mask_image, self.alpha, 0)
            if self.sam_prompts_visible:
                prompts_image = self.image_for_viewer(self.sam_controller.prompts_image)
                prompts_hsv = cv.cvtColor(prompts_image, cv.COLOR_RGB2HSV)
                non_zero = prompts_hsv[:,:,2]
                mask = non_zero > 200
                composit_img[mask] = prompts_image[mask]
    
        self.composite_buffer = composit_img
        self.draw_img(composit_img)

    def image_for_viewer(self, img):
        """
        Returns the image or part of the image currently visible in the viewer adjusted using pan and zoom
        """
        x0, y0, x1, y1 = self.image_box_for_viewer()

        viewer_image = img[y0: y1, x0: x1, :]
        viewer_image = cv.resize(viewer_image, self.graph_size, interpolation=cv.INTER_NEAREST)
        return viewer_image

    def draw_img(self, img):
        # turn our ndarray into a bytesarray of PPM image by adding a simple header:
        # this header is good for RGB. for monochrome, use P5 (look for PPM docs)
        ppm = ('P6 %d %d 255 ' % (img.shape[1], img.shape[0])).encode('ascii') + img.tobytes()
        
        # turn that bytesarray into a PhotoImage object:
        image = tk.PhotoImage(width=img.shape[1], height=img.shape[0], data=ppm, format='PPM')
        
        # for first time, create and attach an image object into the canvas of our sg.Graph:
        if self.graph_elem_image_id is None:
            self.graph_elem_image_id = self.graph_elem.Widget.create_image((0, 0), image=image, anchor=tk.NW)
            # we must mimic the way sg.Graph keeps a track of its added objects:
            self.graph_elem.Images[self.graph_elem_image_id] = image
        else:
            # we reuse the image object, only changing its content
            self.graph_elem.Widget.itemconfig(self.graph_elem_image_id, image=image)
            # we must update this reference too: 
            self.graph_elem.Images[self.graph_elem_image_id] = image

    def image_box_for_viewer(self):
        """
        Returns x0, y0, x1, y1 in image coordinates currently visible in the viewer
        """
        x0 = self.xoffset
        x1 = self.xoffset + int(self.image.shape[1] / self.zoom)
        x1 = min(x1, self.image.shape[1])

        y0 = self.yoffset
        y1 = self.yoffset + int(self.image.shape[0] / self.zoom)
        y1 = min(y1, self.image.shape[0])
        return x0, y0, x1, y1


    def draw_box(self, point1, point2):
        """
        Drawing function for drawing boxes with SAM BBox tool.
        """
        composit_img = copy.copy(self.composite_buffer)
        bbox_image = cv.rectangle(composit_img, point1, point2, (0, 255, 0), thickness=1)
        self.draw_img(bbox_image)

    # Zoom and pan ***************************************************************************

    def zoom_in(self, values):
        if not values['-PAN-'] == True:
            return
        if not isinstance(self.image, np.ndarray):
            return
        if self.zoom >= 1:
            self.zoom += 1
        zoom_text_elem = self.window['-ZOOM-TEXT-']
        zoom_text_elem.update(f'Zoom: {self.zoom}')
        self.draw_composit_image_to_graph()

    def zoom_out(self, values):
        if not values['-PAN-'] == True:
            return
        if not isinstance(self.image, np.ndarray):
            return 
        if self.zoom > 1:
            self.zoom -= 1
        zoom_text_elem = self.window['-ZOOM-TEXT-']
        zoom_text_elem.update(f'Zoom: {self.zoom}')
        self.draw_composit_image_to_graph()

    def pan(self, detla_xy):
        x0, y0, x1, y1 = self.image_box_for_viewer()
        new_x_offset = self.xoffset + detla_xy[0]
        if new_x_offset < self.xoffset:
            self.xoffset = max(0, new_x_offset)
        elif new_x_offset > self.xoffset:
            if x1 + detla_xy[0] <= self.image.shape[1]:
                self.xoffset = new_x_offset
        
        new_y_offset = self.yoffset + detla_xy[1]
        if new_y_offset < self.yoffset:
            self.yoffset = max(0, new_y_offset)
        elif new_y_offset > self.yoffset:
            if y1 + detla_xy[1] <= self.image.shape[0]:
                self.yoffset = new_y_offset

        self.draw_composit_image_to_graph()

    # Tool drawing and related **********************************************************

    def draw(self, point, color):
        """
        Used to draw or erase pixels on an image mask
        """
        image_mask = self.mask_for_anno_id()
        if isinstance(image_mask, ImageMask):
            bool_mask = image_mask.bool_mask
            point = self.image_point_for_graph_point(point)
            rr, cc = draw.disk((point[1], point[0]), self.pen_size, shape = bool_mask.shape)
            bool_mask[rr, cc] = color
        self.draw_composit_image_to_graph()

    def update_pen_size(self, values):
        self.pen_size = int(values['-PEN-SIZE-'])

    def update_alpha(self, values):
        self.alpha = values['-ALPHA-']
        self.draw_composit_image_to_graph()

    # Image selection and Annotation *************************************************

    def select_image(self, values):
        self.update_annotation()
        # reset state vars that need it

        self.image_id = values['-SELECT-IMAGE-'][0]
        image_file_name  = self.coco_data.imgs[self.image_id]['file_name']
        #### add set image_file to samcontroller
        file_name_text_elem = self.window['-IMAGE-NAME-']
        file_name_text_elem.update(image_file_name)
        file_path  = os.path.join(self.coco_dir, 'data',  image_file_name)

        
        annotations = self.coco_data.imgToAnns[self.image_id]
        self.update_annotations_list_element()
        #reset SAM masks list elem
        masks_list_elem = self.window['-SELECT-SAM-MASK-']
        masks_list_elem.update([], set_to_index=[])

        self.masks = []
        for annotation in annotations:
            seg_mask = self.segmentation_mask_for_annotation(annotation)
            self.masks.append(ImageMask(seg_mask))
        
        self.image = cv.imread(file_path) #image is h, w, c
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
        self.sam_controller.set_image(file_path, self.image)

        self.window['-GRAPH-'].set_size((self.image.shape[1], self.image.shape[0]))
        self.window['-GRAPH-'].change_coordinates((0, self.image.shape[0]), (self.image.shape[1], 0))
        self.window['-GRAPH-'].Widget.configure(height=self.image.shape[0], width=self.image.shape[1])
        self.graph_size = (self.image.shape[1], self.image.shape[0])
        
        
        self.draw_composit_image_to_graph()

        
    def update_annotations_list_element(self, annotation_index=0):
        annotations = self.coco_data.imgToAnns[self.image_id]
        if annotation_index == -1:
            annotation_index = len(annotations) - 1
        anno_list_elem = self.window['-SELECT-ANNOTATION-']
        category_list_elem = self.window['-SELECT-CATEGORY-']
        anno_list_elem.update(list(map(lambda anno: anno['id'], annotations)))
        if len(annotations) > 0:
            selected_anno = annotations[annotation_index]
            self.anno_id = selected_anno['id']
            anno_list_elem.update(set_to_index=[annotation_index])
            category_idx = next((idx for idx, cat in enumerate(self.coco_data.dataset['categories']) if cat['id'] == selected_anno['category_id']), None)
            if category_idx != None:
                category_list_elem.update(set_to_index=[category_idx])
            else:
                category_list_elem.update(set_to_index=[])
        else:
            anno_list_elem.update(set_to_index=[])
            category_list_elem.update(set_to_index=[])

    def add_annotation(self, values):
        self.update_annotation()
        self.anno_id = None
        mask_list_elem = self.window['-SELECT-SAM-MASK-']
        selected_mask_idxs  = mask_list_elem.get_indexes()
        if not len(selected_mask_idxs) > 0:
            return
        mask_idx = selected_mask_idxs[0]
        if len(self.coco_data.anns.keys()) > 0:
            anno_id = max(set(self.coco_data.anns.keys())) + 1
        else:
            anno_id = 0
        image_mask =  self.sam_controller.masks[mask_idx]
        segmentation = image_mask.rle
        area = float(cocotoolsmask.area(segmentation))
        bbox = cocotoolsmask.toBbox(segmentation).tolist()
        iscrowd = 0
        annotation = create_coco_detection_annotation(anno_id, self.image_id, 0, segmentation, area, bbox, 0)
        self.coco_data.dataset['annotations'].append(annotation)
        self.coco_data.createIndex()
        self.masks.append(image_mask)
        self.update_annotations_list_element(annotation_index=-1)
        self.reset_sam(None)
        self.select_annotation({'-SELECT-ANNOTATION-': [anno_id]})

    def remove_annotation(self, values):
        """
        Removes the currently selected annotation
        """
        anno_id = self.selected_annotation_id()
        if anno_id != None:
            annotation = self.coco_data.anns[anno_id]
            self.coco_data.dataset['annotations'].remove(annotation)
            self.coco_data.createIndex()
            annotations = self.coco_data.imgToAnns[self.image_id]
            anno_list_elem = self.window['-SELECT-ANNOTATION-']
            anno_list_elem.update(list(map(lambda anno: anno['id'], annotations)))
            anno_mask = self.mask_for_anno_id()
            if anno_mask != None:
                self.masks.remove(anno_mask)
            self.anno_id = None
        self.draw_composit_image_to_graph()
    
    def select_annotation(self, values):
        if len(values['-SELECT-ANNOTATION-']) == 0:
            return
        if self.anno_id != None:
            self.update_annotation(anno_id = self.anno_id) #update and save the current annotation first
        self.anno_id = values['-SELECT-ANNOTATION-'][0]
        annotations = self.coco_data.imgToAnns[self.image_id]
        annotation = next(anno for anno in annotations if anno['id'] == self.anno_id)
        category_idx = next((idx for idx, cat in enumerate(self.coco_data.dataset['categories']) if cat['id'] == annotation['category_id']), None)
        category_list_elem = self.window['-SELECT-CATEGORY-']
        if category_idx != None:
            category_list_elem.update(set_to_index=[category_idx])
        else:
            category_list_elem.update(set_to_index=[])
        self.set_annotation_mask_visibility(True)
        self.set_sam_mask_visibility(False)
        self.set_sam_prompts_visibility(False)
        self.draw_composit_image_to_graph()

    def update_annotation(self, anno_id = None):
        """
        Takes all of the current values of the selected annotation(default) and updates coco_data accordingly.
        """
        if anno_id == None:
            anno_id = self.selected_annotation_id()
        if anno_id == None or self.image_id == None:
            return
        image_mask = self.mask_for_anno_id(anno_id = anno_id)
        category_id  = self.selected_category_id()
        rle_segmentation = image_mask.rle
        area = int(cocotoolsmask.area(rle_segmentation))
        bbox = cocotoolsmask.toBbox(rle_segmentation).tolist()
        annotation = create_coco_detection_annotation(
            anno_id = anno_id,
            image_id = self.image_id,
            category_id = category_id,
            segmentation = rle_segmentation,
            area = area,
            bbox = bbox,
            iscrowd = 0)
        
        anno_idx = next((idx for idx, anno in enumerate(self.coco_data.dataset['annotations']) if anno['id'] == anno_id), None)
        self.coco_data.dataset['annotations'].pop(anno_idx)
        self.coco_data.dataset['annotations'].insert(anno_idx, annotation)
        self.coco_data.createIndex()


    def segmentation_mask_for_annotation(self, annotation):
        segmentation = annotation['segmentation']
        counts = segmentation['counts']
        if not isinstance(counts, str):
            rle = cocotoolsmask.frPyObjects(segmentation, segmentation['size'][0], segmentation['size'][1]) #create RLE from uncompressed counts
        else:
            rle = segmentation
        return cocotoolsmask.decode(rle)

    # SAM
    def load_checkpoint(self, checkpoint, window):
        result = self.sam_controller.load_checkpoint(checkpoint)
        thread_results = {}
        if result == True:
            thread_results['sucess'] = True
            thread_results['error_msg'] = None
        else:
            thread_results['sucess'] = False
            thread_results['error_msg'] = "Failed to load checkpoint!  Ensure checkpoint file has not been renamed and was downloaded from https://github.com/facebookresearch/segment-anything."
        window.write_event_value('-LOAD-CHECKPOINT-FINISHED-', thread_results)


    def handle_load_checkpoint_results(self, values):
        thread_results = values['-LOAD-CHECKPOINT-FINISHED-']
        if thread_results['sucess'] != True:
            sg.popup_error(thread_results['error_msg'], title="Error !")
        else:
            model_name_elem = self.window['-MODEL-NAME-']
            model_name_elem.update(f'Model: {self.sam_controller.model_name}', text_color=sg.theme_element_text_color())
        if self.image_id != None and isinstance(self.image, np.ndarray):
            image_file  = self.coco_data.imgs[self.image_id]['file_name']
            file_path  = os.path.join(self.coco_dir, 'data',  image_file)
            self.sam_controller.set_image(file_path, self.image)

    def reset_sam(self, values):
        self.sam_controller.reset_sam()
        mask_list_elem = self.window['-SELECT-SAM-MASK-']
        mask_list_elem.update([], set_to_index=[])
        self.draw_composit_image_to_graph()

    def set_sam_masks_list(self):
        mask_list_elem = self.window['-SELECT-SAM-MASK-']
        mask_elements = [f"{idx} ({quality:.3f})" for idx, quality in enumerate(self.sam_controller.mask_qualities)]
        mask_list_elem.update(mask_elements)
        if len(self.sam_controller.mask_qualities) > 0:
            best_idx = np.argmax(self.sam_controller.mask_qualities)
            mask_list_elem.update(set_to_index=[best_idx])

    def select_sam_mask(self, values):
        self.set_sam_mask_visibility(True)
        self.set_sam_prompts_visibility(True)
        self.set_annotation_mask_visibility(False)
        self.draw_composit_image_to_graph()

    # UI state helper functions ********************************************************************
    def mask_for_anno_id(self, anno_id = None):
        """
        Returns the annotation mask for the given anno_id, selected value in the -SELECT-ANNOTATION- element, or None if there is no selection
        """
        if anno_id == None:
            anno_id = self.selected_annotation_id()
        if anno_id != None:
            annotations = self.coco_data.imgToAnns[self.image_id]
            anno_idx = next(idx for idx, anno in enumerate(annotations) if anno['id'] == anno_id)
            return self.masks[anno_idx]
        else:
            return None


    def selected_annotation_id(self):
        """
        Returns the 'id' of the currently selected annotation or None if there is no current selection.
        """
        anno_list_elem = self.window['-SELECT-ANNOTATION-']
        anno_id = anno_list_elem.get()
        if len(anno_id) > 0:
            return anno_id[0]
        else:
            return None

    def selected_category_id(self):
        """
        Returns the curretnly selected category_id for an annotation or None
        """
        category_list_elem = self.window['-SELECT-CATEGORY-']
        category_name = category_list_elem.get()
        if len(category_name) == 0:
            return None
        else:
            category_name = category_name[0]
        return next((category['id'] for category in self.coco_data.dataset['categories'] if category['name'] == category_name), None)

    def image_point_for_graph_point(self, point):
        """
        Given a point in the graph's coordinate space returns the cooresponding point on the image.
        """
        x0, y0, x1, y1 = self.image_box_for_viewer()
        x_pixels = x1 - x0
        y_pixels = y1 - y0
        points_per_pixel_x = self.graph_size[0] / x_pixels
        points_per_pixel_y = self.graph_size[1] / y_pixels
        point = (self.xoffset + int(point[0] / points_per_pixel_x), self.yoffset + int(point[1] / points_per_pixel_y))
        return point

    def selected_mask_index(self):
        """
        Returns the currently selected mask index or None if there is no current selection.
        """
        select_mask_elem = self.window['-SELECT-SAM-MASK-']
        selected_items = select_mask_elem.get_indexes()
        if len(selected_items) == 0:
            return None
        else:
            return selected_items[0]

    def set_annotation_mask_visibility(self, visibility):
        anno_vis_elemnt = self.window['-ANNOTATIONS-VISIBLE-']
        anno_vis_elemnt.update(visibility)

    def set_sam_mask_visibility(self, visibility):
        sam_mask_vis_elem = self.window['-SAM-MASKS-VISIBLE-']
        sam_mask_vis_elem.update(visibility)

    def set_sam_prompts_visibility(self, visibility):
        sam_prompts_vis_elem = self.window['-SAM-PROMPTS-VISIBLE-']
        sam_prompts_vis_elem.update(visibility)

