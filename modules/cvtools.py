import cv2
import numpy as np

class frame_objects:
    def __init__(self,area_diff_weight=1,dist_diff_weight=1):
        self.objects = []
        self.area_diff_weight = area_diff_weight
        self.dist_diff_weight = dist_diff_weight
    def process_frame(color,depth_mask):
        im2, contours, hierarchy = cv2.findContours(depth_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)>0:
            if len(self.objects) == 0:
                self.objects = [img_object(cnt) for c in contours]
            else:
                distances  = np.zeros((len(contours),len(self.objects)))
                for i,cnt in enumerate(contours):
                    for j,obj in enumerate(self.objects):
                        centroid_dist,area_percent_diff = obj.calculate_distance(cnt)
                        distances[i][j] = self.area_diff_weight*area_percent_diff + self.dist_diff_weight*centroid_dist
                    #todo: assign detected contours to existing objects
                    #todo: eliminate objects that move out of the frame
        return self.render_color_output(color)
    def render_color_output(self,color):
        # draw bboxes and paths for all objects in frame
        return rendered_color
        
        

class img_object:
    """
    tracks object atribues (contour,bbox,centroid,centroid_path)
    """    
    def __init__(self,contour):
        self.contour = contour
        self.path = []
        self.process_contour(self.contour)
    def process_contour(self,contour):
        """
        takes contour and returns object bounding box and centroid
        """        
        # todo: calculate from contour
        return centroid,bbox,area
    def update(self,contour)
        centroid,bbox,area = self.process_contour(contour)
        self.path.append(centroid)
        self.bbox = bbox
        self.area = area
        self.centroid = centroid
    def calculate_distance(self,candidate_contour):
        candidate_centroid,_,candidate_area = self.process_contour(candidate_contour)
        centroid_dist = np.linalg.norm(self.centroid-candidate_centroid)
        area_percent_diff = abs((self.area-candidate_area)/self.area)
        return centroid_dist,area_percent_diff
    