import cv2
import numpy as np

class frame_objects:
    """keeps track of objects as they move through frame
    """    
    def __init__(self,area_diff_weight:float=1,dist_diff_weight:float=1,min_area=0):
        """
        Args:
            area_diff_weight (float, optional): scalar weight for area diff when calculating contour matches. Defaults to 1.
            dist_diff_weight (float, optional): scalar weight for centroid distance diff when calculating contour matches. Defaults to 1.
            min_area (float,optional): threshold to filter out contours that are too small.
        """        
        self.objects = []
        self.area_diff_weight = area_diff_weight
        self.dist_diff_weight = dist_diff_weight
        self.min_area = min_area
        self.frame_count = 0
    def process_frame(self,color:np.ndarray,depth_mask:np.ndarray):
        """ will find contours and match them to existing objects (if any)

        Args:
            color (np.ndarray): color image array (width,height,3)
            depth_mask (np.ndarray): depth map (width,height)

        Returns:
            [(np.ndarray)]: color image overlayed with object bounding boxes, centroids, and trajectories
        """             
        self.frame_count += 1
        contours, hierarchy = cv2.findContours(depth_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)>0:
            if len(self.objects) == 0:
                self.objects = [img_object(cnt,self.frame_count) for cnt in contours if cv2.moments(cnt)['m00']>self.min_area]
            else:
                contours = [cnt for cnt in contours if cv2.moments(cnt)['m00']>self.min_area]
                distances  = np.zeros((len(contours),len(self.objects)))
                for i,cnt in enumerate(contours):
                    for j,obj in enumerate(self.objects):
                        centroid_dist,area_percent_diff = obj.calculate_distance(cnt)
                        distances[i][j] = self.area_diff_weight*area_percent_diff + self.dist_diff_weight*centroid_dist
                filtered_distances,kept_cnts,unmatched_cnts = self.filter_cnts(distances)
                matched_objects = self.assign_contours(filtered_distances)
                for cnt_index,obj_index in zip(kept_cnts,matched_objects):
                    if distances[cnt_index][obj_index] < 250:
                        self.objects[obj_index].update(contours[cnt_index],self.frame_count)
                for new_cnt_index in unmatched_cnts: 
                    new_cnt = contours[new_cnt_index]
                    M = cv2.moments(new_cnt)
                    if M["m00"]>self.min_area:
                        self.objects.append(img_object(new_cnt,self.frame_count))
        for i,obj in enumerate(self.objects):
            if obj.last_frame+5<self.frame_count:
                del self.objects[i]
        return self.render_color_output(color)
    def render_color_output(self,color:np.ndarray):
        """ overlays bounding boxes and trajectories

        Args:
            color (np.ndarray): color image

        Returns:
            [(np.ndarray)]: rendered color image
        """        
        for obj in self.objects:
            drawing_color = (255,0,0)
            line_thickness = 2
            cv2.rectangle(
                color, 
                (int(obj.bbox[0]), int(obj.bbox[1])), 
                (int(obj.bbox[0]+obj.bbox[2]),int(obj.bbox[1]+obj.bbox[3])), 
                drawing_color, line_thickness)
            path_length = len(obj.path)
            if path_length>2:
                for i in range(1,path_length):
                    cv2.line(color,obj.path[i-1],obj.path[i],drawing_color,line_thickness)
        return color
    @staticmethod
    def find_repeats(l):
        index_set = set()
        repeat_set = set()
        for i in l:
            if i not in index_set:
                index_set.add(i)
            else:
                repeat_set.add(i)
        return repeat_set
    @classmethod
    def assign_contours(cls,m):
        object_candidates = np.zeros((m.shape[0],),dtype=np.int8)
        for i,cnt_dists in enumerate(m):
            max_index = np.where(cnt_dists==np.amin(cnt_dists))[0][0]
            object_candidates[i]=max_index
        repeat_object_indices = cls.find_repeats(object_candidates)
        if len(repeat_object_indices)==0:
            return object_candidates
        else:  
            for r in repeat_object_indices:
                repeat_contour_indices = np.where(object_candidates==r)[0]
                cnts_to_arbitrate = m[repeat_contour_indices,r]
                arbitrated_minimum = np.where(cnts_to_arbitrate==np.amin(cnts_to_arbitrate))[0][0]
                row_filter = [i for i in range(m.shape[0]) if i != arbitrated_minimum]
                m[row_filter,r] = np.amax(m)+1
                return cls.assign_contours(m)
    @staticmethod
    def filter_cnts(m):
        shape = m.shape
        if shape[0]==shape[1]:
            rows_kept = [i for i in range(shape[0])]
            rows_deleted=[]
            return m,rows_kept,rows_deleted
        else:
            rows_deleted = []
            while m.shape[0]>m.shape[1]:
                min_row_vals = np.amin(m,0)
                max_min_val = np.where(min_row_vals==np.amax(min_row_vals))
                rows_deleted.append(max_min_val[0][0])
                m = np.delete(m,max_min_val,0)
            rows_kept = [i for i in range(shape[0]) if i not in rows_deleted]
            return m,rows_kept,rows_deleted
        

class img_object:
    """
    tracks object atribues (contour,bbox,centroid,centroid_path)
    """    
    def __init__(self,contour,frame):
        self.contour = contour
        self.centroid,self.bbox,self.area = self.process_contour(self.contour)
        self.path = [self.centroid]
        self.last_frame = frame
    def process_contour(self,cnt):
        """
        takes contour and returns object bounding box and centroid
        """        
        bbox = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        area = max(M["m00"],1)
        centroid = (int(M["m10"]/area),int(M["m01"]/area))
        return centroid,bbox,area
    def update(self,contour,frame):
        centroid,bbox,area = self.process_contour(contour)
        self.last_frame=frame
        self.path.append(centroid)
        self.bbox = bbox
        self.area = area
        self.centroid = centroid
    def calculate_distance(self,candidate_contour):
        candidate_centroid,_,candidate_area = self.process_contour(candidate_contour)
        centroid_dist = np.linalg.norm(np.array(self.centroid)-np.array(candidate_centroid))
        area_percent_diff = abs((self.area-candidate_area)/self.area)
        return centroid_dist,area_percent_diff
    
