import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import cv2

class LidarCamera:
    def __init__(self,floor_plane=None,fps=30,decimate=False):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
        self.pipeline.start()
        self.profile = self.pipeline.get_active_profile()
        self.depth_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.depth))
        self.depth_intrinsics = self.depth_profile.get_intrinsics()
        self.depth_dims = self.depth_intrinsics.width, self.depth_intrinsics.height #w,h
        self.decimate_option = decimate
        self.decimate = rs.decimation_filter()
        self.decimate.set_option(rs.option.filter_magnitude,2**1)
        self.pipeline.stop()
        self.floor_plane = floor_plane
        align_to = rs.stream.color
        self.align = rs.align(align_to)
    def get_frame(self):
        self.pipeline.start()
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame() 
        if self.decimate_option:
            depth_frame = self.decimate.process(depth_frame)
        depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        self.pipeline.stop()
        return depth_image,color_image
    def show_stream(self):
        self.pipeline.start()
        while True:
            frames = self.pipeline.wait_for_frames()
            frames = self.align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame() 
            # depth_frame = self.decimate.process(depth_frame)
            depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
            w, h = depth_intrinsics.width, depth_intrinsics.height
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            cv2_color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            images = np.hstack((depth_colormap, cv2_color_image))
            cv2.namedWindow('Depth and Color', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Depth and Color', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
        self.pipeline.stop()
    def stream_segment_mask(self,threshold,save_path=None):
        self.pipeline.start()
        if save_path is not None:
                # fourcc = cv2.CV_FOURCC(*'XVID')  # cv2.VideoWriter_fourcc() does not exist
                print(self.color_shape)
                video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MJPG'), 30, self.color_shape,True)
        while True:
            frames = self.pipeline.wait_for_frames()
            frames = self.align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame() 
            if self.decimate_option:
                depth_frame = self.decimate.process(depth_frame)
                color_frame = self.decimate.process(color_frame)
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            mask = (depth_image < (self.floor_plane_img-threshold))*(depth_image > 0)
            mask = mask.astype(np.uint8)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(mask, alpha=0.5), cv2.COLORMAP_JET)
            display_img = np.hstack((depth_colormap,color_image))
            # stacked_mask = np.dstack((mask,mask,mask))
            cv2.namedWindow('Object Mask', cv2.WINDOW_AUTOSIZE)
            # print(stacked_mask.shape)
            cv2.imshow('Object Mask', mask*255)
            cv2.namedWindow('Object Color', cv2.WINDOW_AUTOSIZE)
            # print(stacked_mask.shape)
            cv2.imshow('Object Color', color_image)
            if save_path is not None:
                video_writer.write(color_image)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                if save_path is not None:
                    video_writer.release()
                break
        self.pipeline.stop()
    def show_frame(self):
        depth_image,color_image = self.get_frame()
        plt.figure()
        plt.subplot(211)
        plt.imshow(color_image)
        plt.subplot(212)
        plt.imshow(depth_image)
        plt.show()
    def calibrate_plane(self):
        depth_image,color_image = self.get_frame()
        self.color_shape = color_image.shape[0:2]
        img_rows,img_cols = depth_image.shape
        xyz = []
        G_lstsq = []
        Z_lstsq = []
        for row in range(img_rows):
            for col in range(img_cols):
                depth_element = depth_image[row][col]
                xyz.append([row,col,depth_element])
                if depth_element >0:
                    G_lstsq.append([row,col,1])
                    Z_lstsq.append(depth_element)
        XYZ = np.array(xyz)
        G_lstsq = np.array(G_lstsq)
        Z_lstsq = np.array(Z_lstsq)
        (rows, cols) = XYZ.shape
        G = np.ones((rows, 3))
        G[:, 0] = XYZ[:, 0]  #X
        G[:, 1] = XYZ[:, 1]  #Y
        Z = XYZ[:, 2]
        C,resid,rank,s = np.linalg.lstsq(G_lstsq, Z_lstsq)
        Z_calc  = C[0]*G[:, 0] + C[1]*G[:, 1] + C[2]
        self.floor_plane = C
        self.floor_plane_img = Z_calc.reshape((img_rows,img_cols))
    def plane_segment_mask(self,threshold):
        if self.floor_plane is None:
            raise Exception('call self.calibrate_plane() first')
        depth_image,color_image = self.get_frame()
        mask = (depth_image < (self.floor_plane_img-threshold))*(depth_image > 0)
        # mask = np.where((depth_image<(self.floor_plane_img-threshold) | (depth_image <= 0)),0,1)
        return mask.astype(int)
        