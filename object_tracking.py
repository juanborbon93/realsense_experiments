from modules.camera_tools import  LidarCamera
from matplotlib import pyplot as plt

camera = LidarCamera(decimate=False)
camera.calibrate_plane()
proceed = input('camera calibrated! place object on scene and press ENTER to continue')
mask = camera.stream_tracked_objects(threshold=50,save_path='test.avi')