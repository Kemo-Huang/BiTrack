import numpy as np


class SuscapeObject:

    def __init__(self, obj_id=0, obj_type='Car', lidar_box=None, score=None):
        self.obj_id = obj_id
        self.obj_type = obj_type
        self.position = {
            'x': 0,
            'y': 0,
            'z': 0
        }
        self.rotation = self.position.copy()
        self.scale = self.position.copy()
        self.score = score
        if lidar_box is not None:
            self.from_lidar_box(lidar_box)
    
    def from_lidar_box(self, lidar_box):
        self.position['x'] = lidar_box[0]
        self.position['y'] = lidar_box[1]
        self.position['z'] = lidar_box[2]
        self.scale['x'] = lidar_box[3]
        self.scale['y'] = lidar_box[4]
        self.scale['z'] = lidar_box[5]
        self.rotation['z'] = lidar_box[6]
        return self
    
    def serialize(self) -> dict:
        res = {
            'obj_id': self.obj_id,
            'obj_type': self.obj_type,
            'psr': {
                'position': self.position,
                'rotation': self.rotation,
                'scale': self.scale
            }
        }
        if self.score is not None:
            res['score'] = self.score
        return res