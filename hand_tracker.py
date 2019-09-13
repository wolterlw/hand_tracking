import csv
import cv2
import numpy as np
import tensorflow as tf

class HandTracker():
    r"""
    Class to use Google's Mediapipe HandTracking pipeline from Python.
    So far only detection of a single hand is supported.
    Any any image size and aspect ratio supported.

    Args:
        palm_model: path to the palm_detection.tflite
        joint_model: path to the hand_landmark.tflite
        anchors_path: path to the csv containing SSD anchors
    Ourput:
        (21,2) array of hand joints.
    Examples::
        >>> det = HandTracker(path1, path2, path3)
        >>> input_img = np.random.randint(0,255, 256*256*3).reshape(256,256,3)
        >>> keypoints = det(input_img)
    """
    def __init__(self, palm_model, joint_model, anchors_path):
        self.interp_palm = tf.lite.Interpreter(palm_model)
        self.interp_palm.allocate_tensors()
        self.interp_joint = tf.lite.Interpreter(joint_model)
        self.interp_joint.allocate_tensors()
        
        # reading the SSD anchors
        with open(anchors_path, "r") as csv_f:
            self.anchors = np.r_[
                [x for x in csv.reader(csv_f, quoting=csv.QUOTE_NONNUMERIC)]
            ]
        output_details = self.interp_palm.get_output_details()
        input_details = self.interp_palm.get_input_details()
        
        self.in_idx = input_details[0]['index']
        self.out_reg_idx = output_details[0]['index']
        self.out_clf_idx = output_details[1]['index']
        
        self.in_idx_joint = self.interp_joint.get_input_details()[0]['index']
        self.out_idx_joint = self.interp_joint.get_output_details()[0]['index']
        
        self.R90 = np.r_[[[0,1],[-1,0]]]
        self.target_triangle = np.float32([
                        [128, 128],
                        [128, 0],
                        [0, 128]
                    ])
    
    def _getTriangle(self, kp0, kp2, dist=1):
        dir_v = kp2 - kp0
        dir_v /= np.linalg.norm(dir_v)

        dir_v_r = dir_v @ self.R90.T
        return np.float32([kp2, kp2+dir_v*dist, kp2 + dir_v_r*dist])
    
    @staticmethod
    def _im_normalize(img):
         return np.ascontiguousarray(
             2 * ((img / 255) - 0.5
        ).astype('float32'))
       
    @staticmethod
    def _sigm(x):
        return 1 / (1 + np.exp(-x) )
    
    @staticmethod
    def _pad1(x):
        return np.pad(x, ((0,0),(0,1)), constant_values=1, mode='constant')
    
        
    def _predict_palm(self, img_norm):
        self.interp_palm.set_tensor(self.in_idx, img_norm[None])
        self.interp_palm.invoke()

        out_reg = self.interp_palm.get_tensor(self.out_reg_idx)[0]
        out_clf = self.interp_palm.get_tensor(self.out_clf_idx)[0,:,0]
        
        return out_reg, out_clf
    
    def _predict_joints(self, img_norm):
        self.interp_joint.set_tensor(self.in_idx_joint, img_norm.reshape(1,256,256,3))
        self.interp_joint.invoke()

        joints = self.interp_joint.get_tensor(self.out_idx_joint)
        return joints.reshape(-1,2)
    
    @staticmethod
    def _get_R(center, keypoints, scale=1):
        dir_v = keypoints[0] - keypoints[2]

        rot_deg = np.rad2deg(
            np.arctan(dir_v[0] / dir_v[1])
        )
        R = cv2.getRotationMatrix2D(tuple(center), 180-rot_deg, scale)
        return R

    @staticmethod
    def _get_T(center, to_center=(128,128)):
        offset = center - to_center
        T = np.eye(3)
        T[:2,2] = offset
        return T
    
    def __call__(self, img):
        shape = np.r_[img.shape]
        pad = (shape.max() - shape[:2]).astype('uint32') // 2
        img_pad = np.pad(img, ((pad[0],pad[0]), (pad[1],pad[1]), (0,0)), mode='constant')
        img_small = cv2.resize(img_pad, (256, 256))
        img_small = np.ascontiguousarray(img_small)
        
        img_norm = self._im_normalize(img_small)
        out_reg, out_clf = self._predict_palm(img_norm)
        
        max_idx = np.argmax(out_clf)
        confidence = self._sigm(out_clf[max_idx])
        if confidence < 0.7:
            print("no hand found")
            return
        
        dx,dy,w,h = out_reg[max_idx, :4]
        center_wo_offset = self.anchors[max_idx,:2] * 256
        
        keypoints = center_wo_offset + out_reg[max_idx,4:].reshape(-1,2)        
        side = max(w,h)*1.3
        
        source = self._getTriangle(keypoints[0], keypoints[2], side)
        
        scale = max(shape) / 256
        
        Rtr = cv2.getAffineTransform(
            source * scale,
            self.target_triangle
        )
        
        img_landmark = cv2.warpAffine(
            self._im_normalize(img_pad), Rtr, (256,256)
        )
        
        joints = self._predict_joints(img_landmark)
        
        # adding the [0,0,1] row to make the matrix square
        Rtr = self._pad1(Rtr.T).T
        Rtr[2,:2] = 0

        Rinv = np.linalg.inv(Rtr)

        kp_orig = (self._pad1(joints) @ Rinv.T)[:,:2]
        kp_orig -= pad[::-1]
        
        return kp_orig