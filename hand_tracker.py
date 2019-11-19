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
        >>> keypoints, bbox = det(input_img)
    """

    def __init__(self, palm_model, joint_model, anchors_path,
                box_enlarge=1.5, box_shift=0.2):
        self.box_shift = box_shift
        self.box_enlarge = box_enlarge

        self.interp_palm = tf.lite.Interpreter(palm_model)
        self.interp_palm.allocate_tensors()
        self.interp_joint = tf.lite.Interpreter(joint_model)
        self.interp_joint.allocate_tensors()
        
        # reading the SSD anchors
        with open(anchors_path, "r") as csv_f:
            self.anchors = np.r_[
                [x for x in csv.reader(csv_f, quoting=csv.QUOTE_NONNUMERIC)]
            ]
        # reading tflite model paramteres
        output_details = self.interp_palm.get_output_details()
        input_details = self.interp_palm.get_input_details()
        
        self.in_idx = input_details[0]['index']
        self.out_reg_idx = output_details[0]['index']
        self.out_clf_idx = output_details[1]['index']
        
        self.in_idx_joint = self.interp_joint.get_input_details()[0]['index']
        self.out_idx_joint = self.interp_joint.get_output_details()[0]['index']

        # 90° rotation matrix used to create the alignment trianlge        
        self.R90 = np.r_[[[0,1],[-1,0]]]

        # trianlge target coordinates used to move the detected hand
        # into the right position
        self._target_triangle = np.float32([
                        [128, 128],
                        [128,   0],
                        [  0, 128]
                    ])
        self._target_box = np.float32([
                        [  0,   0, 1],
                        [256,   0, 1],
                        [256, 256, 1],
                        [  0, 256, 1],
                    ])
        self.score = 0

    @staticmethod
    def get_bbox(landmarks, height_width, scale=3):
        hvec = landmarks[[1, 4]]

        center = hvec[0]
        half = max(height_width) * scale / 2
        v1 = hvec[1] - hvec[0]
        norm = np.linalg.norm(v1)
        v1 = v1 / norm
        v2 = np.r_[v1[1], -v1[0]]
        
        p0 = center + v2 * half - v1 * half
        p1 = center + v2 * half + v1 * half
        p2 = center - v2 * half + v1 * half
        p3 = center - v2 * half - v1 * half
    
        return {'box': np.r_[[p0, p1, p2, p3]].astype('int'),
         'center': center,
         'size': half * 2
        }
    
    @staticmethod
    def _im_normalize(img):
         return np.ascontiguousarray(
             2 * ((img / 255) - 0.5
        ).astype('float32'))
       
    @staticmethod
    def _sigm(x):
        # x = x / np.abs(x).max() * 100
        # x = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-x) )
    
    @staticmethod
    def _pad1(x):
        return np.pad(x, ((0,0),(0,1)), constant_values=1, mode='constant')

    @staticmethod
    def agg_hand(hand):
        idx = np.argmax(hand['score'])
        return {k: v[idx] for k,v in hand.items()}
    
    
    def predict_joints(self, img_norm):
        self.interp_joint.set_tensor(
            self.in_idx_joint, img_norm.reshape(1,256,256,3))
        self.interp_joint.invoke()

        joints = self.interp_joint.get_tensor(self.out_idx_joint)
        self.score = float(self.interp_joint.get_tensor(894))
        return joints.reshape(-1,2)

    def detect_hand(self, img_norm, scale=5):
        assert -1 <= img_norm.min() and img_norm.max() <= 1,\
        "img_norm should be in range [-1, 1]"
        assert img_norm.shape == (256, 256, 3),\
        "img_norm shape must be (256, 256, 3)"


        # predict hand location and 7 initial landmarks
        self.interp_palm.set_tensor(self.in_idx, img_norm[None])
        self.interp_palm.invoke()

        reg = self.interp_palm.get_tensor(self.out_reg_idx)[0]
        clf = self.interp_palm.get_tensor(self.out_clf_idx).flatten(
            ).astype('float32').round(1)

        centers = self.anchors[clf > 0][:, :2] * 256
        idx = np.lexsort((centers[:, 1], centers[:, 0]))

        centers = centers[idx]
        dxdy = reg[clf > 0][:,:2][idx]
        wh = reg[clf > 0][:, 2:4][idx]
        landmarks = reg[clf > 0][:, 4:].reshape(-1, 7, 2)[idx]
        max_size = wh.max()

        landmarks = centers[:, None, :] + landmarks
        centers += dxdy[:, :2]

        # nonperfect, but working sample of hand detection
        split_idx = np.argwhere(np.linalg.norm(centers[:-1] - centers[1:], axis=1) > max_size).flatten()
        if np.any(split_idx):
            split_idx += 1

        split_centers = np.split(centers, split_idx)
        split_wh = np.split(wh, split_idx)
        split_landmarks = np.split(landmarks, split_idx)
        split_clf = np.split(clf[clf > 0], split_idx)

        hands = []
        for cnt, wh, lm, scr in zip(
            split_centers, split_wh, split_landmarks, split_clf):
            hands.append({'center': lm[:,1,:] * scale,
                          'height_width': wh * scale,
                          'landmarks': lm * scale,
                          'score': scr})
        return [self.agg_hand(h) for h in hands]

    def preprocess_img(self, img):
        # fit the image into a 256x256 square
        shape = np.r_[img.shape]
        pad = (shape.max() - shape[:2]).astype('uint32') // 2
        img_pad = np.pad(
            img,
            ((pad[0],pad[0]), (pad[1],pad[1]), (0,0)),
            mode='constant')
        img_small = cv2.resize(img_pad, (256, 256))
        img_small = np.ascontiguousarray(img_small)
        
        img_norm = self._im_normalize(img_small)
        return img_pad, img_norm, pad

    def __call__(self, img, boxes=None):
        img_pad, img_norm, pad = self.preprocess_img(img)
        scale = img_pad.shape[0] / img_norm.shape[0]
        
        if boxes is None:
            try:
                hands = self.detect_hand(img_norm, scale=scale)
            except:
                return []
            boxes = [self.get_bbox(h['landmarks'], h['height_width'])\
                     for h in hands]

        for rec in boxes:
            rec['box'] -= pad[::-1]

        return boxes 
        
        # calculating transformation from img_pad coords
        # to img_landmark coords (cropped hand image)
        scale = max(img.shape) / 256
        Mtr = cv2.getAffineTransform(
            source * scale,
            self._target_triangle
        )
        
        img_landmark = cv2.warpAffine(
            self._im_normalize(img_pad), Mtr, (256,256)
        )
        
        joints = self.predict_joints(img_landmark)
        
        # adding the [0,0,1] row to make the matrix square
        Mtr = self._pad1(Mtr.T).T
        Mtr[2,:2] = 0

        Minv = np.linalg.inv(Mtr)

        # projecting keypoints back into original image coordinate space
        kp_orig = (self._pad1(joints) @ Minv.T)[:,:2]
        box_orig = (self._target_box @ Minv.T)[:,:2]
        kp_orig -= pad[::-1]
        box_orig -= pad[::-1]
        
        return kp_orig, box_orig