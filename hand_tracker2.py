import csv
import cv2
import numpy as np
import tensorflow as tf
from sklearn.cluster import DBSCAN

"""
HAND TRACKER WITHOUT ROTATION ESTIMATION AND JOINT DETECTION
"""

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
        self.cluster = DBSCAN(eps=30, min_samples=1)

        self.reset()

        self.track_record = []

    def reset(self):
        self.hands = np.zeros((0,7,2))
        self.sizes = np.zeros((0,1))
        self.dydx = np.zeros((0,2))
        self.n_different = 0

    def track(self, hands):
        lm_new = np.c_[[x['landmarks'] for x in hands]]
        sizes = np.c_[[x['size'] for x in hands]]

        self.track_record.append({'n_inp_hands': len(lm_new)})
        self.track_record[-1]['inp_hands'] = lm_new.copy()

        if len(self.hands) != len(lm_new):
            self.n_different += 1
        if self.n_different > 30:
            self.reset()
            self.track_record[-1]['reset'] = True

        if len(lm_new) == 0:
            for i in range(len(hands)):
                self.hands[i] = self.hands[i] + self.dydx[i]
            return self.hands.copy(), self.sizes.copy(), self.dydx.copy()    

        assert lm_new.shape[1:] == (7,2), lm_new
        
        idx = list(range(len(lm_new)))
        to_update = list(range(len(self.hands))) # tracking which hands were updated
        
        for i, h in enumerate(self.hands):
            dists = self.hand_dist(h, lm_new[idx])
            
            k = np.argmin(dists)
            j = idx.pop(k)
            dydx = lm_new[j][2] - self.hands[i][2]
            if 5 < np.linalg.norm(dydx) < sizes[i]:
                to_update.remove(i)
                self.dydx[i] = dydx

                self.hands[i] = lm_new[j]
                self.sizes[i] = 0.7*self.sizes[i] + 0.3*sizes[j]

            if len(idx) == 0:
                break

        # if there are hands that haven't been found, but we know
        # they had large movement vectors before - just move them 
        # as if by momentum
        self.track_record[-1]['hands not updated'] = to_update
        for i in to_update:
            self.hands[i] = self.hands[i] + self.dydx[i]

        if idx:
            self.hands = np.concatenate([self.hands, lm_new[idx]])
            self.sizes = np.concatenate([self.sizes, sizes[idx]])
            self.dydx = np.concatenate([self.dydx, np.zeros((len(idx), 2))])

        self.track_record[-1]['hand_res'] = self.hands.copy()
        return self.hands.copy(), self.sizes.copy(), self.dydx.copy()

    @staticmethod
    def hand_dist(x,y):
        return np.linalg.norm(x - y, axis=-1).mean(axis=-1)

    @staticmethod
    def add_bbox(hand):
        center = hand['lm'][2]
        half = float(hand['size'] / 2)
        tl = center - [half, half]
        bl = center - [half, -half]
        tr = center - [-half, half]
        br = center + [half, half]
        hand['bbox'] = np.c_[[tl,bl,br,tr]]
        return hand
    
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

    def detect_hand(self, img_norm, scale=5, box_enlarge=3.3):
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



        dydx = reg[clf > 0][:,:2]
        wh = reg[clf > 0][:, 2:4]
        landmarks = reg[clf > 0][:, 4:].reshape(-1, 7, 2)
        clf = clf[clf > 0]
        max_size = wh.max()

        landmarks = centers[:, None, :] + landmarks
        centers += dydx[:, :2]

        split_centers = []
        split_wh = []
        split_landmarks = []
        split_clf = []

        self.cluster.eps = max_size
        clusters = self.cluster.fit_predict(centers)
        for i in set(clusters):
            split_centers.append(centers[clusters == i])
            split_wh.append(wh[clusters == i])
            split_landmarks.append(landmarks[clusters == i])
            split_clf.append(clf[clusters == i])

        hands = []
        for cnt, wh, lm, scr in zip(
            split_centers, split_wh, split_landmarks, split_clf):
            hands.append({'center': lm[:,1,:] * scale,
                          'size': wh.max(axis=-1) * box_enlarge * scale,
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
                
        try:
            hands = self.detect_hand(img_norm, scale=scale)
        except:
            hands = []

        for rec in hands:
            rec['landmarks'] -= pad[::-1]

        tracked = self.track(hands)

        rearranged = [{'lm': x,'size': y, 'dydx': z} for x,y,z in zip(
            tracked[0], tracked[1], tracked[2])]
        hands = [self.add_bbox(h) for h in rearranged]
        return hands