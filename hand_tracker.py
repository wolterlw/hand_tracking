import csv
import cv2
import numpy as np
import tensorflow as tf
from sklearn.cluster import DBSCAN

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
        >>> hands = det(input_img)
    """

    def __init__(self, palm_model, joint_model, anchors_path,
                box_enlarge=3.3, box_shift=0.2):

        self.interp_palm = tf.lite.Interpreter(palm_model)
        self.interp_palm.allocate_tensors()
        self.interp_lm = tf.lite.Interpreter(joint_model)
        self.interp_lm.allocate_tensors()
        
        # reading the SSD anchors
        with open(anchors_path, "r") as csv_f:
            self.anchors = np.r_[
                [x for x in csv.reader(csv_f, quoting=csv.QUOTE_NONNUMERIC)]
            ]
        # reading tflite model paramteres
        output_details = self.interp_palm.get_output_details()
        input_details = self.interp_palm.get_input_details()
        output_details_lm = self.interp_lm.get_output_details()
        
        self.in_idx = input_details[0]['index']
        self.out_reg_idx = output_details[0]['index']
        self.out_clf_idx = output_details[1]['index']
        
        self.lm_reg_idx = output_details_lm[0]['index']
        
        self.cluster = DBSCAN(eps=30, min_samples=1)

        self.reset()
        self._target_box = np.float32([
                        [  0, 256],
                        [256, 256],
                        [256,   0],
                        [  0,   0],
        ]).astype('float32')

    def reset(self):
        """
        resets hand tracking data. Should be used any time there's
        a scene transition in a video.
        """
        self.hands = np.zeros((0,7,2))
        self.sizes = np.zeros((0,1))
        self.dydx = np.zeros((0,2))
        self.n_different = 0

    @staticmethod
    def hand_dist(x,y):
        return np.linalg.norm(x - y, axis=-1).mean(axis=-1)
    
    @staticmethod
    def add_bbox(hand):
        if 'joints' in hand:
            lm = hand['joints'][[0, 5, 9,13,17, 20]]
        else:
            lm = hand['lm']
        hvec = lm[[1, 4]]
        vv = lm[2] - lm[0]

        center = hvec[0]
        
        half = max(hand['size']) / 2
        v1 = hvec[1] - hvec[0]
        norm = np.linalg.norm(v1)
        v1 = v1 / norm
        v2 = np.r_[v1[1], -v1[0]]
        
        if (v2 @ vv) > 0:
            v2 = -v2
        
        p0 = center + v2 * half - v1 * half
        p1 = center + v2 * half + v1 * half
        p2 = center - v2 * half + v1 * half
        p3 = center - v2 * half - v1 * half
    
        hand['bbox'] = np.r_[[p0, p1, p2, p3]].astype('int')
        return hand
    
    @staticmethod
    def _im_normalize(img):
        """
        normalize all values into [-1, 1]
        """
        return np.ascontiguousarray(
             2 * ((img / 255) - 0.5
        ).astype('float32'))
       
    @staticmethod
    def _sigm(x):
        return 1 / (1 + np.exp(-x) )
    
    @staticmethod
    def _pad1(x):
        return np.pad(x, ((0,0),(0,1)), constant_values=1, mode='constant')

    @staticmethod
    def agg_hand(hand):
        """
        method used to choose between multiple
        hand detections in the same cluster
        """
        idx = np.argmax(hand['size']) # seems to work better than score
        return {k: v[idx] for k,v in hand.items()}

    def preprocess_img(self, img):
        """
        fit the image into a 256x256 square and normalizes it
        """
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

    def detect_hand(self, img_norm, scale=5, box_enlarge=2.5):
        """
        runs the palm detection model, decodes possible detections, clusters them
        and aggregates detections in the same clusters
        """
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
                          'lm': lm * scale,
                          'score': scr})

        return [self.agg_hand(h) for h in hands]

    def track(self, hands):
        """
        Makes sure that hands are ordered consistently
        between frames of a video and mitigates jitter.
        """
        lm_new = np.c_[[x['lm'] for x in hands]]
        sizes = np.c_[[x['size'] for x in hands]]

        if len(self.hands) != len(lm_new):
            self.n_different += 1
        if self.n_different > 30:
            self.reset()    

        if len(lm_new) == 0:
            for i in range(len(hands)):
                self.hands[i] = self.hands[i] + self.dydx[i]
            return self.hands.copy(), self.sizes.copy(), self.dydx.copy()    

        assert lm_new.shape[1:] == (7,2), lm_new
        
        idx = list(range(len(self.hands))) # tracking which hands were updated
        idx_new = list(range(len(lm_new))) 
        

        while idx_new:
            if len(idx) == 0:
                break

            i = idx_new.pop(0)
            sz = sizes[i]
            h = lm_new[i]

            dists = self.hand_dist(self.hands[idx], h)
            
            k = np.argmin(dists)
            j = idx.pop(k) # updating tracked hand j

            dydx = h[2] - self.hands[j][2]

            if 10 < np.linalg.norm(dydx) < self.sizes[j]:
                self.dydx[j] = 0.4*self.dydx[j] + 0.6*dydx

                self.hands[j] = h
            self.sizes[j] = 0.7*self.sizes[j] + 0.3*sz

        # if there are hands that haven't been found, but we know
        # they had large movement vectors before - just move them 
        # as if by momentum
        for i in idx:
            self.hands[i] = self.hands[i] + self.dydx[i]

        if idx_new:
            self.hands = np.concatenate([self.hands, lm_new[idx_new]])
            self.sizes = np.concatenate([self.sizes, sizes[idx_new]])
            self.dydx = np.concatenate([self.dydx, np.zeros((len(idx_new), 2))])

        return self.hands.copy(), self.sizes.copy(), self.dydx.copy()
    
    def get_landmarks (self, img, hand):
        """
        crops the hand image according to the bounding box,
        runs hand landmark detection models and projects
        obtained coordinates onto the full-sized image
        """
        source = hand['bbox'].astype('float32')
        
        Mtr = cv2.getAffineTransform(
            source[:3],
            self._target_box[:3]
        )
        img_hand_np = cv2.warpAffine(img, Mtr, (256,256))
        img_hand = self._im_normalize(img_hand_np)
        
        self.interp_lm.set_tensor(0, img_hand[None])
        self.interp_lm.invoke()

        reg = self.interp_lm.get_tensor(self.lm_reg_idx)[0].reshape(21,2)
        
        Mtr = self._pad1(Mtr.T).T
        Mtr[2,:2] = 0
        
        Minv = np.linalg.inv(Mtr)
        kp_orig = (self._pad1(reg) @ Minv.T)[:,:2]
        hand['joints'] = kp_orig
        return hand

    def __call__(self, img, hands=None):
        r"""
        Method used to detect hand poses in individual images
        and track hands in video frames.
        Args:
            img: image as a numpy array of shape (h,w,3)
            hands: detected hands from previous time step
        Ourput:
            hands: ordered list of dictionaries containing
                'bbox': bounding box of a hand
                'joints': (21,2) array of hand joints
            During normal operation hand ordering is preserved
            between video frames
        """
        img_pad, img_norm, pad = self.preprocess_img(img)
        scale = img_pad.shape[0] / img_norm.shape[0]
        
        if hands is None:
            try:
                hands = self.detect_hand(img_norm, scale=scale)
            except:
                hands = []
            for rec in hands:
                rec['lm'] -= pad[::-1]

            tracked = self.track(hands)

            hands = [{'lm': x,'size': y, 'dydx': z} for x,y,z in zip(
                tracked[0], tracked[1], tracked[2])]

        
        hands = [self.add_bbox(h) for h in hands]
        hands = [self.get_landmarks(img, h) for h in hands]
        return hands