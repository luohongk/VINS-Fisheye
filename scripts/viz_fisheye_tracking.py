#!/usr/bin/env python3
"""
viz_fisheye_tracking.py
=======================

A simple stereo-fisheye tracking visualizer. Subscribes to two image topics
(raw or compressed), runs lightweight goodFeaturesToTrack + Lucas-Kanade
optical flow per camera, draws current points and tracks, and publishes a
side-by-side BGR image. Optionally also pops up an OpenCV window.

This is independent of VINS-Fisheye. It's purely "look at the images".
The point counts/tracks shown here are NOT the same as what VINS internally
tracks (VINS works on virtual-pinhole sub-images, not the raw fisheye), but
this visualization tells you whether the cameras see anything trackable
at all.

Usage (auto-generated launch file already does this):
    rosrun vins viz_fisheye_tracking.py \
        _left_topic:=/fisheye/left/image_padded \
        _right_topic:=/fisheye/right/image_padded \
        _display:=true
"""
from __future__ import annotations

import sys
from typing import Optional

import cv2
import message_filters
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image


class StereoFisheyeVisualizer:
    def __init__(self) -> None:
        self.bridge = CvBridge()

        self.left_topic       = rospy.get_param("~left_topic",  "/fisheye/left/image_padded")
        self.right_topic      = rospy.get_param("~right_topic", "/fisheye/right/image_padded")
        self.left_compressed  = bool(rospy.get_param("~left_compressed",  False))
        self.right_compressed = bool(rospy.get_param("~right_compressed", False))
        self.output_topic     = rospy.get_param("~output_topic", "/vins/fisheye_track")
        self.display          = bool(rospy.get_param("~display", True))
        self.max_corners      = int(rospy.get_param("~max_corners", 200))
        self.min_distance     = int(rospy.get_param("~min_distance", 25))
        self.show_width       = int(rospy.get_param("~show_width", 1600))
        self.window_name      = rospy.get_param("~window_name", "Stereo Fisheye Tracking")

        # Optional fisheye masks. If only `mask_path` is given, both cameras
        # use it. The mask is auto-resized + auto-padded to match the runtime
        # image, so a mask drawn for the un-padded sensor still works.
        self.mask_path        = rospy.get_param("~mask_path", "")
        self.mask_left_path   = rospy.get_param("~mask_left_path",  self.mask_path)
        self.mask_right_path  = rospy.get_param("~mask_right_path", self.mask_path)
        # Dim the masked-out region in the visualization (0 = fully black,
        # 1.0 = unchanged). Helps you see at a glance which pixels are
        # being ignored.
        self.mask_dim         = float(rospy.get_param("~mask_dim", 0.25))

        self.prev_left: Optional[np.ndarray]  = None
        self.prev_right: Optional[np.ndarray] = None
        self.left_pts:  Optional[np.ndarray]  = None
        self.right_pts: Optional[np.ndarray]  = None
        self.mask_left:  Optional[np.ndarray] = None  # lazy-loaded on first frame
        self.mask_right: Optional[np.ndarray] = None

        self.pub = rospy.Publisher(self.output_topic, Image, queue_size=2)

        SubTypeL = CompressedImage if self.left_compressed  else Image
        SubTypeR = CompressedImage if self.right_compressed else Image
        sub_l = message_filters.Subscriber(self.left_topic,  SubTypeL, queue_size=10)
        sub_r = message_filters.Subscriber(self.right_topic, SubTypeR, queue_size=10)
        sync = message_filters.ApproximateTimeSynchronizer(
            [sub_l, sub_r], queue_size=10, slop=0.05)
        sync.registerCallback(self.callback)

        rospy.loginfo(
            "[viz_fisheye] L=%s (%s) R=%s (%s)  display=%s  -> %s",
            self.left_topic,  "comp" if self.left_compressed  else "raw",
            self.right_topic, "comp" if self.right_compressed else "raw",
            self.display, self.output_topic,
        )

    # ---------------- helpers ----------------

    def _load_mask(self, path: str, target_shape: tuple) -> Optional[np.ndarray]:
        """Load a mask, scale-by-width and vertically center-pad to match
        target_shape (H, W). Returns a binary uint8 mask (0/255), or None."""
        if not path:
            return None
        raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            rospy.logerr(
                "[viz_fisheye] mask file does NOT exist or could not be "
                "decoded: %r — running WITHOUT mask. Check the path "
                "(remember it must resolve INSIDE the container if you "
                "run in docker).", path)
            return None
        h, w = target_shape[:2]
        mh, mw = raw.shape
        target_aspect = w / float(h)
        mask_aspect   = mw / float(mh)
        if abs(target_aspect - mask_aspect) < 0.01:
            out = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            # Scale by width, vertically center
            scale = w / float(mw)
            new_w = w
            new_h = int(round(mh * scale))
            scaled = cv2.resize(raw, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            out = np.zeros((h, w), dtype=np.uint8)
            if new_h <= h:
                top = (h - new_h) // 2
                out[top:top + new_h] = scaled
            else:
                # Mask taller than target: crop center vertically
                top = (new_h - h) // 2
                out = scaled[top:top + h]
        _, out = cv2.threshold(out, 127, 255, cv2.THRESH_BINARY)
        rospy.loginfo("[viz_fisheye] mask %s loaded -> %dx%d, valid pixels=%.1f%%",
                      path, out.shape[1], out.shape[0],
                      100.0 * float(np.count_nonzero(out)) / out.size)
        return out

    def _to_gray(self, msg, compressed: bool) -> np.ndarray:
        if compressed:
            img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
        else:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _track(self, prev_gray: Optional[np.ndarray],
               prev_pts: Optional[np.ndarray],
               cur_gray: np.ndarray,
               mask: Optional[np.ndarray] = None):
        """Returns (cur_pts, prev_pts_kept). cur_pts is Nx1x2 float32.

        Per user request: features are detected on the MASKED image
        (image AND mask). We also erode the mask inward by ~min_distance/2
        before masking to suppress spurious corners on the mask boundary
        (the hard intensity transition at the boundary would otherwise be
        a strong corner candidate). LK optical flow still runs on the
        ORIGINAL image since masking creates fake gradients that confuse
        the flow at the boundary."""
        # Build an eroded mask + an image masked by it, for feature detection
        if mask is not None:
            er = max(1, self.min_distance // 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er, er))
            inner_mask = cv2.erode(mask, kernel)
            cur_gray_masked = cv2.bitwise_and(cur_gray, inner_mask)
        else:
            inner_mask = None
            cur_gray_masked = cur_gray

        if prev_gray is None or prev_pts is None or len(prev_pts) == 0:
            new = cv2.goodFeaturesToTrack(
                cur_gray_masked, self.max_corners, 0.01, self.min_distance,
                mask=inner_mask)
            return new, None

        # LK on the ORIGINAL (unmasked) frames — masking would create fake
        # gradients at the boundary and break the flow.
        cur_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, cur_gray, prev_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        ok = status.flatten() == 1
        good_cur  = cur_pts[ok]
        good_prev = prev_pts[ok]

        # Drop tracks that landed outside the (eroded) mask
        if inner_mask is not None and len(good_cur) > 0:
            inside = []
            for i, p in enumerate(good_cur.reshape(-1, 2)):
                u, v = int(p[0]), int(p[1])
                if 0 <= u < inner_mask.shape[1] and 0 <= v < inner_mask.shape[0] \
                        and inner_mask[v, u] > 0:
                    inside.append(i)
            inside = np.asarray(inside, dtype=np.int32)
            good_cur  = good_cur.reshape(-1, 2)[inside].reshape(-1, 1, 2) if len(inside) else np.empty((0, 1, 2), dtype=np.float32)
            good_prev = good_prev.reshape(-1, 2)[inside].reshape(-1, 1, 2) if len(inside) else np.empty((0, 1, 2), dtype=np.float32)

        # Top up with new corners — also on the masked image
        target = self.max_corners
        if len(good_cur) < target // 2:
            feat_mask = 255 * np.ones_like(cur_gray)
            if inner_mask is not None:
                feat_mask = cv2.bitwise_and(feat_mask, inner_mask)
            for p in good_cur.reshape(-1, 2):
                cv2.circle(feat_mask, (int(p[0]), int(p[1])), self.min_distance, 0, -1)
            need = target - len(good_cur)
            extra = cv2.goodFeaturesToTrack(
                cur_gray_masked, need, 0.01, self.min_distance, mask=feat_mask)
            if extra is not None:
                good_cur = np.vstack([good_cur.reshape(-1, 2), extra.reshape(-1, 2)])
        return good_cur.reshape(-1, 1, 2).astype(np.float32), good_prev

    def _draw(self, gray: np.ndarray, cur_pts, prev_pts,
              mask: Optional[np.ndarray] = None) -> np.ndarray:
        # Show the masked image: outside-mask region is fully blacked out,
        # so the user sees exactly what was passed to the detector.
        if mask is not None:
            gray = cv2.bitwise_and(gray, mask)
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if mask is not None:
            # Outline the mask boundary so the user sees the valid region
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, (0, 200, 0), 2)
        if cur_pts is None:
            return canvas
        for i, p in enumerate(cur_pts.reshape(-1, 2)):
            cv2.circle(canvas, (int(p[0]), int(p[1])), 4, (0, 255, 255), -1)
            if prev_pts is not None and i < len(prev_pts):
                pp = prev_pts[i].flatten()
                cv2.line(canvas, (int(pp[0]), int(pp[1])),
                         (int(p[0]), int(p[1])), (255, 0, 0), 1)
        return canvas

    # ---------------- main callback ----------------

    def callback(self, msg_l, msg_r) -> None:
        try:
            gray_l = self._to_gray(msg_l, self.left_compressed)
            gray_r = self._to_gray(msg_r, self.right_compressed)
        except Exception as e:
            rospy.logerr_throttle(5.0, f"[viz_fisheye] decode failed: {e}")
            return

        # Lazy-load masks now that we know the runtime image size
        if self.mask_left is None and self.mask_left_path:
            self.mask_left = self._load_mask(self.mask_left_path, gray_l.shape)
        if self.mask_right is None and self.mask_right_path:
            self.mask_right = self._load_mask(self.mask_right_path, gray_r.shape)

        new_left_pts,  prev_left_pts  = self._track(self.prev_left,  self.left_pts,  gray_l, mask=self.mask_left)
        new_right_pts, prev_right_pts = self._track(self.prev_right, self.right_pts, gray_r, mask=self.mask_right)

        viz_l = self._draw(gray_l, new_left_pts,  prev_left_pts,  mask=self.mask_left)
        viz_r = self._draw(gray_r, new_right_pts, prev_right_pts, mask=self.mask_right)

        nl = 0 if new_left_pts  is None else len(new_left_pts)
        nr = 0 if new_right_pts is None else len(new_right_pts)
        cv2.putText(viz_l, f"LEFT  pts={nl}",  (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
        cv2.putText(viz_r, f"RIGHT pts={nr}", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

        combined = np.hstack([viz_l, viz_r])
        if self.show_width > 0 and combined.shape[1] > self.show_width:
            s = self.show_width / float(combined.shape[1])
            combined = cv2.resize(combined, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

        try:
            out_msg = self.bridge.cv2_to_imgmsg(combined, encoding="bgr8")
            out_msg.header = msg_l.header
            self.pub.publish(out_msg)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[viz_fisheye] publish failed: {e}")

        if self.display:
            cv2.imshow(self.window_name, combined)
            cv2.waitKey(1)

        # Update state
        self.prev_left  = gray_l
        self.prev_right = gray_r
        self.left_pts   = new_left_pts
        self.right_pts  = new_right_pts


def main() -> int:
    rospy.init_node("viz_fisheye_tracking", anonymous=True)
    StereoFisheyeVisualizer()
    try:
        rospy.spin()
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
