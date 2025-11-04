import cv2
import numpy as np
import time

# Load reference image (your Paracip photo)
template = cv2.imread("paracip.jpg")
if template is None:
    print("❌ Error: couldn't load paracip.jpg")
    exit()

template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create(nfeatures=1500)
kp_template, des_template = orb.detectAndCompute(template_gray, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Color histogram (to verify color similarity)
template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
template_hist = cv2.calcHist([template_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
cv2.normalize(template_hist, template_hist, 0, 1, cv2.NORM_MINMAX)

cap = cv2.VideoCapture(0)
prev_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    kp_frame, des_frame = orb.detectAndCompute(gray, None)
    detected = False

    if des_frame is not None and len(kp_frame) > 10:
        matches = bf.knnMatch(des_template, des_frame, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:  # slightly relaxed ratio
                good.append(m)

        if len(good) > 25:  # reduced threshold
            src_pts = np.float32([kp_template[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None and mask is not None and np.sum(mask) > 8:
                h, w = template_gray.shape
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                # Extract region and compare color histograms
                x, y, w_, h_ = cv2.boundingRect(np.int32(dst))
                if x > 0 and y > 0 and x+w_ < frame.shape[1] and y+h_ < frame.shape[0]:
                    roi_hsv = hsv[y:y+h_, x:x+w_]
                    roi_hist = cv2.calcHist([roi_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                    cv2.normalize(roi_hist, roi_hist, 0, 1, cv2.NORM_MINMAX)
                    similarity = cv2.compareHist(template_hist, roi_hist, cv2.HISTCMP_CORREL)

                    if similarity > 0.5:  # adjust this threshold if needed
                        frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                        cv2.putText(frame, "✅ Paracip Detected", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        detected = True

    if not detected:
        cv2.putText(frame, "❌ Not Detected", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # FPS counter
    if frame_count >= 10:
        current_time = time.time()
        fps = frame_count / (current_time - prev_time)
        prev_time = current_time
        frame_count = 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Pill Detection (Paracip)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
