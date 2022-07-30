# import necessary packages
import os
import cv2
import numpy as np
import mediapipe as mp
# Open the capture device - change as needed. With MBP, camera accessed via 0
FRONT_FACIAL_CLASSIFIER = cv2.CascadeClassifier('haar_frontalface.xml')
IMAGE_PATH = 'Gg.png'
hat = cv2.imread(IMAGE_PATH, -1)

def overlay(roi, overlay_image):
    # resize the overlay image to fit the ROI target
    resized_dimensions = (roi.shape[1], roi.shape[0])
    resized_overlay_dst = cv2.resize(
        overlay_image,
        resized_dimensions,
        fx=0,
        fy=0,
        interpolation=cv2.INTER_NEAREST)
    bgr_image = resized_overlay_dst[:, :, :3]  # BGR
    alpha_mask_1 = resized_overlay_dst[:, :, 3:]
    # Note: alpha mask 1 + alpha mask 2 = [1,1,1,1...]
    alpha_mask_2 = cv2.bitwise_not(alpha_mask_1)
    three_chan_alpha_mask_1 = cv2.cvtColor(alpha_mask_1, cv2.COLOR_GRAY2BGR)
    three_chan_alpha_mask_2 = cv2.cvtColor(alpha_mask_2, cv2.COLOR_GRAY2BGR)
    # Use inverted alpha mask to multiply pixels from background image by pixel of the inverted alphamask
    # The inverted alpha mask will have 255 set for pixels where the pixel is completely transparent.
    # As a result, during multiplication, the background pixel will have 100%
    # of the weight for that pixel
    background_roi = (roi * 1 / 255.0) * (three_chan_alpha_mask_2 * 1 / 255.0)
    # Use the alpha mask here so the overlay image will have its transparent pixels equal to zero.
    # This is useful when we add the foreground ROI and background ROI together as the correct pixel
    # value from the background  will be used in transparent pixels (alpha=0)
    foreground_roi = (bgr_image * 1 / 255.0) * \
        (three_chan_alpha_mask_1 * 1 / 255.0)
    return cv2.addWeighted(background_roi, 255.0, foreground_roi, 255.0, 0.0)

def overlay_img_above_facial_frame(facial_frame, x, w, y, h, overlay_t_img):
    top_of_frame_coord = max(0, y - h)
    rightmost_frame_coord = x + w
    # Squeeze the ROI to fit within screen bounds
    roi_above_face = facial_frame[top_of_frame_coord:y, x:rightmost_frame_coord]
    overlayed_roi = overlay(roi_above_face, overlay_t_img)
    # Replace frame's ROI with overlayed ROI
    facial_frame[top_of_frame_coord:y, x:rightmost_frame_coord] = overlayed_roi
    pass

def get_gray_and_3_chan_colored_frames(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

def get_face_rects_in_frame(frame):
    return FRONT_FACIAL_CLASSIFIER.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=10, minSize=(75, 75))

# initialize mediapipe
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
# store background images in a list
image_path = 'images'
images = os.listdir(image_path)
image_index= 0
bg_image = cv2.imread(image_path+'/'+images[image_index])
# create videocapture object to access the webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    # flip the frame to horizontal direction
    frame = cv2.flip(frame, 1)
    height , width, channel = frame.shape
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # get the result
    results = selfie_segmentation.process(RGB)
    # extract segmented mask
    mask = results.segmentation_mask
    gray_frame, colored_frame = get_gray_and_3_chan_colored_frames(frame)
    # Detect faces (represented by rectangles)
    face_rects = get_face_rects_in_frame(gray_frame)
    # it returns true or false where the condition applies in the mask
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.6
    # resize the background image to the same size of the original frame
    bg_image = cv2.resize(bg_image, (width, height))
    # combine frame and background image using the condition
    output_image = np.where(condition, frame, bg_image )
    output_image1 = np.where(condition, colored_frame, bg_image)
    # Draw rectangles around all detected faces - this was taken directly
    for (index, (x, y, w, h)) in enumerate(face_rects):
        print('Face at  ' + str(x) + ',' + str(y))
        # Overlay the hat at target ROI at top of detected rectangle
        overlay_img_above_facial_frame(output_image, x, w, y, h, hat)
        overlay_img_above_facial_frame(colored_frame, x, w, y, h, hat)
    # show outputs
    cv2.imshow("Frame Capture", frame)
    cv2.imshow('Frame Capture with Hat Overlayed', colored_frame)
    cv2.imshow("BgFgMask Seperated", mask)
    cv2.imshow("Fg Overlay On Bg Changed", output_image1)
    cv2.imshow("Fg Overlay On Bg Changed with Hat Overlayed", output_image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    # if 'd' key is pressed then change the background image
    elif key == ord('d'):
        if image_index != len(images)-1:
            image_index += 1
        else:
            image_index = 0
        bg_image = cv2.imread(image_path+'/'+images[image_index])
# release the capture object and close all active windows
cap.release()
cv2.destroyAllWindows()



