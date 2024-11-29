import copy
import os
import cv2 as cv
import numpy as np
from pupil_apriltags import Detector
import helper


def draw_tags(image, tags):
    tag_id = -1
    pt_a, pt_b, pt_c, pt_d = (0, 0), (0, 0), (0, 0), (0, 0)

    for tag in tags:
        (pt_a, pt_b, pt_c, pt_d) = tag.corners
        pt_a, pt_b, pt_c, pt_d = map(lambda pt: (int(pt[0]), int(pt[1])), [pt_a, pt_b, pt_c, pt_d])

        helper.line_with_border(image, pt_a, pt_b, (0, 255, 0), 2)
        helper.line_with_border(image, pt_b, pt_c, (0, 255, 0), 2)
        helper.line_with_border(image, pt_c, pt_d, (0, 255, 0), 2)
        helper.line_with_border(image, pt_d, pt_a, (0, 255, 0), 2)

        c_x, c_y = int(tag.center[0]), int(tag.center[1])
        cv.circle(image, (c_x, c_y), 5, (0, 0, 0), -1)

        tag_id = tag.tag_id

    return image, pt_a, pt_b, pt_c, pt_d, tag_id


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv.imread(img_path)
        if img is not None:
            images.append(img)
    return images


def main():
    cap = cv.VideoCapture(0)

    april_detector = Detector(
        families='tag36h11',
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    images = load_images_from_folder("images")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        color_image = copy.deepcopy(frame)
        gray_image = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)

        tags = april_detector.detect(
            gray_image, estimate_tag_pose=False, camera_params=None, tag_size=None
        )

        color_image, ref_pt_bl, ref_pt_br, ref_pt_tr, ref_pt_tl, tag_id = draw_tags(color_image, tags)

        output_image = color_image

        if tag_id != -1 and tag_id < len(images):
            dst_mat = np.array([ref_pt_tl, ref_pt_tr, ref_pt_br, ref_pt_bl])

            source_image = images[tag_id]
            src_h, src_w = source_image.shape[:2]
            src_mat = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]])

            homography_matrix, _ = cv.findHomography(src_mat, dst_mat)
            warped_image = cv.warpPerspective(source_image, homography_matrix, (color_image.shape[1], color_image.shape[0]))

            mask = np.zeros((color_image.shape[0], color_image.shape[1]), dtype="uint8")
            cv.fillConvexPoly(mask, dst_mat.astype("int32"), 255, cv.LINE_AA)

            rect_element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
            mask = cv.dilate(mask, rect_element, iterations=2)

            mask_scaled = mask / 255.0
            mask_scaled = np.dstack([mask_scaled] * 3)

            warped_multiplied = cv.multiply(warped_image.astype("float"), mask_scaled)
            image_multiplied = cv.multiply(frame.astype("float"), 1.0 - mask_scaled)
            output_image = cv.add(warped_multiplied, image_multiplied).astype("uint8")

        cv.imshow("apriltags", output_image)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
