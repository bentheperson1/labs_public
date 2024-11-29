import cv2 as cv

def line_with_border(img: cv.typing.MatLike, pt1: cv.typing.Point, pt2: cv.typing.Point, color: cv.typing.Scalar, thickness: int) -> cv.typing.MatLike:
    cv.line(img, pt1, pt2, (0,0,0), thickness * 3)
    cv.line(img, pt1, pt2, color, thickness)

def text_with_border(img: cv.typing.MatLike, text: str, org: cv.typing.Point, fontFace: int, fontScale: float, color: cv.typing.Scalar, thickness: int = ..., lineType: int = ..., bottomLeftOrigin: bool = ...) -> cv.typing.MatLike:
    cv.putText(img, text, org, fontFace, fontScale, (0, 0, 0), thickness * 3, lineType, bottomLeftOrigin)
    cv.putText(img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin)

