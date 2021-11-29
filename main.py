import numpy as np
import cv2


def reorderPts(pts):
    idx = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[idx]

    if pts[0, 1] > pts[1, 1]:
        pts[[0, 1]] = pts[[1, 0]]
    if pts[2, 1] < pts[3, 1]:
        pts[[2, 3]] = pts[[3, 2]]

    return pts


def on_mouse(event, x, y, flags, param):
    global cnt, src_pts, scanned_img

    if event == cv2.EVENT_LBUTTONDOWN:
        if cnt < 4:
            src_pts[cnt, :] = np.array([x, y]).astype(np.float32)
            cnt += 1

            cv2.circle(src, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('src', src)
        if cnt == 4:
            src_pts = reorderPts(src_pts)
            w = int(src_pts[3, 0] - src_pts[0, 0])
            h = int(src_pts[1, 1] - src_pts[0, 1])
            dst_pts = np.array([[0, 0], [0, h], [w, h], [w, 0]]).astype(np.float32)
            pers_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
            res = cv2.warpPerspective(src, pers_mat, (w, h))
            cv2.imshow('res', res)
            cv2.waitKey(0)
            scanned_img = res


def img_scan(src_img):
    global src_pts

    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for pts in contours:
        if cv2.contourArea(pts) < 1000:
            continue

        approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True) * 0.02, True)

        if len(approx) != 4:
            continue
        src_pts = approx.reshape(4, 2).astype(np.float32)

    src_pts = reorderPts(src_pts)
    w = int(src_pts[3, 0] - src_pts[0, 0])
    h = int(src_pts[1, 1] - src_pts[0, 1])
    dst_pts = np.array([[0, 0], [0, h], [w, h], [w, 0]]).astype(np.float32)
    pers_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    res = cv2.warpPerspective(src, pers_mat, (w, h))

    cv2.imshow('res', res)
    cv2.waitKey(0)

    return res


if __name__ == '__main__':
    src = cv2.imread('test5.jpg', cv2.IMREAD_COLOR)
    if src is None:
        print('Image load failed!')
        exit()
    src_pts = np.array([[0, 0], [0, 0], [0, 0], [0, 0]]).astype(np.float32)
    while True:
        print('a for Auto Mode, m for Manual Mode\n'
              'n for Next step, q for quit')
        mode = input()
        if mode == 'a' or mode == 'A':
            scanned_img = img_scan(src)
        elif mode == 'm' or mode == 'M':
            cnt = 0
            cv2.namedWindow('src')
            cv2.setMouseCallback('src', on_mouse, src)
            cv2.imshow('src', src)
            cv2.waitKey(1)
        elif mode == 'n' or mode == 'N':
            if 'scanned_img' not in locals():
                print('No Doc detected')
                continue
            break
        elif mode == 'q' or mode == 'Q':
            exit()
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    cv2.imshow('scanned_img', scanned_img)
    filtered_img = cv2.GaussianBlur(scanned_img, (5, 5), 0)
    cv2.imshow('filtered_img', filtered_img)
    gray_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
    grad = cv2.morphologyEx(gray_img, cv2.MORPH_GRADIENT, kernel1)
    cv2.imshow('grad', grad)
    _, bin_img = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('bin_img', bin_img)
    close = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel2)
    cv2.imshow('close', close)
    contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 5:
            cv2.rectangle(scanned_img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            temp = scanned_img[y:y+h, x:x+w]
            cv2.imwrite('./scan/scanned_letter' + str(cnt) + '.jpg', temp)
            cnt += 1
    cv2.imshow('rects', scanned_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
