import os
import cv2
import numpy as np
import glob
import pandas as pd

os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

MAX_WIN_W = 1200
MAX_WIN_H = 800
ZOOM_STEP = 0.1
MIN_SCALE = 0.1
MAX_SCALE = 1.0


def _fit_scale(h, w):
    return min(MAX_WIN_W / float(w), MAX_WIN_H / float(h), 1.0)


def _resize(img, scale):
    if scale == 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img,
                      (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


def _draw_points(canvas, roi_pts, scale_pts, scale):
    for p in roi_pts:
        cv2.circle(canvas, (int(p[0]*scale), int(p[1]*scale)), 6, (0, 0, 255), -1)
    for p in scale_pts:
        cv2.circle(canvas, (int(p[0]*scale), int(p[1]*scale)), 6, (255, 0, 0), -1)


def _selection_loop(image, filename):
    h, w = image.shape[:2]
    state = {
        "roi_points":   [],
        "scale_points": [],
        "scale":        _fit_scale(h, w),
        "confirmed":    False,
        "cancelled":    False,
    }

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ox = int(round(x / state["scale"]))
            oy = int(round(y / state["scale"]))
            if len(state["roi_points"]) < 4:
                state["roi_points"].append((ox, oy))
                print(f"ROI {len(state['roi_points'])}: ({ox}, {oy})")
            elif len(state["scale_points"]) < 2:
                state["scale_points"].append((ox, oy))
                print(f"Scale {len(state['scale_points'])}: ({ox}, {oy})")

    win = f"Select ROI and Scale - {filename}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse_cb)

    while True:
        disp = _resize(image, state["scale"])
        _draw_points(disp, state["roi_points"], state["scale_points"], state["scale"])
        cv2.putText(disp, "Left-click: 4 ROI pts then 2 scale pts",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30,30,30), 2, cv2.LINE_AA)
        cv2.putText(disp, "ENTER:confirm  BACKSPACE:undo  +/-:zoom  ESC:skip",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30,30,30), 2, cv2.LINE_AA)
        cv2.imshow(win, disp)
        key = cv2.waitKey(1) & 0xFF

        if key in (13, 10):
            if len(state["roi_points"]) == 4 and len(state["scale_points"]) == 2:
                state["confirmed"] = True
                break
        elif key == 27:
            state["cancelled"] = True
            break
        elif key == 8:
            if state["scale_points"]:
                state["scale_points"].pop()
            elif state["roi_points"]:
                state["roi_points"].pop()
        elif key in (43, ord('=')):
            state["scale"] = min(MAX_SCALE, round(state["scale"] + ZOOM_STEP, 2))
        elif key in (45, ord('_')):
            state["scale"] = max(MIN_SCALE, round(state["scale"] - ZOOM_STEP, 2))

    cv2.destroyWindow(win)
    cv2.waitKey(1)
    return state


def _show_result(image, filename):
    h, w  = image.shape[:2]
    scale = _fit_scale(h, w)
    win   = f"Result - {filename}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        disp = _resize(image, scale)
        cv2.putText(disp, "Zoom: +/-  |  ESC: next image",
                    (10, disp.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30,30,30), 2)
        cv2.imshow(win, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in (43, ord('=')):
            scale = min(MAX_SCALE, round(scale + ZOOM_STEP, 2))
        elif key in (45, ord('_')):
            scale = max(MIN_SCALE, round(scale - ZOOM_STEP, 2))
    cv2.destroyWindow(win)
    cv2.waitKey(1)


def _build_green_mask(roi_img):
    """
    Green mask optimized for greenhouse pot crops.
    Removes white background, metal grid, and yellow stickers/tags.
    """
    lab = cv2.cvtColor(roi_img, cv2.COLOR_BGR2LAB)
    l, a_ch, b_ch = cv2.split(lab)
    clahe  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_eq = cv2.cvtColor(cv2.merge([clahe.apply(l), a_ch, b_ch]), cv2.COLOR_LAB2BGR)
    img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)

    hsv      = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(hsv, np.array([30, 20, 40]), np.array([85, 255, 255]))

    img_f    = img_blur.astype(float)
    ExG      = 2 * img_f[:, :, 1] - img_f[:, :, 0] - img_f[:, :, 2]
    mask_exg = (ExG > 30).astype(np.uint8) * 255

    lab2 = cv2.cvtColor(img_blur, cv2.COLOR_BGR2LAB)
    _, a2, _ = cv2.split(lab2)
    a_f    = a2.astype(float) - 128
    mask_a = (a_f < -10).astype(np.uint8) * 255

    combined = cv2.bitwise_or(mask_hsv, mask_exg)
    combined = cv2.bitwise_or(combined, mask_a)

    white  = cv2.inRange(hsv, np.array([0,   0, 200]), np.array([180,  30, 255]))
    metal  = cv2.inRange(hsv, np.array([0,   0,   0]), np.array([180,  50, 120]))
    yellow = cv2.inRange(hsv, np.array([15, 120, 150]), np.array([30,  255, 255]))
    bg     = cv2.bitwise_or(cv2.bitwise_or(white, metal), yellow)
    combined = cv2.bitwise_and(combined, cv2.bitwise_not(bg))

    k_open  = np.ones((5, 5), np.uint8)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k_open)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)

    return combined


def greenhousepot(
    input_folder,
    output_folder,
    ruler_cm=23,
    min_leaf_area=5.0,
):
    """
    Measure green leaf area of greenhouse pot crops from top-view images.

    Parameters
    ----------
    input_folder : str
        Path to folder containing input .jpg images.
    output_folder : str
        Path to folder where results will be saved.
    ruler_cm : float
        Actual length of the scale bar / ruler in the image (cm). Default 23.
    min_leaf_area : float
        Minimum leaf area in cm2. Clusters smaller than this are ignored. Default 5.0.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        File Name, Leaf Area (cm2), Green Ratio (%), Scale (cm/pixel),
        Min Leaf Area (cm2), Image Path

    Examples
    --------
    >>> from sangchu import greenhousepot
    >>> df = greenhousepot(
    ...     input_folder  = "./images",
    ...     output_folder = "./results",
    ...     ruler_cm      = 23,
    ...     min_leaf_area = 5.0,
    ... )
    >>> print(df)
    """
    os.makedirs(output_folder, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(input_folder, "*.jpg")))
    if not image_paths:
        print(f"No .jpg images found in: {input_folder}")
        return pd.DataFrame()

    print(f"sangchu::greenhousepot()")
    print(f"  input_folder  : {input_folder}")
    print(f"  output_folder : {output_folder}")
    print(f"  ruler_cm      : {ruler_cm}")
    print(f"  min_leaf_area : {min_leaf_area} cm2")
    print(f"  images found  : {len(image_paths)}\n")

    results = []

    for path in image_paths:
        filename = os.path.basename(path)
        image    = cv2.imread(path)
        if image is None:
            print(f"Cannot read: {filename}")
            continue

        height, width = image.shape[:2]
        print(f"{'='*50}")
        print(f"Processing: {filename}  ({width}x{height})")

        # ROI + scale selection
        sel = _selection_loop(image, filename)
        if not sel["confirmed"]:
            print(f"Skipped: {filename}")
            continue

        roi_points   = sel["roi_points"]
        scale_points = sel["scale_points"]

        # cm/pixel
        dist_px = np.linalg.norm(
            np.array(scale_points[0]) - np.array(scale_points[1])
        )
        if dist_px == 0:
            print(f"Invalid scale points: {filename}")
            continue

        cm_per_pixel     = ruler_cm / dist_px
        px_per_cm2       = 1.0 / (cm_per_pixel ** 2)
        min_leaf_area_px = min_leaf_area * px_per_cm2
        print(f"  cm/pixel      : {cm_per_pixel:.5f}")
        print(f"  min leaf area : {min_leaf_area_px:.0f} px2  ({min_leaf_area} cm2)")

        # ROI mask
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(roi_mask, np.array(roi_points, dtype=np.int32), 255)
        roi_img  = cv2.bitwise_and(image, image, mask=roi_mask)

        # Green mask
        green_mask = _build_green_mask(roi_img)
        green_mask = cv2.bitwise_and(green_mask, roi_mask)

        # Contours — keep only largest cluster (one plant per pot)
        contours_all, _ = cv2.findContours(
            green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        leaf_contours = [c for c in contours_all
                         if cv2.contourArea(c) >= min_leaf_area_px]

        if len(leaf_contours) > 1:
            leaf_contours = [max(leaf_contours, key=cv2.contourArea)]

        for i, c in enumerate(leaf_contours):
            print(f"    Cluster {i+1}: {cv2.contourArea(c)*(cm_per_pixel**2):.2f} cm2")

        leaf_area_px    = sum(cv2.contourArea(c) for c in leaf_contours)
        leaf_area_cm2   = leaf_area_px * (cm_per_pixel ** 2)
        roi_area_px     = int(np.sum(roi_mask > 0))
        green_ratio     = (np.sum(green_mask > 0) / roi_area_px * 100) if roi_area_px > 0 else 0.0

        print(f"  Leaf area     : {leaf_area_cm2:.2f} cm2  ({green_ratio:.1f}% of ROI)")

        # Visualization
        output = cv2.bitwise_and(image, image, mask=roi_mask)
        gc     = np.zeros_like(output)
        gc[green_mask > 0] = (0, 255, 80)
        overlay = output.copy()
        cv2.addWeighted(gc, 0.25, overlay, 0.75, 0, output)
        cv2.drawContours(output, leaf_contours, -1, (0, 255, 0), 2)

        cv2.putText(output, f"Leaf area: {leaf_area_cm2:.1f} cm2",
                    (30, 80),  cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,50), 4, cv2.LINE_AA)
        cv2.putText(output, f"Green ratio: {green_ratio:.1f}%",
                    (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,50), 3, cv2.LINE_AA)
        cv2.putText(output, f"Scale: {cm_per_pixel:.5f} cm/px",
                    (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,200,200), 2, cv2.LINE_AA)

        # Save
        base      = os.path.splitext(filename)[0]
        out_path  = os.path.join(output_folder, base + "_analyzed.jpg")
        mask_path = os.path.join(output_folder, base + "_mask.jpg")
        cv2.imwrite(out_path,  output)
        cv2.imwrite(mask_path, green_mask)
        print(f"  Saved: {out_path}")

        # Show result
        _show_result(output, filename)

        results.append({
            "File Name":           filename,
            "Leaf Area (cm2)":     round(leaf_area_cm2, 3),
            "Green Ratio (%)":     round(green_ratio, 2),
            "Scale (cm/pixel)":    round(cm_per_pixel, 6),
            "Min Leaf Area (cm2)": min_leaf_area,
            "Image Path":          path,
        })

    if not results:
        print("\nNo images were processed.")
        return pd.DataFrame()

    df       = pd.DataFrame(results)
    csv_path = os.path.join(output_folder, "analysis_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n{'='*50}")
    print(f"CSV saved: {csv_path}")
    print(df.to_string(index=False))

    return df
