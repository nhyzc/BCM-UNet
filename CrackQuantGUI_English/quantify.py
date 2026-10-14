"""Crack quantification: skeletonisation, junction repair, branch
separation and geometric parameter extraction."""
import numpy as np
from collections import deque
from skimage.morphology import skeletonize, binary_dilation, disk
from skimage.draw import line as sk_line
from scipy.ndimage import label as nd_label, distance_transform_edt

NBR8 = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]


def classify_nodes(skel):
    eps, jns = set(), set()
    h, w = skel.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            if skel[y, x]:
                n = sum(1 for dy, dx in NBR8 if skel[y+dy, x+dx])
                if n == 1: eps.add((y, x))
                elif n >= 3: jns.add((y, x))
    return eps, jns


def extend_endpoints(skel, binary):
    eps, _ = classify_nodes(skel)
    ex = skel.copy()
    for (ey, ex_orig) in eps:
        nbrs = [(ey+dy, ex_orig+dx) for dy, dx in NBR8
                if 0 <= ey+dy < binary.shape[0] and 0 <= ex_orig+dx < binary.shape[1]
                and skel[ey+dy, ex_orig+dx]]
        if len(nbrs) != 1: continue
        ny, nx = nbrs[0]
        vy, vx = ey-ny, ex_orig-nx
        cy, cx = ey, ex_orig
        for s in range(1, 20):
            ny2 = int(round(ey + vy*s)); nx2 = int(round(ex_orig + vx*s))
            if ny2 < 0 or ny2 >= binary.shape[0] or nx2 < 0 or nx2 >= binary.shape[1]: break
            if not binary[ny2, nx2]: break
            cy, cx = ny2, nx2
        if (cy, cx) != (ey, ex_orig):
            rr, cc = sk_line(ey, ex_orig, cy, cx)
            for k in range(len(rr)):
                if 0 <= rr[k] < binary.shape[0] and 0 <= cc[k] < binary.shape[1] and binary[rr[k], cc[k]]:
                    ex[rr[k], cc[k]] = True
    return ex


def line_fit(pts):
    xs = np.array([p[1] for p in pts], dtype=float)
    ys = np.array([p[0] for p in pts], dtype=float)
    if len(xs) < 3: return None, None
    xm, ym = np.mean(xs), np.mean(ys)
    den = np.sum((xs-xm)**2)
    if abs(den) < 1e-10: return None, xm
    slope = np.sum((xs-xm)*(ys-ym)) / den
    return slope, ym - slope*xm


def repair_junctions(skel, binary, jns=None):
    if jns is None:
        _, jns = classify_nodes(skel)
    if not jns: return skel

    h, w = skel.shape
    junc = set(jns)
    for _ in range(6):
        nxt = set(junc)
        for (y, x) in junc:
            for dy, dx in NBR8:
                ny, nx = y+dy, x+dx
                if 0 <= ny < h and 0 <= nx < w and binary[ny, nx]:
                    nxt.add((ny, nx))
        if len(nxt) == len(junc): break
        junc = nxt

    msk = skel.copy()
    for (y, x) in junc: msk[y, x] = False
    lbl, nc = nd_label(msk, structure=np.ones((3, 3)))
    segs = []
    for i in range(1, nc+1):
        ys, xs = np.where(lbl == i)
        if len(ys) >= 5: segs.append(list(zip(ys, xs)))
    if len(segs) < 2: return skel

    jc = (int(np.mean([p[0] for p in junc])), int(np.mean([p[1] for p in junc])))
    pairs = []; used = set()
    for i in range(len(segs)):
        if i in used: continue
        si, _ = line_fit(segs[i])
        if si is None: continue
        ang_i = 90 if si is None else np.degrees(np.arctan(si))
        best_j, best_diff = -1, 999
        for j in range(len(segs)):
            if j == i or j in used: continue
            sj, _ = line_fit(segs[j])
            if sj is None: continue
            ang_j = 90 if sj is None else np.degrees(np.arctan(sj))
            diff = min(abs(ang_i-ang_j), abs(abs(ang_i-ang_j)-180))
            if diff < best_diff and diff < 25:
                best_diff = diff; best_j = j
        if best_j >= 0:
            pairs.append((i, best_j)); used.add(i); used.add(best_j)

    rep = skel.copy()
    for (y, x) in junc: rep[y, x] = False

    for (i, j) in pairs:
        all_pts = segs[i] + segs[j]
        sl, ic = line_fit(all_pts)
        if sl is None and ic is None: continue
        near_i = min(segs[i], key=lambda p: (p[0]-jc[0])**2+(p[1]-jc[1])**2)
        near_j = min(segs[j], key=lambda p: (p[0]-jc[0])**2+(p[1]-jc[1])**2)
        if sl is None:
            xf = int(round(ic))
            rr, cc = sk_line(near_i[0], xf, near_j[0], xf)
        else:
            xi, yi = near_i[1], int(round(sl*near_i[1] + ic))
            xj, yj = near_j[1], int(round(sl*near_j[1] + ic))
            rr, cc = sk_line(yi, xi, yj, xj)
        for k in range(len(rr)):
            if 0 <= rr[k] < h and 0 <= cc[k] < w and (rr[k], cc[k]) in junc:
                rep[rr[k], cc[k]] = True

    for idx in range(len(segs)):
        if idx in used: continue
        near = min(segs[idx], key=lambda p: (p[0]-jc[0])**2+(p[1]-jc[1])**2)
        rr, cc = sk_line(near[0], near[1], jc[0], jc[1])
        for k in range(len(rr)):
            if 0 <= rr[k] < h and 0 <= cc[k] < w and binary[rr[k], cc[k]]:
                rep[rr[k], cc[k]] = True

    return rep


def _neighbors(point, shape):
    y, x = point
    return [(y+dy, x+dx) for dy, dx in NBR8
            if 0 <= y+dy < shape[0] and 0 <= x+dx < shape[1]]


def _second_step(junction, first, skeleton):
    """Pixel one step beyond ``first``, continuing away from ``junction``.

    Used to build the two-pixel outward extension of Section 4.3. The step is
    continued in the same direction when that pixel belongs to the skeleton;
    otherwise the nearest skeleton neighbour of ``first`` is taken.
    """
    step = (first[0] - junction[0], first[1] - junction[1])
    ideal = (first[0] + step[0], first[1] + step[1])

    if (0 <= ideal[0] < skeleton.shape[0] and 0 <= ideal[1] < skeleton.shape[1]
            and skeleton[ideal]):
        return ideal

    options = [n for n in _neighbors(first, skeleton.shape)
               if skeleton[n] and n != junction]
    if not options:
        return first
    return min(options, key=lambda p: (p[0] - ideal[0]) ** 2 + (p[1] - ideal[1]) ** 2)


def _outward_average(junction, first, skeleton):
    """Coordinate average of the two pixels extending outward from a junction."""
    second = _second_step(junction, first, skeleton)
    return (np.asarray(first, dtype=float) + np.asarray(second, dtype=float)) / 2.0


def _direction_angle(vc, vm):
    """beta_m of Eq. (22): angle between the tracking and candidate vectors."""
    norm = float(np.linalg.norm(vc)) * float(np.linalg.norm(vm))
    if norm < 1e-12:
        return 180.0
    cosine = float(np.dot(vc, vm)) / norm
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def separate_branches(skel, eps, jns):
    """Endpoint tracing with the trace-direction judgment of Section 4.3.

    Tracing starts at a fracture endpoint and advances along path points. On
    reaching an intersection the current position is recorded, two pixels are
    extended outward along the tracked branch to give the tracking endpoint
    p_c, and every outgoing branch is sampled the same way to give a candidate
    endpoint p_m. Eq. (22) gives the angle beta_m between v_c and v_m, and the
    candidate closest to 180 degrees is followed as the main fracture branch.
    """
    visited = set()
    branches = []

    def neighbors(y, x):
        return [(y+dy, x+dx) for dy, dx in NBR8
                if 0 <= y+dy < skel.shape[0] and 0 <= x+dx < skel.shape[1] and skel[y+dy, x+dx]]

    for ep in sorted(eps):
        if ep in visited:
            continue
        path = [ep]
        cy, cx = ep
        visited.add(ep)
        prev = None
        visited_branch = {ep}
        for _ in range(10000):
            nbrs = [n for n in neighbors(cy, cx) if n != prev]
            nbrs = [n for n in nbrs if n not in visited or (n in jns and n not in visited_branch)]
            if not nbrs:
                break
            if len(nbrs) >= 2 and (cy, cx) in jns:
                if prev is None or len(path) < 3:
                    nxt = nbrs[0]
                else:
                    junction = (cy, cx)
                    # Tracking endpoint p_c: the two pixels extending outward
                    # along the branch we arrived on.
                    p_c = _outward_average(junction, path[-2], skel)
                    v_c = p_c - np.asarray(junction, dtype=float)

                    best_dev = None
                    nxt = nbrs[0]
                    for n in nbrs:
                        p_m = _outward_average(junction, n, skel)
                        v_m = p_m - np.asarray(junction, dtype=float)
                        deviation = abs(_direction_angle(v_c, v_m) - 180.0)
                        if best_dev is None or deviation < best_dev:
                            best_dev, nxt = deviation, n
            else:
                nxt = nbrs[0]
            prev = (cy, cx)
            cy, cx = nxt
            path.append((cy, cx))
            visited.add((cy, cx))
            visited_branch.add((cy, cx))
        if len(path) >= 3:
            branches.append(path)
    return branches


def calc_length(branch):
    """Trace length: accumulate the Euclidean distance between consecutive
skeleton pixels along the path."""
    if len(branch) < 2: return 0.0
    pts = np.asarray(branch, dtype=float)
    delta = np.diff(pts, axis=0)
    return float(np.sum(np.hypot(delta[:, 0], delta[:, 1])))


def calc_mean_width(branch, distance_map, junction_exclusion=None):
    """Mean branch width.

    For every skeleton pixel the Euclidean distance transform of the binary
    crack region gives the distance d to the nearest background pixel; the
    discrete local width is then 2d-1. Pixels inside a junction neighbourhood
    are excluded, since overlapping cracks would otherwise inflate the width.
    """
    if not branch:
        return 0.0
    widths = []
    h, w = distance_map.shape
    for y, x in branch:
        if not (0 <= y < h and 0 <= x < w):
            continue
        if junction_exclusion is not None and junction_exclusion[y, x]:
            continue
        # d is in pixels; subtract one to match the discrete foreground width.
        widths.append(max(1.0, 2.0 * float(distance_map[y, x]) - 1.0))
    if not widths:
        # Fall back to the whole branch for very short or fully interior ones.
        widths = [max(1.0, 2.0 * float(distance_map[y, x]) - 1.0)
                  for y, x in branch if 0 <= y < h and 0 <= x < w]
    return float(np.mean(widths)) if widths else 0.0


def calc_angle(branch):
    if len(branch) < 2: return 0.0
    xs = np.array([p[1] for p in branch], dtype=float)
    ys = np.array([p[0] for p in branch], dtype=float)
    xm, ym = np.mean(xs), np.mean(ys)
    num = np.sum((xs-xm)*(ys-ym)); den = np.sum((xs-xm)**2)
    if abs(den) < 1e-10: return 90.0
    slope_img = num / den; slope = -slope_img
    ang = np.degrees(np.arctan(slope))
    return float((ang + 180.0) % 180.0)


def quantify(binary_mask, pixel_resolution_um=1.0):
    """Run the full quantification pipeline.

    Images larger than 1024 px on the long side are downscaled for speed; the
    reported pixel lengths and widths are converted back to the original scale.

    Returns (results_list, skeleton_image, overlay_rgb).
    """
    from PIL import Image
    h, w = binary_mask.shape
    scale = 1.0
    max_dim = max(h, w)
    if max_dim > 1024:
        scale = 1024.0 / max_dim
        new_h, new_w = int(h*scale), int(w*scale)
        binary_mask = np.array(Image.fromarray(binary_mask).resize((new_w, new_h), Image.NEAREST))
        pixel_resolution_um /= scale

    h, w = binary_mask.shape

    # Skeletonise, then repair
    skel_raw = skeletonize(binary_dilation(binary_mask, disk(1)))
    skel_raw = extend_endpoints(skel_raw, binary_mask)
    skeleton = repair_junctions(skel_raw, binary_mask)

    # Classify skeleton nodes
    eps, jns = classify_nodes(skeleton)

    # Separate branches
    branches = separate_branches(skeleton, eps, jns)

    # Distance field; junction neighbourhoods are excluded from width stats.
    distance_map = distance_transform_edt(binary_mask)
    junction_mask = np.zeros_like(skeleton, dtype=bool)
    for y, x in jns:
        junction_mask[y, x] = True
    junction_exclusion = binary_dilation(junction_mask, disk(3)) if jns else None

    # Geometric parameters; pixel units are restored to the original scale.
    inv_scale = 1.0 / scale
    results = []
    for i, b in enumerate(branches):
        length_scaled = calc_length(b)
        width_scaled = calc_mean_width(b, distance_map, junction_exclusion)
        le = length_scaled * pixel_resolution_um
        we = width_scaled * pixel_resolution_um
        ang = calc_angle(b)
        results.append({
            'id': i+1,
            'length_px': round(length_scaled * inv_scale, 1),
            'length_um': round(le, 1),
            'width_px': round(width_scaled * inv_scale, 1),
            'width_um': round(we, 1),
            'angle_deg': round(ang, 1),
            'path': b
        })

    # Sort by inclination
    results.sort(key=lambda r: r['angle_deg'])

    # Colour-coded skeleton overlay
    colors = [(255,51,51),(51,204,51),(51,136,255),(255,170,0),(204,51,255)]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[binary_mask] = [60, 60, 60]
    for i, res in enumerate(results):
        c = colors[i % len(colors)]
        for (y, x) in res['path']:
            overlay[y, x] = c

    return results, skeleton, overlay


if __name__ == '__main__':
    # Quick test
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from inference import predict
    from PIL import Image
    test_dir = os.path.dirname(os.path.abspath(__file__))
    results, skel, ov = quantify(np.random.rand(512,512) > 0.9)
    print(f"Found {len(results)} cracks")
