import argparse
import pickle
import cv2
import numpy as np
import os
import seaborn as sns
from skimage.measure import regionprops
from skimage.measure import label as sklabel


# Attempts to find a region center in the layer which is within +-10 pixels in both x & y
# of the center being tested against.
def match_region(regions, center):
    matches = []
    for region in regions:
        if ((int(region.centroid[0] > center[0] - 10) and int(region.centroid[0] < center[0] + 10))
                and (int(region.centroid[1] > center[1] - 10) and int(region.centroid[1] < center[1] + 10))):
            matches.append(region)
    return matches


# Used to map specific corallites back to the raw data.
# Given a list of unique corallite indices and colours, draw the region back onto the raw data.
def visualise_corallite_mapping_on_slices(coral_data, pred_dir, raw_dir, scale_factor, corallite_indices, colours):
    mapped_dir = "../coral_data/full_coral/DISTRIBUTED_LB_Poritres_ZMA-COEL-6781_220kV_Cu2mm_perc/mapped_corallites"
    preds = os.listdir(pred_dir)
    preds.sort()
    raw = [x for x in os.listdir(raw_dir) if x != '.DS_Store']
    raw.sort()
    for i, index in enumerate(corallite_indices):
        corallite = coral_data[index]
        for region in corallite:
            slice_index = int(region['loc'][2] * 10)
            pred_slice = ((cv2.cvtColor(cv2.imread(f"{pred_dir}/{preds[slice_index]}"),
                                        cv2.COLOR_RGB2GRAY) > 0) * 255).astype(np.uint8)
            name = preds[slice_index]
            if os.path.exists(f"{mapped_dir}/{name}"):
                coloured_output = cv2.imread(f"{mapped_dir}/{name}")
            else:
                coloured_output = cv2.imread(f"{raw_dir}/{raw[slice_index]}")

            slice_regions = regionprops(sklabel(pred_slice))

            center = (int(((region['loc'][0] / scale_factor) + pred_slice.shape[0] / 2)),
                      int(((region['loc'][1] / scale_factor) + pred_slice.shape[1] / 2)))
            matched_regions = match_region(slice_regions, center)
            if len(matched_regions) < 1:
                print("No matches")
                print(matched_regions, index)
            else:
                col = [x * 255 for x in colours[i]]
                for pix in matched_regions[0].coords:
                    coloured_output[pix[0]][pix[1]] = col
                # cv2.imshow("test", coloured_output)
                # cv2.waitKey()
                cv2.imwrite(f"{mapped_dir}/{name}", coloured_output)
        print("finished corallite ", index)


# Labels the outputs of the above function.
def label_visualisations(corallites, cols):
    saved_dir = "../coral_data/full_coral/DISTRIBUTED_LB_Poritres_ZMA-COEL-6781_220kV_Cu2mm_perc/mapped_corallites"
    for file in os.listdir(saved_dir):
        img = cv2.imread(f"{saved_dir}/{file}")
        for i, corallite in enumerate(corallites):
            label = "Corallite: " + str(corallite)
            col = [x * 255 for x in cols[i]]
            cv2.putText(img, label, (100, 100 + (50 * i)), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2, cv2.LINE_AA)

        cv2.imwrite(f"{saved_dir}/{file}", img)


# Utility to check the IoU between two corallite regions.
def check_bbox_iou(bb1, bb2):
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(0.000001 + bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


# Given a region, search all regions in the previous layer to find the nearest neighbour.
# Could be optimised to only search a subset of regions.
def find_nn_in_previous_layer(center, bbox, prev_layer, nn_threshold, iou_thresh, use_thresh=True):
    nn = np.infty
    min_dist = np.infty
    center_xyz = np.array([center[0], center[1], center[2]])
    for corallite in prev_layer:
        test_cent_xyz = np.array([corallite['loc'][0], corallite['loc'][1], corallite['loc'][2]])
        dist = np.linalg.norm(center_xyz - test_cent_xyz)
        if use_thresh:
            if dist < min_dist and dist < nn_threshold:
                iou = check_bbox_iou(bbox, corallite['bbox'])
                if iou > iou_thresh:
                    min_dist = dist
                    nn = corallite['id']
        # No thresholds, just find nearest neighbour..
        else:
            if dist < min_dist:
                min_dist = dist
                nn = corallite['id']
    return nn


# Main loop which generates a dictionary of unique corallites based on the parameters determined in main.
def construct_data_from_predictions(pred_dir, scaling_factor, nn_threshold, iou_thresh, search_depth, use_thresh):
    preds = os.listdir(pred_dir)
    preds.sort()
    coral = dict()
    next_id = 0
    this_layer = []
    prev_layers = []
    for layer_num, p in enumerate(preds):
        print(f"Processing layer {layer_num + 1} of {len(preds)}")
        # if layer_num < 100:
        data = ((cv2.cvtColor(cv2.imread(f"{pred_dir}/{p}"), cv2.COLOR_RGB2GRAY) > 0) * 255).astype(np.uint8)
        regions = regionprops(sklabel(data))
        for region in regions:
            # if i <= 600:
            # Shift region centroid to ensure center of render is at world (x:0, y:0)
            center = ((region.centroid[0] - data.shape[0] / 2) * scaling_factor,
                      (region.centroid[1] - data.shape[1] / 2) * scaling_factor, layer_num * 0.1)
            bbox = [region.bbox[0] - data.shape[0] / 2, region.bbox[1] - data.shape[1] / 2,
                    region.bbox[2] - data.shape[0] / 2, region.bbox[3] - data.shape[0] / 2]
            if layer_num == 0:
                id = next_id
                next_id += 1
            else:
                for layer in prev_layers:
                    id = find_nn_in_previous_layer(center, bbox, layer, nn_threshold, iou_thresh,
                                                   use_thresh=use_thresh)
                    # If NN is found, break from loop
                    if id != np.infty:
                        break
                # If no close neighbour in all prev layers, assign a new id
                if id == np.infty:
                    id = next_id
                    next_id += 1
            corallite = {'id': id, 'loc': center, 'bbox': bbox,
                         'shape': ((region.axis_major_length / 2.5) * scaling_factor,
                                   (region.axis_minor_length / 2.5) * scaling_factor, 1),
                         'orient': region.orientation}
            this_layer.append(corallite)
            if id in coral.keys():
                coral[id].append(corallite)
            else:
                coral[id] = [corallite]

        # Maintain a list of previous layers of len==search_depth
        if len(prev_layers) >= search_depth:
            prev_layers.pop(0)
        prev_layers.append(this_layer)
        this_layer = []
    coral['total'] = next_id
    return coral


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str,
                    default='YOUR_PATH/Individual_Corallite_Reconstruction/data/example_species',
                    help='Path to the species directory')
parser.add_argument('--scale', type=float, default=0.01, help="Factor to scale the regions within blender")
parser.add_argument('--nn', type=float, default=0.15, help="Maximum Euclidean distance threshold to consider "
                                                           "a prediction a viable nearest neighbour")
parser.add_argument('--iou', type=float, default=0.3, help="Minimum IoU to consider region overlap as "
                                                           "possible neighbour")
parser.add_argument('--search', type=int, default=3, help="Number of sequential scan layers to search for "
                                                          "a matching region.")
parser.add_argument('--fn', type=str, default=None, help="Save file name")

args = parser.parse_args()

if __name__ == '__main__':
    root_dir = args.root
    raw_dir = f"{root_dir}/complete_raw"
    prediction_dir = f"{root_dir}/predictions"
    save_dir = f"{root_dir}/blender_data"

    scaling_factor = args.scale
    nn_thresh = args.nn
    iou_thresh = args.iou
    search_depth = args.search

    if args.fn is None:
        print("Please provide a file name with the argument --fn {NAME}")
        exit(1)
    else:
        file_name = f"{args.fn}.pkl"

    # # Construct data set using thresholds
    coral = construct_data_from_predictions(prediction_dir, scaling_factor, nn_thresh, iou_thresh, search_depth, True)
    with open(f"{save_dir}/{file_name}", 'wb') as fp:
        pickle.dump(coral, fp)

    # Construct with no thresholds
    # coral = construct_data_from_predictions(prediction_dir, scaling_factor, nn_thresh, iou_thresh, search_depth, False)
    # with open(f"{save_dir}/nothresh_data.pkl", 'wb') as fp:
    #     pickle.dump(coral, fp)

    # Example method to load an already generated corallite map..
    # with open(f"{save_dir}/{file_name}", 'rb') as fp:
    #     coral = pickle.load(fp)
    # print()

    # Used to visualise some unique corallite mappings back onto the raw data.
    # cols = sns.color_palette()
    # corallites = [147, 187, 191, 269, 566, 630, 888, 1045, 1105, 1884]
    # visualise_corallite_mapping_on_slices(coral, prediction_dir, raw_dir, scaling_factor, corallites, cols)
    # label_visualisations(corallites, cols)
