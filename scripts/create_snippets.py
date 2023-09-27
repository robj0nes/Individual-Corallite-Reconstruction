import argparse
import os
import shutil
from random import shuffle
import cv2
import numpy as np
from skimage.measure import label, regionprops


def trial_snippet(image, label_data, snippet_size, filename):
    snippet_center = int(np.floor(snippet_length / 2))
    test_out = cv2.cvtColor(image[0, :, :], cv2.COLOR_GRAY2RGB)
    for i in range(1, snippet_size):
        if i == snippet_center:
            test_ann = cv2.cvtColor(label_data[0], cv2.COLOR_GRAY2RGB)
            test_ann = (test_ann > 0) * np.array([0, 0, 255]).astype(np.uint8)
            test_im = cv2.cvtColor(image[i, :, :], cv2.COLOR_GRAY2RGB)
            combined_test = cv2.addWeighted(test_im, 1, test_ann, 0.4, 0)
            test_out = np.concatenate((test_out, combined_test), axis=1)
        else:
            im = cv2.cvtColor(image[i, :, :], cv2.COLOR_GRAY2RGB)
            test_out = np.concatenate((test_out, im), axis=1)
    cv2.imwrite(f"{check_dir}/{filename}.png", test_out)


# NOTE: This function uses string operations to extract pertinent information, based on teh naming conventions
#   used in the data for the principle project. It will need adjusting for dataset with different naming conventions.
def construct_snippets(snippet_length, resize, tiled, scale_factor, ims, anns, image_dir, label_dir, step_size=None):

    orientation = anns[0].split('Axis_')[1].split('_CentreSlice')[0]                    # TODO: Naming convention specific.

    for i in range(len(ims)):
        if i > len(ims) - snippet_length:
            break

        start = ims[i].split('Slice_')[2].split('_')[0]                                 # TODO: Naming convention specific.
        end = ims[i + snippet_length - 1].split('Slice_')[2].split('_')[0]              # TODO: Naming convention specific.

        if not tiled:
            for j in range(snippet_length):
                image_snippet.append(
                    cv2.resize(
                        cv2.cvtColor(
                            cv2.imread(f"{image_dir}/{ims[i + j]}"),
                            cv2.COLOR_BGR2GRAY),
                        [resize, resize])
                )
            label_index = i + int((np.floor(snippet_length / 2) + 1))
            label_data = [cv2.resize(
                cv2.cvtColor(
                    cv2.imread(f"{label_dir}/{anns[label_index]}"),
                    cv2.COLOR_BGR2GRAY),
                [resize, resize])]
            np_image = np.array(image_snippet)
            np_label = np.array(label_data)
            file_name = f"{orientation}_{start}_{end}"
            np.savez(f"{snippet_dir}/all/{file_name}.npz", image=np_image, label=np_label, seq=file_name)

        else:
            center_index = int(np.floor(snippet_length / 2))

            center_slice_num = ims[i + center_index].split('Slice_')[2].split('_')[0]   # TODO: Naming convention specific.

            for l in anns:
                if center_slice_num in l:
                    label_data = cv2.cvtColor(cv2.imread(f"{label_dir}/{l}"), cv2.COLOR_BGR2GRAY)
                    break

            # Error checks.
            if label_data is None:
                print("unable to find label_data")
                exit(1)
            if step_size is None:
                step_size = resize

            # Get the annotation at the center of the snippet.
            for y in range(int(label_data.shape[0] / step_size)):
                for x in range(int(label_data.shape[0] / step_size)):
                    image_snippet = []
                    # Take a patch of the label_data and check it is non-zero
                    cropped_label = label_data[y * step_size:y * step_size + resize, x * step_size:x * step_size + resize]
                    if np.size(cropped_label) == 0 or np.count_nonzero(cropped_label) < np.size(cropped_label) / 8:
                        continue
                    else:
                        # Up-scaling not recommended unless absolutely necessary as it will add noise to the real data.
                        # Can be introduced by setting scale_factor > 1.
                        upSize = (resize * scale_factor, resize * scale_factor)
                        cropped_label_upscaled = [cv2.resize(cropped_label, dsize=upSize, fx=scale_factor,
                                                            fy=scale_factor,
                                                            interpolation=cv2.INTER_CUBIC)]

                        # Construct snippet length (SL) of input images with dim [SL x step_size x step_size]
                        for m in range(snippet_length):
                            image = cv2.cvtColor(cv2.imread(f"{image_dir}/{ims[i + m]}"), cv2.COLOR_BGR2GRAY)
                            # cropped_image = image[y * resize:(y + 1) * resize, x * resize:(x + 1) * resize]
                            cropped_image = image[y * step_size:y * step_size + resize, x * step_size:x * step_size + resize]
                            cropped_image_upscaled = cv2.resize(cropped_image, dsize=upSize, fx=scale_factor,
                                                                fy=scale_factor,
                                                                interpolation=cv2.INTER_CUBIC)
                            image_snippet.append(cropped_image_upscaled)

                    np_image = np.array(image_snippet)
                    np_label = np.array(cropped_label_upscaled)
                    # Extract region properties for topological loss validation/verification.
                    regions = regionprops(label(cropped_label_upscaled[0]))

                    file_name = f"{orientation}_{start}_{end}_row{y*step_size}_col{x*step_size}"

                    # Output a visual representation of the snippet for manual validation.
                    trial_snippet(np_image, np_label, snippet_length, file_name)

                    print("Saving snippet: " + file_name)
                    np.savez(f"{snippet_dir}/all/{file_name}.npz", image=np_image, label=np_label, regions=regions, seq=file_name)


def run_tt_split(all_snippets):
    props = [0.8, 0.2]

    data_list = [x for x in os.listdir(all_snippets)]

    indices = [x for x in range(len(data_list))]

    num_test = round(len(data_list) * props[1])
    num_train = len(data_list) - num_test
    shuffle(indices)
    train_idx = {"group": "train", "list": indices[:num_train]}
    test_idx = {"group": "test", "list": indices[num_train:]}

    for l in [train_idx, test_idx]:
        for index in l["list"]:
            shutil.copy(f"{all_snippets}/{data_list[index]}", f"{snippet_dir}/{l['group']}/{data_list[index]}")


def write_txt_files(dirs):
    for subdir in dirs:
        with open(f"{listDir}/{subdir}.txt", 'w') as f:
            for line in os.listdir(f"{snippet_dir}/{subdir}"):
                f.write(line + '\n')
        if subdir == 'train':
            shutil.copy(f"{listDir}/{subdir}.txt", f"{listDir}/val.txt")


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str,
                    default='YOUR_PATH/Individual_Corallite_Reconstruction/data/example_species', help='Path to the species directory')
args = parser.parse_args()

if __name__ == '__main__':
    print("-------------------------")
    print("Creating snippets...")

    # data = np.load("/VT-UNet/data/Synapse/train_npz/XA1_2.npz", allow_pickle=True)
    rootDir = args.root
    imDir = f'{rootDir}/training_data/raw_images'
    labelDir = f'{rootDir}/training_data/annotations'

    snippet_length = 5      # Determines the depth of each snippet (along the axis of projection).
    snippet_size = 224      # Determines the width and height of the image tile.
    tiled = True            # Boolean to enable/disable tiling.
    scale_factor = 1        # Factor to facilitate up-scaling the image. [Not required]

    listDir = f"{rootDir}/training_data/data_lists"

    snippet_dir = f"{rootDir}/training_data/snippets"
    check_dir = f"{rootDir}/training_data/snippet_checks"

    # Use this block for full sequence images/annotations in separate directories.
    ims = os.listdir(imDir)
    ims.sort()
    labels = os.listdir(labelDir)
    labels.sort()
    construct_snippets(snippet_length, snippet_size, tiled, scale_factor, ims, labels, imDir, labelDir, step_size=112)
    run_tt_split(f"{snippet_dir}/all")

    write_txt_files(['train', 'test'])

    print("-------------------------")
    print("Snippets made.\n")
