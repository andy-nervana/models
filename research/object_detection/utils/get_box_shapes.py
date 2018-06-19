import numpy as np
import pickle

from multiprocessing import Pool

GT_FILE = '/nfs/site/home/takeller/repo/models/research/object_detection/dataset_tools/train_all_bboxes.pkl'
RESULTS_FILE = '/nfs/site/home/takeller/repo/models/research/object_detection/dataset_tools/best_anchors.pkl'


def get_boxes(min_scale, max_scale, num_layers, aspect_ratios, base_size=[1.0, 1.0]):
    scales = [min_scale + (max_scale - min_scale) * i / (num_layers - 1) for i in range(num_layers)] + [1.0]
    all_scales = sorted(scales + [np.sqrt(scales[i] * scales[i+1]) for i in range(len(scales) - 1)])


    boxes = []
    for ar in aspect_ratios:
        for scale in scales:
            height = scale / np.sqrt(ar) * base_size[0]
            width = scale * np.sqrt(ar) * base_size[1]

            boxes.append([width, height])

    return np.array(boxes)


def smooth_l1_loss(x, delta=1.0):
    loss = 0.0

    for i in range(2):
        if np.abs(x[:, i]) <= delta:
            loss += 0.5 * (x[:, i]**2)
        else:
            return np.abs(x[:, 0]) - 0.5 + np.abs(x[:, 1]**2) - 0.5

def l2_loss(x):
    return np.sqrt(x[:, 0]**2 + x[:, 1]**2)


def min_loss(gt_sizes, anchors, loss_func=l2_loss):
    total_loss = 0.0

    anchor_usage = np.zeros(anchors.shape[0])

    for gtb in gt_sizes:
        dist = anchors - gtb

        # losses for each anchor
        losses = loss_func(dist)

        # get anchor with min loss
        min_loss_idx = np.argmin(losses)
        min_loss = losses[min_loss_idx]
        anchor_usage[min_loss_idx] += 1

        total_loss += min_loss

    return total_loss, anchor_usage


def random_augment(boxes, area_range=[0.1, 1.0], aspect_ratio_range=(0.5, 2.0), aug_prob=0.85, num_iterations=20):
    aug_gt_sizes = []

    for i in range(num_iterations):
        for gt in boxes:
            if np.random.uniform() < aug_prob:
                area = 1.0 / np.random.uniform(area_range[0], area_range[1])
                ar = np.random.uniform(aspect_ratio_range[0], aspect_ratio_range[1])
                aug = [area * np.sqrt(ar), area / np.sqrt(ar)]

                gt_aug = gt * aug

                aug_gt_sizes.append(gt_aug)

            else:
                aug_gt_sizes.append(gt)

    return np.array(aug_gt_sizes)


def run_settings(all_settings):
    settings, aug_gt_sizes, done = all_settings
    anchors = get_boxes(settings['MIN_SCALE'], settings['MAX_SCALE'], settings['NUM_LAYERS'], settings['ASPECT_RATIO'], settings['BASE_SIZE'])
    total_loss, anchor_usage = min_loss(aug_gt_sizes, anchors, loss_func=l2_loss)
    most_used_idxs = np.argsort(anchor_usage)[::-1]
    print("{}, {}".format(done, total_loss))
    return (settings, total_loss, anchor_usage[most_used_idxs], anchors[most_used_idxs])

if __name__ == '__main__':
    with open(GT_FILE, 'rb') as f:
       gt_bboxes = pickle.load(f)

    gt_bboxes = np.array(gt_bboxes)
    gt_widths = gt_bboxes[:, 2] - gt_bboxes[:, 0]
    gt_heights = gt_bboxes[:, 3] - gt_bboxes[:, 1]

    gt_sizes = np.vstack([gt_widths, gt_heights]).T

    aug_gt_sizes = random_augment(gt_sizes)

    MIN_SCALES = [0.005, 0.01, 0.02, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.33, 0.4, 0.5, 0.6]
    MAX_SCALES = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 25]
    NUM_LAYERS = [6]
    ASPECT_RATIOS = [[1.0, 2.0, 3.0, 0.5, 0.33], [1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 0.5, 4.0]]
    BASE_SIZES = [[1.0, 1.0], [0.1, 0.1], [0.5, 0.5], [0.25, 0.25], [0.33, 0.33], [0.05, 0.05], [0.025, 0.025], [0.01, 0.01], [0.075, 0.075], [0.033, 0.033]]

    losses = []

    num_tests = len(MIN_SCALES) * len(MAX_SCALES) * len(NUM_LAYERS) * len(ASPECT_RATIOS) * len(BASE_SIZES)
    done = 0 

    p = Pool(80)
    all_settings = []

    for min_scale in MIN_SCALES:
        for max_scale in MAX_SCALES:
            for num_layers in NUM_LAYERS:
                for aspect_ratios in ASPECT_RATIOS:
                    for base_size in BASE_SIZES:
                        settings = {"MIN_SCALE": min_scale, 
                                    "MAX_SCALE": max_scale, 
                                    "NUM_LAYERS": num_layers, 
                                    "ASPECT_RATIO": aspect_ratios,
                                    "BASE_SIZE": base_size}

                        # anchors = get_boxes(min_scale, max_scale, num_layers, aspect_ratios, base_size)

                        # total_loss, anchor_usage = min_loss(aug_gt_sizes, anchors, loss_func=l2_loss)

                        # most_used_idxs = np.argsort(anchor_usage)[::-1]
                        
                        # losses.append((settings, total_loss, anchor_usage[most_used_idxs], anchors[most_used_idxs]))
                        all_settings.append([settings, aug_gt_sizes, done])

                        done += 1 
                        print("{}/{}".format(done, num_tests))# , losses[-1][1], losses[-1][0]))

    losses = p.map(run_settings, all_settings)

    print(losses)
    best_losses = sorted(losses, key=lambda x: x[1])
    print("Best: {}".format(best_losses[0]))

    with open(RESULTS_FILE, 'wb') as f:
        pickle.dump(best_losses, f)

    import ipdb
    ipdb.set_trace()

