import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
from scipy.spatial import distance as spdis
from tqdm import tqdm
from random import sample
import math


def cornerHarris(image, threshold, k=9, first = True):

    image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask_offset = int(np.floor(k / 2))
    padded_image = cv2.copyMakeBorder(image, mask_offset, mask_offset, mask_offset, mask_offset, cv2.BORDER_CONSTANT)

    # Harris corner
    dest = cv2.cornerHarris(gray, k, 5, 0.05)

    # Reverting back to the original image, with optimal threshold value,  threshold = 0.4
    corner_indices = np.nonzero(dest > threshold * dest.max())

    if len(corner_indices[0]) <= 3500:
        fig, ax = plt.subplots(1, 1)
        ax.imshow(image)
        rows, cols, _ = image.shape
        final = []

        for i, (x, y) in enumerate(zip(corner_indices[0], corner_indices[1])):
            circ = Circle((y, x), edgecolor='red')
            ax.add_patch(circ)
            x+=mask_offset
            y+=mask_offset
            desc = padded_image[x-mask_offset:x + mask_offset + 1, y - mask_offset:y + mask_offset + 1]
            final.append((x, y, desc))



        #plt.show()
        
    else:
        # print(len(corner_indices[0]))
        # fig, ax = plt.subplots(1, 1)
        # ax.imshow(image)
        rows, cols, _ = image.shape
        final = []

        for (x, y) in zip(corner_indices[0], corner_indices[1]):
            # circ = Circle((x, y), edgecolor='red')
            # ax.add_patch(circ)
            desc = np.reshape(padded_image[x-mask_offset:x + mask_offset + 1,
                              y - mask_offset:y + mask_offset + 1], -1)
            final.append((x, y, desc))

        # plt.show()

    return np.array(final)


def computeDistances(desc1, desc2, ncc=False):

    if ncc:
        distances = np.zeros((len(desc1), len(desc2)))
        for index1, (x1, y1, i) in enumerate(tqdm(desc1)):
            for index2, (x2, y2, j) in enumerate(desc2):
                try:
                    distances[index1, index2] = cv2.matchTemplate(i, j, cv2.TM_CCORR_NORMED)
                except:
                    ...
    else:
        aux = desc1[:, 2]
        aux2 = desc2[:, 2]
        norm1 = np.linalg.norm(np.stack(aux), axis=1)
        norm2 = np.linalg.norm(np.stack(aux2), axis=1)

        distances = np.zeros((len(desc1), len(desc2)))
        distances = spdis.cdist(np.stack(aux / norm1), np.stack(aux2 / norm2), 'euclidean')

    return distances


def matchAnalysis(dist, desc1, desc2, ncc=False):

    row_matches = list()

    if not ncc:
        for i, x in enumerate(tqdm(dist)):
            minx = np.argmin(x)
            miny = np.argmin(dist[:, minx])
            if miny == i:
                row_matches.append([desc1[i][:2], desc2[minx][:2], dist[i, minx]])

    else:
        for i, x in enumerate(tqdm(dist)):
            maxy = np.argmax(x)
            maxx = np.argmax(dist[:, maxy])
            if maxx == i:
                row_matches.append([desc1[i][:2], desc2[maxy][:2], dist[i, maxy]])

    return row_matches


def fundamentalMatrix_Error(matches, fund_matrix, threshold=10):

    inliers = []
    error = 0

    for matchA, matchB, _ in matches:
        #Changed order
        desc1_y, desc1_x = matchA
        desc2_y, desc2_x = matchB

        homo_a = np.expand_dims(np.hstack((matchA, 1)), axis=1)
        homo_b = np.expand_dims(np.hstack((matchB, 1)), axis=1)
        sample_error = homo_b - np.dot(fund_matrix, homo_a)
        sample_error = math.hypot(sample_error[0], sample_error[1])
        error += sample_error

        if -threshold < sample_error < threshold:
            inliers.append((matchA, matchB, error))

    return inliers, len(inliers), error


def fundamentalMatrix(matches):

    #fundamental_matrix = None
    fund_mat = np.zeros((6, 6))
    eq_result = np.zeros((6, 1))

    for matchA, matchB in matches:
        # Changed order
        desc1_y, desc1_x = matchA
        desc2_y, desc2_x = matchB

        fund_mat[0, 0] += desc1_x * desc1_x
        fund_mat[0, 1] += desc1_x * desc1_y
        fund_mat[0, 2] += desc1_x
        fund_mat[1, 0] += desc1_x * desc1_y
        fund_mat[1, 1] += desc1_y * desc1_y
        fund_mat[1, 2] += desc1_y
        fund_mat[2, 0] += desc1_x
        fund_mat[2, 1] += desc1_y
        fund_mat[2, 2] += 1

        fund_mat[3, 3] += desc1_x * desc1_x
        fund_mat[3, 4] += desc1_x * desc1_y
        fund_mat[3, 5] += desc1_x
        fund_mat[4, 3] += desc1_x * desc1_y
        fund_mat[4, 4] += desc1_y * desc1_y
        fund_mat[4, 5] += desc1_y
        fund_mat[5, 3] += desc1_x
        fund_mat[5, 4] += desc1_y
        fund_mat[5, 5] += 1

        eq_result[0, 0] += desc2_x * desc1_x
        eq_result[1, 0] += desc2_x * desc1_y
        eq_result[2, 0] += desc2_x
        eq_result[3, 0] += desc2_y * desc1_x
        eq_result[4, 0] += desc2_y * desc1_y
        eq_result[5, 0] += desc2_y

    affine_unknown = np.linalg.lstsq(fund_mat, eq_result)
    fund_matrix = np.reshape(affine_unknown[0], (2, 3))
    fund_matrix = np.vstack((fund_matrix, [0, 0, 1]))

    return fund_matrix

def ransac(matches, threshold=5):

    iterations = 200
    matches = np.array(matches)
    best_model = {'count': -1, 'model': None, 'inliers': None, 'error':-1}

    for i in tqdm(range(iterations)):
        sampled = sample(range(len(matches)), 3)
        coordinates = matches[sampled, :2]
        fund_matrix = fundamentalMatrix(coordinates)
        inliers, inlier_count, error = fundamentalMatrix_Error(matches, fund_matrix, threshold=threshold)
        if inlier_count > best_model['count']:
            best_model['inliers'] = inliers
            best_model['count'] = inlier_count
            best_model['model'] = fund_matrix
            best_model['error'] = error

    return best_model['model'], np.array([f[0] for f in best_model['inliers']]), best_model['error']

def crop(image):

    #crop top
    if not np.sum(image[0]):
        return crop(image[1:])
    #crop top
    if not np.sum(image[-1]):
        return crop(image[:-2])
    #crop top
    if not np.sum(image[:, 0]):
        return crop(image[:, 1:])
    #crop top
    if not np.sum(image[:, -1]):
        return crop(image[:, :-2])

    return image

def stitching(best_model, inliers, image1, image2):

    image1 = cv2.imread(image1)
    image2 = cv2.imread(image2)

    # Apply panorama correction
    height = image2.shape[0] + image1.shape[0]
    width = image2.shape[1] + image1.shape[1]

    if best_model[0, 2] < 0:
        best_model[0, 2] = -best_model[0, 2]
        result = cv2.warpPerspective(image2, best_model, (height, width))
        result[:image1.shape[0], :image1.shape[1]] = image1
    else:
        result = cv2.warpPerspective(image1, best_model, (height, width))
        result[:image1.shape[0], :image1.shape[1]] = image2


    plt.close('all')
    plt.figure(figsize=(10, 8))
    plt.imshow( crop(result) )
    plt.axis('off')
    plt.show()

    return None




if __name__ == '__main__':
    img1_path = '/home/demet/Desktop/000/000000_pitch1_yaw3.jpg'
    img2_path = '/home/demet/Desktop/000/000000_pitch1_yaw2.jpg'
    desc1 = cornerHarris(img1_path, 0.2)
    desc2 = cornerHarris(img2_path, 0.2)
    dist = computeDistances(desc1, desc2, True)
    matches = matchAnalysis(dist, desc1, desc2, True)
    best_model, inliers_img_a, error = ransac(matches, 5.5)
    stitched_img = stitching(best_model, inliers_img_a, img1_path, img2_path)

    ...
