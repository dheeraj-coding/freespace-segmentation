from copy import deepcopy

import numpy as np
import torch
from skimage import segmentation
import tqdm


class SuperPixelAlign(object):
    def __init__(self, scale, sigma, min_size, n_sample_points, feature_shape, img_shape):
        self.scale = scale
        self.sigma = sigma
        self.min_size = min_size
        self.n_sample_points = n_sample_points
        self.feature_shape = feature_shape
        self.img_shape = img_shape
        x_t, y_t = np.meshgrid(np.linspace(-1, 1, self.img_shape[1]), np.linspace(-1, 1, self.img_shape[0]))
        self.sampling_grid = np.vstack([x_t.flatten(), y_t.flatten(), np.ones(x_t.shape[0] * x_t.shape[1])])
        self.x_ind = np.linspace(0, img_shape[1], img_shape[1])
        self.y_ind = np.linspace(0, img_shape[0], img_shape[0])

    def segment(self, img):
        return segmentation.felzenszwalb(img, self.scale, self.sigma, self.min_size)

    def bilinear_interpolate(self, img, features, segments, si):
        f_H, f_W, f_C = features.shape
        H, W, C = img.shape

        scaling_factor = f_H / H
        M = np.array([[scaling_factor, 0, 0], [0, scaling_factor, 0]])

        segments = segments.reshape((H * W))
        superpixels = self.sampling_grid[:, segments == si]
        sample_indices = np.random.randint(0, superpixels.shape[1], size=self.n_sample_points)

        sample_points = superpixels[:, sample_indices]
        original_sample = sample_points.copy()
        sample_points = np.matmul(M, sample_points)

        x_s = sample_points[0, :]
        y_s = sample_points[1, :]
        x = ((x_s + 1.) * f_W) * 0.5
        y = ((y_s + 1.) * f_H) * 0.5

        x0 = np.floor(x).astype(np.int64)
        x1 = x0 + 1
        y0 = np.floor(y).astype(np.int64)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, f_W - 1)
        x1 = np.clip(x1, 0, f_W - 1)
        y0 = np.clip(y0, 0, f_H - 1)
        y1 = np.clip(y1, 0, f_H - 1)

        Ia = features[y0, x0, :]
        Ib = features[y1, x0, :]
        Ic = features[y0, x1, :]
        Id = features[y1, x1, :]

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)
        wa = np.expand_dims(wa, 1)
        wb = np.expand_dims(wb, 1)
        wc = np.expand_dims(wc, 1)
        wd = np.expand_dims(wd, 1)

        interpolated = wa * Ia + wb * Ib + wc * Ic + wd * Id
        interpolated = np.hstack(
            (interpolated, original_sample[0, :].reshape((self.n_sample_points, 1)),
             original_sample[1, :].reshape((self.n_sample_points, 1))))
        return interpolated

    def align(self, img, features):
        segments = self.segment(img)
        unique = np.unique(segments)
        superpixel_aligned = np.zeros((len(unique), self.n_sample_points, self.feature_shape[2] + 2))
        for i in unique:
            superpixel_aligned[i] = self.bilinear_interpolate(img, features, segments, i)

        return superpixel_aligned, segments


class WeightedKMeans(object):
    def __init__(self, k, mu, sigma, thresh=0.0):
        self.k = k
        self.thresh = thresh
        self.mu = mu
        self.sigma = sigma

    def get_weights(self, pixels):
        pxy = pixels[:, -2:]
        pxy = pxy - self.mu
        pxy = np.abs(pxy)
        pxy = np.sum(pxy, axis=1)
        pxy = pxy ** 2
        pxy = pxy / (2 * (self.sigma ** 2))
        pxy = np.exp(-pxy)
        wi = np.sum(pxy)
        wi /= len(pixels)
        return wi

    def dist(self, a, b, axis=1):
        return np.linalg.norm(a - b, axis=axis) ** 2

    def fit(self, sxy):
        si = np.mean(sxy, axis=1)
        si = si[:, :-2]
        weights = np.array([self.get_weights(sxy[i]) for i in range(len(sxy))])
        median = np.median(weights)
        labels = np.array([0 if weights[i] > median else np.random.randint(1, self.k) for i in range(len(sxy))])
        c = si[np.random.randint(0, len(si), size=self.k)]
        c_old = np.zeros((self.k, si.shape[1]))

        while self.dist(c, c_old, None) > self.thresh:
            c_old = deepcopy(c)
            temp = weights[labels == 0].reshape((-1, 1)) * si[labels == 0]
            c[0] = np.sum(temp, axis=0) / (np.sum(weights[labels == 0]) + 1e-10)
            for i in range(1, self.k):
                temp = (1 - weights[labels == i]).reshape((-1, 1)) * si[labels == i]
                c[i] = np.sum(temp, axis=0) / (np.sum(1 - weights[labels == i]) + 1e-10)
            for j in range((len(si))):
                labels[j] = np.argmin((self.dist(c, si[j].reshape((1, -1)))).reshape((-1)))
        return c, labels


class LabelGenerator(object):
    def __init__(self, img_shape, feature_shape, scale=100, sigma=0.5, min_size=50, n_sample_points=15, k=5,
                 mu=(0.5, 0.), psigma=0.1, thresh=0.0001):
        self.img_shape = img_shape
        self.feature_shape = feature_shape
        self.scale = scale
        self.sigma = sigma
        self.min_size = min_size
        self.n_sample_points = n_sample_points
        self.k = k
        self.mu = np.array([[mu[0], mu[1]]])
        self.psigma = psigma
        self.thresh = thresh
        self.aligner = SuperPixelAlign(self.scale, self.sigma, self.min_size, self.n_sample_points, self.feature_shape,
                                       self.img_shape)
        self.kmeans = WeightedKMeans(self.k, self.mu, self.psigma, self.thresh)

    def process_batch(self, prediction, xo):
        result = None
        batch = xo.size()[0]
        for i in range(batch):
            superpixel_aligned, segments = self.aligner.align(xo[i], prediction[i])
            c, labels = self.kmeans.fit(superpixel_aligned)
            for j in np.unique(segments):
                segments[segments == j] = labels[j]
            if result is None:
                result = np.zeros((1, self.img_shape[0], self.img_shape[1]))
                result[0] = segments
            else:
                result = np.vstack((result, np.expand_dims(segments, axis=0)))
        return result

    def generate(self, dataloader, model):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        progress = tqdm.tqdm()
        images = None
        result = None
        for x, xo in dataloader:
            x = x.to(device)
            prediction = model(x)
            prediction = prediction.permute(0, 2, 3, 1)
            prediction = prediction.cpu().detach().numpy()
            batch_result = self.process_batch(prediction, xo)
            if result is None:
                result = batch_result
            else:
                result = np.vstack((result, batch_result))
            if images is None:
                images = xo
            else:
                images = np.vstack((images, xo))
            progress.update()
        return result, images
