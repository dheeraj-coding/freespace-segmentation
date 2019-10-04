import numpy as np
from skimage import segmentation


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
            (interpolated, x_s.reshape((self.n_sample_points, 1)), y_s.reshape((self.n_sample_points, 1))))
        return interpolated

    def align(self, img, features):
        segments = self.segment(img)
        unique = np.unique(segments)
        superpixel_aligned = np.zeros((len(unique), self.n_sample_points, self.feature_shape[2] + 2))
        for i in unique:
            superpixel_aligned[i] = self.bilinear_interpolate(img, features, segments, i)

        return superpixel_aligned, segments
