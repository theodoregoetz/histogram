# coding: utf-8
from __future__ import division

import os
from os import path

import numpy as np
from numpy import random as rand

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image

from histogram import Histogram


class ImageComparator(object):
    def __init__(self,
                 baseline_directory='image_comparisons/baseline',
                 test_directory='image_comparisons/test',
                 diff_directory='image_comparisons/diff',
                 tol=1e-3):
        self.baseline_directory = baseline_directory
        self.test_directory = test_directory
        self.diff_directory = diff_directory
        self.tol = tol
        for d in [self.baseline_directory,
                  self.test_directory,
                  self.diff_directory]:
            if not path.exists(d):
                os.makedirs(d)

    @staticmethod
    def _read(fpath, dtype=None):
        with Image.open(fpath, 'r') as img:
            data = np.asarray(img.getdata(), dtype=dtype)
            data.shape = (img.height, img.width, -1)
        return data[:, :, :3]

    @staticmethod
    def _rms(baseline_image, test_image):
        baseline_image = baseline_image.astype(np.int16)
        test_image = test_image.astype(np.int16)
        diff_image = abs(baseline_image - test_image)
        histogram = np.bincount(diff_image.ravel(), minlength=256)
        sum_of_squares = np.sum(histogram * np.arange(len(histogram))**2)
        return np.sqrt(sum_of_squares / baseline_image.size)

    @staticmethod
    def _diff(baseline_fpath, test_fpath):
        baseline_image = ImageComparator._read(baseline_fpath, np.float32)
        test_image = ImageComparator._read(test_fpath, np.float32)
        if baseline_image.shape != test_image.shape:
            diff_image = test_image.sum(2)
            diff_image = diff_image * 255 / diff_image.max()
            diff_image = diff_image.astype(np.uint8)
        else:
            diff_image = abs(baseline_image - test_image)
            diff_image = 255 * 10 * diff_image.sum(2)
            diff_image = np.clip(diff_image, 0, 255).astype(np.uint8)
        return diff_image

    def compare_images(self, savefig, fname, tol=None):
        tol = self.tol if tol is None else tol
        baseline_fpath = path.join(self.baseline_directory, fname)
        test_fpath = path.join(self.test_directory, fname)

        savefig(test_fpath)

        good_match = False
        if path.exists(baseline_fpath):
            baseline_image = ImageComparator._read(baseline_fpath, np.uint8)
            test_image = ImageComparator._read(test_fpath, np.uint8)
            if tol <= 0.0:
                good_match = np.array_equal(baseline_image, test_image)
            else:
                if baseline_image.shape != test_image.shape:
                    good_match = False
                else:
                    rms = ImageComparator._rms(baseline_image, test_image)
                    good_match = rms <= tol

            diff_fpath = path.join(self.diff_directory, fname)
            if good_match:
                if path.exists(diff_fpath):
                    os.remove(diff_fpath)
            else:
                diff_image = ImageComparator._diff(baseline_fpath, test_fpath)
                with Image.fromarray(diff_image, 'L') as img:
                    img.save(diff_fpath)

        return good_match


if __name__ == '__main__':
    rand.seed(1)

    h = Histogram(40, [0,1], 'χ (cm)', 'counts', 'Some Data')
    h.fill(rand.normal(0.5, 0.1, 2000))

    fig, ax = plt.subplots()
    pt = ax.plothist(h)

    comparator = ImageComparator()
    assert comparator.compare_images(fig.savefig, 'hist_1d.png')

