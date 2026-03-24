#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def small_rgb_image():
    return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def small_grayscale_mask():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255
    return mask


@pytest.fixture
def small_semantic_mask():
    mask = np.zeros((64, 64, 3), dtype=np.uint8)
    mask[0:20, :, 0] = 255
    mask[20:44, :, 1] = 255
    mask[44:64, :, 2] = 255
    return mask


@pytest.fixture
def empty_mask():
    return np.zeros((64, 64), dtype=np.uint8)


@pytest.fixture
def full_mask():
    return np.ones((64, 64), dtype=np.uint8) * 255


@pytest.fixture
def small_haralick_image():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[8:24, 8:24] = 128
    return img


@pytest.fixture
def mock_tensorflow_model():
    class MockModel:
        def __init__(self):
            self._input_shape = (None, 256, 256, 3)
            self._output_shape = (None, 256, 256, 3)

        @property
        def input(self):
            class MockInput:
                shape = self._input_shape
            return MockInput()

        @property
        def output(self):
            return np.zeros((1, 256, 256, 3))

        def load_weights(self, path):
            pass

        def predict(self, x, verbose=0):
            batch_size = x.shape[0]
            h, w = x.shape[1], x.shape[2]
            pred = np.random.rand(batch_size, h, w, 3).astype(np.float32)
            return pred

    return MockModel()
