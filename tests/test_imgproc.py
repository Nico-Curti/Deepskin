#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
from deepskin.imgproc import imfill, get_perilesion_mask


class TestImfill:

    def test_imfill_basic_hole_filling(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[8:24, 8:24] = 255
        img[12:20, 12:20] = 0
        
        result = imfill(img)
        
        assert result.shape == img.shape
        assert result.dtype == np.uint8
        assert result[15, 15] == 255

    def test_imfill_already_filled(self):
        img = np.ones((32, 32), dtype=np.uint8) * 255
        
        result = imfill(img)
        
        assert np.all(result == 255)

    def test_imfill_empty_image(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        
        result = imfill(img)
        
        assert result.shape == img.shape
        assert np.all(result == 0)

    def test_imfill_multiple_holes(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        img[10:30, 10:30] = 255
        img[15:25, 15:25] = 0
        img[35:55, 35:55] = 255
        img[40:50, 40:50] = 0
        
        result = imfill(img)
        
        assert result[20, 20] == 255
        assert result[45, 45] == 255

    def test_imfill_preserves_dtype(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[8:24, 8:24] = 255
        
        result = imfill(img)
        
        assert result.dtype == np.uint8

    def test_imfill_output_shape_matches_input(self):
        for shape in [(16, 16), (32, 64), (100, 50)]:
            img = np.zeros(shape, dtype=np.uint8)
            img[4:-4, 4:-4] = 255
            
            result = imfill(img)
            
            assert result.shape == shape


class TestGetPerilesionMask:

    def test_get_perilesion_mask_basic(self, small_grayscale_mask):
        result = get_perilesion_mask(small_grayscale_mask)
        
        assert result.shape == small_grayscale_mask.shape
        assert result.dtype == np.uint8

    def test_get_perilesion_mask_returns_ring_like_structure(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255
        
        result = get_perilesion_mask(mask, ksize=(10, 10))
        
        assert result[50, 50] == 0
        assert np.any(result > 0)

    def test_get_perilesion_mask_empty_input(self, empty_mask):
        result = get_perilesion_mask(empty_mask)
        
        assert np.all(result == 0)

    def test_get_perilesion_mask_full_input(self, full_mask):
        result = get_perilesion_mask(full_mask, ksize=(5, 5))
        
        assert result.shape == full_mask.shape

    def test_get_perilesion_mask_custom_ksize(self, small_grayscale_mask):
        result_default = get_perilesion_mask(small_grayscale_mask)
        result_custom = get_perilesion_mask(small_grayscale_mask, ksize=(10, 10))
        
        assert result_default.shape == result_custom.shape
        assert not np.array_equal(result_default, result_custom)

    def test_get_perilesion_mask_small_kernel(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:44, 20:44] = 255
        
        result = get_perilesion_mask(mask, ksize=(3, 3))
        
        assert result.shape == mask.shape

    def test_get_perilesion_mask_large_kernel(self):
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[40:88, 40:88] = 255
        
        result = get_perilesion_mask(mask, ksize=(30, 30))
        
        assert result.shape == mask.shape
        assert np.any(result > 0)

    def test_get_perilesion_mask_non_square_kernel(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:44, 20:44] = 255
        
        result = get_perilesion_mask(mask, ksize=(5, 15))
        
        assert result.shape == mask.shape

    def test_get_perilesion_mask_output_range(self, small_grayscale_mask):
        result = get_perilesion_mask(small_grayscale_mask)
        
        assert np.all(result >= 0)
        assert np.all(result <= 255)

    def test_get_perilesion_mask_preserves_edges(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:54, 10:54] = 255
        
        result = get_perilesion_mask(mask, ksize=(10, 10))
        
        assert result[0, 0] == 0
        assert result[63, 63] == 0
