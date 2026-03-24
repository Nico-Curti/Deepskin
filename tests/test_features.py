#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
from deepskin.features import (
    get_haralick,
    get_rgb_channel_stats,
    get_hsv_channel_stats,
    get_lab_channel_stats,
    get_park_redness,
    get_amparo_redness,
    evaluate_features,
)


class TestGetHaralick:

    def test_get_haralick_returns_correct_shape(self, small_rgb_image, small_grayscale_mask):
        import cv2
        masked = cv2.bitwise_and(small_rgb_image, small_rgb_image, mask=small_grayscale_mask)
        
        result = get_haralick(masked)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (13,)

    def test_get_haralick_returns_numeric(self, small_rgb_image, small_grayscale_mask):
        import cv2
        masked = cv2.bitwise_and(small_rgb_image, small_rgb_image, mask=small_grayscale_mask)
        
        result = get_haralick(masked)
        
        assert np.issubdtype(result.dtype, np.floating)

    def test_get_haralick_insufficient_points_returns_zeros(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[5:7, 5:7] = 128
        
        result = get_haralick(img)
        
        assert np.all(result == 0)
        assert result.shape == (13,)

    def test_get_haralick_empty_masked_image(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        
        result = get_haralick(img)
        
        assert np.all(result == 0)

    def test_get_haralick_uniform_image(self):
        img = np.ones((32, 32, 3), dtype=np.uint8) * 128
        
        result = get_haralick(img)
        
        assert result.shape == (13,)


class TestGetRgbChannelStats:

    def test_get_rgb_channel_stats_returns_correct_types(self, small_rgb_image, small_grayscale_mask):
        avg, std = get_rgb_channel_stats(small_rgb_image, small_grayscale_mask)
        
        assert isinstance(avg, np.ndarray)
        assert isinstance(std, np.ndarray)

    def test_get_rgb_channel_stats_returns_correct_shape(self, small_rgb_image, small_grayscale_mask):
        avg, std = get_rgb_channel_stats(small_rgb_image, small_grayscale_mask)
        
        assert avg.shape == (3, 1)
        assert std.shape == (3, 1)

    def test_get_rgb_channel_stats_values_in_range(self, small_rgb_image, small_grayscale_mask):
        avg, std = get_rgb_channel_stats(small_rgb_image, small_grayscale_mask)
        
        assert np.all(avg >= 0) and np.all(avg <= 1)
        assert np.all(std >= 0) and np.all(std <= 1)

    def test_get_rgb_channel_stats_empty_mask(self, small_rgb_image, empty_mask):
        avg, std = get_rgb_channel_stats(small_rgb_image, empty_mask)
        
        assert avg.shape == (3, 1)
        assert std.shape == (3, 1)

    def test_get_rgb_channel_stats_full_mask(self, small_rgb_image, full_mask):
        avg, std = get_rgb_channel_stats(small_rgb_image, full_mask)
        
        assert np.all(avg > 0)


class TestGetHsvChannelStats:

    def test_get_hsv_channel_stats_returns_correct_types(self, small_rgb_image, small_grayscale_mask):
        avg, std = get_hsv_channel_stats(small_rgb_image, small_grayscale_mask)
        
        assert isinstance(avg, np.ndarray)
        assert isinstance(std, np.ndarray)

    def test_get_hsv_channel_stats_returns_correct_shape(self, small_rgb_image, small_grayscale_mask):
        avg, std = get_hsv_channel_stats(small_rgb_image, small_grayscale_mask)
        
        assert avg.shape == (3, 1)
        assert std.shape == (3, 1)

    def test_get_hsv_channel_stats_values_in_range(self, small_rgb_image, small_grayscale_mask):
        avg, std = get_hsv_channel_stats(small_rgb_image, small_grayscale_mask)
        
        assert np.all(avg >= 0) and np.all(avg <= 1)
        assert np.all(std >= 0) and np.all(std <= 1)

    def test_get_hsv_channel_stats_empty_mask(self, small_rgb_image, empty_mask):
        avg, std = get_hsv_channel_stats(small_rgb_image, empty_mask)
        
        assert avg.shape == (3, 1)


class TestGetLabChannelStats:

    def test_get_lab_channel_stats_returns_correct_types(self, small_rgb_image, small_grayscale_mask):
        avg, std = get_lab_channel_stats(small_rgb_image, small_grayscale_mask)
        
        assert isinstance(avg, np.ndarray)
        assert isinstance(std, np.ndarray)

    def test_get_lab_channel_stats_returns_correct_shape(self, small_rgb_image, small_grayscale_mask):
        avg, std = get_lab_channel_stats(small_rgb_image, small_grayscale_mask)
        
        assert avg.shape == (3, 1)
        assert std.shape == (3, 1)

    def test_get_lab_channel_stats_values_in_range(self, small_rgb_image, small_grayscale_mask):
        avg, std = get_lab_channel_stats(small_rgb_image, small_grayscale_mask)
        
        assert np.all(avg >= 0) and np.all(avg <= 1)
        assert np.all(std >= 0) and np.all(std <= 1)


class TestGetParkRedness:

    def test_get_park_redness_returns_float(self, small_rgb_image, small_grayscale_mask):
        result = get_park_redness(small_rgb_image, small_grayscale_mask)
        
        assert isinstance(result, (float, np.floating))

    def test_get_park_redness_value_in_valid_range(self, small_rgb_image, small_grayscale_mask):
        result = get_park_redness(small_rgb_image, small_grayscale_mask)
        
        assert -0.5 <= result <= 1.0

    def test_get_park_redness_empty_mask_returns_default(self, small_rgb_image, empty_mask):
        result = get_park_redness(small_rgb_image, empty_mask)
        
        assert result == -0.5

    def test_get_park_redness_red_dominant_image(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[:, :, 0] = 255
        img[:, :, 1] = 0
        img[:, :, 2] = 0
        mask = np.ones((32, 32), dtype=np.uint8) * 255
        
        result = get_park_redness(img, mask)
        
        assert result > 0

    def test_get_park_redness_blue_dominant_image(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[:, :, 0] = 0
        img[:, :, 1] = 0
        img[:, :, 2] = 255
        mask = np.ones((32, 32), dtype=np.uint8) * 255
        
        result = get_park_redness(img, mask)
        
        assert result < 0


class TestGetAmparoRedness:

    def test_get_amparo_redness_returns_float(self, small_rgb_image, small_grayscale_mask):
        result = get_amparo_redness(small_rgb_image, small_grayscale_mask)
        
        assert isinstance(result, (float, np.floating))

    def test_get_amparo_redness_value_in_valid_range(self, small_rgb_image, small_grayscale_mask):
        result = get_amparo_redness(small_rgb_image, small_grayscale_mask)
        
        assert 0.0 <= result <= 1.0

    def test_get_amparo_redness_empty_mask_returns_zero(self, small_rgb_image, empty_mask):
        result = get_amparo_redness(small_rgb_image, empty_mask)
        
        assert result == 0.0

    def test_get_amparo_redness_uniform_image(self):
        img = np.ones((32, 32, 3), dtype=np.uint8) * 128
        mask = np.ones((32, 32), dtype=np.uint8) * 255
        
        result = get_amparo_redness(img, mask)
        
        assert 0.0 <= result <= 1.0


class TestEvaluateFeatures:

    def test_evaluate_features_returns_dict(self, small_rgb_image, small_grayscale_mask):
        result = evaluate_features(small_rgb_image, small_grayscale_mask, prefix='test_')
        
        assert isinstance(result, dict)

    def test_evaluate_features_contains_expected_keys(self, small_rgb_image, small_grayscale_mask):
        result = evaluate_features(small_rgb_image, small_grayscale_mask, prefix='w_')
        
        expected_prefixes = ['w_haralick', 'w_avgR', 'w_avgG', 'w_avgB',
                           'w_stdR', 'w_stdG', 'w_stdB',
                           'w_avgH', 'w_avgS', 'w_avgV',
                           'w_stdH', 'w_stdS', 'w_stdV',
                           'w_avgL', 'w_avga', 'w_avgb',
                           'w_stdL', 'w_stda', 'w_stdb',
                           'w_park', 'w_amparo']
        
        for prefix in expected_prefixes:
            assert any(k.startswith(prefix) for k in result.keys()), f"Missing key starting with {prefix}"

    def test_evaluate_features_prefix_applied(self, small_rgb_image, small_grayscale_mask):
        result = evaluate_features(small_rgb_image, small_grayscale_mask, prefix='custom_')
        
        for key in result.keys():
            assert key.startswith('custom_')

    def test_evaluate_features_haralick_count(self, small_rgb_image, small_grayscale_mask):
        result = evaluate_features(small_rgb_image, small_grayscale_mask, prefix='w_')
        
        haralick_keys = [k for k in result.keys() if 'haralick' in k]
        assert len(haralick_keys) == 13

    def test_evaluate_features_all_values_are_numeric(self, small_rgb_image, small_grayscale_mask):
        result = evaluate_features(small_rgb_image, small_grayscale_mask, prefix='w_')
        
        for key, value in result.items():
            assert isinstance(value, (int, float, np.floating, np.integer)), f"Key {key} has non-numeric value: {type(value)}"

    def test_evaluate_features_empty_mask(self, small_rgb_image, empty_mask):
        result = evaluate_features(small_rgb_image, empty_mask, prefix='w_')
        
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_evaluate_features_different_prefixes(self, small_rgb_image, small_grayscale_mask):
        result_w = evaluate_features(small_rgb_image, small_grayscale_mask, prefix='w_')
        result_p = evaluate_features(small_rgb_image, small_grayscale_mask, prefix='p_')
        
        w_keys = [k for k in result_w.keys() if k.startswith('w_')]
        p_keys = [k for k in result_p.keys() if k.startswith('p_')]
        
        assert len(w_keys) == len(p_keys)

    def test_evaluate_features_full_mask(self, small_rgb_image, full_mask):
        result = evaluate_features(small_rgb_image, full_mask, prefix='test_')
        
        assert len(result) > 0
        assert all(isinstance(v, (int, float, np.floating, np.integer)) for v in result.values())

    def test_evaluate_features_deterministic(self, small_rgb_image, small_grayscale_mask):
        result1 = evaluate_features(small_rgb_image, small_grayscale_mask, prefix='w_')
        result2 = evaluate_features(small_rgb_image, small_grayscale_mask, prefix='w_')
        
        for key in result1.keys():
            assert np.isclose(result1[key], result2[key]), f"Non-deterministic result for key {key}"
