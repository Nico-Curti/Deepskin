#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for segmentation post-processing functions.
Tests use artificially constructed probability maps to verify:
- Class mutual exclusivity
- Correct output shape / dtype
- Empty wound scenario handling
- JSON report field correctness
"""

import unittest
import numpy as np
import json
from deepskin.segmentation import (
    seg_to_labels,
    seg_to_onehot,
    seg_to_rgb_mask,
    generate_analysis_report,
    CLASS_COLORS,
    CLASS_NAMES
)


class TestSegmentationPostProcessing(unittest.TestCase):
    """Test cases for segmentation post-processing functions"""

    def setUp(self):
        """Set up test fixtures"""
        self.h, self.w = 100, 100
        self.num_classes = 3
        
        # Create a simple test probability map
        self.prob_map = np.zeros((self.h, self.w, self.num_classes), dtype=np.float32)
        
        # Create regions: background (top half), body (bottom left), wound (bottom right)
        half_h = self.h // 2
        half_w = self.w // 2
        
        # Background (class 0)
        self.prob_map[:half_h, :, 0] = 0.9
        self.prob_map[:half_h, :, 1] = 0.05
        self.prob_map[:half_h, :, 2] = 0.05
        
        # Body (class 1)
        self.prob_map[half_h:, :half_w, 0] = 0.05
        self.prob_map[half_h:, :half_w, 1] = 0.9
        self.prob_map[half_h:, :half_w, 2] = 0.05
        
        # Wound (class 2)
        self.prob_map[half_h:, half_w:, 0] = 0.05
        self.prob_map[half_h:, half_w:, 1] = 0.05
        self.prob_map[half_h:, half_w:, 2] = 0.9
        
        # Create empty wound scenario (no wound class)
        self.prob_map_no_wound = np.zeros((self.h, self.w, self.num_classes), dtype=np.float32)
        self.prob_map_no_wound[:, :, 0] = 0.5
        self.prob_map_no_wound[:, :, 1] = 0.5
        self.prob_map_no_wound[:, :, 2] = 0.0

    def test_seg_to_labels_shape_and_dtype(self):
        """Test labels output shape and dtype"""
        labels = seg_to_labels(self.prob_map)
        
        # Check shape is 2D
        self.assertEqual(labels.shape, (self.h, self.w))
        
        # Check dtype is uint8
        self.assertEqual(labels.dtype, np.uint8)

    def test_seg_to_labels_class_values(self):
        """Test labels contain valid class values only"""
        labels = seg_to_labels(self.prob_map)
        
        # Check all values are within valid class range
        self.assertTrue(np.all(np.logical_or.reduce([
            labels == 0, labels == 1, labels == 2
        ])))

    def test_seg_to_labels_class_mutual_exclusivity(self):
        """Test that each pixel belongs to exactly one class (mutual exclusivity)"""
        labels = seg_to_labels(self.prob_map)
        
        # Create one-hot encoding from labels
        onehot_from_labels = seg_to_onehot(labels)
        
        # Each pixel should have exactly one channel with 255 (active class)
        pixel_sums = np.sum(onehot_from_labels, axis=-1)
        self.assertTrue(np.all(pixel_sums == 255))
        
        # Alternative: verify argmax produces single label per pixel
        unique_per_pixel = np.max(labels == labels, axis=-1)
        self.assertTrue(np.all(unique_per_pixel))

    def test_seg_to_labels_region_correctness(self):
        """Test correct class assignment for different regions"""
        labels = seg_to_labels(self.prob_map)
        half_h = self.h // 2
        half_w = self.w // 2
        
        # Check background region (class 0)
        self.assertTrue(np.all(labels[:half_h, :] == 0))
        
        # Check body region (class 1)
        self.assertTrue(np.all(labels[half_h:, :half_w] == 1))
        
        # Check wound region (class 2)
        self.assertTrue(np.all(labels[half_h:, half_w:] == 2))

    def test_seg_to_onehot_shape_and_dtype(self):
        """Test onehot output shape and dtype"""
        labels = seg_to_labels(self.prob_map)
        onehot = seg_to_onehot(labels, self.num_classes)
        
        # Check shape is 3D (H, W, num_classes)
        self.assertEqual(onehot.shape, (self.h, self.w, self.num_classes))
        
        # Check dtype is uint8
        self.assertEqual(onehot.dtype, np.uint8)

    def test_seg_to_onehot_binary_values(self):
        """Test onehot output contains only 0 and 255"""
        labels = seg_to_labels(self.prob_map)
        onehot = seg_to_onehot(labels)
        
        # Check all values are either 0 or 255
        self.assertTrue(np.all(np.logical_or(onehot == 0, onehot == 255)))

    def test_seg_to_onehot_backward_compatibility(self):
        """Test that onehot format is compatible with existing PWAT calculation"""
        labels = seg_to_labels(self.prob_map)
        onehot = seg_to_onehot(labels)
        
        # Check it can be split into 3 channels like original mask
        bg_mask, body_mask, wound_mask = np.split(onehot, 3, axis=-1)
        
        # Each mask should be 2D (after squeeze)
        self.assertEqual(bg_mask.squeeze().shape, (self.h, self.w))

    def test_seg_to_rgb_mask_shape_and_dtype(self):
        """Test RGB mask output shape and dtype"""
        labels = seg_to_labels(self.prob_map)
        rgb_mask = seg_to_rgb_mask(labels)
        
        # Check shape is 3D (H, W, 3)
        self.assertEqual(rgb_mask.shape, (self.h, self.w, 3))
        
        # Check dtype is uint8
        self.assertEqual(rgb_mask.dtype, np.uint8)

    def test_seg_to_rgb_mask_color_correctness(self):
        """Test RGB mask colors match class definitions"""
        labels = seg_to_labels(self.prob_map)
        rgb_mask = seg_to_rgb_mask(labels)
        half_h = self.h // 2
        half_w = self.w // 2
        
        # Check background color (class 0: blue)
        bg_color = rgb_mask[0, 0]
        self.assertTrue(np.all(bg_color == CLASS_COLORS[0]))
        
        # Check body color (class 1: green)
        body_color = rgb_mask[half_h + 1, 0]
        self.assertTrue(np.all(body_color == CLASS_COLORS[1]))
        
        # Check wound color (class 2: red)
        wound_color = rgb_mask[half_h + 1, half_w + 1]
        self.assertTrue(np.all(wound_color == CLASS_COLORS[2]))

    def test_generate_analysis_report_structure(self):
        """Test JSON report structure contains all required fields"""
        labels = seg_to_labels(self.prob_map)
        report = generate_analysis_report(self.prob_map, labels)
        
        # Check required top-level keys exist
        required_keys = ['image_size', 'total_pixels', 'classes', 'has_wound', 'wound_bbox']
        for key in required_keys:
            self.assertIn(key, report)

    def test_generate_analysis_report_pixel_counts(self):
        """Test pixel count calculations"""
        labels = seg_to_labels(self.prob_map)
        report = generate_analysis_report(self.prob_map, labels)
        
        # Total pixels
        self.assertEqual(report['total_pixels'], self.h * self.w)
        
        # Check each class has pixel_count and area_percentage
        for class_name in CLASS_NAMES:
            self.assertIn('pixel_count', report['classes'][class_name])
            self.assertIn('area_percentage', report['classes'][class_name])
            
            # Area percentage should be between 0 and 100
            area_pct = report['classes'][class_name]['area_percentage']
            self.assertTrue(0 <= area_pct <= 100)

    def test_generate_analysis_report_wound_detection(self):
        """Test wound detection (has_wound flag)"""
        # Test with wound
        labels_with_wound = seg_to_labels(self.prob_map)
        report_with_wound = generate_analysis_report(self.prob_map, labels_with_wound)
        self.assertTrue(report_with_wound['has_wound'])
        self.assertIsNotNone(report_with_wound['wound_bbox'])
        
        # Test without wound
        labels_no_wound = seg_to_labels(self.prob_map_no_wound)
        report_no_wound = generate_analysis_report(self.prob_map_no_wound, labels_no_wound)
        self.assertFalse(report_no_wound['has_wound'])
        self.assertIsNone(report_no_wound['wound_bbox'])

    def test_generate_analysis_report_bbox_correctness(self):
        """Test wound bbox calculations"""
        labels = seg_to_labels(self.prob_map)
        report = generate_analysis_report(self.prob_map, labels)
        
        half_h = self.h // 2
        half_w = self.w // 2
        
        # Bbox should cover bottom right quadrant
        bbox = report['wound_bbox']
        self.assertEqual(bbox['x_min'], half_w)
        self.assertEqual(bbox['y_min'], half_h)
        self.assertEqual(bbox['x_max'], self.w - 1)
        self.assertEqual(bbox['y_max'], self.h - 1)
        self.assertEqual(bbox['width'], half_w)
        self.assertEqual(bbox['height'], half_h)

    def test_generate_analysis_report_mean_confidence(self):
        """Test mean confidence calculation"""
        labels = seg_to_labels(self.prob_map)
        report = generate_analysis_report(self.prob_map, labels)
        
        # Check each class has mean_confidence (approximately 0.9 in our test case)
        for class_name in CLASS_NAMES:
            if report['classes'][class_name]['pixel_count'] > 0:
                self.assertIn('mean_confidence', report['classes'][class_name])
                # Should be approximately 0.9
                conf = report['classes'][class_name]['mean_confidence']
                self.assertAlmostEqual(conf, 0.9, delta=0.1)

    def test_empty_wound_scenario(self):
        """Test handling of images with no wound"""
        labels = seg_to_labels(self.prob_map_no_wound)
        report = generate_analysis_report(self.prob_map_no_wound, labels)
        
        # No wound detected
        self.assertFalse(report['has_wound'])
        self.assertIsNone(report['wound_bbox'])
        
        # Wound class should have 0 pixels
        self.assertEqual(report['classes']['wound']['pixel_count'], 0)
        self.assertEqual(report['classes']['wound']['area_percentage'], 0.0)

    def test_report_json_serializable(self):
        """Test that report can be serialized to JSON"""
        labels = seg_to_labels(self.prob_map)
        report = generate_analysis_report(self.prob_map, labels)
        
        # Should not raise any JSON serialization errors
        try:
            json_str = json.dumps(report, ensure_ascii=False, indent=2)
            self.assertIsInstance(json_str, str)
        except Exception as e:
            self.fail(f"JSON serialization failed: {e}")

    def test_all_formats_consistent(self):
        """Test that all output formats are consistent with each other"""
        labels = seg_to_labels(self.prob_map)
        onehot = seg_to_onehot(labels)
        rgb_mask = seg_to_rgb_mask(labels)
        
        # Reconstruct labels from onehot and verify consistency
        labels_from_onehot = np.argmax(onehot, axis=-1).astype(np.uint8)
        self.assertTrue(np.all(labels == labels_from_onehot))
        
        # Verify RGB colors correspond to class labels
        for i in range(self.num_classes):
            class_pixels_rgb = rgb_mask[labels == i]
            expected_color = CLASS_COLORS[i]
            self.assertTrue(np.all(class_pixels_rgb == expected_color))


if __name__ == '__main__':
    unittest.main(verbosity=2)
