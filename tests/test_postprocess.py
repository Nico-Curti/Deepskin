#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for postprocess module.

这些测试使用人工构造的概率图，不依赖真实模型权重。
"""

import unittest
import numpy as np
import json
import tempfile
import os

from deepskin.postprocess import (
    softmax_to_label_map,
    label_map_to_onehot,
    label_map_to_rgb,
    label_map_to_mask,
    SegmentationReport,
    convert_mask_format,
    CLASS_NAMES,
    CLASS_INDICES,
    CLASS_COLORS,
)


class TestSoftmaxToLabelMap(unittest.TestCase):
    """测试softmax转label map功能"""

    def test_basic_conversion(self):
        """测试基本的softmax到label map转换"""
        # 构造一个3x3x3的softmax输出
        # 每个像素只有一个类别概率为1，其他为0
        softmax = np.zeros((3, 3, 3), dtype=np.float32)
        softmax[0, 0, 0] = 1.0  # background
        softmax[0, 1, 1] = 1.0  # body
        softmax[0, 2, 2] = 1.0  # wound
        softmax[1, 0, 0] = 1.0
        softmax[1, 1, 1] = 1.0
        softmax[1, 2, 2] = 1.0
        softmax[2, 0, 0] = 1.0
        softmax[2, 1, 1] = 1.0
        softmax[2, 2, 2] = 1.0

        label_map = softmax_to_label_map(softmax)

        # 验证输出shape
        self.assertEqual(label_map.shape, (3, 3))
        self.assertEqual(label_map.dtype, np.uint8)

        # 验证每个像素的类别
        self.assertEqual(label_map[0, 0], 0)  # background
        self.assertEqual(label_map[0, 1], 1)  # body
        self.assertEqual(label_map[0, 2], 2)  # wound

    def test_class_mutual_exclusivity(self):
        """测试类别互斥性：每个像素只能属于一个类别"""
        # 构造softmax输出，每个像素概率最高的类别应该被选中
        softmax = np.array([
            [[0.7, 0.2, 0.1],  # 应该是background
             [0.1, 0.8, 0.1]], # 应该是body
            [[0.2, 0.3, 0.5],  # 应该是wound
             [0.6, 0.3, 0.1]]  # 应该是background
        ], dtype=np.float32)

        label_map = softmax_to_label_map(softmax)

        # 验证每个像素只有一个类别
        self.assertEqual(label_map[0, 0], 0)
        self.assertEqual(label_map[0, 1], 1)
        self.assertEqual(label_map[1, 0], 2)
        self.assertEqual(label_map[1, 1], 0)

    def test_invalid_input(self):
        """测试无效输入处理"""
        # 2D输入应该报错
        with self.assertRaises(ValueError):
            softmax_to_label_map(np.zeros((3, 3)))

        # 4D输入应该报错
        with self.assertRaises(ValueError):
            softmax_to_label_map(np.zeros((1, 3, 3, 3)))


class TestLabelMapToOnehot(unittest.TestCase):
    """测试label map转onehot功能"""

    def test_basic_conversion(self):
        """测试基本的label map到onehot转换"""
        label_map = np.array([
            [0, 1, 2],
            [2, 0, 1]
        ], dtype=np.uint8)

        onehot = label_map_to_onehot(label_map, num_classes=3)

        # 验证输出shape
        self.assertEqual(onehot.shape, (2, 3, 3))
        self.assertEqual(onehot.dtype, np.uint8)

        # 验证每个通道的值
        # channel 0 (background)
        self.assertEqual(onehot[0, 0, 0], 255)
        self.assertEqual(onehot[1, 1, 0], 255)

        # channel 1 (body)
        self.assertEqual(onehot[0, 1, 1], 255)
        self.assertEqual(onehot[1, 2, 1], 255)

        # channel 2 (wound)
        self.assertEqual(onehot[0, 2, 2], 255)
        self.assertEqual(onehot[1, 0, 2], 255)

    def test_mutual_exclusivity(self):
        """测试onehot编码的类别互斥性"""
        label_map = np.random.randint(0, 3, size=(10, 10))
        onehot = label_map_to_onehot(label_map, num_classes=3)

        # 对于每个像素，所有通道的和应该等于255
        pixel_sums = np.sum(onehot, axis=-1)
        np.testing.assert_array_equal(pixel_sums, np.full((10, 10), 255))

    def test_invalid_input(self):
        """测试无效输入处理"""
        # 3D输入应该报错
        with self.assertRaises(ValueError):
            label_map_to_onehot(np.zeros((3, 3, 3)))


class TestLabelMapToRgb(unittest.TestCase):
    """测试label map转RGB功能"""

    def test_basic_conversion(self):
        """测试基本的label map到RGB转换"""
        label_map = np.array([
            [0, 1, 2],
            [2, 0, 1]
        ], dtype=np.uint8)

        rgb = label_map_to_rgb(label_map)

        # 验证输出shape
        self.assertEqual(rgb.shape, (2, 3, 3))
        self.assertEqual(rgb.dtype, np.uint8)

        # 验证颜色
        # background = 蓝色 (0, 0, 255)
        np.testing.assert_array_equal(rgb[0, 0], [0, 0, 255])
        # body = 绿色 (0, 255, 0)
        np.testing.assert_array_equal(rgb[0, 1], [0, 255, 0])
        # wound = 红色 (255, 0, 0)
        np.testing.assert_array_equal(rgb[0, 2], [255, 0, 0])

    def test_all_colors_present(self):
        """测试所有类别颜色都正确映射"""
        label_map = np.array([
            [0, 1, 2]
        ], dtype=np.uint8)

        rgb = label_map_to_rgb(label_map)

        expected_colors = [
            CLASS_COLORS['background'],  # blue
            CLASS_COLORS['body'],        # green
            CLASS_COLORS['wound'],       # red
        ]

        for i, expected_color in enumerate(expected_colors):
            np.testing.assert_array_equal(rgb[0, i], expected_color)


class TestLabelMapToMask(unittest.TestCase):
    """测试label map转mask功能（与原API兼容）"""

    def test_basic_conversion(self):
        """测试基本的label map到mask转换"""
        label_map = np.array([
            [0, 1, 2],
            [2, 0, 1]
        ], dtype=np.uint8)

        mask = label_map_to_mask(label_map)

        # 验证输出shape与原API一致
        self.assertEqual(mask.shape, (2, 3, 3))
        self.assertEqual(mask.dtype, np.uint8)

        # 验证与原onehot格式一致
        expected_onehot = label_map_to_onehot(label_map, num_classes=3)
        np.testing.assert_array_equal(mask, expected_onehot)


class TestSegmentationReport(unittest.TestCase):
    """测试分割报告功能"""

    def test_pixel_counts(self):
        """测试像素计数"""
        # 构造一个已知像素分布的label map
        # 3x3 = 9 pixels total
        # background (0): positions (0,0), (0,1), (2,0) = 3 pixels
        # body (1): positions (0,2), (1,0) = 2 pixels
        # wound (2): positions (1,1), (1,2), (2,1), (2,2) = 4 pixels
        label_map = np.array([
            [0, 0, 1],
            [1, 2, 2],
            [0, 2, 2]
        ], dtype=np.uint8)

        report = SegmentationReport(label_map)
        data = report.generate()

        # 验证像素计数
        self.assertEqual(data['pixel_counts']['background'], 3)
        self.assertEqual(data['pixel_counts']['body'], 2)
        self.assertEqual(data['pixel_counts']['wound'], 4)

    def test_area_ratios(self):
        """测试面积占比计算"""
        label_map = np.array([
            [0, 1],
            [2, 0]
        ], dtype=np.uint8)

        report = SegmentationReport(label_map)
        data = report.generate()

        # 总面积=4
        self.assertAlmostEqual(data['area_ratios']['background'], 0.5)  # 2/4
        self.assertAlmostEqual(data['area_ratios']['body'], 0.25)       # 1/4
        self.assertAlmostEqual(data['area_ratios']['wound'], 0.25)      # 1/4

    def test_wound_bbox(self):
        """测试wound边界框计算"""
        label_map = np.zeros((10, 10), dtype=np.uint8)
        # 在(3,4)到(6,7)区域设置wound
        label_map[3:7, 4:8] = 2

        report = SegmentationReport(label_map)
        data = report.generate()

        self.assertTrue(data['wound_detected'])
        self.assertEqual(data['wound_bbox']['x_min'], 4)
        self.assertEqual(data['wound_bbox']['y_min'], 3)
        self.assertEqual(data['wound_bbox']['x_max'], 7)
        self.assertEqual(data['wound_bbox']['y_max'], 6)
        self.assertEqual(data['wound_bbox']['width'], 4)
        self.assertEqual(data['wound_bbox']['height'], 4)

    def test_no_wound(self):
        """测试没有wound的情况"""
        label_map = np.array([
            [0, 1],
            [1, 0]
        ], dtype=np.uint8)

        report = SegmentationReport(label_map)
        data = report.generate()

        self.assertFalse(data['wound_detected'])
        self.assertNotIn('wound_bbox', data)

    def test_wound_confidence(self):
        """测试wound置信度计算"""
        label_map = np.array([
            [0, 2],
            [2, 0]
        ], dtype=np.uint8)

        # 构造softmax输出
        softmax = np.array([
            [[0.9, 0.05, 0.05], [0.1, 0.1, 0.8]],
            [[0.2, 0.1, 0.7], [0.85, 0.1, 0.05]]
        ], dtype=np.float32)

        report = SegmentationReport(label_map, softmax)
        data = report.generate()

        # wound像素在(0,1)和(1,0)，对应的softmax值分别为0.8和0.7
        expected_confidence = (0.8 + 0.7) / 2
        self.assertAlmostEqual(data['wound_confidence'], expected_confidence, places=6)

    def test_no_confidence_without_softmax(self):
        """测试没有softmax时不返回置信度"""
        label_map = np.array([
            [0, 2],
            [2, 0]
        ], dtype=np.uint8)

        report = SegmentationReport(label_map)  # 不提供softmax
        data = report.generate()

        self.assertNotIn('wound_confidence', data)

    def test_report_save(self):
        """测试报告保存功能"""
        label_map = np.array([
            [0, 1, 2],
            [2, 0, 1]
        ], dtype=np.uint8)

        report = SegmentationReport(label_map)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            report.save(temp_path)

            # 验证文件内容
            with open(temp_path, 'r') as f:
                loaded_data = json.load(f)

            data = report.generate()
            self.assertEqual(loaded_data['wound_detected'], data['wound_detected'])
            self.assertEqual(loaded_data['pixel_counts'], data['pixel_counts'])
        finally:
            os.unlink(temp_path)

    def test_report_to_json(self):
        """测试报告转JSON字符串"""
        label_map = np.array([
            [0, 1, 2]
        ], dtype=np.uint8)

        report = SegmentationReport(label_map)
        json_str = report.to_json()

        # 验证是有效的JSON
        data = json.loads(json_str)
        self.assertIn('image_shape', data)
        self.assertIn('pixel_counts', data)


class TestConvertMaskFormat(unittest.TestCase):
    """测试mask格式转换功能"""

    def test_labels_format(self):
        """测试labels格式"""
        label_map = np.array([[0, 1, 2]], dtype=np.uint8)
        result = convert_mask_format(label_map, 'labels')
        np.testing.assert_array_equal(result, label_map)

    def test_onehot_format(self):
        """测试onehot格式"""
        label_map = np.array([[0, 1, 2]], dtype=np.uint8)
        result = convert_mask_format(label_map, 'onehot')
        expected = label_map_to_onehot(label_map)
        np.testing.assert_array_equal(result, expected)

    def test_rgb_format(self):
        """测试rgb格式"""
        label_map = np.array([[0, 1, 2]], dtype=np.uint8)
        result = convert_mask_format(label_map, 'rgb')
        expected = label_map_to_rgb(label_map)
        np.testing.assert_array_equal(result, expected)

    def test_mask_format(self):
        """测试mask格式（与原API兼容）"""
        label_map = np.array([[0, 1, 2]], dtype=np.uint8)
        result = convert_mask_format(label_map, 'mask')
        expected = label_map_to_mask(label_map)
        np.testing.assert_array_equal(result, expected)

    def test_invalid_format(self):
        """测试无效格式"""
        with self.assertRaises(ValueError):
            convert_mask_format(np.array([[0]]), 'invalid')


class TestEmptyWoundScenario(unittest.TestCase):
    """测试空wound场景"""

    def test_no_wound_detection(self):
        """测试没有wound时的检测"""
        # 只有background和body，没有wound
        label_map = np.array([
            [0, 0, 1],
            [1, 0, 0]
        ], dtype=np.uint8)

        report = SegmentationReport(label_map)
        data = report.generate()

        self.assertFalse(data['wound_detected'])
        self.assertEqual(data['pixel_counts']['wound'], 0)
        self.assertEqual(data['area_ratios']['wound'], 0.0)
        self.assertNotIn('wound_bbox', data)

    def test_no_wound_confidence(self):
        """测试没有wound时的置信度"""
        label_map = np.array([
            [0, 0],
            [1, 1]
        ], dtype=np.uint8)

        softmax = np.random.rand(2, 2, 3).astype(np.float32)
        softmax = softmax / softmax.sum(axis=-1, keepdims=True)  # 归一化

        report = SegmentationReport(label_map, softmax)
        data = report.generate()

        self.assertNotIn('wound_confidence', data)


class TestOutputShapes(unittest.TestCase):
    """测试输出shape正确性"""

    def test_various_input_shapes(self):
        """测试各种输入shape"""
        shapes = [(32, 32), (64, 128), (256, 256), (480, 640)]

        for h, w in shapes:
            # 构造softmax输出
            softmax = np.random.rand(h, w, 3).astype(np.float32)
            softmax = softmax / softmax.sum(axis=-1, keepdims=True)

            label_map = softmax_to_label_map(softmax)
            self.assertEqual(label_map.shape, (h, w))

            onehot = label_map_to_onehot(label_map)
            self.assertEqual(onehot.shape, (h, w, 3))

            rgb = label_map_to_rgb(label_map)
            self.assertEqual(rgb.shape, (h, w, 3))


class TestDtypes(unittest.TestCase):
    """测试输出dtype正确性"""

    def test_label_map_dtype(self):
        """测试label map的dtype"""
        softmax = np.random.rand(10, 10, 3).astype(np.float32)
        label_map = softmax_to_label_map(softmax)
        self.assertEqual(label_map.dtype, np.uint8)

    def test_onehot_dtype(self):
        """测试onehot的dtype"""
        label_map = np.random.randint(0, 3, size=(10, 10)).astype(np.uint8)
        onehot = label_map_to_onehot(label_map)
        self.assertEqual(onehot.dtype, np.uint8)

    def test_rgb_dtype(self):
        """测试RGB的dtype"""
        label_map = np.random.randint(0, 3, size=(10, 10)).astype(np.uint8)
        rgb = label_map_to_rgb(label_map)
        self.assertEqual(rgb.dtype, np.uint8)


if __name__ == '__main__':
    unittest.main()
