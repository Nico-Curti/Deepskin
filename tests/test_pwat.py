#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from deepskin.pwat import evaluate_PWAT_score


class TestEvaluatePWATScore:

    def test_evaluate_pwat_score_returns_float(self, small_rgb_image, small_semantic_mask):
        result = evaluate_PWAT_score(
            img=small_rgb_image,
            mask=small_semantic_mask,
            verbose=False
        )
        
        assert isinstance(result, (float, np.floating))

    def test_evaluate_pwat_score_basic_call(self, small_rgb_image, small_semantic_mask):
        result = evaluate_PWAT_score(
            img=small_rgb_image,
            mask=small_semantic_mask,
            ksize=(10, 10),
            verbose=False
        )
        
        assert np.isfinite(result)

    def test_evaluate_pwat_score_verbose_mode(self, small_rgb_image, small_semantic_mask, capsys):
        result = evaluate_PWAT_score(
            img=small_rgb_image,
            mask=small_semantic_mask,
            verbose=True
        )
        
        captured = capsys.readouterr()
        assert isinstance(result, (float, np.floating))

    def test_evaluate_pwat_score_custom_ksize(self, small_rgb_image, small_semantic_mask):
        result_default = evaluate_PWAT_score(
            img=small_rgb_image,
            mask=small_semantic_mask,
            ksize=(20, 20),
            verbose=False
        )
        
        result_custom = evaluate_PWAT_score(
            img=small_rgb_image,
            mask=small_semantic_mask,
            ksize=(5, 5),
            verbose=False
        )
        
        assert isinstance(result_default, (float, np.floating))
        assert isinstance(result_custom, (float, np.floating))

    def test_evaluate_pwat_score_deterministic(self, small_rgb_image, small_semantic_mask):
        np.random.seed(42)
        result1 = evaluate_PWAT_score(
            img=small_rgb_image.copy(),
            mask=small_semantic_mask.copy(),
            verbose=False
        )
        
        np.random.seed(42)
        result2 = evaluate_PWAT_score(
            img=small_rgb_image.copy(),
            mask=small_semantic_mask.copy(),
            verbose=False
        )
        
        assert np.isclose(result1, result2)

    def test_evaluate_pwat_score_empty_wound_mask(self, small_rgb_image):
        mask = np.zeros((64, 64, 3), dtype=np.uint8)
        mask[:, :, 0] = 255
        
        result = evaluate_PWAT_score(
            img=small_rgb_image,
            mask=mask,
            verbose=False
        )
        
        assert isinstance(result, (float, np.floating))

    def test_evaluate_pwat_score_full_wound_mask(self, small_rgb_image):
        mask = np.zeros((64, 64, 3), dtype=np.uint8)
        mask[:, :, 2] = 255
        
        result = evaluate_PWAT_score(
            img=small_rgb_image,
            mask=mask,
            verbose=False
        )
        
        assert isinstance(result, (float, np.floating))

    def test_evaluate_pwat_score_uses_constants(self, small_rgb_image, small_semantic_mask):
        from deepskin.constants import Deepskin_PWAT_BIAS
        
        result = evaluate_PWAT_score(
            img=small_rgb_image,
            mask=small_semantic_mask,
            verbose=False
        )
        
        assert isinstance(result, (float, np.floating))

    def test_evaluate_pwat_score_different_image_sizes(self):
        for size in [32, 64, 128]:
            img = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
            mask = np.zeros((size, size, 3), dtype=np.uint8)
            mask[:size//3, :, 0] = 255
            mask[size//3:2*size//3, :, 1] = 255
            mask[2*size//3:, :, 2] = 255
            
            result = evaluate_PWAT_score(
                img=img,
                mask=mask,
                verbose=False
            )
            
            assert isinstance(result, (float, np.floating))

    def test_evaluate_pwat_score_with_body_and_wound(self, small_rgb_image):
        mask = np.zeros((64, 64, 3), dtype=np.uint8)
        mask[10:50, 10:50, 1] = 255
        mask[20:40, 20:40, 2] = 255
        
        result = evaluate_PWAT_score(
            img=small_rgb_image,
            mask=mask,
            verbose=False
        )
        
        assert isinstance(result, (float, np.floating))

    def test_evaluate_pwat_score_integration_with_features(self, small_rgb_image, small_semantic_mask):
        result = evaluate_PWAT_score(
            img=small_rgb_image,
            mask=small_semantic_mask,
            verbose=False
        )
        
        assert np.isfinite(result)

    def test_evaluate_pwat_score_non_square_image(self):
        img = np.random.randint(0, 256, (64, 128, 3), dtype=np.uint8)
        mask = np.zeros((64, 128, 3), dtype=np.uint8)
        mask[:, :42, 0] = 255
        mask[:, 42:85, 1] = 255
        mask[:, 85:, 2] = 255
        
        result = evaluate_PWAT_score(
            img=img,
            mask=mask,
            verbose=False
        )
        
        assert isinstance(result, (float, np.floating))

    def test_evaluate_pwat_score_uses_perilesion_mask(self, small_rgb_image, small_semantic_mask):
        with patch('deepskin.pwat.get_perilesion_mask') as mock_perilesion:
            mock_perilesion.return_value = np.zeros((64, 64), dtype=np.uint8)
            
            result = evaluate_PWAT_score(
                img=small_rgb_image,
                mask=small_semantic_mask,
                verbose=False
            )
            
            mock_perilesion.assert_called_once()
            assert isinstance(result, (float, np.floating))

    def test_evaluate_pwat_score_uses_evaluate_features(self, small_rgb_image, small_semantic_mask):
        with patch('deepskin.pwat.evaluate_features') as mock_features:
            mock_features.return_value = {
                'w_haralick0': 0.1,
                'w_avgR': 0.5,
                'w_park': 0.3,
                'p_haralick0': 0.2,
                'p_avgR': 0.6,
                'p_park': 0.4,
            }
            
            result = evaluate_PWAT_score(
                img=small_rgb_image,
                mask=small_semantic_mask,
                verbose=False
            )
            
            assert mock_features.call_count == 2
