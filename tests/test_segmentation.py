#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import os
from unittest.mock import patch, MagicMock


class TestWoundSegmentation:

    def test_wound_segmentation_basic_call(self, small_rgb_image, mock_tensorflow_model):
        with patch('deepskin.segmentation.deepskin_model') as mock_model_fn:
            with patch('deepskin.segmentation.download_model_weights'):
                with patch('os.path.exists', return_value=True):
                    mock_model_fn.return_value = mock_tensorflow_model
                    
                    from deepskin.segmentation import wound_segmentation
                    
                    result = wound_segmentation(
                        img=small_rgb_image,
                        tol=0.5,
                        verbose=False
                    )
                    
                    assert isinstance(result, np.ndarray)
                    assert result.shape[:2] == small_rgb_image.shape[:2]
                    assert result.shape[2] == 3

    def test_wound_segmentation_output_dtype(self, small_rgb_image, mock_tensorflow_model):
        with patch('deepskin.segmentation.deepskin_model') as mock_model_fn:
            with patch('deepskin.segmentation.download_model_weights'):
                with patch('os.path.exists', return_value=True):
                    mock_model_fn.return_value = mock_tensorflow_model
                    
                    from deepskin.segmentation import wound_segmentation
                    
                    result = wound_segmentation(
                        img=small_rgb_image,
                        verbose=False
                    )
                    
                    assert result.dtype == np.uint8

    def test_wound_segmentation_output_values_binary(self, small_rgb_image, mock_tensorflow_model):
        with patch('deepskin.segmentation.deepskin_model') as mock_model_fn:
            with patch('deepskin.segmentation.download_model_weights'):
                with patch('os.path.exists', return_value=True):
                    mock_model_fn.return_value = mock_tensorflow_model
                    
                    from deepskin.segmentation import wound_segmentation
                    
                    result = wound_segmentation(
                        img=small_rgb_image,
                        tol=0.5,
                        verbose=False
                    )
                    
                    unique_values = np.unique(result)
                    assert all(v in [0, 255] for v in unique_values)

    def test_wound_segmentation_custom_tolerance(self, small_rgb_image, mock_tensorflow_model):
        with patch('deepskin.segmentation.deepskin_model') as mock_model_fn:
            with patch('deepskin.segmentation.download_model_weights'):
                with patch('os.path.exists', return_value=True):
                    mock_model_fn.return_value = mock_tensorflow_model
                    
                    from deepskin.segmentation import wound_segmentation
                    
                    result_low = wound_segmentation(
                        img=small_rgb_image,
                        tol=0.1,
                        verbose=False
                    )
                    
                    result_high = wound_segmentation(
                        img=small_rgb_image,
                        tol=0.9,
                        verbose=False
                    )
                    
                    assert result_low.shape == result_high.shape

    def test_wound_segmentation_verbose_mode(self, small_rgb_image, mock_tensorflow_model, capsys):
        with patch('deepskin.segmentation.deepskin_model') as mock_model_fn:
            with patch('deepskin.segmentation.download_model_weights'):
                with patch('os.path.exists', return_value=True):
                    mock_model_fn.return_value = mock_tensorflow_model
                    
                    from deepskin.segmentation import wound_segmentation
                    
                    result = wound_segmentation(
                        img=small_rgb_image,
                        verbose=True
                    )
                    
                    captured = capsys.readouterr()
                    assert isinstance(result, np.ndarray)

    def test_wound_segmentation_downloads_weights_if_missing(self, small_rgb_image, mock_tensorflow_model):
        with patch('deepskin.segmentation.deepskin_model') as mock_model_fn:
            with patch('deepskin.segmentation.download_model_weights') as mock_download:
                with patch('os.path.exists', return_value=False):
                    mock_model_fn.return_value = mock_tensorflow_model
                    
                    from deepskin.segmentation import wound_segmentation
                    
                    result = wound_segmentation(
                        img=small_rgb_image,
                        verbose=False
                    )
                    
                    mock_download.assert_called_once()

    def test_wound_segmentation_different_image_sizes(self, mock_tensorflow_model):
        with patch('deepskin.segmentation.deepskin_model') as mock_model_fn:
            with patch('deepskin.segmentation.download_model_weights'):
                with patch('os.path.exists', return_value=True):
                    mock_model_fn.return_value = mock_tensorflow_model
                    
                    from deepskin.segmentation import wound_segmentation
                    
                    for h, w in [(32, 32), (64, 128), (100, 50)]:
                        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
                        
                        result = wound_segmentation(
                            img=img,
                            verbose=False
                        )
                        
                        assert result.shape[:2] == (h, w)

    def test_wound_segmentation_uses_model_predict(self, small_rgb_image, mock_tensorflow_model):
        with patch('deepskin.segmentation.deepskin_model') as mock_model_fn:
            with patch('deepskin.segmentation.download_model_weights'):
                with patch('os.path.exists', return_value=True):
                    mock_model_fn.return_value = mock_tensorflow_model
                    mock_tensorflow_model.predict = MagicMock(return_value=np.random.rand(1, 256, 256, 3).astype(np.float32))
                    
                    from deepskin.segmentation import wound_segmentation
                    
                    result = wound_segmentation(
                        img=small_rgb_image,
                        verbose=False
                    )
                    
                    mock_tensorflow_model.predict.assert_called_once()

    def test_wound_segmentation_three_channel_output(self, small_rgb_image, mock_tensorflow_model):
        with patch('deepskin.segmentation.deepskin_model') as mock_model_fn:
            with patch('deepskin.segmentation.download_model_weights'):
                with patch('os.path.exists', return_value=True):
                    mock_model_fn.return_value = mock_tensorflow_model
                    
                    from deepskin.segmentation import wound_segmentation
                    
                    result = wound_segmentation(
                        img=small_rgb_image,
                        verbose=False
                    )
                    
                    assert result.ndim == 3
                    assert result.shape[2] == 3

    def test_wound_segmentation_preserves_spatial_dimensions(self, mock_tensorflow_model):
        with patch('deepskin.segmentation.deepskin_model') as mock_model_fn:
            with patch('deepskin.segmentation.download_model_weights'):
                with patch('os.path.exists', return_value=True):
                    mock_model_fn.return_value = mock_tensorflow_model
                    
                    from deepskin.segmentation import wound_segmentation
                    
                    original_sizes = [(50, 50), (100, 200), (128, 64)]
                    
                    for h, w in original_sizes:
                        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
                        
                        result = wound_segmentation(
                            img=img,
                            verbose=False
                        )
                        
                        assert result.shape[0] == h
                        assert result.shape[1] == w

    def test_wound_segmentation_model_weights_path(self, small_rgb_image, mock_tensorflow_model):
        with patch('deepskin.segmentation.deepskin_model') as mock_model_fn:
            with patch('deepskin.segmentation.download_model_weights'):
                with patch('os.path.exists', return_value=True) as mock_exists:
                    mock_model_fn.return_value = mock_tensorflow_model
                    
                    from deepskin.segmentation import wound_segmentation
                    
                    result = wound_segmentation(
                        img=small_rgb_image,
                        verbose=False
                    )
                    
                    assert mock_exists.called

    def test_wound_segmentation_resizes_input_for_model(self, small_rgb_image, mock_tensorflow_model):
        with patch('deepskin.segmentation.deepskin_model') as mock_model_fn:
            with patch('deepskin.segmentation.download_model_weights'):
                with patch('os.path.exists', return_value=True):
                    mock_model_fn.return_value = mock_tensorflow_model
                    
                    predict_called_with = None
                    original_predict = mock_tensorflow_model.predict
                    
                    def capture_predict(x, verbose=0):
                        nonlocal predict_called_with
                        predict_called_with = x.shape
                        return original_predict(x, verbose)
                    
                    mock_tensorflow_model.predict = capture_predict
                    
                    from deepskin.segmentation import wound_segmentation
                    
                    result = wound_segmentation(
                        img=small_rgb_image,
                        verbose=False
                    )
                    
                    assert predict_called_with is not None
                    assert predict_called_with[1] == 256
                    assert predict_called_with[2] == 256
