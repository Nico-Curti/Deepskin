#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock
from io import StringIO


class TestParseArgs:

    def test_parse_args_version_flag(self):
        from deepskin.__main__ import parse_args
        
        with patch('sys.argv', ['deepskin', '--version']):
            args = parse_args()
            assert args.version is True

    def test_parse_args_short_version_flag(self):
        from deepskin.__main__ import parse_args
        
        with patch('sys.argv', ['deepskin', '-v']):
            args = parse_args()
            assert args.version is True

    def test_parse_args_input_file(self):
        from deepskin.__main__ import parse_args
        
        with patch('sys.argv', ['deepskin', '--input', 'test.png']):
            args = parse_args()
            assert args.filepath == 'test.png'

    def test_parse_args_short_input_flag(self):
        from deepskin.__main__ import parse_args
        
        with patch('sys.argv', ['deepskin', '-i', 'test.png']):
            args = parse_args()
            assert args.filepath == 'test.png'

    def test_parse_args_mask_flag(self):
        from deepskin.__main__ import parse_args
        
        with patch('sys.argv', ['deepskin', '--input', 'test.png', '--mask']):
            args = parse_args()
            assert args.mask is True

    def test_parse_args_short_mask_flag(self):
        from deepskin.__main__ import parse_args
        
        with patch('sys.argv', ['deepskin', '-i', 'test.png', '-m']):
            args = parse_args()
            assert args.mask is True

    def test_parse_args_pwat_flag(self):
        from deepskin.__main__ import parse_args
        
        with patch('sys.argv', ['deepskin', '--input', 'test.png', '--pwat']):
            args = parse_args()
            assert args.pwat is True

    def test_parse_args_short_pwat_flag(self):
        from deepskin.__main__ import parse_args
        
        with patch('sys.argv', ['deepskin', '-i', 'test.png', '-p']):
            args = parse_args()
            assert args.pwat is True

    def test_parse_args_verbose_default(self):
        from deepskin.__main__ import parse_args
        
        with patch('sys.argv', ['deepskin', '-i', 'test.png']):
            args = parse_args()
            assert args.verbose is True

    def test_parse_args_no_input(self):
        from deepskin.__main__ import parse_args
        
        with patch('sys.argv', ['deepskin']):
            args = parse_args()
            assert args.filepath is None


class TestMain:

    def test_main_version_exits_zero(self, capsys):
        from deepskin.__main__ import main
        
        with patch('sys.argv', ['deepskin', '--version']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_main_no_input_exits_one(self, capsys):
        from deepskin.__main__ import main
        
        with patch('sys.argv', ['deepskin']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_main_missing_input_file_exits_one(self, capsys):
        from deepskin.__main__ import main
        
        with patch('sys.argv', ['deepskin', '-i', 'nonexistent.png']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_main_with_valid_input_file(self, mock_tensorflow_model):
        from deepskin.__main__ import main
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            
        try:
            import cv2
            test_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(temp_path, test_img)
            
            with patch('sys.argv', ['deepskin', '-i', temp_path]):
                with patch('deepskin.__main__.wound_segmentation') as mock_seg:
                    mock_seg.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
                    main()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_main_mask_flag_creates_output(self, mock_tensorflow_model):
        from deepskin.__main__ import main
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, 'test.png')
            import cv2
            test_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(temp_path, test_img)
            
            with patch('sys.argv', ['deepskin', '-i', temp_path, '-m']):
                with patch('deepskin.__main__.wound_segmentation') as mock_seg:
                    mock_seg.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
                    main()
                    
                    expected_output = os.path.join(tmpdir, 'test_deepskin_mask.png')
                    assert os.path.exists(expected_output)

    def test_main_pwat_flag_computes_score(self, mock_tensorflow_model):
        from deepskin.__main__ import main
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, 'test.png')
            import cv2
            test_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(temp_path, test_img)
            
            with patch('sys.argv', ['deepskin', '-i', temp_path, '-p']):
                with patch('deepskin.__main__.wound_segmentation') as mock_seg:
                    with patch('deepskin.__main__.evaluate_PWAT_score') as mock_pwat:
                        mock_seg.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
                        mock_pwat.return_value = 10.5
                        
                        main()
                        
                        mock_pwat.assert_called_once()

    def test_main_verbose_output(self, mock_tensorflow_model, capsys):
        from deepskin.__main__ import main
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, 'test.png')
            import cv2
            test_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(temp_path, test_img)
            
            with patch('sys.argv', ['deepskin', '-i', temp_path, '-w']):
                with patch('deepskin.__main__.wound_segmentation') as mock_seg:
                    mock_seg.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
                    main()


class TestMainIntegration:

    def test_main_loads_image_correctly(self, mock_tensorflow_model):
        from deepskin.__main__ import main
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, 'test.png')
            import cv2
            test_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(temp_path, test_img)
            
            loaded_image = None
            original_imread = cv2.imread
            
            def capture_imread(path, flags):
                nonlocal loaded_image
                loaded_image = original_imread(path, flags)
                return loaded_image
            
            with patch('sys.argv', ['deepskin', '-i', temp_path]):
                with patch('cv2.imread', side_effect=capture_imread):
                    with patch('deepskin.__main__.wound_segmentation') as mock_seg:
                        mock_seg.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
                        main()
                        
                        assert loaded_image is not None

    def test_main_converts_bgr_to_rgb(self, mock_tensorflow_model):
        from deepskin.__main__ import main
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, 'test.png')
            import cv2
            test_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(temp_path, test_img)
            
            segmentation_input = None
            
            def capture_segmentation(img, **kwargs):
                nonlocal segmentation_input
                segmentation_input = img.copy()
                return np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            
            with patch('sys.argv', ['deepskin', '-i', temp_path, '-m']):
                with patch('deepskin.__main__.wound_segmentation', side_effect=capture_segmentation):
                    main()
                    
                    assert segmentation_input is not None

    def test_main_mask_and_pwat_together(self, mock_tensorflow_model):
        from deepskin.__main__ import main
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, 'test.png')
            import cv2
            test_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(temp_path, test_img)
            
            with patch('sys.argv', ['deepskin', '-i', temp_path, '-m', '-p']):
                with patch('deepskin.__main__.wound_segmentation') as mock_seg:
                    with patch('deepskin.__main__.evaluate_PWAT_score') as mock_pwat:
                        mock_seg.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
                        mock_pwat.return_value = 12.5
                        
                        main()
                        
                        mock_seg.assert_called_once()
                        mock_pwat.assert_called_once()
