#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
import os


class TestDeepskinModel:

    def test_deepskin_model_constant_exists(self):
        from deepskin.model import MODEL_CHECKPOINT
        
        assert isinstance(MODEL_CHECKPOINT, str)
        assert len(MODEL_CHECKPOINT) > 0

    def test_deepskin_model_builds_successfully(self):
        from deepskin.model import deepskin_model
        
        model = deepskin_model(verbose=False)
        
        assert model is not None
        assert hasattr(model, 'input')
        assert hasattr(model, 'output')

    def test_deepskin_model_input_shape(self):
        from deepskin.model import deepskin_model
        
        model = deepskin_model(verbose=False)
        
        input_shape = model.input.shape
        assert input_shape[1] == 256
        assert input_shape[2] == 256
        assert input_shape[3] == 3

    def test_deepskin_model_output_shape(self):
        from deepskin.model import deepskin_model
        
        model = deepskin_model(verbose=False)
        
        output_shape = model.output.shape
        assert output_shape[3] == 3

    def test_deepskin_model_verbose_mode(self, capsys):
        from deepskin.model import deepskin_model
        
        model = deepskin_model(verbose=True)
        
        captured = capsys.readouterr()
        assert model is not None


class TestModelBlocks:

    def test_model_has_encoder_layers(self):
        from deepskin.model import deepskin_model
        
        model = deepskin_model(verbose=False)
        
        layer_names = [layer.name for layer in model.layers]
        assert len(layer_names) > 0

    def test_model_has_decoder_layers(self):
        from deepskin.model import deepskin_model
        
        model = deepskin_model(verbose=False)
        
        layer_names = [layer.name for layer in model.layers]
        decoder_layers = [name for name in layer_names if 'decoder' in name]
        assert len(decoder_layers) > 0

    def test_model_has_softmax_output(self):
        from deepskin.model import deepskin_model
        
        model = deepskin_model(verbose=False)
        
        layer_names = [layer.name for layer in model.layers]
        assert 'softmax' in layer_names


class TestCheckpoints:

    def test_download_model_weights_handles_exception(self):
        with patch('gdown.download', side_effect=Exception('Download failed')):
            from deepskin.checkpoints import download_model_weights
            
            result = download_model_weights(
                Id='test_id',
                model_name='test_model'
            )
            
            assert result is None

    def test_download_model_weights_file_not_created(self):
        with patch('gdown.download'):
            with patch('os.path.exists', return_value=False):
                from deepskin.checkpoints import download_model_weights
                
                result = download_model_weights(
                    Id='test_id',
                    model_name='test_model'
                )
                
                assert result is None

    def test_download_model_weights_success_flow(self):
        with patch('gdown.download') as mock_download:
            with patch('os.path.exists', return_value=True):
                with patch('deepskin.checkpoints.ZipFile') as mock_zipfile_class:
                    mock_zip = MagicMock()
                    mock_zipfile_class.return_value.__enter__ = MagicMock(return_value=mock_zip)
                    mock_zipfile_class.return_value.__exit__ = MagicMock(return_value=False)
                    with patch('os.makedirs'):
                        with patch('os.rename'):
                            with patch('os.remove'):
                                from deepskin.checkpoints import download_model_weights
                                
                                result = download_model_weights(
                                    Id='test_id',
                                    model_name='test_model'
                                )
                                
                                mock_download.assert_called_once()
