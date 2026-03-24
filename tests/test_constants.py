#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
from deepskin.constants import (
    CRLF,
    IMG_SIZE,
    GREEN_COLOR_CODE,
    ORANGE_COLOR_CODE,
    VIOLET_COLOR_CODE,
    RED_COLOR_CODE,
    RESET_COLOR_CODE,
    Deepskin_CENTER,
    Deepskin_SCALE,
    Deepskin_PWAT_PARAMS,
    Deepskin_PWAT_BIAS,
)


class TestConstants:

    def test_img_size_value(self):
        assert IMG_SIZE == 256
        assert isinstance(IMG_SIZE, int)

    def test_color_codes_are_strings(self):
        assert isinstance(GREEN_COLOR_CODE, str)
        assert isinstance(ORANGE_COLOR_CODE, str)
        assert isinstance(VIOLET_COLOR_CODE, str)
        assert isinstance(RED_COLOR_CODE, str)
        assert isinstance(RESET_COLOR_CODE, str)

    def test_crlf_is_string(self):
        assert isinstance(CRLF, str)

    def test_color_codes_contain_ansi_escape(self):
        assert GREEN_COLOR_CODE.startswith('\033')
        assert RED_COLOR_CODE.startswith('\033')
        assert RESET_COLOR_CODE == '\033[0m'

    def test_deepskin_center_is_dict(self):
        assert isinstance(Deepskin_CENTER, dict)
        assert len(Deepskin_CENTER) > 0

    def test_deepskin_scale_is_dict(self):
        assert isinstance(Deepskin_SCALE, dict)
        assert len(Deepskin_SCALE) > 0

    def test_deepskin_pwat_params_is_dict(self):
        assert isinstance(Deepskin_PWAT_PARAMS, dict)
        assert len(Deepskin_PWAT_PARAMS) > 0

    def test_deepskin_pwat_bias_is_numeric(self):
        assert isinstance(Deepskin_PWAT_BIAS, (int, float))
        assert Deepskin_PWAT_BIAS != 0

    def test_center_keys_have_prefixes(self):
        wound_keys = [k for k in Deepskin_CENTER.keys() if k.startswith('w_')]
        peri_keys = [k for k in Deepskin_CENTER.keys() if k.startswith('p_')]
        
        assert len(wound_keys) > 0
        assert len(peri_keys) > 0

    def test_scale_keys_have_prefixes(self):
        wound_keys = [k for k in Deepskin_SCALE.keys() if k.startswith('w_')]
        peri_keys = [k for k in Deepskin_SCALE.keys() if k.startswith('p_')]
        
        assert len(wound_keys) > 0
        assert len(peri_keys) > 0

    def test_pwat_params_keys_have_prefixes(self):
        wound_keys = [k for k in Deepskin_PWAT_PARAMS.keys() if k.startswith('w_')]
        peri_keys = [k for k in Deepskin_PWAT_PARAMS.keys() if k.startswith('p_')]
        
        assert len(wound_keys) > 0
        assert len(peri_keys) > 0

    def test_center_values_are_numeric(self):
        for key, value in Deepskin_CENTER.items():
            assert isinstance(value, (int, float)), f"Key {key} has non-numeric value"

    def test_scale_values_are_numeric(self):
        for key, value in Deepskin_SCALE.items():
            assert isinstance(value, (int, float)), f"Key {key} has non-numeric value"

    def test_pwat_params_values_are_numeric(self):
        for key, value in Deepskin_PWAT_PARAMS.items():
            assert isinstance(value, (int, float)), f"Key {key} has non-numeric value"

    def test_scale_values_non_zero(self):
        for key, value in Deepskin_SCALE.items():
            assert value != 0, f"Scale value for {key} is zero (would cause division by zero)"

    def test_center_contains_haralick_features(self):
        haralick_keys = [k for k in Deepskin_CENTER.keys() if 'haralick' in k]
        assert len(haralick_keys) >= 13

    def test_center_contains_rgb_features(self):
        rgb_keys = [k for k in Deepskin_CENTER.keys() if 'avg' in k or 'std' in k]
        r_keys = [k for k in rgb_keys if 'R' in k]
        g_keys = [k for k in rgb_keys if 'G' in k]
        b_keys = [k for k in rgb_keys if 'B' in k]
        
        assert len(r_keys) > 0
        assert len(g_keys) > 0
        assert len(b_keys) > 0

    def test_center_contains_park_amparo(self):
        park_keys = [k for k in Deepskin_CENTER.keys() if 'park' in k]
        amparo_keys = [k for k in Deepskin_CENTER.keys() if 'amparo' in k]
        
        assert len(park_keys) > 0
        assert len(amparo_keys) > 0
