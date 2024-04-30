#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
# disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# constant values
from .constants import CRLF
from .constants import IMG_SIZE
from .constants import RESET_COLOR_CODE
from .constants import GREEN_COLOR_CODE

__author__  = ['Nico Curti', 'Gianluca Carlini']
__email__ = ['nico.curti2@unibo.it', 'gianluca.carlini3@unibo.it']

__all__ = [
  'deepskin_model',
  'MODEL_CHECKPOINT',
]

MODEL_CHECKPOINT = '1it-fXhSTFp49kS6I0_ceykZ8jqtYLPkL'

def transpose_bn_block (inputs, filters, stage, activation='relu'):

  transpose_name = f'decoder_stage_{stage}a_transpose'
  bn_name = f'decoder_stage_{stage}a_bn'
  act_name = f'decoder_stage_{stage}a_{activation}'

  x = inputs
  x = tf.keras.layers.Conv2DTranspose(
    filters=filters,
    kernel_size=(4, 4),
    strides=(2, 2),
    padding='same',
    name=transpose_name,
    use_bias=False,
  )(inputs)

  x = tf.keras.layers.BatchNormalization(
    axis=-1,
    name=bn_name
  )(x)

  x = tf.keras.layers.Activation(
    activation=activation,
    name=act_name
  )(x)

  return x


def conv_bn_block (inputs, filters, stage='', activation='relu', k_size=3, dilation_rate=1, use_bias=False, name=None):

  if name is not None:
    conv_block_name = name

  else:
    conv_block_name = f'decoder_stage_{stage}b'

  x = inputs
  x = tf.keras.layers.Conv2D(
    filters=filters,
    kernel_size=k_size,
    kernel_initializer='he_uniform',
    padding='same',
    use_bias=use_bias,
    dilation_rate=dilation_rate,
    name=f'{conv_block_name}_conv',
  )(x)

  x = tf.keras.layers.BatchNormalization(
    axis=-1,
    name=f'{conv_block_name}_bn'
  )(x)

  x = tf.keras.layers.Activation(
    activation=activation,
    name=f'{conv_block_name}_{activation}'
  )(x)

  return x


def decoder_block (inputs, filters, stage, skip=None, activation='relu'):

  x = inputs
  x = transpose_bn_block(
    inputs=x,
    filters=filters,
    stage=stage,
    activation=activation
  )

  if skip is not None:
    x = tf.keras.layers.Concatenate(
      axis=-1,
      name=f'decoder_stage_{stage}concat'
    )([x, skip])

  x = conv_bn_block(
    inputs=x,
    filters=filters,
    stage=stage,
    activation=activation,
    k_size=3,
    dilation_rate=1,
    use_bias=False,
    name=None,
  )

  return x

def deepskin_model (verbose : bool = False) -> tf.keras.Model :
  '''
  Build the Deepskin segmentation model.

  Parameters
  ----------
    verbose : bool (default := False)
      Enable/Disable the logging of the steps.
  '''

  if verbose:
    print(f'Build the deepskin model for semantic image segmentation',
      end='',
      flush=True,
    )

  filters = [256, 128, 64, 32, 16]

  # define the efficientnetb3 model as encoder part
  encoder = tf.keras.applications.efficientnet.EfficientNetB3(
    include_top=False,
    weights=None, # remove pre-processing ImageNet layers
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
  )

  # get the feature layers
  layers = [
    'block6a_expand_activation',
    'block4a_expand_activation',
    'block3a_expand_activation',
    'block2a_expand_activation',
  ]

  x = encoder.output

  skip_connections = []
  for layer in layers:
    skip_connections.append(
      encoder.get_layer(layer).output
    )

  for i, skip in enumerate(skip_connections):
    x = decoder_block(
      inputs=x,
      filters=filters[i],
      stage=i,
      skip=skip,
      activation='relu',
    )

  x = decoder_block(
    inputs=x,
    filters=filters[-1],
    stage=4,
    activation='linear',
  )

  x = tf.keras.layers.Conv2D(
    filters=3,
    kernel_size=(1, 1),
    padding='same',
    name='final_conv',
    dtype='float32',
  )(x)
  x = tf.keras.layers.Activation(
    activation='softmax',
    name='softmax',
    dtype='float32'
  )(x)

  model = tf.keras.models.Model(
    inputs=encoder.input,
    outputs=x
  )

  if verbose:
    print(f'{CRLF} {GREEN_COLOR_CODE}[DONE]{RESET_COLOR_CODE}                   ',
      end='\n',
      flush=True,
    )

  return model
