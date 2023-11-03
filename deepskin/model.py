#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
# disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# download model weights
import requests
from zipfile import ZipFile
from time import time as now

import numpy as np
import tensorflow as tf

# image pre/post processing algorithms
from .imgproc import imfill
# constant values
from .constants import CRLF
from .constants import IMG_SIZE
from .constants import RESET_COLOR_CODE
from .constants import GREEN_COLOR_CODE

__author__  = ['Nico Curti', 'Gianluca Carlini']
__email__ = ['nico.curti2@unibo.it', 'gianluca.carlini3@unibo.it']

__all__ = [
  'deepskin_model',
  'wound_segmentation'
]


def download_file_from_google_drive (Id : int,
                                     destination : str,
                                     total_length : int,
                                     ):
  '''
  Download file from google drive page.

  Parameters
  ----------
    Id : int
      File Id in Google Drive page

    destination : str
      Destination path of the download

    total_lenght : int
      File dimension in bytes

  Returns
  -------
    None

  Notes
  -----
  ..note::
    The file Id can be extracted from the google drive page when the file is shared.
    The total length is useful only for graphics.
  '''

  url = 'https://docs.google.com/uc?export=download&confirm=1'

  def get_confirm_token (response):
    '''
    Check token validity.
    '''
    for key, value in response.cookies.items():
      if key.startswith('download_warning'):
        return value

    return None

  def save_response_content (response, destination):
    '''
    Download the file chunk by chunk and plot the progress
    '''
    chunk_size = 32768
    with open(destination, 'wb') as fp:
      dl = 0
      start = now()
      download = now()

      for chunk in response.iter_content(chunk_size):

        dl += len(chunk)
        done = int(50 * dl / total_length)
        progress = "█" * done + " " * (50 - done)
        perc = int(dl / total_length * 100)
        mb = dl / 1000000
        print((
          f'{CRLF}Downloading Deepskin model ... '
          f'|{progress}| {perc:.0f}% ({mb:.1f} Mb) {now() - start:<3.1f} sec'
          ),
          end='',
          flush=True
        )
        download = now()

        if chunk: # filter out keep-alive new chunks
          fp.write(chunk)

    print(f'{CRLF}Downloading Deepskin model ... {GREEN_COLOR_CODE}[DONE]{RESET_COLOR_CODE}',
      end='\n',
      flush=True,
    )

  session = requests.Session()
  response = session.get(
    url,
    params={'id' : Id},
    stream=True
  )
  token = get_confirm_token(response)

  if token:
    params = {
      'id' : Id,
      'confirm' : token
    }
    response = session.get(
      url,
      params=params,
      stream=True
    )

  save_response_content(response, destination)


def download_model_weights ():
  '''
  Download the model files from google-drive repository
  and unpack the files in the 'data' directory.
  '''
  # define the model name
  model_name = 'efficientnetb3_deepskin'
  print (f'Downloading Deepskin model ... ', end='', flush=True)
  download_file_from_google_drive(
    Id='1VCde7xlczbqJVgjIpTKQ4VjB45y5qq_D',
    destination=f'{model_name}.zip',
    total_length=66262365
  )
  print ('Extracting files ... ', end='', flush=True)

  with ZipFile(f'{model_name}.zip') as zipper:
    zipper.extractall('.')

  print (f'{GREEN_COLOR_CODE}[DONE]{RESET_COLOR_CODE}')

  local = os.path.dirname(os.path.abspath(__file__))
  outdir = os.path.join(local, '..', 'checkpoints')
  os.makedirs(outdir, exist_ok=True)

  os.rename(f'{model_name}.h5', os.path.join(outdir, f'{model_name}.h5'))
  os.remove(f'{model_name}.zip')

  return

def decoder_block (inputs, skip, filters, stage):

  x = tf.keras.layers.UpSampling2D(
    size=2,
    name=f'decoder_stage_{stage}_upsampling'
  )(inputs)
  x = tf.keras.layers.Concatenate(
    axis=-1,
    name=f'decoder_stage_{stage}concat'
  )([x, skip])
  x = tf.keras.layers.Conv2D(
    filters,
    kernel_size=(3, 3),
    kernel_initializer='he_uniform',
    padding='same',
    name=f'decoder_stage_{stage}a_conv',
    use_bias=False
  )(x)
  x = tf.keras.layers.BatchNormalization(
    axis=-1,
    name=f'decoder_stage_{stage}a_bn'
  )(x)
  x = tf.keras.layers.Activation(
    'relu',
    name=f'decoder_stage_{stage}a_relu'
  )(x)
  x = tf.keras.layers.Conv2D(
    filters,
    kernel_size=(3, 3),
    kernel_initializer='he_uniform',
    padding='same',
    name=f'decoder_stage_{stage}b_conv',
    use_bias=False
  )(x)
  x = tf.keras.layers.BatchNormalization(
    axis=-1,
    name=f'decoder_stage_{stage}b_bn'
  )(x)
  x = tf.keras.layers.Activation(
    'relu',
    name=f'decoder_stage_{stage}b_relu'
  )(x)

  return x

def transp_conv_block (inputs, filters, stage):

  x = inputs
  x = tf.keras.layers.UpSampling2D(
    size=2,
    name=f'decoder_stage_{stage}_upsampling'
  )(inputs)
  x = tf.keras.layers.Conv2D(
    filters,
    kernel_size=(3, 3),
    kernel_initializer='he_uniform',
    padding='same',
    name=f'decoder_stage_{stage}a_conv',
    use_bias=False
  )(x)
  x = tf.keras.layers.BatchNormalization(
    axis=-1,
    name=f'decoder_stage_{stage}a_bn'
  )(x)
  x = tf.keras.layers.Activation(
    'relu',
    name=f'decoder_stage_{stage}a_relu'
  )(x)
  x = tf.keras.layers.Conv2D(
    filters,
    kernel_size=(3, 3),
    kernel_initializer='he_uniform',
    padding='same',
    name=f'decoder_stage_{stage}b_conv',
    use_bias=False
  )(x)
  x = tf.keras.layers.BatchNormalization(
    axis=-1,
    name=f'decoder_stage_{stage}b_bn'
  )(x)
  x = tf.keras.layers.Activation(
    'relu',
    name=f'decoder_stage_{stage}b_relu'
  )(x)

  return x

def deepskin_model (verbose : bool = False) -> tf.keras.Model :
  '''
  Build the Deepskin segmentation model with the
  appropriated weights.
  '''

  if verbose:
    print(f'Build the deepskin model for wound image segmentation',
      end='\n',
      flush=True,
    )

  model = tf.keras.applications.efficientnet.EfficientNetB3(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling=None,
    classes=1,
    classifier_activation='sigmoid'
  )

  skip_connections = []

  for i in range(2, 7):
    if i == 5:
      continue
    skip_connections.append(model.get_layer(f'block{i}a_expand_activation').output)

  skip_connections.reverse()

  x = model.get_layer('top_activation').output
  filters = [256, 128, 64, 32]
  for i, skip in enumerate(skip_connections):
    x = decoder_block(
      inputs=x,
      skip=skip,
      filters=filters[i],
      stage=i
    )

  x = transp_conv_block(
    inputs=x,
    filters=16,
    stage=4
  )
  x = tf.keras.layers.Conv2D(
    filters=1,
    kernel_size=(3, 3),
    padding='same',
    name='final_conv'
  )(x)
  x = tf.keras.layers.Activation(
    'sigmoid',
    name='sigmoid'
  )(x)

  model = tf.keras.models.Model(
    inputs=model.layers[3].input,
    outputs=x
  )

  # load model weights

  local = os.path.dirname(os.path.abspath(__file__))
  # build the weights filepath
  weightspath = os.path.join(
    local,
    '..',
    'checkpoints',
    'efficientnetb3_deepskin.h5'
  )
  # if the weights file does not exists
  if not os.path.exists(weightspath):
    download_model_weights()

  # load the weights
  model.load_weights(weightspath)

  return model


def wound_segmentation (img : np.ndarray,
                        tol : float = 0.5,
                        verbose : bool = False
                        ) -> np.ndarray :
  '''
  Perform the wound image segmentation using the
  Deepskin model.

  Parameters
  ----------
    img : np.ndarray
      Input image to analyze in RGB format

    tol : float (default := 0.5)
      Threshold to apply on the resulting mask for the
      output binarization

    verbose : bool (default := False)
      Enable/Disable the logging of the steps

  Returns
  -------
    pred : np.ndarray
      Output image mask obtained by the model
  '''

  tic = now()
  step = 'Perform the wound image segmentation... '

  if verbose:
    print(f'{step}',
      end='\r',
      flush=True,
    )

  # build the model for the wound segmentation
  model = deepskin_model()

  # get the model input shape
  _, h, w, c = model.input.shape

  # pre-process the input

  # resize the image into the shape required by the model
  if verbose:
    print(f'{CRLF}{step} {GREEN_COLOR_CODE}[1/3] pre-process the input image{RESET_COLOR_CODE}',
      end='',
      flush=True,
    )
  resized = cv2.resize(
    img,
    dsize=(h, w),
    interpolation=cv2.INTER_CUBIC
  )
  # convert the image into floating-point values
  resized = np.float32(resized)
  # normalize the image into [0, 1] range
  resized *= 1. / 255
  # extend the dimensionality of the input array
  # to the [batch, h, w, c] format
  resized = resized.reshape(1, *resized.shape)

  # apply the segmentation model

  if verbose:
    print(f'{CRLF}{step} {GREEN_COLOR_CODE}[2/3] apply the segmentation model{RESET_COLOR_CODE}',
      end='',
      flush=True,
    )
  # apply the model to get the prediction
  pred = model.predict(resized)
  # remove useless dimensions from the image
  pred = np.squeeze(pred)
  # filter the mask output to binary format
  pred = np.where(pred > tol, 255, 0)
  # convert the mask into uint8 fmt
  pred = np.uint8(pred)

  # post-process the mask

  if verbose:
    print(f'{CRLF}{step} {GREEN_COLOR_CODE}[3/3] post-process the segmentation mask{RESET_COLOR_CODE}',
      end='',
      flush=True,
    )
  # fill possible holes and artifacts
  pred = imfill(pred)
  # resize the output mask to the same
  # shape of the original image, with an
  # appropriated interpolation algorithm
  pred = cv2.resize(
    pred,
    dsize=(img.shape[1], img.shape[0]),
    interpolation=cv2.INTER_NEAREST_EXACT
  )

  toc = now()
  if verbose:
    print(f'{CRLF}{step} {GREEN_COLOR_CODE}[DONE]{RESET_COLOR_CODE} ({toc - tic:.3f} sec)',
      end='\n',
      flush=True,
    )

  return pred
