#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
# download model weights
import gdown
from zipfile import ZipFile

# constant values
from .constants import RESET_COLOR_CODE
from .constants import GREEN_COLOR_CODE
from .constants import RED_COLOR_CODE

__author__  = ['Nico Curti']
__email__ = ['nico.curti2@unibo.it']

__all__ = [
  'download_model_weights',
]

def download_model_weights (Id : str, model_name : str):
  '''
  Download the model files from google-drive repository
  and unpack the files in the 'data' directory.

  Parameters
  ----------
  Id : str
    Google Drive Id of the weights file to download

  model_name : str
    Output filename without extension of the model weights
  '''

  print (f'Downloading Deepskin model ... ', end='', flush=True)
  try:
    gdown.download(id=Id, output=f'{model_name}.zip', quiet=True)
    if os.path.exists(f'{model_name}.zip'):
      print(f'{GREEN_COLOR_CODE}[DONE]{RESET_COLOR_CODE}',
        end='\n',
        flush=True,
      )
    else:
      raise ValueError
  except Exception as e:
    print(f'{RED_COLOR_CODE}[FAILED]{RESET_COLOR_CODE}',
      end='\n',
      flush=True,
    )
    return

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
