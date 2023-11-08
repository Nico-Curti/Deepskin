#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
# download model weights
import requests
from zipfile import ZipFile
from time import time as now

# constant values
from .constants import CRLF
from .constants import IMG_SIZE
from .constants import RESET_COLOR_CODE
from .constants import GREEN_COLOR_CODE

__author__  = ['Nico Curti']
__email__ = ['nico.curti2@unibo.it']

__all__ = [
  'download_model_weights',
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


def download_model_weights (Id : str, model_name : str):
  '''
  Download the model files from google-drive repository
  and unpack the files in the 'data' directory.
  '''

  print (f'Downloading Deepskin model ... ', end='', flush=True)
  download_file_from_google_drive(
    Id=Id,
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
