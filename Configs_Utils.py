import torch
import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm  # tqdm can be used in for-loops to show a progress bar
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import glob
import json
import os
import matplotlib.patches as patches
from PIL import ImageDraw, ImageFont
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath("Configs & Utils.py"))
DATA_PATH = os.path.join(BASE_DIR, "Vehicle Images")

SPLIT_TRAIN = "train_3_classi"
TRAIN_IMAGES_PATH = os.path.join(DATA_PATH, SPLIT_TRAIN, "images")
TRAIN_LABELS_PATH = os.path.join(DATA_PATH, SPLIT_TRAIN, "labels")


BATCH_SIZE = 32
EPOCHS =  25
WARMUP_EPOCHS = 0
LEARNING_RATE = 1E-4 
EPSILON = 1E-6
IMAGE_SIZE = (448, 448)

S = 7       
B = 2       


import yaml

CLASSES_PATH = os.path.join(DATA_PATH, "dataset_archive_3_classi.yaml")

# Load YAML

with open(CLASSES_PATH, 'r') as f:
    data = yaml.safe_load(f)

C = data['nc']        # number of classes
class_array = data['names']     # list: index -> class name
class_dict = {name: i for i, name in enumerate(class_array)}  # name -> index

###############
## UTILITIES ##
###############

def get_iou(p, a): # p è la bounding box predetta invece a quella reale
  """Calculates the Intersection-over-Union between two boxes"""
  p_tl, p_br = bbox_to_coords(p)          # (batch, S, S, B, 2)
  a_tl, a_br = bbox_to_coords(a)
  """
  B è il numero di Bounding Box
  2 sono le coordinate del vertice
  """

  # Largest top-left corner and smallest bottom-right corner give the intersection
  coords_join_size = (-1, -1, -1, B, B, 2)
  tl = torch.max(
      p_tl.unsqueeze(4).expand(coords_join_size),         # (batch, S, S, B, 1, 2) -> (batch, S, S, B, B, 2)
                                                            # quindi aggiungo una dimensioni dopo la quarta ovvero aggiungo quell'1
                                                            # poi con espand espando 1 -> B perchè deve tendere a (-1, -1, -1, B, B, 2)
                                                            # dove i -1 significa che lascio quelle dimensioni come sono
      a_tl.unsqueeze(3).expand(coords_join_size)          # (batch, S, S, 1, B, 2) -> (batch, S, S, B, B, 2)
                                                            # quindi aggiungouna dimensioni dopo la terza ovvero aggiungo quell'1
                                                            # poi con espand espando 1 -> B perchè deve tendere a (-1, -1, -1, B, B, 2)
                                                            # dove i -1 significa che lascio quelle dimensioni come sono
  )
  br = torch.min(
      p_br.unsqueeze(4).expand(coords_join_size),
      a_br.unsqueeze(3).expand(coords_join_size)
  )



  intersection_sides = torch.clamp(br - tl, min=0.0) # br - tl così trovo la larghezza e l'altezza dell'intersezione
  intersection = intersection_sides[..., 0] \
                  * intersection_sides[..., 1]       
  
  

# per trovare IoU ovvero intersezione/unione, ora devo calcolare l'unione, per farlo dev prima trovare le aree delle singole bounding box
  p_area = bbox_attr(p, 2) * bbox_attr(p, 3)                  # (batch, S, S, B)
  # questo è base per altezza (w * h), ovvero è l'area di ciascun box predetto
  p_area = p_area.unsqueeze(4).expand_as(intersection)        # (batch, S, S, B, 1) -> (batch, S, S, B, B)
  # uso unsqueeze e expand così che p_area coincida in dimensioni con intersection

  a_area = bbox_attr(a, 2) * bbox_attr(a, 3)                  # (batch, S, S, B)
  # area di ciascun box reale
  a_area = a_area.unsqueeze(3).expand_as(intersection)        # (batch, S, S, 1, B) -> (batch, S, S, B, B)

  union = p_area + a_area - intersection

  # Catch division-by-zero
  zero_unions = (union == 0.0)
  union[zero_unions] = EPSILON
  # se union = 0 allora dividerei per 0 allora al posto di 0 metto epsilon, quantià molto piccola
  intersection[zero_unions] = 0.0
  # dove union = 0 metto l'intersezione rispettiva uguale a 0

  return intersection / union


def bbox_to_coords(t):
  """Changes format of bounding boxes from [x, y, width, height] to ([x1, y1], [x2, y2])."""

  width = bbox_attr(t, 2)
  x = bbox_attr(t, 0)
  x1 = x - width / 2.0
  x2 = x + width / 2.0

  height = bbox_attr(t, 3)
  y = bbox_attr(t, 1)
  y1 = y - height / 2.0
  y2 = y + height / 2.0

  return torch.stack((x1, y1), dim=4), torch.stack((x2, y2), dim=4)


def bbox_attr(data, i):
 

  attr_start = C + i
  return data[..., attr_start::5]


def get_overlap(a, b):
  
  a_tl, a_width, a_height, _, _ = a
  b_tl, b_width, b_height, _, _ = b

  i_tl = (
      max(a_tl[0], b_tl[0]),
      max(a_tl[1], b_tl[1])
  )
  i_br = (
      min(a_tl[0] + a_width, b_tl[0] + b_width),
      min(a_tl[1] + a_height, b_tl[1] + b_height),
  )
  

  intersection = max(0, i_br[0] - i_tl[0]) \
                  * max(0, i_br[1] - i_tl[1])
  

  a_area = a_width * a_height
  b_area = b_width * b_height

  a_intersection = b_intersection = intersection
  if a_area == 0:
      a_intersection = 0
      a_area = EPSILON
  if b_area == 0:
      b_intersection = 0
      b_area = EPSILON
# Serve per evitare divisione per zero: se una box ha area zero (ad esempio box vuota), usa una piccola costante EPSILON
  return torch.max(
      a_intersection / a_area,
      b_intersection / b_area
  ).item() # prende il maggiore dei due (cioè la sovrapposizione più significativa)


def plot_boxes(data, labels, classes, color='orange', min_confidence=0.2, max_overlap=0.5, file=None, return_image=False):
  """Plots bounding boxes on the given image (data) based on the labels.
  The image is displayed, or saved to file if file argument is set."""

  grid_size_x = data.size(dim=2) / S # dimensione totale diviso il numero di celle in orizzontale
  grid_size_y = data.size(dim=1) / S # dimensione totale diviso il numero di celle in veticale
  m = labels.size(dim=0)
  n = labels.size(dim=1)

  bboxes = []
  for i in range(m):
      for j in range(n):
          for k in range((labels.size(dim=2) - C) // 5):
              bbox_start = 5 * k + C
              bbox_end = 5 * (k + 1) + C
              bbox = labels[i, j, bbox_start:bbox_end]
              # Questo prende i 5 attributi relativi alla k-esima box di quella cella.
              class_index = torch.argmax(labels[i, j, :C]).item() # prende l'indice della classe più probabile
              confidence = labels[i, j, class_index].item() * bbox[4].item()          # pr(c) * IoU
              """
              labels[i, j, class_index] → probabilità di quella classe in quella cella.
              bbox[4] → objectness o IoU score del box (quanto il box “contiene” davvero un oggetto).
              quindi confidence = probabilità combinata che:
              nella cella ci sia un oggetto,
              e che l’oggetto appartenga a quella classe.
              """
              if confidence > min_confidence: # Se la confidenza totale è troppo bassa, scarta il box
                  width = bbox[2] * IMAGE_SIZE[0]
                  height = bbox[3] * IMAGE_SIZE[1]
                  
                  tl = (
                      bbox[0] * IMAGE_SIZE[0] + j * grid_size_x - width / 2,
                      bbox[1] * IMAGE_SIZE[1] + i * grid_size_y - height / 2
                  )
                  # Trova il centro del box nell’immagine, poi risali di metà larghezza
                  # e metà altezza per ottenere l’angolo in alto a sinistra.
                  bboxes.append([tl, width, height, confidence, class_index])

  # Sort by highest to lowest confidence
  bboxes = sorted(bboxes, key=lambda x: x[3], reverse=True)
  # Ordinare decrescente è fondamentale per la Non-Max Suppression (NMS): terremo prima le box più affidabili

  # Calculate IoUs between each pair of boxes
  num_boxes = len(bboxes)
  iou = [[0 for _ in range(num_boxes)] for _ in range(num_boxes)]
  for i in range(num_boxes):
      for j in range(num_boxes):
          iou[i][j] = get_overlap(bboxes[i], bboxes[j])
# iou[i][j] = misura di sovrapposizione tra la box i e la box j.
  # Non-Maximal Suppression and render image
  image = T.ToPILImage()(data)
  draw = ImageDraw.Draw(image) # pennello per disegnare rettangoli e testo sull’immagine.
  discarded = set()
  for i in range(num_boxes):
      if i not in discarded:
          tl, width, height, confidence, class_index = bboxes[i]
          
          # Decrease confidence of other conflicting bboxes
          for j in range(num_boxes):
              other_class = bboxes[j][4]
              if j != i and other_class == class_index and iou[i][j] > max_overlap: # se la sovrapposizione supera la soglia (es. 0.5), j viene scartata.
                  discarded.add(j)
                # questo funziona sempre perchè essendo ordinate per confidenza, quando arrivi a i, tutte le j successive hanno confidenza minore
          # Annotate image
          draw.rectangle((tl, (tl[0] + width, tl[1] + height)), outline='orange')
          text_pos = (max(0, tl[0]), max(0, tl[1] - 11))
          #print(f"class_index: {class_index}")
          #print(f"classes: {classes}")
          text = f'{classes[class_index]} {round(confidence * 100, 1)}%'
          text_bbox = draw.textbbox(text_pos, text)
          draw.rectangle(text_bbox, fill='orange')
          draw.text(text_pos, text)
  if file is not None:
      output_dir = os.path.dirname(file)
      if output_dir and not os.path.exists(output_dir):
          os.makedirs(output_dir)
      if not file.endswith('.png'):
          file += '.png'
      image.save(file)
  elif not return_image:
      display(image)

  if return_image:
      return image
