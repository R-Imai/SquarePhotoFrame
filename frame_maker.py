import os
import cv2
import glob
import tqdm
import numpy as np
from sklearn.cluster import KMeans
import argparse

def parse_arguments():
  parser = argparse.ArgumentParser(prog="SquarePhotoFrame", description="画像に適当な色の枠をつけて正方形にします。", epilog="フォルダもしくはファイルのどちらかの指定は必須です。")
  parser.add_argument(
    "-d", "--dir",
    dest="dir",
    type=str,
    help="取り込み対象のフォルダ"
  )
  parser.add_argument(
    "-f", "--fig",
    dest="fig",
    type=str,
    help="取り込み対象の画像ファイル"
  )
  parser.add_argument(
    "output",
    type=str,
    help="出力先フォルダ"
  )
  return parser.parse_args(), parser

def resize_image(img):
  width = img.shape[1]
  height = img.shape[0]
  max_val = max([width, height])
  rate = 300 / max_val
  size = (int(width*rate), int(height*rate))
  resize_img = cv2.resize(img ,size, interpolation = cv2.INTER_AREA)
  return resize_img

def import_image(file_name):
  img = cv2.imread(file_name)
  return img

def get_colors_cluster(img, cluster_num):
  flat_img = np.array([x for row in img for x in row])
  cls = KMeans(n_clusters=cluster_num)
  pred = cls.fit_predict(flat_img)
  _, counts = np.unique(pred, return_counts=True)
  return cls.cluster_centers_[np.argsort(counts)[::-1]]

def hsv_to_bgr(h, s, v):
    bgr = cv2.cvtColor(np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
    return [int(bgr[0]), int(bgr[1]), int(bgr[2])]

def bgr_to_hsv(b, g, r):
    hsv = cv2.cvtColor(np.array([[[b, g, r]]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]
    return [hsv[0], hsv[1], hsv[2]]

def get_bright_colors(colors):
  THRESHOLD = 256.0
  hsv = bgr_to_hsv(*colors)
  hsv[1] = 64 if hsv[1] > 64 else hsv[1]
  hsv[2] = 210 if hsv[2] < 210 else hsv[2]
  return hsv_to_bgr(*hsv)

def analysis_image(file_name, output_dir):
  CLUSTER_NUM = 3
  img = import_image(file_name)
  img_y = int(img.shape[0])
  img_x = int(img.shape[1])

  t_img = np.array(img) if img_x > img_y else np.array(img).transpose(1, 0, 2)
  cluster_centers1 = get_colors_cluster(resize_image(t_img[:int(len(t_img)/3)]), CLUSTER_NUM)
  color1 = get_bright_colors(cluster_centers1[0])
  cluster_centers2 = get_colors_cluster(resize_image(t_img[int(len(t_img)*2/3):]), CLUSTER_NUM)
  color2 = get_bright_colors(cluster_centers2[0])

  if img_x > img_y:
    draw_space_height = int((img_x - img_y)/2)
    img_tmp = cv2.copyMakeBorder(img, draw_space_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=[color1[0], color1[1], color1[2]])
    output_img = cv2.copyMakeBorder(img_tmp, 0, draw_space_height, 0, 0, cv2.BORDER_CONSTANT, value=[color2[0], color2[1], color2[2]])
    cv2.imwrite(f"{output_dir}/{os.path.basename(file_name)}", output_img)
  else:
    draw_space_width = int((img_y - img_x)/2)
    img_tmp = cv2.copyMakeBorder(img, 0, 0, draw_space_width, 0, cv2.BORDER_CONSTANT, value=[color1[0], color1[1], color1[2]])
    output_img = cv2.copyMakeBorder(img_tmp, 0, 0, 0, draw_space_width, cv2.BORDER_CONSTANT, value=[color2[0], color2[1], color2[2]])
    cv2.imwrite(f"{output_dir}/{os.path.basename(file_name)}", output_img)

def main():
  args, parser = parse_arguments()
  if args.dir is None and args.fig is None:
    parser.print_help()
    exit()
  if args.dir is not None:
    path_list = glob.glob(f"{args.dir}/*jpg")
    for path in tqdm.tqdm(path_list):
      analysis_image(path, args.output)
  if args.fig is not None:
    analysis_image(args.fig, args.output)
  

if __name__ == '__main__':
  main()