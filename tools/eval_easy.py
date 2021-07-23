# -*- coding: UTF-8 -*-
# -------------------- 导入必要的包 -------------------- #
# ----- Python2 到 Python3 的兼容 ----- #
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ----- 系统操作相关 ----- #
import os
import os.path as osp
import sys
import time

# ----- 读取文件相关 ----- #
import json
import h5py

# ----- 数据处理相关 ----- #
import numpy as np

# ----- 美化打印 ----- #
from pprint import pprint

# ----- 命令行交互 ----- #
import argparse

# ----- 自定义的包 ----- #
# model
import _init_paths
from layers.joint_match import JointMatching
from loaders.gt_mrcn_loader import GtMRCNLoader
import models.eval_easy_utils as eval_utils

# ----- 模型相关 ----- #
# torch
import torch
import torch.nn as nn

# -------------------- 加载模型 -------------------- #
def load_model(checkpoint_path, opt):
  tic = time.time()
  model = JointMatching(opt)
  checkpoint = torch.load(checkpoint_path)
  model.load_state_dict(checkpoint['model'].state_dict())
  model.eval()
  model.cuda()
  print('model loaded in %.2f seconds' % (time.time()-tic))
  return model

# -------------------- 测试过程 -------------------- #
def evaluate(params):
  # ----- 建立数据加载器 ----- #
  # set up loader
  data_json = osp.join('/home/data/yjgroup/qxy/qxy/CM-Att-Erase/cache/prepro', params['dataset_splitBy'], 'data.json')
  data_h5 = osp.join('/home/data/yjgroup/qxy/qxy/CM-Att-Erase/cache/prepro', params['dataset_splitBy'], 'data.h5')
  loader = GtMRCNLoader(data_h5=data_h5, data_json=data_json)

  # ----- 加载模型 ----- #
  # load mode info
  model_prefix = osp.join('/home/data/yjgroup/qxy/qxy/CM-Att-Erase/output', params['dataset_splitBy'], params['id'])
  infos = json.load(open(model_prefix+'.json'))
  model_opt = infos['opt']
  model_path = model_prefix + '.pth'
  model = load_model(model_path, model_opt)

  # ----- 加载特征 ----- #
  # loader's feats
  feats_dir = '%s_%s_%s' % (model_opt['net_name'], model_opt['imdb_name'], model_opt['tag'])
  args.imdb_name = model_opt['imdb_name']
  args.net_name = model_opt['net_name']
  args.tag = model_opt['tag']
  args.iters = model_opt['iters']
  loader.prepare_mrcn(head_feats_dir=osp.join('/home/data/yjgroup/qxy/qxy/CM-Att-Erase/cache/feats/', model_opt['dataset_splitBy'], 'mrcn', feats_dir), 
                      args=args) 
  ann_feats = osp.join('/home/data/yjgroup/qxy/qxy/CM-Att-Erase/cache/feats', model_opt['dataset_splitBy'], 'mrcn', 
                       '%s_%s_%s_ann_feats.h5' % (model_opt['net_name'], model_opt['imdb_name'], model_opt['tag']))
  loader.loadFeats({'ann': ann_feats})

  # ----- 核查模型信息 ----- #
  # check model_info and params
  assert model_opt['dataset'] == params['dataset']
  assert model_opt['splitBy'] == params['splitBy']

  # ----- 在分割集上测试 ----- #
  # evaluate on the split, 
  # predictions = [{sent_id, sent, gd_ann_id, pred_ann_id, pred_score, sub_attn, loc_attn, weights}]
  split = params['split']
  model_opt['num_sents'] = params['num_sents']
  model_opt['verbose'] = params['verbose']
  crit = None
  val_loss, acc, predictions, overall = eval_utils.eval_split(loader, model, crit, split, model_opt)
  print('Comprehension on %s\'s %s (%s sents) is %.2f%%' % \
        (params['dataset_splitBy'], params['split'], len(predictions), acc*100.)) 
  print('attribute precision : %.2f%%' % (overall['precision']*100.0))
  print('attribute recall    : %.2f%%' % (overall['recall']*100.0))
  print('attribute f1        : %.2f%%' % (overall['f1']*100.0))       

  # ----- 保存运行的信息 ----- #
  # save
  out_dir = osp.join('cache', 'results', params['dataset_splitBy'], 'easy')
  if not osp.isdir(out_dir):
    os.makedirs(out_dir)
  out_file = osp.join(out_dir, params['id']+'_'+params['split']+'.json')
  with open(out_file, 'w') as of:
    json.dump({'predictions': predictions, 'acc': acc, 'overall': overall}, of)

  # ----- 将结果写到 results.txt 中 ----- #
  # write to results.txt
  f = open('experiments/easy_results.txt', 'a')
  f.write('[%s][%s], id[%s]\'s acc is %.2f%%\n' % \
          (params['dataset_splitBy'], params['split'], params['id'], acc*100.0))


# -------------------- main 函数执行 -------------------- #
if __name__ == '__main__':
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='refcoco', help='dataset name: refclef, refcoco, refcoco+, refcocog')
  parser.add_argument('--splitBy', type=str, default='unc', help='splitBy: unc, google, berkeley')
  parser.add_argument('--split', type=str, default='testA', help='split: testAB or val, etc')
  parser.add_argument('--id', type=str, default='0', help='model id name')
  parser.add_argument('--num_sents', type=int, default=-1, help='how many sentences to use when periodically evaluating the loss? (-1=all)')
  parser.add_argument('--verbose', type=int, default=1, help='if we want to print the testing progress')
  args = parser.parse_args()
  params = vars(args)

  # make other options
  params['dataset_splitBy'] = params['dataset'] + '_' + params['splitBy']
  evaluate(params)
