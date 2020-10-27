# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
import datetime
import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.data import datasets
from ..utils.comm import is_main_process
from ..utils.comm import all_gather
from ..utils.comm import synchronize

import multiprocessing as mp
import queue
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
import numpy as np
import copy

def prepare_for_coco_segmentation_batch(in_q, out_q, dataset, finish_input):
    import pycocotools.mask as mask_util
    import numpy as np

    masker = Masker(threshold=0.5, padding=1)
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    while(not finish_input.is_set()):
#        if(not in_q.empty()):
        try:
            out = in_q.get(False)
         #   out_ = copy.deepcopy(out)
         #   except queue.Empty:
         #       continue
            for image_id, prediction_list in out.items():
          #      print("is it in segmentation for loop ", len(prediction), flush=True)
                
                original_id = dataset.id_to_img_map[image_id]
                if len(prediction_list) == 0:
                    continue

                img_info = dataset.get_img_info(image_id)
                image_width = img_info["width"]
                image_height = img_info["height"]
            #    prediction_ = copy.deepcopy(prediction)
                prediction_ = prediction_list[3]
                prediction_ = prediction_.resize((image_width, image_height))
               # print(prediction)
             #   np.resize(prediction, (image_width, image_height))
             #   print("what is this value after resize", prediction_)
           #     masks = prediction_.get_field("mask")
             #   print("this is after resize", prediction_)
                masks = torch.from_numpy(prediction_list[0])
             #   masks = prediction_list[0]
                # Masker is necessary only if masks haven't been already resized.
            #    if list(masks.shape[-2:]) != [image_height, image_width]:
            #        masks_n = masker(torch.from_numpy(masks).expand(1, -1, -1, -1, -1), prediction_)
            #        masks_ = masks_n[0]
              #      print("in mask compute ", flush=True)
                # prediction = prediction.convert('xywh')
            #    print("what is this value ", prediction)
                # boxes = prediction.bbox.tolist()
              #  scores = prediction_.get_field("scores").tolist()
              #  labels = prediction_.get_field("labels").tolist()

                # rles = prediction.get_field('mask')
             #   print("started encoding ", flush=True)

               # masks = prediction_list[0]
                scores = prediction_list[1]
                labels = prediction_list[2]

                rles = [
                    mask_util.encode(np.array(mask[0, :, :, np.newaxis],dtype=np.uint8, order="F"))[0]
                    for mask in masks
                ]
                for rle in rles:
                    rle["counts"] = rle["counts"].decode("utf-8")

          #      print("decoding done", flush=True)
          #      print("after decoding ", rles, flush=True)

                mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

                coco_results.extend(
                    [
                        {
                            "image_id": original_id,
                            "category_id": mapped_labels[k],
                            "segmentation": rle,
                            "score": scores[k],
                        }
                        for k, rle in enumerate(rles)
                    ]
                )
        except queue.Empty:
            pass
    print("putting coco results ")
    out_q.put(coco_results)
    return

#def compute_on_dataset(model, data_loader, dataset, device):
def compute_on_dataset(model, data_loader, device):
 #   in_q = mp.Queue()
 #   out_q = mp.Queue()
 #   stop_event = mp.Event()
 #   stop_event.clear()
 #   post_proc = mp.Process(target=prepare_for_coco_segmentation_batch, args=(in_q, out_q, dataset, stop_event))
 #   post_proc2 = mp.Process(target=prepare_for_coco_segmentation_batch, args=(in_q, out_q, dataset, stop_event))
 #   post_proc3 = mp.Process(target=prepare_for_coco_segmentation_batch, args=(in_q, out_q, dataset, stop_event))
 #   post_proc.start()
 #   post_proc2.start()
 #   post_proc3.start()

    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
 #   bb_op = {}
   # masker = Masker(threshold=0.5, padding=1)
    
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        images = images.to(device)
        with torch.no_grad():
            output = model(images)
            output = [o.to(cpu_device) for o in output]

  #      for img_id, pred in zip(image_ids, output):
    #        img_info = dataset.get_img_info(img_id)
   #         image_width = img_info["width"]
   #         image_height = img_info["height"]
   #         pred = pred.resize((image_width, image_height))
   #         masks = pred.get_field("mask")
           # if list(masks.shape[-2:]) != [image_height, image_width]:
           #     masks = masker(masks.expand(1, -1, -1, -1, -1), pred)
           #     masks = masks[0]
    #        scores = pred.get_field("scores").tolist()
     #       labels = pred.get_field("labels").tolist()
           # print(type(masks))
          #  print("from boxlist values ", i.fields())
     #       bb_op.update({img_id: [masks.numpy(), scores, labels, pred]})
           # bb_op.append([o.to_numpy() for o in output])
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
      #  preds = copy.deepcopy(results_dict)
        #bb_op.update({img_id: [masks.numpy(), scores, labels]})
#        print("started putting items ", flush=True)
      #  in_q.put(bb_op, False)
 #   stop_event.set()
 #   converted_predictions = out_q.get()
  #  converted_predictions = out_q.get() + out_q.get() + out_q.get()
  #  post_proc.join()
  #  post_proc2.join()
  #  post_proc3.join()
  #  print("Q is empty ", in_q.empty())
   # return results_dict, converted_predictions
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    data_load_start = time.time()
    dataset = data_loader.dataset
    data_load_time = time.time() - data_load_start
    logger.info("Data loading time {} ".format(data_load_time))
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    start_time = time.time()
 #   synchronize()
  #  predictions, prepare_segm = compute_on_dataset(model, data_loader, dataset, device)
#    prepare_segm = {}
    predictions = compute_on_dataset(model, data_loader, device)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info("dataset length: {}  ".format(len(dataset)))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    # We have an optimised path for COCO which takes advantage of more parallelism.
    # If not using COCO, fall back to regular path: gather predictions from all ranks
    # and call evaluate on those results.
    if not isinstance(dataset, datasets.COCODataset):
        acc_start = time.time()
        predictions = _accumulate_predictions_from_multiple_gpus(predictions)
        acc_total = time.time() - acc_start
        logger.info("Accumulating predictions {} ".format(acc_total))
        if not is_main_process():
            return

    if output_folder and is_main_process():
        save_pred_start = time.time()
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))
        save_pred = time.time() - save_pred_start
        logger.info("Time to save predictions {} ".format(save_pred))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
 #       prepare_segm=prepare_segm,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
