import time
import logging
import random
import string
from copy import copy

import yaml
from kubernetes import client, config

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

def submit_job(batch, env_dict):
    """
    Submit a job to the cluster.
    Params:
        batch: K8S batch client
        env_dict: list of dicts where each dict is of the form name:<name>, value:<value> these will get added to the
            containers environment variables and pod and job labels
    """
    job_dict = yaml.load(open('gpu_job_ssd.yaml'))

    # Add parameter to containers environment variables
    job_dict['spec']['template']['spec']['containers'][0]['env'] += env_dict

    # Also we want to add it to the job and pods labels
    for pair in env_dict:
        k,v = pair['name'], pair['value']
        assert isinstance(v, (str, bytes))
        job_dict['metadata']['labels'][k] = v.replace('/', 'S')[-60:] # Add parameter to the job label
        job_dict['spec']['template']['metadata']['labels'][k] = v.replace('/', 'S')[-60:] # Add parameter to the pod label

    resp = batch.create_namespaced_job(body=job_dict, namespace='tareknas')
    logger.info("Job submitted with parameters: {}".format(', '.join(["{}:{}".format(setting['name'], setting['value']) for setting in env_dict])))
    # print(resp) # Uncomment for debugging failed jobs

if __name__ == '__main__':
    config.load_kube_config()
    batch = client.BatchV1Api()
    v1 = client.CoreV1Api()
    logger.info("Connected to cluster")

    models_dir = '/mnt/repo/models/research/object_detection/ferrari/models/'

    param_settings = [
      {'CONFIG_PATH' : models_dir + 'ssd_mobilenet_v2_coco_2018_6_6/pipeline.config',
       'TRAIN_PATH' :  models_dir + 'ssd_mobilenet_v2_coco_2018_6_6/train/',
       'EVAL_PATH' : models_dir + 'ssd_mobilenet_v2_coco_2018_6_6/eval/',
       'MODEL_PATH' : models_dir + 'ssd_mobilenet_v2_coco_2018_6_6/'},

      # {'CONFIG_PATH' : models_dir + 'ssd_mobilenet_v2_coco_focal_2018_6_6/pipeline.config',
      #  'TRAIN_PATH' :  models_dir + 'ssd_mobilenet_v2_coco_focal_2018_6_6/train/'},

      # {'CONFIG_PATH' : models_dir + 'ssdlite_mobilenet_v2_coco_2018_6_6/pipeline.config',
      #  'TRAIN_PATH' :  models_dir + 'ssdlite_mobilenet_v2_coco_2018_6_6/train/'},

      # {'CONFIG_PATH' : models_dir + 'ssdlite_mobilenet_v2_coco_focal_2018_6_6/pipeline.config',
      #  'TRAIN_PATH' :  models_dir + 'ssdlite_mobilenet_v2_coco_focal_2018_6_6/train/'},

      # {'CONFIG_PATH' : models_dir + 'ssd_inception_v2_coco_2018_6_6/pipeline.config',
      #  'TRAIN_PATH' :  models_dir + 'ssd_inception_v2_coco_2018_6_6/train/'},

      # {'CONFIG_PATH' : models_dir + 'ssd_inception_v2_coco_focal_2018_6_6/pipeline.config',
      #  'TRAIN_PATH' :  models_dir + 'ssd_inception_v2_coco_focal_2018_6_6/train/'},

      # {'CONFIG_PATH' : models_dir + 'faster_rcnn_inception_v2_coco_2018_6_6/pipeline.config',
      #  'TRAIN_PATH' :  models_dir + 'faster_rcnn_inception_v2_coco_2018_6_6/train/'},

      # {'CONFIG_PATH' : models_dir + 'faster_rcnn_resnet101_kitti_2018_6_6/pipeline.config',
      #  'TRAIN_PATH' :  models_dir + 'faster_rcnn_resnet101_kitti_2018_6_6/train/'}
    ]

    ## CHANGE THESE VALUES
    in_flight_count = 40
    sleep_time = 10
    ## /CHANGE THESE VALUES

    done = False
    job_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))
    label_selector = 'jobid=' + job_id

    while not done:
        jobs = batch.list_job_for_all_namespaces(label_selector=label_selector)
        live_job_count = len([job for job in jobs.items if job.status.succeeded is None])
        logger.info("Found %d jobs with label %s", live_job_count, label_selector)
        for i in range(in_flight_count - live_job_count):
            params = param_settings.pop()
            env_dict = [dict(name=p_key, value=str(p_val)) for p_key, p_val in params.items()] + [dict(name='jobid', value=job_id)]
            
            logger.warn("Submitting job for param{}".format(', '.join(["{}:{}".format(setting['name'], setting['value']) for setting in env_dict])))
            submit_job(batch, env_dict)
            if len(param_settings) == 0:
                done = True
                break
        
        logger.info("Sleeping for %d", sleep_time)
        time.sleep(sleep_time)

logger.info("All jobs done, to clean up your jobs and pods, run `kubectl delete jobs,pods -l jobid=%s`", job_id)
