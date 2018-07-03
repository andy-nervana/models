import time
import logging
import random
import string

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
    # Add parameter to containers environment variables
    job_dict['spec']['template']['spec']['containers'][0]['env'].extend(env_dict)

    # Also we want to add it to the job and pods labels
    for pair in env_dict:
        k,v = pair['name'], pair['value']
        assert isinstance(v, (str, bytes))
        job_dict['metadata']['labels'][k] = v # Add parameter to the job label
        job_dict['spec']['template']['metadata']['labels'][k] = v # Add parameter to the pod label

    resp = batch.create_namespaced_job(body=job_dict, namespace='ailab-users-tareknas')
    logger.info("Job submitted")
    # print(resp) # Uncomment for debugging failed jobs

if __name__ == '__main__':
    config.load_kube_config()
    batch = client.BatchV1Api()
    v1 = client.CoreV1Api()
    logger.info("Connected to cluster")


    job_dict = yaml.load(open('gpu_job.yaml'))

    ## CHANGE THESE VALUES
    # models_dir = '/dataset/TF_models/current_ssd_models/' # change this, this should be /workspace/...
    models_dir = ''
    #models = ['ssd_mobilenet_v2_coco_cotafix_lr015_dr05_ds20k_512_noSigmoid']
    models = [' ']

    param_key = 'MODEL_PATH'
    params = [models_dir + model for model in models]

    in_flight_count = 2
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
            param = str(params.pop())
            env_dict = [dict(name=param_key, value=param),
                        dict(name='jobid', value=job_id)]
            logger.warn("Submitting job for param %s", param)
            submit_job(batch, env_dict)
            if len(params) == 0:
                done = True
                break
        logger.info("Sleeping for %d", sleep_time)
        time.sleep(sleep_time)
    logger.info("All jobs done, to clean up your jobs and pods, run `kubectl delete jobs,pods -l jobid=%s`", job_id)
