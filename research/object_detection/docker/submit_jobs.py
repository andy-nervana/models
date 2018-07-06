import time
import logging
import random
import string

import yaml
from kubernetes import client, config

# from aikube import set_environment_variable
import urllib3; urllib3.disable_warnings()

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

def submit_job(batch, command):
    """
    Submit a job to the cluster.
    Params:
        batch: K8S batch client
        env_dict: list of dicts where each dict is of the form name:<name>, value:<value> these will get added to the
            containers environment variables and pod and job labels
    """
    # Add parameter to containers environment variables
    # job_dict['spec']['template']['spec']['containers'][0]['env'].extend(env_dict)

    b_dict = yaml.load(open('gpu_job.yaml'))

    current_cmd = b_dict['spec']['template']['spec']['containers'][0]['command'] 
    b_dict['spec']['template']['spec']['containers'][0]['command'] = current_cmd + command
    # b_dict['spec']['template']['spec']['containers'][0]['args'] = ['-c'] + command

    # Also we want to add it to the job and pods labels
    #for pair in env_dict:
    #    k,v = pair['name'], pair['value']
    #    assert isinstance(v, (str, bytes))
    #    job_dict['metadata']['labels'][k] = v # Add parameter to the job label
    #    job_dict['spec']['template']['metadata']['labels'][k] = v # Add parameter to the pod label

    resp = batch.create_namespaced_job(body=b_dict, namespace='ailab-users-tareknas')
    logger.info("Job submitted")
    # print(resp) # Uncomment for debugging failed jobs

if __name__ == '__main__':
    config.load_kube_config()
    batch = client.BatchV1Api()
    v1 = client.CoreV1Api()
    logger.info("Connected to cluster")


    job_dict = yaml.load(open('gpu_job.yaml'))

    ## CHANGE THESE VALUES
    #param_key = 'MATRIX_SIZE'
    #params = [10, 20, 30]

    param_key = 'MODEL_PATH'
    params = ['blah blah']

    train_cmd = 'python /home/models/research/object_detection/train_script.py /home/models/research/object_detection/train_args.txt '
    model_dir = '/workspace/'

    models = ['01_ssd_mobilenet_v2_coco_cotafix_lr015_dr05_ds20k_512_foc_a010_g500',
             '02_ssd_mobilenet_v2_coco_cotafix_lr050_dr05_ds30k_512_foc_a025_b200']
    models = [model_dir + model for model in models]
    commands = [[train_cmd + model] for model in models]

    in_flight_count = 10
    sleep_time = 5
    ## /CHANGE THESE VALUES

    done = False
    job_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))
    label_selector = 'jobid=' + job_id

    while not done:
        jobs = batch.list_job_for_all_namespaces(label_selector=label_selector)
        live_job_count = len([job for job in jobs.items if job.status.succeeded is None])
        logger.info("Found %d jobs with label %s", live_job_count, label_selector)
        for i in range(in_flight_count - live_job_count):
            # param = str(params.pop())
            command = commands.pop()
            # env_dict = [dict(name=param_key, value=param),
            #             dict(name='jobid', value=job_id)]
            # logger.warn("Submitting job for param %s", param)
            logger.warn("Submitting job for command " + str(command))
            submit_job(batch, command)
            if len(commands) == 0:
                done = True
                break
        logger.info("Sleeping for %d", sleep_time)
        time.sleep(sleep_time)
    logger.info("All jobs done, to clean up your jobs and pods, run `kubectl delete jobs,pods -l jobid=%s`", job_id)
