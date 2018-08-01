# USAGE:
#   python train_script.py train_args.txt MODEL_PATH

import os
import sys
import time
import subprocess

def step_mod(num_steps, file_in, file_out=None):

    if file_out == None:
        file_out = 'pipeline.config'

    with open(file_in, 'r') as f:
        x = f.readlines()

    for i in range(len(x)):
        if 'num_steps' in x[i]:
            num_steps_index = i

    x[num_steps_index] = '  num_steps: ' + str(num_steps) + '\n'
    # import pdb; pdb.set_trace()


    with open(file_out, 'w') as f:
        for line in x:
            f.write(line)

def main():
    assert len(sys.argv) == 3 # INSUFFICIENT ARGUMENTS, USAGE: $python train_script.py train_args.txt MODEL_PATH

    arg_file = sys.argv[1]
    model_path = sys.argv[2]

    with open(arg_file, 'r') as f:
        args = f.readlines()

    args = [line.strip() for line in args]

    # max_steps = 400000
    # chunk_size = 50000

    # train_file = '/nfs/site/home/tareknas/models/research/object_detection/train.py'
    # train_dir = 'train/'
    # eval_file = '/nfs/site/home/tareknas/models/research/object_detection/eval.py'
    # eval_dir = 'eval/'
    # config_file = 'pipeline.config'

    train_file = args[0]
    eval_file = args[1]
    train_dir = args[2]
    eval_dir = args[3]
    config_file = args[4]
    max_steps = int(args[5])
    chunk_size = int(args[6])

    # train_command = 'python ' + train_file + ' --logtostderr --pipeline_config_path=' + config_file + ' --train_dir=' + train_dir
    # eval_command = 'python ' + eval_file + ' --logtostderr --pipeline_config_path=' + config_file + ' --checkpoint_dir=' + train_dir + ' --eval_dir=' + eval_dir + '  --run_once True' 

    train_command = 'python ' + train_file + ' --logtostderr --pipeline_config_path=' + 'pipeline.config' + ' --train_dir=' + train_dir
    eval_command = 'python ' + eval_file + ' --logtostderr --pipeline_config_path=' + 'pipeline.config' + ' --checkpoint_dir=' + train_dir + ' --eval_dir=' + eval_dir + '  --run_once True'

    initial_path = os.getcwd()
    os.chdir(model_path)

    i = 0
    while i < max_steps:
        i += chunk_size
        step_mod(i, config_file)
        # os.system("python /nfs/site/home/tareknas/models/research/object_detection/train.py --logtostderr --pipeline_config_path=pipeline.config --train_dir=train/")
        # os.system("python /nfs/site/home/tareknas/models/research/object_detection/eval.py --logtostderr --pipeline_config_path=pipeline.config --checkpoint_dir=train/ --eval_dir=eval/ --run_once True")
        print("TRAINING TO " + str(i) + " STEPS.")
        # subprocess.call(train_command.split())

        process = subprocess.Popen(train_command.split(), stdout=subprocess.PIPE)
        train_output, train_error = process.communicate()

        time.sleep(15)
        print("STARTING EVAL")
        with open('ode_results.txt', 'a') as f:
            f.write('STEPS: ' + str(i) + ' ')

        subprocess.call(eval_command.split())
        process = subprocess.Popen(train_command.split(), stdout=subprocess.PIPE)
        eval_output, eval_error = process.communicate()

    os.chdir(initial_path)

if __name__ == '__main__':
    main()
