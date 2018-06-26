# USAGE:
#   python train_script.py TRAIN_FILE EVAL_FILE TRAIN_DIRECTORY EVAL_DIRECTORY CONFIG_FILE
import os
import sys

def step_mod(num_steps, file_in, file_out=None):

    if file_out == None:
        file_out = file_in

    with open(file_in, 'r') as f:
        x = f.readlines()

    for i in range(len(x)):
        if 'num_steps' in x[i]:
            num_steps_index = i

    x[num_steps_index] = '  num_steps: ' + str(num_steps) + '\n'

    with open(file_out, 'w') as f:
        for line in x:
            f.write(line)

def main():
    assert len(sys.argv) == 2 # INSUFFICIENT ARGUMENTS

    arg_file = sys.argv[1]

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

    train_command = 'python ' + train_file + ' --logtostderr --pipeline_config_path=' + config_file + ' --train_dir=' + train_dir
    eval_command = 'python ' + eval_file + ' --logtostderr --pipeline_config_path=' + config_file + ' --checkpoint_dir=' + train_dir + ' --eval_dir=' + eval_dir + '  --run_once True' 

    i = 0
    while i < max_steps:
        i += chunk_size
        step_mod(i, config_file)
        # os.system("python /nfs/site/home/tareknas/models/research/object_detection/train.py --logtostderr --pipeline_config_path=pipeline.config --train_dir=train/")
        # os.system("python /nfs/site/home/tareknas/models/research/object_detection/eval.py --logtostderr --pipeline_config_path=pipeline.config --checkpoint_dir=train/ --eval_dir=eval/ --run_once True")
        os.system(train_command)
        os.system(eval_command)

if __name__ == '__main__':
    main()