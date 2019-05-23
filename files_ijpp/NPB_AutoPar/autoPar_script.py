import argparse
import os
import subprocess
import distutils
import time
from distutils import dir_util

def run_test(f_name, dest_dir):
    print ("starting test "+ f_name)
    sub_proc = subprocess.Popen(['autoPar', '-I../common', '-rose:o', f_name, f_name], cwd=dest_dir)
    sub_proc.wait()
    print ("done test")

def main(dir):

    dir = os.path.abspath(dir)
    dir_name = os.path.basename(dir)

    for root, dirs, files in os.walk(dir):
        for name in files:
            if os.path.splitext(name)[1] == '.c':
                run_test(name, dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs tests with varying input sizes.')
    parser.add_argument('-dir',
        dest='dir',
        help='Path to the directory containing the tests.')
    args = parser.parse_args()

    main(args.dir)

