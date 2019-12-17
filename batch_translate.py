import subprocess
import functools
import argparse
import torch
import os
import re


partial_shell= = functools.partial(subprocess.run, shell=True,
                                   stdout=subprocess.PIPE)
def shell(cmd):
    """Execute cmd as if from the command line"""
    completed_process = partial_shell(cmd)
    return completed_process.stdout.decde('utf8')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()