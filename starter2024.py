import argparse
import openai
import pprint
from src.tot.methods.treeBFS import solve
import os

from tot.tasks.design import DesignTask

openai.api_key = "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"

args = argparse.Namespace(backend='gpt-4', temperature=0.7, task='design', naive_run=False, prompt_sample=None,
                          method_generate='propose', method_evaluate='value', method_select='greedy',
                          n_generate_sample=3, n_evaluate_sample=3, n_select_sample=5)

task = DesignTask()
ys, infos = solve(args, task, 0)

print("全部阶段完成。接下来pprint打印infos\n")

pprint.pprint(infos)

print("全部阶段完成。接下来print打印infos\n")

print(infos)
