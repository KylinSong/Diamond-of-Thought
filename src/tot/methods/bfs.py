import json
import itertools
import numpy as np
from functools import partial
from tot.models import gpt
import pprint


def get_former_part_str_in_str(y: str) -> str:
    y_dict = json.loads(y)
    if len(y_dict) <= 1:
        return ""
    else:
        y_dict.popitem()
        return json.dumps(y_dict, ensure_ascii=False)


# 合并ys内部所有y
def merge_similar_jsons_str_to_str(ys: []) -> str:
    # 确保 ys 中至少有一个元素
    if not ys:
        return "{}"
    # 将 JSON 字符串转换为字典
    dict_list = [json.loads(y) for y in ys]
    # 提取除了最后一个键以外的共同部分
    common_part = {k: v for k, v in dict_list[0].items() if k != list(dict_list[0])[-1]}
    # 合并最后一个键的数组
    last_key = list(dict_list[0])[-1]
    combined_last_values = []
    for d in dict_list:
        combined_last_values.extend(d[last_key])
    # 构建合并后的字典
    merged_dict = {**common_part, last_key: combined_last_values}
    # 将合并后的字典转换回 JSON 字符串
    return json.dumps(merged_dict, ensure_ascii=False)


# 合并头尾，弥合两个字符
def merge_head_and_tail_str_to_str(json_str1: str, json_str2: str) -> str:
    # 将 JSON 字符串转换为字典
    dict1 = json.loads(json_str1)
    dict2 = json.loads(json_str2)

    # 合并两个字典
    merged_dict = {**dict1, **dict2}

    # 将合并后的字典转换回 JSON 字符串
    return json.dumps(merged_dict, ensure_ascii=False)


def get_proposals(task, step: int, x: str, y: str) -> []:
    # 因为我们规定采用json格式传输，所以x y 拼一下，直接传给gpt生成的就是json，可以直接拼在一起作为下一次的y（实现y再循环中迭代）
    if task.env.name == "design":
        print("进入get——proposal并且进入design task的定制处理\n")
        propose_prompt = task.propose_prompt_wrap(step, x, y)
        proposals = gpt(propose_prompt, n=3, stop=None)
        # 将 y 从 JSON 字符串转换为 Python 对象
        # 检查字符串是否为空
        if step > 1:
            y_data = json.loads(y)
        else:
            y_data = {}  # 或者其他合适的默认值，如 None 或 []

        # 使用列表推导式创建一个新的列表，存储合并后的结果
        # dump 从字典到json（str）
        ys = [json.dumps({**y_data, **json.loads(proposal)}) for proposal in proposals]
        # 返回合并后的结果列表
        return ys

    # 【y】和【gpt生成proposal的集合中的元素】分别结合
    # 形成新的集合 并return
    propose_prompt = task.propose_prompt_wrap(step, x, y)
    # gpt 直接得到的是一个json（当前步骤的一些解法）
    proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')[0]
    proposals_dict = json.loads(proposals)
    y_dict = json.loads(y)
    merged_dict = {**y_dict, **proposals_dict}
    merged_y = json.dumps(merged_dict, ensure_ascii=False)
    return merged_y
    # 一个str（json）


def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    # if task.env.name == "design" --> 设定合并json 在输出
    if task.env.name == "design":
        y_data = json.loads(y)
        # 使用列表推导式创建一个新的列表，存储合并后的结果
        merged_results = [json.dumps({**y_data, **json.loads(proposal)}) for proposal in samples]
        return merged_results
    return [y + _ for _ in samples]


def get_selected_proposals(task, step, new_ys: str, x: str) -> str:
    print("现在在处理select工作，伴随而来的是一个新的节点生成\n")
    prompt_tmp = task.get_select_wrap(task, step, x, new_ys)
    former_part_json = get_former_part_str_in_str(new_ys)
    selected_new_y = gpt(prompt_tmp)[0]
    if former_part_json == "":
        return selected_new_y
    else:
        return merge_head_and_tail_str_to_str(former_part_json, selected_new_y)


def solve(args, task, idx, to_print=True):
    print("现在在处理solve前期工作\n")
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)
    ys = ''  # current output candidates
    infos = []

    for step in range(task.steps):
        # 下面是针对不需要筛选的项目 直接进行节点生成
        # if task.env.name == "design":
        # step0: Interact with Database to get competitors
        # 对x进行拓展，不对 ys 列表进行任何扩充

        if step == 0:
            # 1️⃣ step0 ： 仅对x做扩充，局部变量和类内变量都扩充
            print("现在在处理step0\n")
            task.handle4step0(task, x)
            # 👇 垃圾制造填满 infos ，用来输出日志
            ys = []
            new_ys = []
            values = []
            select_new_ys = []
            # 👆 垃圾制造停止符
            ys = select_new_ys
            x = task.get_input(idx)
            infos.append(
                {'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
            continue

        # 3️⃣ step4 最后处理 ： 信息整合
        if step == 4:
            x = task.get_input(idx)
            # step4 已经脱离了 bfs的逻辑，现在暂用来整合信息
            # 之后考虑用来调用stable diffusion
            results = [task.handle4step4(x, y) for y in ys]
            for y in ys:
                results += task.handle4step4(x, y)
            # 👇 垃圾制造填满 infos ，用来输出日志
            values = []
            new_ys = []
            select_new_ys = results
            # 👆 垃圾制造停止符
            infos.append(
                {'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys,
                 'results': results})
            # 其实这里break也行，跳出循环了
            print("下面将会直接打印三个全流程设计结果\n")
            result_json1 = json.dumps(results, ensure_ascii=False)
            result_json2 = json.dumps(results)
            print(result_json1)
            file_path_log1 = '/Users/kylinsong/Desktop/prompt/实验/BodyoidDesignAgentSystem/logs/design/task1output.json'
            file_path_log2 = '/Users/kylinsong/Desktop/prompt/实验/BodyoidDesignAgentSystem/logs/design/task1output.json'
            # Writing the JSON string to the file
            with open(file_path_log1, 'w') as file:
                file.write(result_json1)
            with open(file_path_log2, 'w') as file:
                file.write(result_json2)

            print("\n\n\n\n")
            return ys, {'steps': infos}

        # generation
        if args.method_generate == 'sample':
            new_ys = [
                get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step])
                for y in ys]
        if args.method_generate == 'propose':
            print("现在在处理step{step}的generation\n".format(step=step))
            # 对每一个ys中的y 平等的生成平等的节点 ——— new_y ，并形成 二阶数量的new_ys集合
            #  new_ys = (merge_similar_jsons_str_to_str([get_proposals(task, step, x, y) for y in ys]))
            if not ys:
                ys = [""]
            proposals = []
            for y in ys:
                proposals += get_proposals(task, step, x, y)
            new_ys = merge_similar_jsons_str_to_str(proposals)
            # new_ys = (merge_similar_jsons_str_to_str([get_proposals(task, step, x, y) for y in ys]))

        # evaluation  +  # selection for design
        print("现在在处理step{step}的select\n".format(step=step))
        if task.env.name == 'design':
            select_new_ys = [get_selected_proposals(task, step, new_ys, x) for i in range(3)]

        infos.append(
            {'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'select_new_ys': select_new_ys})
        ys = select_new_ys
        print("现在是{step}第步，ys是：".format(step=step))
        print(select_new_ys)

    if to_print:
        pprint.pprint(ys)

    # ys 是最终结果，最终输出
    # {'steps': infos} 是一组什么玩意儿，这是一个字典，它包含了解决过程的详细信息
    print("接下来是全部日志\n")
    return ys, {'steps': infos}


def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}
