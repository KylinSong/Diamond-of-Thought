import json
import itertools
import numpy as np
from functools import partial
from tot.models import gpt
import pprint
import os


def get_former_part_str_in_str(y: str) -> str:
    y_dict = json.loads(y)
    if len(y_dict) <= 1:
        return ""
    else:
        y_dict.popitem()
        return json.dumps(y_dict, ensure_ascii=False)


def my_switch_case(state: int) -> str:
    if state == 0:
        return "[0]原始数据"
    elif state == 1:
        return "[1]竞品分析"
    elif state == 2:
        return "[2]需求分析"
    elif state == 3:
        return "[3]功能定义"
    elif state == 4:
        return "[4]技术方案"
    elif state == 5:
        return "[5]设计说明"
    else:
        return "出错了，不想写异常，自己查吧"


# 写入公共内存 Tree
# 输入 state 和 字典
def writer(state: int, dict_in: []):
    tree_abs_path = "/Users/kylinsong/Desktop/prompt/实验/BodyoidDesignAgentSystem/src/tot/methods/Tree.json"
    # 检查文件是否为空
    if os.path.getsize(tree_abs_path) == 0:
        dict_tree = {}  # 如果文件为空，则初始化为空字典
    else:
        with open(tree_abs_path, 'r') as tree:
            dict_tree = json.load(tree)  # 如果文件不为空，则正常加载数据
    key_tmp = my_switch_case(state)
    dict_tree[key_tmp] = dict_in
    with open(tree_abs_path, 'w') as tree:
        json.dump(dict_tree, tree, indent=4, ensure_ascii=False)
    print("共享内存已记录：{x}\n".format(x=key_tmp))


# 额外处理子结构套字典的结构
def custom_merge_dicts(dict_list):
    # 初始化一个空字典用于存放最终合并的结果
    merged = {}
    # 遍历每个字典
    for d in dict_list:
        for key, value in d.items():
            # 检查当前键是否已经存在于合并后的字典中
            if key in merged:
                # 如果当前键的值是字典，并且已合并的字典中相应的键的值也是字典，则将它们放入列表中
                if isinstance(value, dict) and isinstance(merged[key], dict):
                    # 如果已经是列表，则直接添加
                    if isinstance(merged[key], list):
                        merged[key].append(value)
                    else:
                        # 否则，创建一个新列表
                        merged[key] = [merged[key], value]
                else:
                    # 对于非字典类型，或者已处理为列表的情况，只需保留原始逻辑
                    merged[key] = value
            else:
                # 如果当前键在合并后的字典中不存在，直接添加
                merged[key] = value
    return merged


# 合并ys内部所有y
# 将前面的共用，当前的new_ys拆开揉碎，放在一个集合里。
# 输出为json
def merge_similar_jsons_str_to_str(ys: []) -> str:
    # 如果输入列表为空，返回空字典
    if not ys:
        return "{}"

    # 首先将JSON字符串列表转换为字典列表
    dict_list = [json.loads(y) for y in ys]

    # 使用第一个字典来初始化结果字典，对于最后一个键，初始化为空列表
    merged = {key: [] if key == list(dict_list[0].keys())[-1] else dict_list[0][key] for key in dict_list[0]}

    # 最后一个键的名称
    last_key = list(merged.keys())[-1]

    # 遍历字典列表
    for d in dict_list:
        # 将最后一个键的值加入到列表中
        merged[last_key].append(d[last_key])
        # 对于其他键，假设它们在所有字典中的值都相同，所以不需要额外操作

    # 将合并后的字典转换回JSON字符串
    return json.dumps(merged, ensure_ascii=False)


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
        ys = [json.dumps({**y_data, **json.loads(proposal)}, ensure_ascii=False) for proposal in proposals]
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
        merged_results = [json.dumps({**y_data, **json.loads(proposal)}, ensure_ascii=False) for proposal in samples]
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

    # 0 原始写入
    dict_x = json.loads(x)
    writer(0, dict_x)

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

            dict_x = json.loads(x)
            dict_competitor = {"competitors": dict_x["competitors"]}
            writer(1, dict_competitor)

            infos.append(
                {'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
            continue

        # 3️⃣ step4 最后处理 ： 信息整合
        if step == 4:
            x = task.get_input(idx)
            # step4 已经脱离了 bfs的逻辑，现在暂用来整合信息
            # 之后考虑用来调用stable diffusion
            results = task.handle4step4(x, ys)

            dict_result = json.loads(results)
            dict_description = {"design_description": dict_result["design_description"]}
            writer(5, dict_description)

            # for y in ys:
            # results += task.handle4step4(x, y)
            # 👇 垃圾制造填满 infos ，用来输出日志
            values = []
            new_ys = []
            select_new_ys = results
            # 👆 垃圾制造停止符
            infos.append(
                {'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys,
                 'results': results})
            # 其实这里break也行，跳出循环了
            print("下面将会直接打印全流程设计结果\n")
            result_json1 = json.dumps(results, ensure_ascii=False)
            result_json2 = json.dumps(results, ensure_ascii=False)
            print(result_json1)
            file_path_log1 = '//logs/design/task4output.json'
            file_path_log2 = '//logs/design/json_format_test4.json'
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
                ys = ""
            proposals = get_proposals(task, step, x, ys)
            # debug
            print("proposals:\n")
            print(proposals)

            if step == 2:
                # new_ys = merge4step2(proposals)
                new_ys = merge_similar_jsons_str_to_str(proposals)
            else:
                new_ys = merge_similar_jsons_str_to_str(proposals)
            # new_ys = (merge_similar_jsons_str_to_str([get_proposals(task, step, x, y) for y in ys]))
            print("new_ys: \n")
            print(new_ys)

        # evaluation  +  # selection for design
        print("现在在处理step{step}的select\n".format(step=step))
        if task.env.name == 'design':
            # select_new_ys = [get_selected_proposals(task, step, new_ys, x) for i in range(3)]
            select_new_ys = get_selected_proposals(task, step, new_ys, x)
        infos.append(
            {'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'select_new_ys': select_new_ys})

        dict_selected_new_ys = json.loads(select_new_ys)
        state = step + 1
        key_circle = 0
        if state == 2:
            key_circle = "requirements"
        elif state == 3:
            key_circle = "product_definition"
        elif state == 4:
            key_circle = "technologies"

        dict_circle_selection = {key_circle: dict_selected_new_ys[key_circle]}

        print("new_ys:\n")
        print(new_ys)
        print("type of new_ys:\n")
        print(type(new_ys))

        if new_ys == [''] or not new_ys:
            dict_circle_generation = {}
        else:
            print("(json.loads(new_ys))[key_circle]:\n")
            print((json.loads(new_ys))[key_circle])
            if step == 2:
                dict_circle_generation = {key_circle: (json.loads(new_ys))[key_circle]}
            else:
                dict_circle_generation = {key_circle: (json.loads(new_ys))[key_circle]}

        dict_circle = {"generation": dict_circle_generation, "selection": dict_circle_selection}
        writer(state, dict_circle)

        ys = select_new_ys
        print("现在是{step}第步，ys是：".format(step=step))
        print(select_new_ys)

    if to_print:
        pprint.pprint(new_ys)

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
