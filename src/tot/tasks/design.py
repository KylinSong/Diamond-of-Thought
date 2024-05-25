import json
import os
import pandas as pd
from tot.models import gpt
import numpy as np  # for cosine_similarity —— 向量相似度计算（余弦值计算）
import re
from tot.tasks.base import Task
from tot.prompts.design import *  # 假设prompt模板定义在这里
import openai
import ast


# 计算余弦相似度
def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))


def search_in_df(input_str, top_n=3):
    # 读取本地向量数据库（csv）
    path_vectorized_database_csv = '/Users/kylinsong/Desktop/prompt/实验/BodyoidDesignAgentSystem/可穿戴数据库（向量版）.csv'
    # 输出: my_csv_db 是一个 pandas DataFrame
    my_csv_db = pd.read_csv(path_vectorized_database_csv)

    # 使用 OpenAI 的 Embedding API 获取输入字符串的向量，格式为字典dict
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=[input_str]
    )

    # 取字典的第一条，毕竟只有一条
    # 输出: embedding 是一个列表 (list) ，包含数字（嵌入向量的元素）
    embedding = response['data'][0]['embedding']

    # 转换字符串形式的向量为数值列表
    def string_to_list(string):
        return np.fromstring(string.strip('[]'), sep=', ')

    # 计算相似度并返回最相似的项
    # 这里使用的 lambda 函数会将 my_csv_db DataFrame 中的 'ada_embedding' 列中的每个元素（应为列表）
    # 与刚刚获得的 embedding 向量进行余弦相似度计算
    # 输出: my_csv_db['similarities'] 是一个 pandas Series，包含计算得到的相似度分数
    my_csv_db['similarities'] = my_csv_db['ada_embedding'].apply(
        lambda x: cosine_similarity(string_to_list(x), embedding))

    # 对 DataFrame 根据相似度排序并获取前 top_n 个条目
    # 输出: 返回的是一个 pandas DataFrame，包含了排序后的前 top_n 个最相似的条目
    return my_csv_db.sort_values('similarities', ascending=False).head(top_n)


class DesignEnv:
    def __init__(self, file='designTask4.json'):
        # DONE
        # json文本input预处理
        file_path = os.path.join('/Users/kylinsong/Desktop/prompt/实验/BodyoidDesignAgentSystem/src/tot/data/design',
                                 file)

        with open(file_path, 'r') as json_file:
            self.raw_data = json.load(json_file)

        # 长度 不一定会用到( len_y 可以用来判定层级）
        def __len__(self) -> int:
            return len(self.data)

        self.name = "design"

        # 创建数据结构，存放中间数据
        self.data = {
            "background": self.raw_data.get('background'),
            "target_audience": self.raw_data.get('target_audience'),
            "target_problems": self.raw_data.get('target_problems'),
            # "competitive_analysis": None,  # 如果这里也有 raw_data 对应的值，可以同样处理
            # "requirements": [],
            # "functions": [],
            # "technologies": [],
            # "final_plan": None,
            # "competitors": []  # 假设这里没有对应的 raw_data 值
        }
        self.data_x = self.data
        # 状态码 ( 0 ~ 5 )
        self.status = -1


class DesignTask(Task):
    def __init__(self,
                 file='/Users/kylinsong/Desktop/prompt/实验/BodyoidDesignAgentSystem/src/tot/data/design/designTask4.json'):
        super().__init__()
        self.env = DesignEnv(file)
        self.steps = 5  # 根据设计任务的复杂性设定步骤数

    def __len__(self) -> int:
        return len(self.env.data)

    def get_input(self, idx: int) -> str:
        # 这里有个很重要的问题，因为x在solve中是不随循环变化的，所以只提取已有的部分
        # 所以设计 data_x = data ，作为保险
        extracted_data = self.env.data_x
        # 将提取的数据转换为 JSON 字符串
        return json.dumps(extracted_data, ensure_ascii=False)  # 使用 indent 参数美化输出： 每层会有4格缩进

    def test_output(self, idx: int, output: str):
        # 测试输出的有效性并更新环境状态
        # 用正则表达式检验是否data的所有个子项都有内容
        test_output = gpt(test_output_prompt)
        # 准备x
        input_x_json = json.dumps(self.env.data, ensure_ascii=False)
        res = test_output[0].format(method=test_output_method_prompt, eg=test_output_eg_prompt, input_x=input_x_json,
                                    input_y=output)
        return {'r', res}

    @staticmethod
    def standard_prompt_wrap(x: str, y: str = '') -> str:
        return propose_prompt.format(input_x=x, input_y=y)

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = '') -> str:
        return propose_prompt.format(input_x=x, input_y=y)

    @staticmethod
    def propose_prompt_wrap(step: int, x: str, y: str = '') -> str:
        # 生成提议的提示词：注意！这里的return将直接作为gpt的输入
        # DONE
        if step == 1:
            y = "暂无"
            # len(json.loads(y)) == 0:
            print("现在在处理step1的propose\n")
            return propose_prompt.format(eg=propose_prompt_eg_requirements, input_x=x, input_y=y,
                                         input_tmp_task="requirements", method=propose_method_requirements)
        elif step == 2:
            print("现在在处理step2的propose\n")
            return propose_prompt.format(eg=propose_prompt_eg_product_definition, input_x=x, input_y=y,
                                         input_tmp_task="product_definition", method=propose_method_product_definition)
        elif step == 3:
            print("现在在处理step3的propose\n")
            return propose_prompt.format(eg=propose_prompt_eg_technologies, input_x=x, input_y=y,
                                         input_tmp_task="technologies", method=propose_method_technologies)
        return "propose_prompt_wrap - error"

    # 这一步的gpt()之后会导致y从无到有，len(y) ++
    @staticmethod
    def handle4step0(self, x):
        # 调用GPT4，让GPT给出产品名
        tmp_prompt = fuzzy_product_name_prompt.format(input=x)
        predicted_product_name = gpt(tmp_prompt)[0]
        print(
            "predicted_product_name = {predicted_product_name}\n".format(predicted_product_name=predicted_product_name))

        # 整个匹配工作封装出去，return的是dataFrame
        similar_res = search_in_df(predicted_product_name, 3)

        # 将取回的data frame行进行字符串化
        competitors = []
        for index, row in similar_res.iterrows():
            text_str = '; '.join(
                [f"{col}:{row[col]}" for col in row.index if col not in ['ada_embedding', 'similarities']])
            competitors.append(text_str)
        self.env.data_x["competitors"] = competitors

    @staticmethod
    def handle4step4(x: str, y: str) -> str:
        # 通过替换可变信息，更新、格式化设计说明提示词
        design_description = gpt(description_prompt.format(input_x=x, input_y=y, eg=description_eg))[0]
        # y转化成为字典类型的y_dict
        y_dict = json.loads(y)
        # 字典添加一堆key-value
        y_dict["design_description"] = design_description
        # 将总字典（已更新）y_dict转化成str（json）并输出，记得格式化（缩进4）
        return json.dumps(y_dict, ensure_ascii=False)

    @staticmethod
    def get_select_wrap(self, step: int, x, merged_ys: str) -> str:
        # x 是json（str） merged_ys是存放了很多json的集合 (wrong？)
        y_dict = json.loads(merged_ys)
        # str --> dict

        print("y_dict:\n")
        print(y_dict)

        if step == 1:
            return select_prompt.format(eg=select_prompt_eg_requirements, input_x=x, input_y1="暂无，当前为设计第一步",
                                        input_y2=merged_ys, input_tmp_task="requirements",
                                        method=select_method_requirements)
        elif step == 2:
            # 提取最后一个键
            last_key = list(y_dict.keys())[-1]
            # 分割字典，前半部分是除了最后一个键的其他部分
            first_part = {k: v for k, v in y_dict.items() if k != last_key}
            second_part = y_dict[last_key]
            y_former_json = json.dumps(first_part, ensure_ascii=False)
            y_later_json = json.dumps(second_part, ensure_ascii=False)
            return select_prompt.format(eg=select_prompt_eg_product_definition, input_x=x, input_y1=y_former_json,
                                        input_y2=y_later_json, input_tmp_task="product_definition",
                                        method=select_method_product_definition)
        elif step == 3:
            # 提取最后一个键
            last_key = list(y_dict.keys())[-1]
            # 分割字典，前半部分是除了最后一个键的其他部分
            first_part = {k: v for k, v in y_dict.items() if k != last_key}
            second_part = y_dict[last_key]
            y_former_json = json.dumps(first_part, ensure_ascii=False)
            y_later_json = json.dumps(second_part, ensure_ascii=False)
            return select_prompt.format(eg=select_prompt_eg_technologies, input_x=x, input_y1=y_former_json,
                                        input_y2=y_later_json, input_tmp_task="technologies",
                                        method=select_method_technologies)
        else:
            return "something wrong with the step 'get_select_wrap' "
