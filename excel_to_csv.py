import pandas as pd
import numpy as np
import openai
import json
import time

# 设置您的 OpenAI API 密钥
openai.api_key = "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"


# 从 Excel 读取数据转换为 DataFrame
def read_excel_to_df(file_path):
    return pd.read_excel(file_path)


# 批量向量化，但是降低单次api数据量，增加调用次数
def batch_vectorize_to_df(df, batch_size=100):
    text_list = df.apply(lambda row: '; '.join([f"{col}:{row[col]}" for col in df.columns]),
                         axis=1).tolist()  # 将每一行转成一个text
    all_embeddings = []
    api_calls = 0
    for i in range(0, len(text_list), batch_size):
        api_calls += 1
        batch_texts = text_list[i:i + batch_size]
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=batch_texts
        )
        time.sleep(1)
        batch_embeddings = [item['embedding'] for item in response['data']]
        all_embeddings.extend(batch_embeddings)
    df['ada_embedding'] = all_embeddings
    return df, api_calls


# 主程序
def main():
    file_path = '//可穿戴数据库（主要信息）.xlsx'
    df = read_excel_to_df(file_path)
    df, api_calls = batch_vectorize_to_df(read_excel_to_df(file_path))

    # 将结果保存为 CSV 文件
    output_csv_path = '可穿戴数据库（向量版）.csv'
    df.to_csv(output_csv_path, index=False)
    print("全部完成，共调用成功 {} 次API".format(api_calls))


# 运行主程序
if __name__ == "__main__":
    main()

# 备注：因为采用了老版本的openai API 所以降低了版本
#  python -m pip install openai==0.28

# 一定要在虚拟环境下安装所需的包，所以运行时记得先进入虚拟环境
# conda activate myenv

# 安装时记得 -m 保证安装在虚拟环境中


# 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

# gpt4 : "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"
