import csv


def remove_odd_rows(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        lines = list(reader)

    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        for i, row in enumerate(lines):
            # 保留前两行和偶数行
            if i < 2 or i % 2 == 0:
                writer.writerow(row)


# 使用函数
input_csv = '/Users/kylinsong/Desktop/prompt/实验/BodyoidDesignAgentSystem/可穿戴数据库（向量版）.csv'  # 替换为你的原始 CSV 文件路径
output_file = '//可穿戴数据库（单行向量版）.csv'  # 替换为你希望保存结果的新文件路径
remove_odd_rows(input_csv, output_file)
