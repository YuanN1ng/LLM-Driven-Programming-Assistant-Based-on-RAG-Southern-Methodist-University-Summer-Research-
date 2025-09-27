import pandas as pd

import cc_indexing




df = (
    pd.read_csv("data/benchmark.csv", encoding="utf-8-sig", usecols=['Query', 'Code'])  # 读两列
      .dropna(subset=['Query', 'Code'])        # ① 删掉 Query/Code 有缺失的行
      .astype(str)                             # ② 全部转成 str，防止 float nan
)

query_list = df['Query'].str.strip().tolist()
code_list  = df['Code'].str.strip().tolist()
# 构建向量库
cc_indexing.create_db_gemini(code_list, query_list)
cc_indexing.create_db_graphcodebert(code_list, query_list)
print("向量库已构建完毕！")







