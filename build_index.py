# build_index.py
import pathlib, c_indexing

txt = pathlib.Path("data/codes_top50.txt")
c_indexing.create_db_gemini(txt)
c_indexing.create_db_graphcodebert(txt)
print("向量库已构建完毕！")
