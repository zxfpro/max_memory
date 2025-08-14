
import re
import json
def load_json(file_path = ""):
    with open(file_path,'r') as f:
        x = f.read()
        return json.loads(x)


def extract_python_code(text: str) -> str:
    """从文本中提取python代码
    Args:
        text (str): 输入的文本。
    Returns:
        str: 提取出的python文本
    """
    pattern = r"```json([\s\S]*?)```"
    matches = re.findall(pattern, text)
    if matches:
        return matches[0].strip()  # 添加strip()去除首尾空白符
    else:
        return ""  # 返回空字符串或抛出异常，此处返回空字符串

