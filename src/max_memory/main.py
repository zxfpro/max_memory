'''
Author: 823042332@qq.com 823042332@qq.com
Date: 2025-08-12 10:42:37
LastEditors: 823042332@qq.com 823042332@qq.com
LastEditTime: 2025-08-12 14:35:12
FilePath: /max_memory/src/max_memory/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#1
import uuid

## ID_RANDOM_POOL

ID_RANDOM_POOL = [str(uuid.uuid4())[:16] for i in range(100)]

