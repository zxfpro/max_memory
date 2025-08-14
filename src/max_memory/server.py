
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from max_memory.memory import Memory
from max_memory.indexs import get_index
from max_memory.graphs import Graphs, DiGraphs
from max_memory.prompt import ptt, user_rule, data_struct
from max_memory.utils import extract_python_code
import uuid
import json
app = FastAPI(
    title="LLM Service",
    description="Provides an OpenAI-compatible API for custom large language models.",
    version="1.0.1",
)

origins = [
    "*", # Allows all origins (convenient for development, insecure for production)
    # Add the specific origin of your "别的调度" tool/frontend if known
    # e.g., "http://localhost:5173" for a typical Vite frontend dev server
    # e.g., "http://127.0.0.1:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Specifies the allowed origins
    allow_credentials=True, # Allows cookies/authorization headers
    allow_methods=["*"],    # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],    # Allows all headers (Content-Type, Authorization, etc.)
)


memory= Memory()
index = get_index(collection_name = 'test_1')
graph = Graphs("main_graph.pickle")
digraph = DiGraphs("main_degraph.pickle")


@app.get("/")
async def root():
    """ x """
    return {"message": "Service is running."}

@app.get(
    "/update_text",
    description="将聊天或者文本内容上传到记忆中",
    )
def update_text(text:str):
    ID_RANDOM_POOL = ",".join([str(uuid.uuid4())[:16] for i in range(100)])
    prompt_1 = ptt.format(
    user_rule = user_rule,
    data_struct = data_struct,
    ID_RANDOM_POOL = ID_RANDOM_POOL,
    )

    result_gener = memory.bx.product_stream(prompt_1 + "\n" + text)
    result = ""
    for result_i in result_gener:
        result += result_i

    result = extract_python_code(result)
    data_dict = json.loads(result)
    memory.update(index,graph,digraph,data_dict)
    # save 一下
    graph.save_graph()
    digraph.save_graph()

@app.get(
    "/show_graph",
    description="展示图",
    )
def show_graphs():
    graph.show_graph("main_graph.html")
    digraph.show_graph("main_degraph.html")


@app.get(
    "/build",
    description="搜索前的准备工作, 必须要build 完毕才能再构建",
    )
def build(similarity_top_k:int = 2, similarity_cutoff:float=0.8):

    memory.build(
            index,graph,digraph,
            similarity_top_k = similarity_top_k,
            similarity_cutoff=similarity_cutoff
                )
    # save 一下
    return {'message':'success'}


@app.get("/search",
    description="根据关键词搜索, 返回system prompt",
    )
def search(prompt_no_history:str,depth: int = 2)->str:

    system_prompt = memory.get_prompts_search(prompt_no_history = prompt_no_history,depth = depth)

    return system_prompt



@app.get("/update_session",
    description="在一个聊天session结束后 做一个后处理",
    )
def update_session(prompt_with_history:str)->str:
    try:
        memory.update_session(prompt_with_history = prompt_with_history)
    except Exception as e:
        return "failed"

    return "update_session successful"


@app.get("/thinking",
    description="思考, 整理脑中的信息, 针对一个主题进行集中式的优化",
    )
def thinking(topic:str)->str:
    try:
        memory.thinking(topic = topic)
    except Exception as e:
        return "failed"

    return "thingking builded"


if __name__ == "__main__":
    # 这是一个标准的 Python 入口点惯用法
    # 当脚本直接运行时 (__name__ == "__main__")，这里的代码会被执行
    # 当通过 python -m YourPackageName 执行 __main__.py 时，__name__ 也是 "__main__"
    import argparse
    import uvicorn
    from .log import Log

    parser = argparse.ArgumentParser(
        description="Start a simple HTTP server similar to http.server."
    )
    parser.add_argument(
        'port',
        metavar='PORT',
        type=int,
        nargs='?', # 端口是可选的
        default=8022,
        help='Specify alternate port [default: 8000]'
    )

    parser.add_argument(
        '--env',
        type=str,
        default='dev', # 默认是开发环境
        choices=['dev', 'prod'],
        help='Set the environment (dev or prod) [default: dev]'
    )

    args = parser.parse_args()

    port = args.port
    print(args.env)
    if args.env == "dev":
        port += 100
        Log.reset_level('debug',env = args.env)
        reload = True
        app_import_string = "src.max_memory.server:app" # <--- 关键修改：传递导入字符串
    elif args.env == "prod":
        Log.reset_level('info',env = args.env)# ['debug', 'info', 'warning', 'error', 'critical']
        reload = False
        app_import_string = app
    else:
        reload = False
        app_import_string = app

    # 使用 uvicorn.run() 来启动服务器
    # 参数对应于命令行选项
    uvicorn.run(
        app_import_string,
        # app, # 要加载的应用，格式是 "module_name:variable_name"
        host="0.0.0.0",
        port=port,
        reload=reload  # 启用热重载
    )
