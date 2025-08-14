from max_memory.graphs import Graphs,DiGraphs, Entity_Graph, Event_Graph
from max_memory.utils import extract_python_code
import json

import uuid
from llmada.core import BianXieAdapter
import json
import re


class Memory():
    def __init__(self):
        bx = BianXieAdapter()
        model_name = "gemini-2.5-flash-preview-05-20-nothinking"
        bx.model_pool.append(model_name)
        bx.set_model(model_name=model_name)
        self.bx = bx

        
        # 历史的记忆
        self.entity_graph = Entity_Graph()
        self.event_graph = Event_Graph()
        # 临时的记录
        self.G = Graphs("temp_g.pickle")
        self.DG = DiGraphs("temp_dg.pickle")
    
    def update(self,index, graph, digraph, data_dict):
        self.entity_graph.update(index,graph,data_dict)
        self.event_graph.update(index,digraph,data_dict)
        
    def build(self, index, graph, digraph, similarity_top_k: int = 2, similarity_cutoff=0.8):
        self.entity_graph.build(index,graph,similarity_top_k = similarity_top_k, similarity_cutoff=similarity_cutoff)
        self.event_graph.build(index,digraph,similarity_top_k = similarity_top_k, similarity_cutoff=similarity_cutoff)


    def resolve(self,text):
        prompt_22 = """
        用户会与你聊天, 从用户的话语中有特定含义的实体 与主要events(主要事件应该是一个简单句)

        输入为json
        ```json
        {"entities":[<string>],
        "events":[<string>]
        }
        ```
        user: """
        result = self.bx.product(prompt_22 + text)
        result = extract_python_code(result)
        return json.loads(result)
    
    def get_prompts_search(self,prompt_no_history,depth = 2):
        # Talk  search
        # 得到system_prompt  之后使用 加上 chat_history
        # 分解 
        message_dict = self.resolve(prompt_no_history)
        # {'entities': ['软件系统'], 'events': ['用户想了解软件系统']}
        
        # search()
        entity_prompt = self.entity_graph.search(message_dict.get("entities"),depth,output_type = "prompt")
        event_prompt = self.event_graph.search(message_dict.get("events"),depth,output_type = "prompt")
        
        system_prompt = f'''
你是一个聊天机器人, 相比你的大模型记忆来说, 下面的事件和概念陈述更加重要.

{event_prompt}
{entity_prompt}
'''
        entity_graph_entity = self.entity_graph.search(message_dict.get("entities"),depth,output_type = "entity")
        event_graph_entity = self.event_graph.search(message_dict.get("events"),depth,output_type = "entity")
        
        # 存储检索到的数据,并合并
        self.G = Graphs()# +
        self.DG = DiGraphs()# +
        
        return system_prompt
    
    def update_session(self, prompt_with_history):
        
        # 1 chat_history 生成return2_2
        prompt_1 = '\n**角色:**你是一名高效、精确的信息抽取与知识图谱构建专家。你的任务是从用户提供的文章中,按照严格定义的结构化要求,抽取并组织事件信息。\n\n**目标:**产出符合特定JSON Schema的事件知识图谱,该图谱包含:\n\n1. **细粒度的基本事件列表 (basic_events)**:作为知识图谱的原子节点。\n2. **优先基于文章分级标题的结构化关系图 (event_relations_graph)**:描述事件间的层次和关系, 次之就自己总结层次关系。\n---\n\n# 延展规则:\n\n\n\n1 提取的粒度要收敛, 提取的内容重点集中在"最具备价值"这点上,\n\n何为价值, 有如下定义:\n1 新概念,核心概念,对应解释\n2 主要原理与论点\n3 相对肯定的统计性规律和结论\n4 作者重点强调的相当肯定的句子\n\n总之就是, 可以带来新认知的内容.\n\n\n---\n\n**输出JSON Schema:**\n\n请严格按照以下JSON结构输出,不得有任何额外内容或偏差。\n\n\n```json\n{\n  "entities": [ // 文章中的核心对象, 要求相同概念合并, 如 人工智能与AI\n  {\n   "id": "string", // 唯一标识符,从 ID_RANDOM_POOL 中抽取\n   "name": "string", // 核心对象\n   "aliases": ["string"] // 存放同一概念的别名\n   "describe": ["string","string"] // 对核心对象的描述\n  }],\n  "entities_relations": [ // 从文章中抽取出的entities之间的关系\n   {\n  "subject_id": "string", // 发出对象的唯一标识符,与entities对应\n  "relation": "MANIFESTS_AS", // 关系类型\n  "object_id": "获取知识和信息的方式", // 接收对象的唯一标识符,与entities对应\n  }],\n "events": [ // 细粒度的基本事件列表,每个事件都是一个原子信息单元\n  {\n   "id": "string", // 唯一标识符,从 ID_RANDOM_POOL 中抽取\n   "name": "string", // 抽取出的内容的抽象,减少非必要的形容词。移除所有特殊字符（如Markdown语法、图片链接等）,只保留文本和基本标点。 注意与entities的概念统一, 处理好aliases, 同时代词要替换为明确的词\n   "involved_entities": ["string", "string"] // 列表,包含事件中交互的1到2个核心对象。核心对象应该与entities的概念统一, 如果句子无法识别出1个或2个有代表性的实体,则该字段留空 []。\n   "describe": ["string","string"] // 对该事件的描述\n  }],\n  "events_relations": [ // 细粒度的基本事件列表,每个事件都是一个原子信息单元\n  {\n  "subject_id": "string", // 发出对象的唯一标识符,与events对应  \n  "sub_events_id": ["string"] // 列表,子事件的id列表 子事件的定义很广, 按照逻辑, 可以归类到某一事件的事件都属于其子事件\n  }],\n```\n\n---\n\n关系抽取 (RELATION_EXTRACTION) (利用马克思主义启发的分类法)**\n  *   **输入 (INPUT):** `CONSOLIDATED_ENTITIES`（来自阶段1）。\n  *   **逻辑 (LOGIC):**\n      1.  定义 `马克思主义知识图谱关系分类法 (MARXIST_KG_RELATION_TAXONOMY)`:\n          *   **本质-现象/具象化 (ESSENCE_OF / MANIFESTS_AS):** (X, MANIFESTS_AS, Y) / (Y, IS_ESSENCE_OF, X)。语义：X 是 Y 的具体表达/实例；Y 是 X 的根本性质/抽象概念。关键词："是", "一种", "系统", "方法", "表现", "本质"。\n          *   **内容-形式/构成 (CONTAINS / IS_COMPONENT_OF):** (X, CONTAINS, Y) / (Y, IS_COMPONENT_OF, X)。语义：X 包含 Y；Y 是 X 的一部分/要素。关键词："包含", "由...构成", "存在", "具有"。\n          *   **矛盾-对立/互补 (CONTRADICTS / COMPLEMENTS):** (X, CONTRADICTS, Y) / (X, COMPLEMENTS, Y)。语义：X 与 Y 对立/不同；X 与 Y 协同/增强。关键词："与...不同", "对立", "互补", "协同"。\n          *   **原因-结果/目的 (CAUSES / RESULTS_IN / AIMS_TO_ACHIEVE):** (X, CAUSES, Y) / (X, RESULTS_IN, Y) / (X, AIMS_TO_ACHIEVE, Y)。语义：X 导致 Y；X 产生 Y；X 的目的是 Y。关键词："导致", "产生", "用于", "旨在", "实现", "为了"。\n          *   **实践-理论/应用 (APPLIES_TO / IS_APPLIED_BY / GUIDES):** (X, APPLIES_TO, Y) / (X, IS_APPLIED_BY, Y) / (X, GUIDES, Y)。语义：X 在 Y 中应用/相关；Y 使用 X；X 为 Y 提供指导。关键词："应用于", "使用", "指导", "针对"。\n          *   **发展-历史/创造 (DEVELOPED_BY / PROPOSED_IN / DISCOVERED_IN):** (X, DEVELOPED_BY, Y) / (X, PROPOSED_IN, Y) / (X, DISCOVERED_IN, Y)。语义：X 的起源/创造；X 在 Y 时间被提出/发现。关键词："由...开发", "提出", "发现", "于...年"。\n      2.  **关系抽取策略:**\n          a.  遍历 `CONSOLIDATED_ENTITIES` 中的每个 `实体 (entity)`。\n          b.  对于每个 `实体`，分析其 `describe` 列表。\n          c.  对于 `entity.describe` 中的每个 `描述字符串 (description_string)`:\n              i.  执行关键词匹配和浅层语义解析（例如，如果可用，进行依存句法分析，否则使用基于规则的模式匹配）以识别潜在的主语-关系-宾语三元组。\n              ii. **主语 (Subject):** 默认设置为当前 `entity.name`。\n              iii. **宾语 (Object):** 尝试在 `描述字符串` 中识别其他 `CONSOLIDATED_ENTITIES.name` 或 `CONSOLIDATED_ENTITIES.aliases`。如果未找到其他实体，则从描述中提取与关系类型对齐的重要短语或概念。\n              iv. **关系 (Relation):** 将识别到的关键词/模式映射到 `MARXIST_KG_RELATION_TAXONOMY`。优先考虑特异性匹配。通过推断处理双向关系（例如，如果 X MANIFESTS_AS Y，则推断 Y IS_ESSENCE_OF X，除非已明确抽取）。\n              v.  确保抽取的三元组的唯一性（主语、关系、宾语的组合）。\n          d.  **三元组结构:** 每个三元组将是一个字典：`{"subject": "实体名称A", "relation": "关系类型", "object": "实体名称B_或_概念", "description": "支持该关系的原始文本片段"}`。\n  *   **输出 (OUTPUT):** `EXTRACTED_RELATIONS`: 结构化关系三元组字典列表。\n\n---\nID_RANDOM_POOL\n\nc3093f8c-efbb-42,7dff62b2-d811-46,2ba39efe-04ca-42,8c34a179-8d31-44,43c047bf-f4da-42,89630cfe-1967-42,f3cedf55-0a3d-46,dc2779b1-04de-4c,21e77936-aa6b-46,8bfcfeb6-f6df-4c,b641ff47-20f0-49,3893542e-7be7-4b,73216785-bdc5-41,82817a9a-7037-4a,330f1c6c-77b5-4b,06943ece-cf88-47,a6693478-2b62-46,4d2a457e-293d-48,16535773-0b9a-4f,500eab24-1b79-46,9ec072b4-4ff6-4f,0e0d955e-d2f3-4a,47b37f6a-f067-40,6d4b204b-0948-46,25dfdc98-bd28-43,5d39a546-2c68-47,5dde459f-f5fb-4d,015007eb-a06a-4d,3f5fd3ae-add6-4e,0c5628c1-16ed-4e,c78c6fe6-e9da-4d,a641bb2a-7290-40,09edb3b3-1d9e-45,095616c0-a7e1-47,61441e18-6900-42,d7b5bab9-b127-40,c142373c-8082-43,0742e741-a9d8-45,d5f896fd-a3e8-41,cab237c1-4723-42,1c1b6b58-c78f-4c,86326cc4-ee46-4e,e89711bb-7d8c-46,15cefc12-2b0e-47,55dfe099-6771-42,4b5c934d-4677-4e,9dc90e57-b407-4d,c295b15c-1023-4e,86047024-b2f1-47,5a581ca9-125c-42,8e234902-e48e-41,c0c81f77-2c90-4b,2fbe06c4-db50-4c,7ba03456-89b2-4a,f729e5cb-7d44-46,e2b2bfb5-4993-47,3b4ff42b-e55f-49,4127f82f-974d-47,fbbf019f-4be6-4f,3d83db65-0800-47,f928c1fe-6ae6-47,ff4c4ca8-e4cf-4b,d3002b65-7509-44,228535f7-8eee-4a,c4f8a5c8-0e88-46,96bcd276-e102-4c,cf447bb2-224a-46,4a0a8e89-901f-4b,d4d53f3d-98e1-46,60cfd24d-77c9-4c,47d68846-a3c8-4f,51e46303-d84a-4b,c9efd684-f319-46,9a530ae2-5cd3-4a,425b2f8e-6395-40,5d5491e5-1301-47,8650cd2c-40d8-44,fa5d5d3a-51ac-4b,440fbe23-615d-40,e779fb75-440e-49,7afd245a-f768-4d,8906887f-f780-47,b91e52d1-8e5d-40,9cd58cf3-8ac9-42,d692fc5c-37f6-4f,36b58af4-4a66-49,92dcff23-f4c1-45,6623bb0f-dd77-4b,6b605700-20af-41,00c39bdd-05a5-4b,e6235665-5e58-4c,66e9a2e5-5b8d-4c,ecc0ec4c-b737-4e,1e125d0a-8298-4a,275e0399-9357-4a,9efe382d-745a-44,a259e2b3-37e1-4c,81b03c23-3927-4b,d211b6d3-8081-4b,aeffca7d-6c5a-44\n\n---\n\n**自省与验证（内部检查机制,请严格遵循）:**\n\n  \n*  **格式检查**:输出的JSON是否完全符合定义的Schema?所有字段名、类型、列表结构是否正确?\n\n*  **特殊字符检查**:`entities.name`, `events.name`, 字段是否已完全清除特殊字符和Markdown语法?\n\n*  **ID一致性**:所有 `id` 字段是否唯一?`entities_relations` 中的 `subject_id` 和 `object_id` 是否都指向 `entities` 中存在的ID? `sub_events_id` 是否都指向 `events`中存在的ID?\n\n*  **作者结构**:检查是否有作者的结构信息?\n\n*  **实体一致性**:\n\n  *  `events` 中的 `involved_entities` 在整个文档中是否保持一致?是否尽可能地具体化了通用词汇?**元素数量是否严格不超过2个?**\n\n  *  所有的 `involved_entities` 中的元素是否都来自于entities?  aliases 的是否进行了统一?\n  \n*  **关系语义**:每个 `relation` 的选择是否准确反映了 `subject_id` 和 `object_id` 之间的语义关系?\n\n*  **覆盖度**:文章的关键信息点是否都被提取为基本事件?\n'

        result_gener = self.bx.product_stream(prompt_1 + "\n" + prompt_with_history)
        result = ""
        for result_i in result_gener:
            result += result_i

        result = extract_python_code(result)
        result_2_2 = json.loads(result)
        
        # 2 利用return2_2 制作Graphs1 和 DiGraphs1
        
        ent_graph_1 = Entity_Graph()
        evt_graph_1 = Event_Graph()
        g_1 = Graphs()
        dg_1 = DiGraphs()
        
        ent_graph_1.update(index,g_1,result_2_2)
        evt_graph_1.update(index,dg_1,result_2_2)
        
        
        # 3 将Graphs1 he DiGraphs1 上传合并到 主的 Graphs
        # trans_dict 大模型融合
        
        self.G.merge_other_graph(g_1,trans_dict)
        self.DG.merge_other_graph(dg_1,trans_dict_2)
        
        
        
        # 4 将 self.entity_graph = Entity_Graph()
            # self.event_graph = Event_Graph()
            # 上传合并到 主的 Graphs
            
    
    def thinking(self,topic):
        # 主脑自主决策
        # TODO 深度搜寻, 广度搜索, 或者更自由的搜寻, 或者强化学习 GNN
        pass
