'''
Author: 823042332@qq.com 823042332@qq.com
Date: 2025-08-01 14:31:16
LastEditors: 823042332@qq.com 823042332@qq.com
LastEditTime: 2025-08-07 16:54:44
FilePath: /max_memory/src/max_memory/core.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# core 
from pyvis.network import Network
import networkx as nx
import re
import uuid
import json
from llama_index.core import VectorStoreIndex
from llama_index.core import Document
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from llama_index.core.vector_stores import FilterOperator, FilterCondition
from llama_index.core.postprocessor import SimilarityPostprocessor


import json

# L2
# class Entity:
#     """
#     链表中的一个节点。
#     每个节点包含一个值和指向下一个节点的引用。
#     """
#     def __init__(self, data):
#         self.data = data  # 节点存储的原始数据字典
#         self.id = data.get('id')
#         self.name = data.get('name')
#         self.aliases = data.get("aliases")
#         self.describe = ';'.join(data.get('describe', []))
#         # self.next 现在存放的是其他 Entity 的 ID 字符串列表
#         # 初始值通常应该从 data 中获取，如果 data 中没有，则为空列表
#         self.next = data.get('next', []) # 假设原始数据中next就是ID列表

#     def __repr__(self):
#         """用于调试，方便打印节点信息"""
#         # 由于 next 是 ID 字符串列表，直接打印即可
#         return (f"Entity(id='{self.id}', name='{self.name}', "
#                 f"aliases={self.aliases}, describe='{self.describe}', "
#                 f"next_ids={self.next})")

#     def __str__(self):
#         return self.__repr__()

#     def to_dict(self):
#         """
#         将 Entity 实例转换为一个 Python 字典。
#         next 属性直接作为 ID 字符串列表存储。
#         """
#         entity_dict = {
#             "id": self.id,
#             "name": self.name,
#             "aliases": self.aliases,
#             "describe": self.describe,
#             # 将 next 属性直接作为 ID 列表添加到字典中
#             "next": self.next,
#             # 可以选择性地保留原始的 data 字段，如果它包含额外的信息
#             "data": self.data
#         }
#         return entity_dict

#     @classmethod
#     def from_dict(cls, data_dict):
#         """
#         从一个字典创建 Entity 实例。
#         next 属性直接从字典中获取 ID 字符串列表。
#         :param data_dict: 包含 Entity 数据的字典。
#         :return: Entity 实例。
#         """
#         # 首先，我们需要从字典中还原原始的data字段，或者直接构造一个用于初始化的字典
#         # 如果 to_dict 中包含了 'data' 字段，我们优先使用它作为原始 data
#         original_data_for_init = data_dict.get('data', {})

#         # 如果没有 'data' 字段，或者 'data' 字段是空的，那么从传入的 data_dict 中构建
#         if not original_data_for_init:
#             original_data_for_init = {
#                 'id': data_dict.get('id'),
#                 'name': data_dict.get('name'),
#                 'aliases': data_dict.get('aliases'),
#                 # 如果describe是字符串，需要将其还原为列表以便于原始数据结构
#                 'describe': data_dict.get('describe').split(';') if data_dict.get('describe') else [],
#                 'next': data_dict.get('next', []) # 重要的是将 next 也还原到原始 data
#             }

#         # 创建 Entity 实例
#         # 注意：这里我们假设 Entity 的 __init__ 可以直接处理包含 'next' 键的 data 字典
#         # 如果你的 Entity.__init__ 不会从 data 中自动设置 self.next，
#         # 则需要在 Entity 实例创建后手动设置：entity_instance.next = data_dict.get('next', [])
#         entity_instance = cls(original_data_for_init)

#         # 确保 next 属性被正确设置，以防原始 data 中没有
#         if 'next' in data_dict and isinstance(data_dict['next'], list):
#             entity_instance.next = data_dict['next']
#         else:
#             entity_instance.next = [] # 确保即使没有next字段，也初始化为空列表

#         return entity_instance


# class Events:
#     """
#     链表中的一个节点。
#     每个节点包含一个值和指向下一个节点的引用。
#     """
#     def __init__(self, data):
#         self.data = data  # 节点存储的数据
#         self.id = data.get('id')
#         self.name = data.get('name')
#         self.involved_entities = data.get("involved_entities")
#         self.describe = ';'.join(data.get('describe',[]))
#         self.next = []  # 指向下一个节点的引用，初始为 None

#     def __repr__(self):
#         """用于调试，方便打印节点信息"""
#         return f"Events: ({self.data})({self.next})"
    


# import json

# class Relation:
#     """
#     表示两个实体之间关系的类。
#     每个关系包含 subject_id, object_id, proportion，以及可选的额外数据。
#     """
#     def __init__(self, data):
#         """
#         初始化 Relation 实例。
#         :param data: 包含关系所有数据的字典。
#                      期望包含 'subject_id', 'object_id', 'proportion'。
#                      可以包含 'relation_type' 或其他自定义字段。
#         """
#         if not isinstance(data, dict):
#             raise TypeError("Initialization data must be a dictionary.")

#         self.data = data.copy()  # 存储原始数据，使用 .copy() 防止外部修改影响内部状态

#         # 从原始数据中提取核心属性
#         self.subject_id = self.data.get("subject_id")
#         self.object_id = self.data.get("object_id")
#         self.proportion = self.data.get("proportion") # 可以是 None，如果原始数据中没有

#         # 校验核心属性是否存在
#         if self.subject_id is None:
#             raise ValueError("Relation data must contain 'subject_id'.")
#         if self.object_id is None:
#             raise ValueError("Relation data must contain 'object_id'.")

#         # 针对 proportion 进行更健壮的处理：
#         # 如果原始数据中没有 proportion 或者不是数字，可以给一个默认值
#         if not isinstance(self.proportion, (int, float)):
#             # 如果 proportion 不存在或类型不正确，给一个默认值 1.0
#             print(f"Warning: 'proportion' missing or invalid in data for subject_id={self.subject_id}. Defaulting to 1.0.")
#             self.proportion = 1.0
#         else:
#             self.proportion = float(self.proportion) # 确保是浮点数

#         # 示例：你可以在这里提取其他你关心的属性，比如 relation_type
#         self.relation_type = self.data.get("relation_type")


#     def __repr__(self):
#         """用于调试，方便打印关系信息"""
#         type_str = f", type='{self.relation_type}'" if self.relation_type else ""
#         return (f"Relation(subject_id='{self.subject_id}', "
#                 f"object_id='{self.object_id}', "
#                 f"proportion={self.proportion}{type_str})")

#     def __str__(self):
#         return self.__repr__()

#     def to_dict(self):
#         """
#         将 Relation 实例及其存储的原始数据转换为一个 Python 字典，便于序列化。
#         直接返回 self.data 的副本，确保所有原始信息被保留。
#         """
#         return self.data.copy()

#     @classmethod
#     def from_dict(cls, data_dict):
#         """
#         从一个字典创建 Relation 实例。
#         直接将传入的字典作为原始数据传递给 __init__ 方法。
#         :param data_dict: 包含关系数据的字典。
#         :return: Relation 实例。
#         """
#         if not isinstance(data_dict, dict):
#             raise TypeError("Input for from_dict must be a dictionary.")
#         # 直接将 data_dict 传递给 __init__，因为它就是我们需要的原始数据字典
#         return cls(data_dict)

    
#     # --- 实现去重功能的核心 ---

#     def __eq__(self, other):
#         """
#         定义两个 Relation 对象相等的条件。
#         如果 subject_id, object_id, 和 relation_type 都相同，则认为它们相等。
#         """
#         if not isinstance(other, Relation):
#             return NotImplemented # 或者返回 False
#         return (self.subject_id == other.subject_id and
#                 self.object_id == other.object_id and
#                 self.relation_type == other.relation_type)

#     def __hash__(self):
#         """
#         定义 Relation 对象的哈希值。
#         哈希值基于 subject_id, object_id, 和 relation_type。
#         只有可哈希的对象才能作为 set 的元素或 dict 的键。
#         """
#         # 使用 tuple 的 hash 值，因为 tuple 是不可变的且可哈希的
#         return hash((self.subject_id, self.object_id, self.relation_type))

import pickle
import os
class Graphs():
    def __init__(self,path = "save.pickle"):
        self.G = nx.Graph()
        # self.entities = []
        # self.entities_relations = {}
        # self.id2entities = {}
        # self.name2entities = {}
        #TODO2
        self.name2id = {}
        self.id2entities = {}
        self.path = path
        # self.json_path = json_path


        if os.path.exists(self.path):
            self.load_graph()

    def save_graph(self): # 2
        with open(self.path, "wb") as f:
            pickle.dump(self.G, f)
        # nx.write_graphml(self.G, self.path)
        # data = {
        #     "entities_relations":{v.to_dict() for v in self.entities_relations}, 
        #     "id2entities":{ k : v.to_dict() for k,v in self.id2entities.items()},
        #     "name2entities":{ k : v.to_dict() for k,v in self.name2entities.items()},
        # }
        # with open(self.json_path,'w') as f:
        #     f.write(json.dumps(data,ensure_ascii=False))

    def load_graph(self): # 3

        with open(self.path, "rb") as f:
            self.G = pickle.load(f)
        # print("Loaded G (from pickle) nodes:", loaded_graph_pickle.nodes(data=True))
        # self.G = nx.read_graphml(self.path)
        # with open(self.json_path,'r') as f:
        #     result = f.read()
        # data = json.loads(result)
        # self.entities_relations = {Relation.from_dict(v) for v in data.get("entities_relations")}
        # self.id2entities = {k : Entity.from_dict(v) for k,v in data.get("id2entities").items()}
        # self.name2entities = {k : Entity.from_dict(v) for k,v in data.get("name2entities").items()}

    def show_graph(self,path = "basic.html"): # 4
        nt = Network('1000px', '1000px')
        nt.from_nx(self.G)
        nt.write_html(path, open_browser=False,notebook=False)


    #TODO6 要考虑到融合
    #TODO1
    def update(self,entities_relations, id2entities, name2id):
        # self.id2entities.update(id2entities)
        # self.name2entities.update(name2entities)
        # self.entities_relations.update(entities_relations)

        # edges_graph = [(self.id2entities[i.get('subject_id')].name,
        #                   self.id2entities[i.get('object_id')].name,
        #                   {'title':i.get("relation")}) 
        #                for i in self.entities_relations]
        # nodes_graph = [(name,{"title":obj.describe}) 
        #                for name,obj in self.name2entities.items()]
        # self.G.add_nodes_from(nodes_graph)
        # self.G.add_edges_from(edges_graph)
        # self.G.save_graph()


        # edges_graph = [(i['subject_id'],i["object_id"],
        #                 {'relation':i.get("relation")})
        #                     for i in entities_relations]

        nodes_graph = [(id,dic) for id,dic in id2entities.items()]

        self.G.add_nodes_from(nodes_graph)
        # self.G.add_edges_from(edges_graph)
        self.save_graph()
        self.name2id = name2id
        self.id2entities = id2entities


    def find_nodes_by_attribute(self,graph, attribute_name, attribute_value):
        """
        通过指定属性名和属性值查找图中的所有匹配节点。

        Args:
            graph (nx.Graph): 要搜索的NetworkX图。
            attribute_name (str): 要查找的属性的名称 (例如 'name', 'type')。
            attribute_value (any): 要匹配的属性值。

        Returns:
            list of tuple: 包含 (node_id, node_data_dict) 的列表，所有匹配的节点。
        """
        matching_nodes = []
        for node_id, node_data in graph.nodes(data=True):
            if node_data.get(attribute_name) == attribute_value:
                matching_nodes.append((node_id, node_data))
        return matching_nodes
    
    def get_nodes_by_name(self,name):
        return self.find_nodes_by_attribute(self.G,"name",name)

    def search_graph(self,result:[str],depth:int = 2, output_type = "prompt")-> set:
        # result 输入 retriver 以后产出的Documents 然后只保留其i.text 构成列表,  理论上, 这些都是图的结点
        # result = self._retriver_search(text)
        entities = set()
        for i in result:
            # try:
            all_entity, _ = self.search_networkx_depth_2(self.G,i) # 深度为2的搜索  
            print(all_entity,'all_entity')
            # except ValueError as e:
            #     continue
            entities |= {i}
            entities |= all_entity
        if output_type == 'prompt':
            result = self.get_prompt(entities)
        elif output_type == 'entity':
            result = self.get_entitys(entities)
        else:
            raise TypeError('错误')
        return result

    
    def search_networkx_depth_1(self,graph, start_node):
        """
        深度为1的搜索
        """
        return list(graph.neighbors(start_node))
        
    def search_networkx_depth_2(self,graph, start_node):
        """
        在 NetworkX 图中实现深度为 2 的搜索。

        Args:
            graph (nx.Graph 或 nx.DiGraph): 要搜索的 NetworkX 图。
            start_node: 搜索的起始节点。

        Returns:
            set: 包含从起始节点可达的深度为 1 和深度为 2 的所有节点的集合。
                 不包括起始节点本身。
            set: 仅包含深度为 2 的节点的集合。
        """
        if start_node not in graph:
            raise ValueError(f"起始节点 '{start_node}' 不存在于图中。")

        # 存储所有可达节点 (深度1和深度2)
        all_reachable_nodes = set()
        # 存储仅深度2的节点
        depth_2_nodes = set()

        # 深度1：获取起始节点的所有邻居
        # for neighbor_1 in graph.neighbors(start_node):
        #     all_reachable_nodes.add(neighbor_1)

        # 优化：可以直接使用 comprehension 方式获取邻居，并进行后续处理
        neighbors_depth_1 = set(graph.neighbors(start_node))
        all_reachable_nodes.update(neighbors_depth_1)

        # 深度2：遍历深度1的邻居，获取它们的邻居
        for node_depth_1 in neighbors_depth_1:
            for node_depth_2 in graph.neighbors(node_depth_1):
                # 确保节点不是起始节点，也不是深度1的节点（避免重复计算或包含起始节点）
                if node_depth_2 != start_node and node_depth_2 not in neighbors_depth_1:
                    all_reachable_nodes.add(node_depth_2)
                    depth_2_nodes.add(node_depth_2)
                # 如果 node_depth_2 是深度1的节点，它已经包含在 all_reachable_nodes 中了
                elif node_depth_2 in neighbors_depth_1:
                    pass # 已经处理过，不做额外操作

        return all_reachable_nodes, depth_2_nodes

    
    def get_prompt(self,entities_names:list[str]) -> str:
        """拼接prompt

        Args:
            entities_names (str]): [str]  

        Returns:
            _type_: _description_
        """
        entity_prompt = "## 名词解释" + '\n    '
        for i in entities_names:
            self.id2entities[self.name2id[i]]
            entity_prompt += i + ":" + (self.name2entities[i].describe or "常规理解")  + "\n    "
        return entity_prompt
    
    def get_entitys(self,entities_names:set[str]):
        # entities_names -> [Entity]
        return [self.get_entity_by_name(i) for i in entities_names]
    
    def get_entity_by_id(self,id:str):
        return self.id2entities.get(id)
    
    def get_entity_by_name(self,name:str):
        return self.name2entities.get(name)


class Entity_Graph():
    """
    有两种使用方式 
        1 没有data_dict 那么就直接build 然后使用, 使用的是之前在G 和 index 中的存留数据
        2 有data_dict 那么就要在之前使用update 来做以下, 然后build 和使用
    """
    def __init__(self):
        self._build =  False

    def update(self,index,graph,data_dict):
        entities_relations, id2entities, name2id = self._process(data_dict)
        graph.update(entities_relations, id2entities, name2id)
        for i in list(graph.G.nodes):
            doc = Document(text = graph.G.nodes[i].get("name"),
                            metadata = {'type':"entity","id":i},
                            excluded_embed_metadata_keys = ['type','id'],
                            id_=i)
            index.update(document=doc)

    def build(self,index,G,similarity_top_k:int = 2,similarity_cutoff = 0.8):
        self.postprocess = SimilarityPostprocessor(similarity_cutoff = similarity_cutoff)
        self.retriver = index.as_retriever(similarity_top_k=similarity_top_k,
                                            filters = MetadataFilters(
                                                        filters=[MetadataFilter(key="type", operator=FilterOperator.EQ, value="entity"),]
                                            ))
        self.G = G
        self._build = True

    def search(self,text,depth = 2,output_type = "prompt"):
        # output_type == prompt or entity
        assert self._build == True
        result = self.postprocess.postprocess_nodes(self.retriver.retrieve(text))
        result_text = [i.text for i in result]
        result = self.G.search_graph(result_text,depth =depth, output_type = output_type)
        return result

    #TODO7
    def _process(self,data_dict:dict):
        if data_dict:
            x = []
            for i in data_dict.get('entities_relations'):
                if self._identify_string_type(i.get('object_id')) == "GENERIC_STRING":
                    x.append({'id':str(uuid.uuid4())[:16],
                            "name":i.get('object_id')})

            entities = x + data_dict.get('entities')

            id2entities = {i.get('id'): i for i in entities}
            name2id = {i.get('name'):i.get('id') for i in entities}


            entities_relations = []
            for i in data_dict.get('entities_relations'):
                if self._identify_string_type(i.get('object_id')) == "GENERIC_STRING":
                    object_id = name2id[i.get('object_id')]
                else:
                    object_id = i.get('object_id')

                # id2entities[i.get('subject_id')].next.append(object_id)
                entities_relations.append(
                        {
                        "subject_id":i.get('subject_id'),
                        "proportion":0.8,
                        "object_id":object_id,
                        })
            return entities_relations, id2entities, name2id
        else:
            return [],{},{}

    def _identify_string_type(self,text: str) -> str:
        """
        根据给定的字符串，判断其是UUID的一部分（或完整UUID）还是一个普通字符串。
        假设普通字符串包含中文或非UUID字符。

        参数:
            text (str): 需要判断的字符串。

        返回:
            str: 'UUID_ENTITY' 如果看起来是UUID的一部分或完整UUID。
                 'GENERIC_STRING' 如果是普通字符串（例如包含中文）。
        """
        if not isinstance(text, str):
            return 'GENERIC_STRING' # 处理非字符串输入

        text_lower = text.lower()

        # 1. 尝试转换为完整的UUID对象 (最高优先级判断)
        # 这一步可以捕获完整的UUID，无论是否带连字符
        try:
            uuid_obj = uuid.UUID(text_lower)
            # 确保转换后的UUID字符串形式与原始输入一致或仅连字符有差异
            # 这一步是为了避免 'test' 被解析成 '00000000-0000-0000-0000-000000000000'
            # 但如果已知输入类型，可以简化
            if str(uuid_obj) == text_lower or str(uuid_obj).replace('-', '') == text_lower.replace('-', ''):
                return 'UUID_ENTITY' # 这是一个完整的UUID
        except ValueError:
            pass # 不是一个完整的UUID，继续检查

        # 2. 检查字符集：是否只包含十六进制字符和连字符
        # UUID_CHARS_PATTERN: 包含十六进制数字 (0-9, a-f), 字母 (a-f), 和连字符 (-)
        # 如果字符串包含任何其他字符（例如中文、空格、标点符号等），它就不是UUID的一部分
        uuid_char_pattern = re.compile(r"^[0-9a-f-]+$")

        if uuid_char_pattern.fullmatch(text_lower):
            # 额外检查，确保不是空字符串或纯连字符，且有足够的长度像UUID
            # 因为 'a' 也会匹配这个模式，但它太短了，不像UUID part
            # '5b697312-2701-4b' 至少有 2 个字符
            if len(text) > 1 and any(c.isalnum() for c in text): # 确保不为空且包含至少一个字母数字
                return 'UUID_ENTITY' # 看起来是UUID的一部分

        # 3. 如果以上条件都不满足，则认为是普通字符串
        return 'GENERIC_STRING'


# class Event_Graph():
#     def __init__(self,data_dict:dict):
#         self.G = nx.DiGraph() # DG
#         self.postprocess = SimilarityPostprocessor(similarity_cutoff=0.9)
#         events, events_relations, id2events, name2events = self.process(data_dict)
#         self.events = events
#         self.events_relations = events_relations
#         self.id2events = id2events
#         self.name2events = name2events
        
#     def add_graph(self):# 1
#         edges_graph = [(self.id2events[i.get('subject_id')].name,
#                           self.id2events[i.get('object_id')].name,
#                           {'title':i.get("relation","")}) 
#                        for i in self.events_relations]
#         nodes_graph = [(name,{"title":obj.describe}) 
#                        for name,obj in self.name2events.items()]
#         self.G.add_nodes_from(nodes_graph)
#         self.G.add_edges_from(edges_graph)
    
#     def save_graph(self,path = "save_event.gml"): # 2
#         nx.write_graphml(self.G, path)
    
#     def load_graph(self,path = "save_event.gml"): # 3
#         self.G = nx.read_graphml(path)
    
#     def show_graph(self,path = "basic_event.html"): # 4
#         nt = Network('1000px', '1000px')
#         nt.set_options("""
# {
#   "layout": {
#     "hierarchical": {
#       "enabled": true,
#       "direction": "UD",
#       "sortMethod": "directed"
#     }
#   },
#   "edges": {
#     "color": {
#       "inherit": true
#     },
#     "smooth": {
#       "enabled": true,
#       "type": "dynamic"
#     }
#   },
#   "nodes": {
#       "shape": "box",
#       "widthConstraint": { "maximum": 120 },
#       "font": {"size": 10}
#   }
# }
# """)
#         nt.from_nx(self.G)
#         nt.write_html(path, open_browser=False,notebook=False)
#         # content  = ""
#         # for i in self.events:
#         #     content += f"{i.get('name')}" + "\n"
#         #     if i.get('sub_events_id'):
#         #         for j in i.get('sub_events_id'):
#         #             # print(j)
#         #             x = self.get_event_by_id(j)
#         #             x = x or {}
#         #             content += f"----{x.get('name','none')}" + "\n"

    
#     def process(self, data_dict:dict):
#         events = data_dict.get('events')

#         id2events = {i.get('id'): Events(i) for i in events}
#         name2events = {v.name: v for v in id2events.values()}


#         events_relations = self.deal_(data_dict.get("events_relations"))

#         for i in events_relations:
#             object_id = i.get('object_id')
#             id2events[i.get('subject_id')].next.append(object_id)

#         return events, events_relations, id2events, name2events

#     def deal_(self,data):
#         result = []
#         for entry in data:
#             subject_id = entry['subject_id']
#             sub_events_ids = entry['sub_events_id']
#             for sub_event_id in sub_events_ids:
#                 result.append({'subject_id': subject_id,
#                                "proportion":0.8,
#                                'object_id': sub_event_id})

#         return result
#     def update_index(self,index):
#         for i in list(self.G.nodes):
#             doc = Document(text = i,metadata = {'type':"events"},excluded_embed_metadata_keys = ['type'],id_=self.name2events.get(i).id)
#             index.update(document=doc)
    
    
        
#     def build_retriver(self,index,similarity_top_k = 2):
#         self.retriver = index.as_retriever(similarity_top_k=similarity_top_k,
#                                             filters = MetadataFilters(
#                                             filters=[MetadataFilter(key="type", operator=FilterOperator.EQ, value="events"),]
#                                             ))
    
#     def retriver_search(self,text):
#         result = self.postprocess.postprocess_nodes(self.retriver.retrieve(text))
#         return result

    
#     def retrive(self,text,depth = 2):
#         result = self.retriver_search(text)
#         xxp = []
#         for i in result:
#             result = self.find_nodes_by_depth(self.G, i.text, max_depth = depth)
#             xxp.append(result)
#         return xxp

  
#     def find_nodes_by_depth(self,graph, start_node, max_depth):
#         """
#         从起始节点出发，沿有向边查找指定深度内的所有节点，并按层级组织。

#         Args:
#             graph (nx.DiGraph): 有向图。
#             start_node: 起始节点。
#             max_depth (int): 最大查找深度（0表示起始节点本身，1表示直接后继，依此类推）。

#         Returns:
#             dict: 一个字典，键是深度（int），值是该深度下可达的节点列表。
#                   例如：{0: [start_node], 1: [node1, node2], 2: [node3, node4]}
#                   如果起始节点不存在，返回空字典。
#         """
#         if start_node not in graph:
#             print(f"错误: 起始节点 '{start_node}' 不存在于图中。")
#             return {}

#         # 使用 BFS 算法
#         visited = {start_node}  # 记录已访问节点，避免循环和重复
#         queue = [(start_node, 0)]  # 队列，存储 (node, current_depth)

#         # 结果字典，按深度存储节点
#         result_by_depth = {0: [start_node]} 

#         head = 0  # 队列的头指针，代替 pop(0) 以提高效率
#         while head < len(queue):
#             current_node, current_depth = queue[head]
#             head += 1

#             # 如果当前深度已达到最大深度，则不再探索其后继
#             if current_depth >= max_depth:
#                 continue

#             # 探索当前节点的直接后继
#             for neighbor in graph.successors(current_node):
#                 if neighbor not in visited:
#                     visited.add(neighbor)
#                     next_depth = current_depth + 1

#                     # 将后继节点添加到结果字典的对应深度层
#                     if next_depth not in result_by_depth:
#                         result_by_depth[next_depth] = []
#                     result_by_depth[next_depth].append(neighbor)

#                     # 将后继节点加入队列，以便后续探索
#                     queue.append((neighbor, next_depth))

#         return result_by_depth
    
#     def get_prompt(self,events:list[dict]):
#         events_prompt = '## 过往事件\n'
#         for i in events:
#             if i.get(0):
#                 for i1 in i.get(0):
#                     events_prompt += '----'
#                     events_prompt += i1
#                     events_prompt += '\n'

#             if i.get(1):
#                 for i2 in i.get(1):
#                     events_prompt += '----'
#                     events_prompt += '----'
#                     events_prompt += i2
#                     events_prompt += '\n'

#         return events_prompt
    
#     def get_events(self,events)->list[Events]:
#         x = []
#         for i in events:
#             if i.get(0):
#                 for j in i.get(0):
#                     x.append(self.get_entity_by_name(j))
#             if i.get(1):
#                 for j2 in i.get(1):
#                     x.append(self.get_entity_by_name(j2))

#         return x
    
#     def get_entity_by_id(self,id:str)->Events:
#         return self.id2events.get(id)
    
#     def get_entity_by_name(self,name:str)->Events:
#         return self.name2events.get(name)
    