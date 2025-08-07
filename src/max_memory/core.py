'''
Author: 823042332@qq.com 823042332@qq.com
Date: 2025-08-01 14:31:16
LastEditors: 823042332@qq.com 823042332@qq.com
LastEditTime: 2025-08-07 10:00:57
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



class Entity:
    """
    链表中的一个节点。
    每个节点包含一个值和指向下一个节点的引用。
    """
    def __init__(self, data):
        self.data = data  # 节点存储的数据
        self.id = data.get('id')
        self.name = data.get('name')
        self.aliases = data.get("aliases")
        self.describe = ';'.join(data.get('describe',[]))
        self.next = []  # 指向下一个节点的引用，初始为 None

    def __repr__(self):
        """用于调试，方便打印节点信息"""
        return f"Entity: ({self.data})({self.next})"

class Events:
    """
    链表中的一个节点。
    每个节点包含一个值和指向下一个节点的引用。
    """
    def __init__(self, data):
        self.data = data  # 节点存储的数据
        self.id = data.get('id')
        self.name = data.get('name')
        self.involved_entities = data.get("involved_entities")
        self.describe = ';'.join(data.get('describe',[]))
        self.next = []  # 指向下一个节点的引用，初始为 None

    def __repr__(self):
        """用于调试，方便打印节点信息"""
        return f"Events: ({self.data})({self.next})"

import os
class Graphs():
    def __init__(self,path = "save.gml"):
        self.G = nx.Graph()
        self.path = path
        if os.path.exists(self.path):
            self.load_graph()

    def save_graph(self): # 2
        nx.write_graphml(self.G, self.path)
        
    def load_graph(self): # 3
        self.G = nx.read_graphml(self.path)

    def show_graph(self,path = "basic.html"): # 4
        nt = Network('1000px', '1000px')
        nt.from_nx(self.G)
        nt.write_html(path, open_browser=False,notebook=False)

    def as_g_retriver(self):
        pass

        

class Entity_Graph():
    def __init__(self,data_dict:dict = None,similarity_cutoff = 0.9):
        self.postprocess = SimilarityPostprocessor(similarity_cutoff = similarity_cutoff)
        entities, entities_relations, id2entities, name2entities = self.process(data_dict)
        self.entities = entities
        self.entities_relations = entities_relations
        self.id2entities = id2entities
        self.name2entities = name2entities

    def process(self,data_dict:dict):
        if data_dict:
            x = []
            for i in data_dict.get('entities_relations'):
                if self._identify_string_type(i.get('object_id')) == "GENERIC_STRING":
                    x.append({'id':str(uuid.uuid4())[:16],
                            "name":i.get('object_id')})

            entities = x + data_dict.get('entities')

            id2entities = {i.get('id'): Entity(i) for i in entities}
            name2entities = {v.name: v for v in id2entities.values()}


            entities_relations = []
            for i in data_dict.get('entities_relations'):
                if self._identify_string_type(i.get('object_id')) == "GENERIC_STRING":
                    object_id = name2entities[i.get('object_id')].id
                else:
                    object_id = i.get('object_id')

                id2entities[i.get('subject_id')].next.append(object_id)
                entities_relations.append({
                "subject_id":i.get('subject_id'),
                "proportion":0.8,
                "object_id":object_id,
                })
            return entities, entities_relations, id2entities, name2entities
        else:
            return [],[],{},{}

    def update(self,index,G):
        self._update_graph(G=G)
        self._update_index(index,G =G)

    def _update_graph(self,G):# 1
        edges_graph = [(self.id2entities[i.get('subject_id')].name,
                          self.id2entities[i.get('object_id')].name,
                          {'title':i.get("relation")}) 
                       for i in self.entities_relations]
        nodes_graph = [(name,{"title":obj.describe}) 
                       for name,obj in self.name2entities.items()]
        G.add_nodes_from(nodes_graph)
        G.add_edges_from(edges_graph)
        G.save_graph()

    def _update_index(self,index,G):
        for i in list(G.nodes):
            doc = Document(text = i,metadata = {'type':"entity"},excluded_embed_metadata_keys = ['type'],id_=self.name2entities.get(i).id)
            index.update(document=doc)

    def build(self,index,G,similarity_top_k:int = 2):
        self.retriver = index.as_retriever(similarity_top_k=similarity_top_k,
                                            filters = MetadataFilters(
                                                        filters=[MetadataFilter(key="type", operator=FilterOperator.EQ, value="entity"),]
                                            ))
        self.G = G

    def _retriver_search(self,text:str)->object:
        result = self.postprocess.postprocess_nodes(self.retriver.retrieve(text))
        return result



    def search(self,text:str,depth:int = 2, output_type = "prompt")-> set:
        entities = set()
        result = self._retriver_search(text)
        for i in result:
            try:
                all_entity, _ = self.search_networkx_depth_2(self.G,i.text) # 深度为2的搜索  
            except ValueError as e:
                continue
            entities |= {i.text}
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
            entity_prompt += i + ":" + (self.name2entities[i].describe or "常规理解")  + "\n    "
        return entity_prompt
    
    def get_entitys(self,entities_names:set[str]) -> list[Entity]:
        # entities_names -> [Entity]
        return [self.get_entity_by_name(i) for i in entities_names]

    def get_entity_by_id(self,id:str)->Entity:
        return self.id2entities.get(id)
    
    def get_entity_by_name(self,name:str) -> Entity:
        return self.name2entities.get(name)
    
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
    

class Event_Graph():
    def __init__(self,data_dict:dict):
        self.G = nx.DiGraph() # DG
        self.postprocess = SimilarityPostprocessor(similarity_cutoff=0.9)
        events, events_relations, id2events, name2events = self.process(data_dict)
        self.events = events
        self.events_relations = events_relations
        self.id2events = id2events
        self.name2events = name2events
        
    def add_graph(self):# 1
        edges_graph = [(self.id2events[i.get('subject_id')].name,
                          self.id2events[i.get('object_id')].name,
                          {'title':i.get("relation","")}) 
                       for i in self.events_relations]
        nodes_graph = [(name,{"title":obj.describe}) 
                       for name,obj in self.name2events.items()]
        self.G.add_nodes_from(nodes_graph)
        self.G.add_edges_from(edges_graph)
    
    def save_graph(self,path = "save_event.gml"): # 2
        nx.write_graphml(self.G, path)
    
    def load_graph(self,path = "save_event.gml"): # 3
        self.G = nx.read_graphml(path)
    
    def show_graph(self,path = "basic_event.html"): # 4
        nt = Network('1000px', '1000px')
        nt.set_options("""
{
  "layout": {
    "hierarchical": {
      "enabled": true,
      "direction": "UD",
      "sortMethod": "directed"
    }
  },
  "edges": {
    "color": {
      "inherit": true
    },
    "smooth": {
      "enabled": true,
      "type": "dynamic"
    }
  },
  "nodes": {
      "shape": "box",
      "widthConstraint": { "maximum": 120 },
      "font": {"size": 10}
  }
}
""")
        nt.from_nx(self.G)
        nt.write_html(path, open_browser=False,notebook=False)
        # content  = ""
        # for i in self.events:
        #     content += f"{i.get('name')}" + "\n"
        #     if i.get('sub_events_id'):
        #         for j in i.get('sub_events_id'):
        #             # print(j)
        #             x = self.get_event_by_id(j)
        #             x = x or {}
        #             content += f"----{x.get('name','none')}" + "\n"

    
    def process(self, data_dict:dict):
        events = data_dict.get('events')

        id2events = {i.get('id'): Events(i) for i in events}
        name2events = {v.name: v for v in id2events.values()}


        events_relations = self.deal_(data_dict.get("events_relations"))

        for i in events_relations:
            object_id = i.get('object_id')
            id2events[i.get('subject_id')].next.append(object_id)

        return events, events_relations, id2events, name2events

    def deal_(self,data):
        result = []
        for entry in data:
            subject_id = entry['subject_id']
            sub_events_ids = entry['sub_events_id']
            for sub_event_id in sub_events_ids:
                result.append({'subject_id': subject_id,
                               "proportion":0.8,
                               'object_id': sub_event_id})

        return result
    def update_index(self,index):
        for i in list(self.G.nodes):
            doc = Document(text = i,metadata = {'type':"events"},excluded_embed_metadata_keys = ['type'],id_=self.name2events.get(i).id)
            index.update(document=doc)
    
    
        
    def build_retriver(self,index,similarity_top_k = 2):
        self.retriver = index.as_retriever(similarity_top_k=similarity_top_k,
                                            filters = MetadataFilters(
                                            filters=[MetadataFilter(key="type", operator=FilterOperator.EQ, value="events"),]
                                            ))
    
    def retriver_search(self,text):
        result = self.postprocess.postprocess_nodes(self.retriver.retrieve(text))
        return result

    
    def retrive(self,text,depth = 2):
        result = self.retriver_search(text)
        xxp = []
        for i in result:
            result = self.find_nodes_by_depth(self.G, i.text, max_depth = depth)
            xxp.append(result)
        return xxp

  
    def find_nodes_by_depth(self,graph, start_node, max_depth):
        """
        从起始节点出发，沿有向边查找指定深度内的所有节点，并按层级组织。

        Args:
            graph (nx.DiGraph): 有向图。
            start_node: 起始节点。
            max_depth (int): 最大查找深度（0表示起始节点本身，1表示直接后继，依此类推）。

        Returns:
            dict: 一个字典，键是深度（int），值是该深度下可达的节点列表。
                  例如：{0: [start_node], 1: [node1, node2], 2: [node3, node4]}
                  如果起始节点不存在，返回空字典。
        """
        if start_node not in graph:
            print(f"错误: 起始节点 '{start_node}' 不存在于图中。")
            return {}

        # 使用 BFS 算法
        visited = {start_node}  # 记录已访问节点，避免循环和重复
        queue = [(start_node, 0)]  # 队列，存储 (node, current_depth)

        # 结果字典，按深度存储节点
        result_by_depth = {0: [start_node]} 

        head = 0  # 队列的头指针，代替 pop(0) 以提高效率
        while head < len(queue):
            current_node, current_depth = queue[head]
            head += 1

            # 如果当前深度已达到最大深度，则不再探索其后继
            if current_depth >= max_depth:
                continue

            # 探索当前节点的直接后继
            for neighbor in graph.successors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_depth = current_depth + 1

                    # 将后继节点添加到结果字典的对应深度层
                    if next_depth not in result_by_depth:
                        result_by_depth[next_depth] = []
                    result_by_depth[next_depth].append(neighbor)

                    # 将后继节点加入队列，以便后续探索
                    queue.append((neighbor, next_depth))

        return result_by_depth
    
    def get_prompt(self,events:list[dict]):
        events_prompt = '## 过往事件\n'
        for i in events:
            if i.get(0):
                for i1 in i.get(0):
                    events_prompt += '----'
                    events_prompt += i1
                    events_prompt += '\n'

            if i.get(1):
                for i2 in i.get(1):
                    events_prompt += '----'
                    events_prompt += '----'
                    events_prompt += i2
                    events_prompt += '\n'

        return events_prompt
    
    def get_events(self,events)->list[Events]:
        x = []
        for i in events:
            if i.get(0):
                for j in i.get(0):
                    x.append(self.get_entity_by_name(j))
            if i.get(1):
                for j2 in i.get(1):
                    x.append(self.get_entity_by_name(j2))

        return x
    
    def get_entity_by_id(self,id:str)->Events:
        return self.id2events.get(id)
    
    def get_entity_by_name(self,name:str)->Events:
        return self.name2events.get(name)
    