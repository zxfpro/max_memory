import re
import os
import uuid
import json
import pickle
import networkx as nx
from pyvis.network import Network
from llama_index.core.vector_stores import FilterCondition
from llama_index.core import VectorStoreIndex
from llama_index.core import Document
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)


def merge_graphs_with_advanced_aliases(
    graph1: nx.Graph,
    graph2: nx.Graph,
    node_mapping: dict = None,  # 新增参数：{original_name_in_graph: target_name_in_merged_graph}
) -> nx.Graph:
    """
    合并两个 NetworkX 图，支持预定义的节点别名映射，并处理同名节点为别名关系。

    原则：
    1. 节点重命名/别名映射：
       - `node_mapping` 参数允许在合并前将特定节点名称映射到统一的名称。
       - 例如：{"join": "David"} 会将 graph1 中的 "join" 视为 "David"。
       - 如果映射的目标名称在 graph1 中已存在，则该节点成为 graph1 对应节点的别名。
    2. 同名节点处理：
       - 如果 graph2 中的节点（或经过映射后的节点）在 graph1 中已存在，则 graph2 的该节点
         被视为 graph1 中同名节点的别名。新图中，该节点保留 graph1 的属性，并增加一个
         'aliases' 属性来记录 graph2 中同名节点的原始ID，同时保留 'all_aliases_details'
         来存储完整的原始信息。
       - 'aliases' 属性格式：如果只有一个别名，为字符串；如果有多个别名，为字符串列表。
       - 'all_aliases_details' 属性格式：列表，每个元素是一个字典，包含：
         {'original_id': '原始节点ID', 'source_graph': '来源图', 'original_data': '原始节点属性'}
    3. 线的关系继承：
       - 所有来自 graph1 和 graph2 的边都会被添加到新图中。在添加边之前，其端点会根据
         node_mapping 进行调整。

    Args:
        graph1 (nx.Graph): 第一个图（优先级较高，其节点和属性为主）。
        graph2 (nx.Graph): 第二个图（其节点可能成为 graph1 的别名）。
        node_mapping (dict, optional): 一个字典，定义了节点名称的映射关系。
                                       键是原始节点名称，值是目标节点名称。
                                       例如：{"join": "David", "Robert": "Bob"}。
                                       默认为 None，表示不进行额外映射。

    Returns:
        nx.Graph: 合并后的新图。
    """
    merged_graph = nx.Graph()
    node_mapping = node_mapping if node_mapping is not None else {}

    # --- 辅助函数：根据映射获取节点的新名称 ---
    def get_mapped_node_name(node_name):
        return node_mapping.get(node_name, node_name)

    # --- 辅助函数：添加别名到节点 ---
    def _add_alias_to_node(target_node_id, alias_id, source_graph_name, original_data):
        node_data = merged_graph.nodes[target_node_id]

        # 维护简洁的 'aliases' 属性 (字符串或字符串列表)
        if "aliases" not in node_data:
            node_data["aliases"] = []  # 先初始化为列表

        # 避免重复添加别名ID到简洁列表
        if alias_id not in node_data["aliases"]:
            node_data["aliases"].append(alias_id)

        # 维护详细的 'all_aliases_details' 属性
        if "all_aliases_details" not in node_data:
            node_data["all_aliases_details"] = []

        node_data["all_aliases_details"].append(
            {
                "original_id": alias_id,
                "source_graph": source_graph_name,
                "original_data": original_data,
            }
        )

    # 1. 处理 graph1 的所有节点和属性
    for original_node_g1, data_g1 in graph1.nodes(data=True):
        mapped_node_g1 = get_mapped_node_name(original_node_g1)

        if not merged_graph.has_node(mapped_node_g1):
            # 如果映射后的节点在 merged_graph 中不存在，直接添加为主节点
            merged_graph.add_node(mapped_node_g1, **data_g1)
            # 如果 original_node_g1 不同于 mapped_node_g1 (即发生了映射)
            # 那么 original_node_g1 是 mapped_node_g1 的一个别名
            if original_node_g1 != mapped_node_g1:
                _add_alias_to_node(mapped_node_g1, original_node_g1, "graph1", data_g1)
        else:
            # 如果映射后的节点在 merged_graph 中已存在 (这意味着 G1 内部映射导致重合)
            # 此时 original_node_g1 应该作为 mapped_node_g1 的别名。
            if original_node_g1 != mapped_node_g1:
                _add_alias_to_node(mapped_node_g1, original_node_g1, "graph1", data_g1)

    # 2. 处理 graph2 的节点：别名机制和映射
    for original_node_g2, data_g2 in graph2.nodes(data=True):
        mapped_node_g2 = get_mapped_node_name(original_node_g2)

        if merged_graph.has_node(mapped_node_g2):
            # mapped_node_g2 在 merged_graph 中已存在 (可能是来自G1的节点，或者G2内部映射)
            # 此时 original_node_g2 应该作为 mapped_node_g2 的一个别名
            _add_alias_to_node(mapped_node_g2, original_node_g2, "graph2", data_g2)

            # 属性合并策略：如果graph2的属性在主节点中不存在，则合并
            for key, value in data_g2.items():
                if key not in merged_graph.nodes[mapped_node_g2]:
                    merged_graph.nodes[mapped_node_g2][key] = value
        else:
            # mapped_node_g2 在 merged_graph 中不存在
            # 这意味着 mapped_node_g2 是一个全新的主节点，由 original_node_g2 映射而来
            merged_graph.add_node(mapped_node_g2, **data_g2)
            # 如果 original_node_g2 != mapped_node_g2，则 original_node_g2 也是一个别名
            if original_node_g2 != mapped_node_g2:
                _add_alias_to_node(mapped_node_g2, original_node_g2, "graph2", data_g2)

    # 3. 继承边的关系
    # 对 graph1 的边进行处理
    for u, v, data in graph1.edges(data=True):
        mapped_u = get_mapped_node_name(u)
        mapped_v = get_mapped_node_name(v)
        # 确保边连接的端点在合并图中都存在 (映射后可能有所不同)
        if merged_graph.has_node(mapped_u) and merged_graph.has_node(mapped_v):
            merged_graph.add_edge(mapped_u, mapped_v, **data)
        else:
            print(
                f"Warning: Edge ({u}, {v}) from graph1 skipped. Mapped nodes ({mapped_u}, {mapped_v}) not found in merged graph."
            )

    # 对 graph2 的边进行处理
    for u, v, data in graph2.edges(data=True):
        mapped_u = get_mapped_node_name(u)
        mapped_v = get_mapped_node_name(v)
        if merged_graph.has_node(mapped_u) and merged_graph.has_node(mapped_v):
            merged_graph.add_edge(mapped_u, mapped_v, **data)
        else:
            print(
                f"Warning: Edge ({u}, {v}) from graph2 skipped. Mapped nodes ({mapped_u}, {mapped_v}) not found in merged graph."
            )

    # 最后处理 'aliases' 属性，如果只有一个元素，将其转换为字符串
    for node, data in merged_graph.nodes(data=True):
        if "aliases" in data and isinstance(data["aliases"], list):
            if len(data["aliases"]) == 1:
                data["aliases"] = data["aliases"][0]  # 如果只有一个别名，简化为字符串
            elif len(data["aliases"]) == 0:
                del data["aliases"]  # 如果没有别名，删除属性

    return merged_graph


def find_related_edges_greedy_flexible_networkx(
    graph: nx.Graph | nx.DiGraph, nodes_to_check: list
) -> list[tuple]:
    """
    在一个 NetworkX 图中，以“贪婪”方式查找与给定任意数量节点相关的**所有**边。

    “贪婪”意味着只要一条边的任一端点在 `nodes_to_check` 集合中，
    这条边就会被包含在结果中。

    Args:
        graph: 一个 NetworkX Graph (无向图) 或 DiGraph (有向图) 对象。
        nodes_to_check: 一个包含所有目标节点的列表、元组或集合。

    Returns:
        一个包含所有符合条件的边的列表。每条边表示为一个元组 (u, v)。
        对于无向图，返回的边会进行标准化（例如，总是 (min_node, max_node)）
        以确保结果的唯一性和一致性。
    """
    # 将目标节点列表转换为一个集合，以便进行 O(1) 的快速查找
    # 这一步非常重要，无论输入是列表、元组还是集合，都能正确处理
    target_nodes_set = set(nodes_to_check)

    # 使用集合来存储找到的边，自动处理重复和顺序问题
    found_edges = set()

    # 遍历图中的所有边
    for u, v in graph.edges():
        # 检查边的任一端点是否在目标节点集合中
        if u in target_nodes_set or v in target_nodes_set:
            if graph.is_directed():
                # 对于有向图，直接添加边 (u, v)
                found_edges.add((u, v))
            else:
                # 对于无向图，为了保证结果的唯一性，
                # 将边标准化为 (较小节点, 较大节点) 的形式
                standard_edge = tuple(sorted((u, v)))
                found_edges.add(standard_edge)

    # 将集合转换回列表并返回
    return list(found_edges)


class Graphs:
    def __init__(self, path="save.pickle"):
        self.G = nx.Graph()
        self.name2id = {}
        self.id2entities = {}
        self.path = path
        # if os.path.exists(self.path):
        #     self.load_graph()

    def save_graph(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.G, f)

    def load_graph(self):
        with open(self.path, "rb") as f:
            self.G = pickle.load(f)

    def show_graph(self, path="basic.html"):
        nt = Network("1000px", "1000px")
        # 遍历图中的所有节点，将 'name' 属性设置为节点的 'label'
        for node_id, node_data in self.G.nodes(data=True):
            if "name" in node_data:
                self.G.nodes[node_id]["label"] = node_data["name"]
        nt.from_nx(self.G)
        nt.write_html(path, open_browser=False, notebook=False)

    def update(self, entities_relations, id2entities, name2id):
        nodes_graph = [(id, dic) for id, dic in id2entities.items()]
        edges_graph = [
            (
                i.get("subject_id"),
                i.get("object_id"),
                {"proportion": i.get("proportion", 1)},
            )
            for i in entities_relations
        ]

        self.G.add_nodes_from(nodes_graph)
        self.G.add_edges_from(edges_graph)
        self.save_graph()
        self.name2id.update(name2id)  # 合并 name2id
        self.id2entities.update(id2entities)  # 合并 id2entities

    def merge_other_graph(
        self, other_graph_instance: "Graphs", node_mapping_by_name: dict = None
    ):
        """
        将另一个 Graphs 实例的图合并到当前实例中。
        同时会更新当前实例的 name2id 和 id2entities 映射。
        for example : node_mapping_by_name={"蚂蚁2":"蚂蚁集团"} 新增nodes : 旧nodes
        Args:
            other_graph_instance (Graphs): 另一个 Graphs 实例。
            node_mapping_by_name (dict, optional): 节点名称的映射关系。
                                           键是 other_graph_instance 中的原始节点名称，
                                           值是合并图中目标节点名称。
                                           例如：{"join": "David", "Robert": "Bob"}。
                                           默认为 None，表示不进行额外映射。
        """
        # 将基于名称的映射转换为基于ID的映射，以适应 merge_graphs_with_advanced_aliases 函数
        node_mapping_by_id = {}
        if node_mapping_by_name:
            for (
                original_name_in_other,
                target_name_in_merged,
            ) in node_mapping_by_name.items():
                # 获取 other_graph_instance 中原始名称对应的ID
                original_id_in_other_list = other_graph_instance.get_nodes_by_name(
                    original_name_in_other
                )
                if not original_id_in_other_list:
                    print(
                        f"Warning: Node '{original_name_in_other}' not found in other_graph_instance. Skipping mapping."
                    )
                    continue
                # 假设 get_nodes_by_name 返回的列表中的第一个元素就是我们想要映射的节点ID
                original_id_in_other = original_id_in_other_list[0][
                    0
                ]  # (node_id, node_data)

                # 获取当前图 self.G 中目标名称对应的ID
                target_id_in_merged_list = self.get_nodes_by_name(target_name_in_merged)
                if not target_id_in_merged_list:
                    # 如果目标名称在当前图中不存在，则使用目标名称本身作为ID（这将导致新节点创建）
                    # 或者，如果原函数期望的是一个ID，这里需要确保传入的是一个有效的ID格式
                    # 由于 merge_graphs_with_advanced_aliases 接受的是 node_name 作为键值，
                    # 这里的 target_name_in_merged 可以直接作为目标ID使用，
                    # 因为它最终会被映射到 merged_graph 中的对应节点。
                    target_id_in_merged = target_name_in_merged
                else:
                    target_id_in_merged = target_id_in_merged_list[0][
                        0
                    ]  # (node_id, node_data)

                node_mapping_by_id[original_id_in_other] = target_id_in_merged

        # 使用 merge_graphs_with_advanced_aliases 函数合并图
        # 当前实例的 G 作为 graph1 (优先级高), other_graph_instance.G 作为 graph2
        merged_nx_graph = merge_graphs_with_advanced_aliases(
            self.G, other_graph_instance.G, node_mapping_by_id
        )
        self.G = merged_nx_graph

        # 更新 name2id 和 id2entities
        # 遍历合并后的图的节点，重新构建或更新 name2id 和 id2entities
        new_name2id = {}
        new_id2entities = {}

        # 优先保留当前实例的映射，然后合并其他实例的
        # 注意：如果 other_graph_instance 中的节点通过映射与 self.G 中的节点合并，
        # 那么 merged_nx_graph 中的节点ID将是 self.G 中的ID或映射后的ID。
        # 因此，需要根据 merged_nx_graph 的实际节点来更新映射。

        for node_id, node_data in self.G.nodes(data=True):
            # 优先使用合并后图中节点的 'name' 属性来更新 name2id
            if "name" in node_data:
                new_name2id[node_data["name"]] = node_id
            new_id2entities[node_id] = node_data

        # 处理别名：如果一个节点有别名，确保所有别名也指向同一个主ID
        # 这个逻辑已经在 merge_graphs_with_advanced_aliases 中处理了，
        # 这里的 new_name2id 和 new_id2entities 应该直接从 merged_nx_graph 构建即可。
        # 对于别名，其原始ID不会作为主ID出现在 merged_nx_graph 的 nodes 列表中，
        # 而是作为主节点的 'aliases' 或 'all_aliases_details' 属性。
        # 因此，我们只需要确保所有“主”节点及其属性被正确记录。
        # 如果需要通过别名名称也能找到主节点，则需要额外处理：
        for node_id, node_data in self.G.nodes(data=True):
            if "all_aliases_details" in node_data:
                for alias_detail in node_data["all_aliases_details"]:
                    original_alias_id = alias_detail["original_id"]
                    original_alias_data = alias_detail["original_data"]
                    if "name" in original_alias_data:
                        # 确保别名名称也映射到主节点的ID
                        new_name2id[original_alias_data["name"]] = node_id

        self.name2id = new_name2id
        self.id2entities = new_id2entities
        self.save_graph()

    def find_nodes_by_attribute(self, graph, attribute_name, attribute_value):
        matching_nodes = []
        for node_id, node_data in graph.nodes(data=True):
            if node_data.get(attribute_name) == attribute_value:
                matching_nodes.append((node_id, node_data))
        return matching_nodes

    def get_nodes_by_name(self, name):
        return self.find_nodes_by_attribute(self.G, "name", name)

    def search_graph(
        self, result_names: list[str], depth: int = 2, output_type: str = "prompt"
    ) -> set:
        """
        根据节点名称列表，在图中搜索相关实体。

        Args:
            result_names (list[str]): 待搜索的节点名称列表。
            depth (int): 搜索深度。
            output_type (str): 输出类型，'prompt' 或 'entity'。

        Returns:
            set: 根据 output_type 返回相应的结果集合。
        """
        all_found_entity_names = set()

        for name in result_names:
            node_id = self.name2id.get(name)  # 获取名称对应的ID
            if node_id:  # 只有当名称对应的ID存在时才进行搜索
                # all_entity 包含的是节点ID
                all_entity_ids, _ = self.search_networkx_depth_2(self.G, node_id)
                # 将找到的实体ID转换为名称并添加到集合中
                for entity_id in all_entity_ids:
                    # 从 id2entities 获取实体字典，再获取其name
                    entity_data = self.id2entities.get(entity_id)
                    if entity_data and "name" in entity_data:
                        all_found_entity_names.add(entity_data["name"])
                all_found_entity_names.add(name)  # 确保原始查询的名称也被包含进来
            else:
                print(f"Warning: Node with name '{name}' not found in name2id mapping.")

        if output_type == "prompt":
            # get_prompt 期望的是实体名称列表
            result = self.get_prompt(list(all_found_entity_names))
        elif output_type == "entity":
            # get_entitys 期望的是实体名称集合
            result = self.get_entitys(all_found_entity_names)
        else:
            raise TypeError('Invalid output_type. Must be "prompt" or "entity".')
        return result

    def search_networkx_depth_1(self, graph, start_node):
        return list(graph.neighbors(start_node))

    def search_networkx_depth_2(self, graph, start_node):
        if start_node not in graph:
            raise ValueError(f"起始节点 '{start_node}' 不存在于图中。")

        all_reachable_nodes = set()
        depth_2_nodes = set()

        neighbors_depth_1 = set(graph.neighbors(start_node))
        all_reachable_nodes.update(neighbors_depth_1)

        for node_depth_1 in neighbors_depth_1:
            for node_depth_2 in graph.neighbors(node_depth_1):
                if node_depth_2 != start_node and node_depth_2 not in neighbors_depth_1:
                    all_reachable_nodes.add(node_depth_2)
                    depth_2_nodes.add(node_depth_2)
                elif node_depth_2 in neighbors_depth_1:
                    pass

        return all_reachable_nodes, depth_2_nodes

    def get_prompt(self, entities_names: list[str]) -> str:
        entity_prompt = "## 名词解释" + "\n    "
        for i in entities_names:
            # 这里的self.name2entities 不存在了，需要改为从 self.id2entities 通过 self.name2id 获取
            # 确保获取到的描述是字符串类型
            entity_id = self.name2id.get(i)
            if entity_id and entity_id in self.id2entities:
                entity_data = self.id2entities[entity_id]
                # 假设 describe 字段在 id2entities 存储的字典中是字符串或可以安全转换为字符串
                describe_text = ";".join(entity_data.get("describe")) or "常规理解"
                entity_prompt += i + ":" + str(describe_text) + "\n    "
            else:
                entity_prompt += i + ": 未知实体描述\n    "
        return entity_prompt

    def get_entitys(self, entities_names: set[str]):
        # get_entity_by_name 在 Graphs 中不存在，需要从 self.id2entities 通过 self.name2id 获取
        # 且返回的是原始的实体字典，而不是 Entity 对象
        found_entities = []
        for name in entities_names:
            entity_id = self.name2id.get(name)
            if entity_id and entity_id in self.id2entities:
                found_entities.append(self.id2entities[entity_id])
        return found_entities

    def get_entity_by_id(self, id: str):
        return self.id2entities.get(id)

    def get_entity_by_name(self, name: str):
        entity_id = self.name2id.get(name)
        if entity_id:
            return self.id2entities.get(entity_id)
        return None

    def find_related_edges(self, nodes_to_check: list) -> list[tuple]:
        """
        在当前 Graphs 实例的 NetworkX 图中，以“贪婪”方式查找与给定任意数量节点相关的**所有**边。
        此方法封装了顶层的 find_related_edges_greedy_flexible_networkx 函数。

        Args:
            nodes_to_check: 一个包含所有目标节点的列表、元组或集合。

        Returns:
            一个包含所有符合条件的边的列表。每条边表示为一个元组 (u, v)。
            对于无向图，返回的边会进行标准化（例如，总是 (min_node, max_node)）
            以确保结果的唯一性和一致性。
        """
        return find_related_edges_greedy_flexible_networkx(self.G, nodes_to_check)


class DiGraphs(Graphs):
    def __init__(self, path="save_digraph.pickle"):
        super().__init__(path)
        self.G = nx.DiGraph()  # 将图类型改为有向图

    def show_graph(self, path="basic_digraph.html"):
        nt = Network("1000px", "1000px")
        # 遍历图中的所有节点，将 'name' 属性设置为节点的 'label'
        for node_id, node_data in self.G.nodes(data=True):
            if "name" in node_data:
                self.G.nodes[node_id]["label"] = node_data["name"]

        # 为有向图设置分层布局选项
        nt.set_options(
            """
        {
        "layout": {
            "hierarchical": {
            "enabled": true,
            "direction": "UD",
            "sortMethod": "directed"
            }
        },
        "edges": {
            "arrows": {
            "to": { "enabled": true, "scaleFactor": 1 }
            },
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
        """
        )
        nt.from_nx(self.G)
        nt.write_html(path, open_browser=False, notebook=False)

    def search_graph(
        self, result_names: list[str], depth: int = 2, output_type: str = "prompt"
    ) -> set:
        """
        根据节点名称列表，在图中搜索相关实体。

        Args:
            result_names (list[str]): 待搜索的节点名称列表。
            depth (int): 搜索深度。
            output_type (str): 输出类型，'prompt' 或 'entity'。

        Returns:
            set: 根据 output_type 返回相应的结果集合。
        """
        all_found_entity_names = set()

        for name in result_names:
            node_id = self.name2id.get(name)  # 获取名称对应的ID
            if node_id:  # 只有当名称对应的ID存在时才进行搜索
                # all_entity 包含的是节点ID
                all_entity = self.find_nodes_by_depth(self.G, node_id, depth)
                # 将找到的实体ID转换为名称并添加到集合中
            else:
                print(f"Warning: Node with name '{name}' not found in name2id mapping.")

        if output_type == "prompt":
            # get_prompt 期望的是实体名称列表
            result = self.get_prompt(all_entity)
        elif output_type == "entity":
            # get_entitys 期望的是实体名称集合
            result = self.get_entitys(all_entity)
        else:
            raise TypeError('Invalid output_type. Must be "prompt" or "entity".')
        return result

    def get_entitys(self, events):
        x = {}
        for level, content_list in events.items():
            x[level] = [self.get_entity_by_id(i) for i in content_list]
        return x

    def get_prompt(self, events: list[dict]):
        events_prompt = "## 过往事件\n"
        for level, content_list in events.items():
            if level == 0:
                for i in content_list:
                    print(self.get_entity_by_id(i).get("name"))
                    events_prompt += ""
                    events_prompt += self.get_entity_by_id(i).get("name")
                    events_prompt += " : "
                    events_prompt += ";".join(self.get_entity_by_id(i).get("describe"))
                    events_prompt += "\n"
            elif level == 1:
                for i in content_list:
                    events_prompt += "----"
                    events_prompt += self.get_entity_by_id(i).get("name")
                    events_prompt += " : "
                    events_prompt += ";".join(self.get_entity_by_id(i).get("describe"))
                    events_prompt += "\n"
            elif level == 2:
                for i in content_list:
                    events_prompt += "--------"
                    events_prompt += self.get_entity_by_id(i).get("name")
                    events_prompt += " : "
                    events_prompt += ";".join(self.get_entity_by_id(i).get("describe"))
                    events_prompt += "\n"

        return events_prompt

    def find_nodes_by_depth(self, graph, start_node, max_depth):
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

class Entity_Graph:
    def __init__(self):
        self._build = False

    def update(self, index, graph, data_dict):
        entities_relations, id2entities, name2id = self._process(data_dict)
        graph.update(entities_relations, id2entities, name2id)
        # 在更新后，确保 G.nodes 中有 'name' 属性，因为 Graph.update 接受的是 (id, dic)
        # 这里的 graph.G.nodes[i].get("name") 应该能正确获取
        for i in list(graph.G.nodes):
            node_data = graph.G.nodes[i]  # 获取节点属性字典
            doc = Document(
                text=node_data.get("name"),  # 使用节点的 'name' 属性作为文本
                metadata={"type": "entity", "id": i},
                excluded_embed_metadata_keys=["type", "id"],
                id_=i,
            )
            index.update(document=doc)

    def build(self, index, G, similarity_top_k: int = 2, similarity_cutoff=0.8):
        self.postprocess = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
        self.retriver = index.as_retriever(
            similarity_top_k=similarity_top_k,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="type", operator=FilterOperator.EQ, value="entity"
                    ),
                ]
            ),
        )
        self.G = G
        self._build = True

    def search(self, text, depth=2, output_type="prompt"):
        assert self._build == True
        result_nodes = self.postprocess.postprocess_nodes(self.retriver.retrieve(text))
        # result_text 现在是包含节点名称的列表，可以直接传递给 Graphs.search_graph
        result_names = [node.text for node in result_nodes]
        result = self.G.search_graph(result_names, depth=depth, output_type=output_type)
        return result

    def _process(self, data_dict: dict):
        if data_dict:
            x = []
            for i in data_dict.get("entities_relations"):
                if self._identify_string_type(i.get("object_id")) == "GENERIC_STRING":
                    x.append(
                        {
                            "id": str(uuid.uuid4())[:16],
                            "name": i.get("object_id"),  # 确保新生成的实体有name属性
                            "describe": [],  # 为新生成的实体添加默认的 describe 字段，与现有结构保持一致
                        }
                    )

            entities = x + data_dict.get("entities")

            id2entities = {i.get("id"): i for i in entities}
            name2id = {
                i.get("name"): i.get("id")
                for i in entities
                if i.get("name") is not None
            }  # 确保name存在

            entities_relations = []
            for i in data_dict.get("entities_relations"):
                subject_id = i.get("subject_id")
                object_id_raw = i.get("object_id")

                if self._identify_string_type(object_id_raw) == "GENERIC_STRING":
                    object_id = name2id.get(object_id_raw)  # 从name2id获取object_id
                    if object_id is None:  # 如果未能找到，可能需要创建或跳过
                        print(
                            f"Warning: Object '{object_id_raw}' not found in name2id map during relation processing. Skipping relation."
                        )
                        continue
                else:
                    object_id = object_id_raw

                # 检查 subject_id 和 object_id 是否都存在于 id2entities 中
                if subject_id in id2entities and object_id in id2entities:
                    entities_relations.append(
                        {
                            "subject_id": subject_id,
                            "proportion": 0.8,
                            "object_id": object_id,
                        }
                    )
                else:
                    print(
                        f"Warning: Relation involving unknown subject_id '{subject_id}' or object_id '{object_id}'. Skipping relation."
                    )

            return entities_relations, id2entities, name2id
        else:
            return [], {}, {}

    def _identify_string_type(self, text: str) -> str:
        if not isinstance(text, str):
            return "GENERIC_STRING"

        text_lower = text.lower()

        try:
            uuid_obj = uuid.UUID(text_lower)
            if str(uuid_obj) == text_lower or str(uuid_obj).replace(
                "-", ""
            ) == text_lower.replace("-", ""):
                return "UUID_ENTITY"
        except ValueError:
            pass

        uuid_char_pattern = re.compile(r"^[0-9a-f-]+$")

        if uuid_char_pattern.fullmatch(text_lower):
            if len(text) > 1 and any(c.isalnum() for c in text):
                return "UUID_ENTITY"

        return "GENERIC_STRING"

class Event_Graph:
    def __init__(self):
        self._build = False

    def update(self, index, graph: DiGraphs, data_dict):
        events_relations, id2events, name2id = self._process(data_dict)
        graph.update(events_relations, id2events, name2id)
        for i in list(graph.G.nodes):
            node_data = graph.G.nodes[i]
            doc = Document(
                text=node_data.get("name"),
                metadata={"type": "event", "id": i},
                excluded_embed_metadata_keys=["type", "id"],
                id_=i,
            )
            index.update(document=doc)

    def build(self, index, G, similarity_top_k: int = 2, similarity_cutoff=0.8):
        self.postprocess = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
        self.retriver = index.as_retriever(
            similarity_top_k=similarity_top_k,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="type", operator=FilterOperator.EQ, value="event"
                    ),
                ]
            ),
        )
        self.G = G
        self._build = True

    def search(self, text, depth=2, output_type="prompt"):
        assert self._build == True
        result_nodes = self.postprocess.postprocess_nodes(self.retriver.retrieve(text))
        result_names = [node.text for node in result_nodes]
        result = self.G.search_graph(result_names, depth=depth, output_type=output_type)
        return result

    def _process(self, data_dict: dict):
        if data_dict:
            events = data_dict.get("events", [])
            events_relations_raw = data_dict.get("events_relations", [])

            id2events = {event.get("id"): event for event in events if event.get("id")}
            name2id = {
                event.get("name"): event.get("id")
                for event in events
                if event.get("name") is not None
            }

            # 处理 events_relations，将其转换为统一的 subject_id, object_id 格式
            processed_relations = []
            for entry in events_relations_raw:
                subject_id = entry.get("subject_id")
                sub_events_ids = entry.get(
                    "sub_events_id", []
                )  # 假设 sub_events_id 是一个列表

                for sub_event_id in sub_events_ids:
                    # 确保 subject_id 和 sub_event_id (作为 object_id) 都存在于 id2events 中
                    if subject_id in id2events and sub_event_id in id2events:
                        processed_relations.append(
                            {
                                "subject_id": subject_id,
                                "proportion": 0.8,
                                "object_id": sub_event_id,
                            }
                        )
                    else:
                        print(
                            f"Warning: Event relation involving unknown subject_id '{subject_id}' or object_id '{sub_event_id}'. Skipping relation."
                        )

            return processed_relations, id2events, name2id
        else:
            return [], {}, {}
