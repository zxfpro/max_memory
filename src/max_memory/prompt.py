import uuid
from llama_index.core import PromptTemplate

## 对应适配的prompt

user_rule = """

1 提取的粒度要收敛, 提取的内容重点集中在"最具备价值"这点上,

何为价值, 有如下定义:
1 新概念,核心概念,对应解释
2 主要原理与论点
3 相对肯定的统计性规律和结论
4 作者重点强调的相当肯定的句子

总之就是, 可以带来新认知的内容.

"""

## 数据结构定义

data_struct = """
```json
{
  "entities": [ // 文章中的核心对象, 要求相同概念合并, 如 人工智能与AI
  {
   "id": "string", // 唯一标识符,从 ID_RANDOM_POOL 中抽取
   "name": "string", // 核心对象
   "aliases": ["string"] // 存放同一概念的别名
   "describe": ["string","string"] // 对核心对象的描述
  }],
  "entities_relations": [ // 从文章中抽取出的entities之间的关系
   {
  "subject_id": "string", // 发出对象的唯一标识符,与entities对应
  "relation": "MANIFESTS_AS", // 关系类型
  "object_id": "获取知识和信息的方式", // 接收对象的唯一标识符,与entities对应
  }],
 "events": [ // 细粒度的基本事件列表,每个事件都是一个原子信息单元
  {
   "id": "string", // 唯一标识符,从 ID_RANDOM_POOL 中抽取
   "name": "string", // 抽取出的内容的抽象,减少非必要的形容词。移除所有特殊字符（如Markdown语法、图片链接等）,只保留文本和基本标点。 注意与entities的概念统一, 处理好aliases, 同时代词要替换为明确的词
   "involved_entities": ["string", "string"] // 列表,包含事件中交互的1到2个核心对象。核心对象应该与entities的概念统一, 如果句子无法识别出1个或2个有代表性的实体,则该字段留空 []。
   "describe": ["string","string"] // 对该事件的描述
  }],
  "events_relations": [ // 细粒度的基本事件列表,每个事件都是一个原子信息单元
  {
  "subject_id": "string", // 发出对象的唯一标识符,与events对应  
  "sub_events_id": ["string"] // 列表,子事件的id列表 子事件的定义很广, 按照逻辑, 可以归类到某一事件的事件都属于其子事件
  }],
```
"""

prompt = """
**角色:**你是一名高效、精确的信息抽取与知识图谱构建专家。你的任务是从用户提供的文章中,按照严格定义的结构化要求,抽取并组织事件信息。

**目标:**产出符合特定JSON Schema的事件知识图谱,该图谱包含:

1. **细粒度的基本事件列表 (basic_events)**:作为知识图谱的原子节点。
2. **优先基于文章分级标题的结构化关系图 (event_relations_graph)**:描述事件间的层次和关系, 次之就自己总结层次关系。
---

# 延展规则:

{user_rule}
---

**输出JSON Schema:**

请严格按照以下JSON结构输出,不得有任何额外内容或偏差。

{data_struct}
---

关系抽取 (RELATION_EXTRACTION) (利用马克思主义启发的分类法)**
  *   **输入 (INPUT):** `CONSOLIDATED_ENTITIES`（来自阶段1）。
  *   **逻辑 (LOGIC):**
      1.  定义 `马克思主义知识图谱关系分类法 (MARXIST_KG_RELATION_TAXONOMY)`:
          *   **本质-现象/具象化 (ESSENCE_OF / MANIFESTS_AS):** (X, MANIFESTS_AS, Y) / (Y, IS_ESSENCE_OF, X)。语义：X 是 Y 的具体表达/实例；Y 是 X 的根本性质/抽象概念。关键词："是", "一种", "系统", "方法", "表现", "本质"。
          *   **内容-形式/构成 (CONTAINS / IS_COMPONENT_OF):** (X, CONTAINS, Y) / (Y, IS_COMPONENT_OF, X)。语义：X 包含 Y；Y 是 X 的一部分/要素。关键词："包含", "由...构成", "存在", "具有"。
          *   **矛盾-对立/互补 (CONTRADICTS / COMPLEMENTS):** (X, CONTRADICTS, Y) / (X, COMPLEMENTS, Y)。语义：X 与 Y 对立/不同；X 与 Y 协同/增强。关键词："与...不同", "对立", "互补", "协同"。
          *   **原因-结果/目的 (CAUSES / RESULTS_IN / AIMS_TO_ACHIEVE):** (X, CAUSES, Y) / (X, RESULTS_IN, Y) / (X, AIMS_TO_ACHIEVE, Y)。语义：X 导致 Y；X 产生 Y；X 的目的是 Y。关键词："导致", "产生", "用于", "旨在", "实现", "为了"。
          *   **实践-理论/应用 (APPLIES_TO / IS_APPLIED_BY / GUIDES):** (X, APPLIES_TO, Y) / (X, IS_APPLIED_BY, Y) / (X, GUIDES, Y)。语义：X 在 Y 中应用/相关；Y 使用 X；X 为 Y 提供指导。关键词："应用于", "使用", "指导", "针对"。
          *   **发展-历史/创造 (DEVELOPED_BY / PROPOSED_IN / DISCOVERED_IN):** (X, DEVELOPED_BY, Y) / (X, PROPOSED_IN, Y) / (X, DISCOVERED_IN, Y)。语义：X 的起源/创造；X 在 Y 时间被提出/发现。关键词："由...开发", "提出", "发现", "于...年"。
      2.  **关系抽取策略:**
          a.  遍历 `CONSOLIDATED_ENTITIES` 中的每个 `实体 (entity)`。
          b.  对于每个 `实体`，分析其 `describe` 列表。
          c.  对于 `entity.describe` 中的每个 `描述字符串 (description_string)`:
              i.  执行关键词匹配和浅层语义解析（例如，如果可用，进行依存句法分析，否则使用基于规则的模式匹配）以识别潜在的主语-关系-宾语三元组。
              ii. **主语 (Subject):** 默认设置为当前 `entity.name`。
              iii. **宾语 (Object):** 尝试在 `描述字符串` 中识别其他 `CONSOLIDATED_ENTITIES.name` 或 `CONSOLIDATED_ENTITIES.aliases`。如果未找到其他实体，则从描述中提取与关系类型对齐的重要短语或概念。
              iv. **关系 (Relation):** 将识别到的关键词/模式映射到 `MARXIST_KG_RELATION_TAXONOMY`。优先考虑特异性匹配。通过推断处理双向关系（例如，如果 X MANIFESTS_AS Y，则推断 Y IS_ESSENCE_OF X，除非已明确抽取）。
              v.  确保抽取的三元组的唯一性（主语、关系、宾语的组合）。
          d.  **三元组结构:** 每个三元组将是一个字典：`{"subject": "实体名称A", "relation": "关系类型", "object": "实体名称B_或_概念", "description": "支持该关系的原始文本片段"}`。
  *   **输出 (OUTPUT):** `EXTRACTED_RELATIONS`: 结构化关系三元组字典列表。

---
ID_RANDOM_POOL

{ID_RANDOM_POOL}

---

**自省与验证（内部检查机制,请严格遵循）:**

  
*  **格式检查**:输出的JSON是否完全符合定义的Schema?所有字段名、类型、列表结构是否正确?

*  **特殊字符检查**:`entities.name`, `events.name`, 字段是否已完全清除特殊字符和Markdown语法?

*  **ID一致性**:所有 `id` 字段是否唯一?`entities_relations` 中的 `subject_id` 和 `object_id` 是否都指向 `entities` 中存在的ID? `sub_events_id` 是否都指向 `events`中存在的ID?

*  **作者结构**:检查是否有作者的结构信息?

*  **实体一致性**:

  *  `events` 中的 `involved_entities` 在整个文档中是否保持一致?是否尽可能地具体化了通用词汇?**元素数量是否严格不超过2个?**

  *  所有的 `involved_entities` 中的元素是否都来自于entities?  aliases 的是否进行了统一?
  
*  **关系语义**:每个 `relation` 的选择是否准确反映了 `subject_id` 和 `object_id` 之间的语义关系?

*  **覆盖度**:文章的关键信息点是否都被提取为基本事件?
"""

ptt = PromptTemplate(prompt)
