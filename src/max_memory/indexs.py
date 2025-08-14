# Embedding

import qdrant_client
from typing import Any, List
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from volcenginesdkarkruntime import Ark


class VolcanoEmbedding(BaseEmbedding):
    _model = PrivateAttr()
    _ark_client = PrivateAttr()
    _encoding_format = PrivateAttr()

    def __init__(
        self,
        model_name: str = "doubao-embedding-text-240715",
        api_key: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._ark_client = Ark(api_key=api_key)
        self._model = model_name
        self._encoding_format = "float"
    @classmethod
    def class_name(cls) -> str:
        return "ark"
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """
        获取查询字符串的 embedding。
        通常查询和文档使用相同的 embedding 模型。
        """
        
        resp = self._ark_client.embeddings.create(
            model=self._model,
            input=[query],
            encoding_format=self._encoding_format,
        )
        return resp.data[0].embedding

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        获取单个文档字符串的 embedding。
        """
        resp = self._ark_client.embeddings.create(
            model=self._model,
            input=[text],
            encoding_format=self._encoding_format,
        )
        return resp.data[0].embedding

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取文档字符串的 embedding。
        如果你的火山模型支持批量推理，强烈建议实现此方法以提高效率。
        否则，它可以简单地循环调用 _get_text_embedding。
        """
        resp = self._ark_client.embeddings.create(
            model=self._model,
            input=texts,
            encoding_format=self._encoding_format,
        )
        return [i.embedding for i in resp.data]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)



def get_index(collection_name = 'large_test5',api_key ="ac89ee8d-ba2a-4e31-bad7-021cf411c673"):
    client = qdrant_client.QdrantClient(
        host="localhost",
        port=6333,
    )
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    embed_model = VolcanoEmbedding(model_name = "doubao-embedding-large-text-250515",
                                api_key =api_key )


    index = VectorStoreIndex.from_vector_store(vector_store,
                                            embed_model=embed_model
                                            )
    return index