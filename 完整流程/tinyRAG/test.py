from typing import List,Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
import tqdm
import bm25s
from transformers import pipeline
import faiss

class BaseEmbeddings:
    """
    Base class for embeddings
    """
    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path
        self.is_api = is_api
    
    def get_embedding_List(self, texts:  List[str]) -> List[List[float]]:
        raise NotImplementedError
    def get_embedding_List(self, text: str) -> List[float]:
        raise NotImplementedError
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude


class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass

# MyEmbedding类，可以是bert也可以是大模型的embedding层，实例化后对文本进行embedding，传入vs.get_vector(embedding_model)
class MyEmbedding(BaseEmbeddings):
    """
    class for My embeddings
    """
    def __init__(self, path: str = '', is_api: bool = False) -> None:
        super().__init__(path, is_api)
        self.path = path
        self.model = SentenceTransformer(path)
    
    def get_embedding_List(self, texts: List[str]) -> List[List[float]]:
        embeddings  = self.model.encode(texts)
        return embeddings
    def get_embedding(self, text: str) -> List[float]:
        embeddings  = self.model.encode(text)
        return embeddings


def read_file_content( file_path: str) -> List[str]: 
    with open(file_path, 'r') as file:
        content = []
        for line in file:
            content.append(line.strip())
    return content



# 数据库类，可以实现持久化存储，加载，查询等功能
class VectorStore:
    def __init__(self, document: List[str] = [''],d = 1024,nlist = 32) -> None:
        # 可以在初始化的时候储存数据，可以为空
        self.document = document
        self.nlist = nlist
        self.d = d
        self.quantizer = faiss.IndexFlatL2(self.d)
        self.index = faiss.IndexIVFFlat(self.quantizer, self.d, self.nlist)
        self.vectors = None
        if not document:
            print("No document found")
    
    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        # 将存储的数据进行embedding，并储存
        self.vectors = EmbeddingModel.get_embedding_List(self.document)
        l = self.vectors.shape
        self.index.train(self.vectors)
        self.index.add(self.vectors)
        self.vectors = None
        return l

    def persist(self, path = 'storage'):
        # 将embedding后的数据存储到指定路径下
        faiss.write_index(self.index, path + '/faiss_ivf.index') 
        with open(f"{path}/document.json", 'w', encoding='utf-8') as f:
            json.dump(self.document, f, ensure_ascii=False,indent=2)

    def load_vector(self, path: str = 'storage'):
        # 从指定路径下加载embedding数据
        self.index = faiss.read_index(path+ '/faiss_ivf.index')
        with open(f"{path}/document.json", 'r', encoding='utf-8') as f:
            self.document = json.load(f)
    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        # 计算两个向量之间的相似度
        return BaseEmbeddings.cosine_similarity(vector1, vector2)

    def querys(self, querys: list[str], EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
        # 查询函数，返回与query最相似的k个文档（稠密查询）
        query_vector = EmbeddingModel.get_embedding_List(querys)
        D,I = self.index.search(query_vector, k)
        print(self.index.ntotal)
        print(I)
        for q_idx, neighbor_idx in enumerate(I):
            print(f"查询{q_idx} 的 top-5 文档：")
            for rank, doc_idx in enumerate(neighbor_idx):
                print(f"  {rank+1}. {self.document[doc_idx]}")
       
