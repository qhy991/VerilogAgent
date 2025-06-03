#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于FAISS的RAG系统
需要安装: pip install faiss-cpu sentence-transformers ollama numpy
"""

import ollama
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
from typing import List, Dict, Any, Tuple
import json

class FaissRAGSystem:
    def __init__(self, 
                 ollama_host: str = "http://10.130.149.18:11434",
                 model_name: str = "qwen2.5-coder:14b",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 index_file: str = "./faiss_index"):
        """
        基于FAISS的RAG系统
        
        Args:
            ollama_host: Ollama服务器地址
            model_name: 生成模型名称
            embedding_model: 嵌入模型名称
            index_file: FAISS索引文件路径前缀
        """
        self.ollama_host = ollama_host
        self.model_name = model_name
        self.embedding_model_name = embedding_model
        self.index_file = index_file
        
        # 存储文档的数据结构
        self.documents = []  # 存储原始文档
        self.metadatas = []  # 存储元数据
        self.doc_ids = []    # 存储文档ID
        
        print("=" * 60)
        print("初始化FAISS-RAG系统")
        print("=" * 60)
        
        # 初始化组件
        self._init_ollama()
        self._init_embedding_model()
        self._init_faiss_index()
        
        print("✅ FAISS-RAG系统初始化完成！")
    
    def _init_ollama(self):
        """初始化Ollama客户端"""
        try:
            print(f"🔗 连接Ollama服务器: {self.ollama_host}")
            self.ollama_client = ollama.Client(host=self.ollama_host)
            
            # 测试连接
            response = self.ollama_client.list()
            print(f"✅ Ollama连接成功")
            
            # 检查模型
            model_found = False
            if 'models' in response:
                for model in response['models']:
                    name = model.get('name', model.get('model', ''))
                    if self.model_name in name:
                        model_found = True
                        break
            
            if model_found:
                print(f"✅ 模型 {self.model_name} 可用")
            else:
                print(f"⚠️  模型 {self.model_name} 未找到")
                
        except Exception as e:
            print(f"❌ Ollama连接失败: {e}")
            raise
    
    def _init_embedding_model(self):
        """初始化嵌入模型"""
        try:
            print(f"📚 加载嵌入模型: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # 获取嵌入维度
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            print(f"✅ 嵌入模型加载成功，维度: {self.embedding_dim}")
            
        except Exception as e:
            print(f"❌ 嵌入模型加载失败: {e}")
            raise
    
    def _init_faiss_index(self):
        """初始化FAISS索引"""
        try:
            print("🗄️  初始化FAISS索引...")
            
            # 尝试加载现有索引
            if self._load_index():
                print("✅ 加载现有FAISS索引")
            else:
                # 创建新索引
                print("创建新的FAISS索引...")
                # 使用L2距离的平面索引
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                print("✅ 新FAISS索引创建成功")
                
        except Exception as e:
            print(f"❌ FAISS索引初始化失败: {e}")
            raise
    
    def _load_index(self) -> bool:
        """加载现有的FAISS索引和元数据"""
        try:
            index_path = f"{self.index_file}.index"
            metadata_path = f"{self.index_file}.pkl"
            
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                # 加载FAISS索引
                self.index = faiss.read_index(index_path)
                
                # 加载元数据
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.metadatas = data['metadatas']
                    self.doc_ids = data['doc_ids']
                
                print(f"📊 加载了 {len(self.documents)} 个文档")
                return True
                
        except Exception as e:
            print(f"⚠️  加载索引失败: {e}")
        
        return False
    
    def save_index(self):
        """保存FAISS索引和元数据"""
        try:
            # 保存FAISS索引
            index_path = f"{self.index_file}.index"
            faiss.write_index(self.index, index_path)
            
            # 保存元数据
            metadata_path = f"{self.index_file}.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadatas': self.metadatas,
                    'doc_ids': self.doc_ids
                }, f)
            
            print(f"💾 索引已保存到 {self.index_file}")
            
        except Exception as e:
            print(f"❌ 保存索引失败: {e}")
    
    def add_document(self, text: str, doc_id: str, metadata: dict = None):
        """添加文档到FAISS索引"""
        try:
            print(f"📄 添加文档: {doc_id}")
            
            # 分块处理长文档
            chunks = self._chunk_text(text, chunk_size=500)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                
                # 生成嵌入向量
                embedding = self.embedding_model.encode([chunk])
                embedding = embedding.astype('float32')  # FAISS需要float32
                
                # 添加到FAISS索引
                self.index.add(embedding)
                
                # 存储文档和元数据
                chunk_metadata = metadata or {}
                chunk_metadata.update({
                    'source_doc': doc_id,
                    'chunk_index': i,
                    'chunk_id': chunk_id,
                    'text_length': len(chunk)
                })
                
                self.documents.append(chunk)
                self.metadatas.append(chunk_metadata)
                self.doc_ids.append(chunk_id)
            
            print(f"✅ 文档 {doc_id} 添加完成 ({len(chunks)} 个片段)")
            
        except Exception as e:
            print(f"❌ 添加文档失败: {e}")
    
    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """文本分块"""
        if len(text) <= chunk_size:
            return [text.strip()]
        
        # 按句子分割（简单版本）
        sentences = text.replace('。', '。\n').replace('！', '！\n').replace('？', '？\n').split('\n')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def search_documents(self, query: str, k: int = 3) -> List[Tuple[str, float, dict]]:
        """搜索相关文档"""
        try:
            if self.index.ntotal == 0:
                print("⚠️  索引中没有文档")
                return []
            
            # 生成查询嵌入
            query_embedding = self.embedding_model.encode([query])
            query_embedding = query_embedding.astype('float32')
            
            # 在FAISS中搜索
            distances, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx >= 0 and idx < len(self.documents):
                    similarity = 1 / (1 + distance)  # 转换为相似度分数
                    results.append((
                        self.documents[idx],
                        similarity,
                        self.metadatas[idx]
                    ))
            
            return results
            
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return []
    
    def ask(self, question: str, k: int = 3) -> str:
        """RAG问答"""
        try:
            print(f"🤔 问题: {question}")
            print(f"🔍 在FAISS索引中搜索相关文档...")
            
            # 搜索相关文档
            search_results = self.search_documents(question, k)
            
            if not search_results:
                return "抱歉，没有找到相关文档来回答您的问题。"
            
            print(f"📖 找到 {len(search_results)} 个相关文档片段")
            
            # 构建上下文
            context_docs = [doc for doc, score, metadata in search_results]
            context = "\n\n".join(context_docs)
            
            # 构建提示词
            prompt = f"""请基于以下上下文信息回答问题。请给出准确、详细的回答。

上下文信息：
{context}

问题：{question}

回答："""
            
            print(f"🤖 使用 {self.model_name} 生成回答...")
            
            # 生成回答 - 添加内存优化选项
            response = self.ollama_client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.3,
                    'top_p': 0.8,
                    'num_ctx': 2048,  # 限制上下文长度
                    'num_predict': 512,  # 限制输出长度
                    'num_thread': 4,  # 限制线程数
                }
            )
            
            return response['response']
            
        except Exception as e:
            if "CUDA error: out of memory" in str(e):
                return f"⚠️  GPU内存不足，建议：\n1. 减少上下文长度\n2. 使用更小的模型\n3. 切换到CPU模式\n\n检索到的相关信息：\n{context}"
            return f"回答生成失败: {e}"
    
    def chat(self, message: str) -> str:
        """简单对话"""
        try:
            response = self.ollama_client.generate(
                model=self.model_name,
                prompt=message,
                options={
                    'num_ctx': 1024,  # 减少上下文
                    'num_predict': 256,  # 减少输出长度
                }
            )
            return response['response']
        except Exception as e:
            return f"对话失败: {e}"
    
    def get_stats(self):
        """获取索引统计信息"""
        return {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal,
            'embedding_dimension': self.embedding_dim
        }


def main():
    """主测试函数"""
    try:
        # 初始化FAISS-RAG系统
        rag = FaissRAGSystem()
        
        # 准备测试文档
        test_documents = {
            "python_basics": """
            Python是一种高级编程语言，具有简洁明了的语法。
            它支持多种编程范式，包括面向对象、函数式和过程式编程。
            Python在数据科学、Web开发、人工智能等领域有广泛应用。
            其强大的标准库和丰富的第三方包使得开发效率很高。
            Python的设计哲学强调代码的可读性和简洁性。
            """,
            
            "machine_learning": """
            机器学习是人工智能的一个重要分支。
            它让计算机能够从数据中学习模式，而无需明确编程。
            常见的机器学习算法包括线性回归、决策树、神经网络等。
            机器学习分为监督学习、无监督学习和强化学习三大类。
            监督学习使用标记数据，无监督学习发现数据中的隐藏模式。
            强化学习通过与环境交互来学习最优策略。
            """,
            
            "deep_learning": """
            深度学习是机器学习的子集，使用多层神经网络。
            它在图像识别、自然语言处理、语音识别等领域表现突出。
            深度学习模型包括卷积神经网络、循环神经网络、transformer等。
            深度学习的成功得益于大数据、强大的计算能力和算法改进。
            它能够自动学习特征表示，减少了人工特征工程的需求。
            """
        }
        
        print("\n" + "=" * 60)
        print("添加测试文档到FAISS索引")
        print("=" * 60)
        
        # 添加文档
        for doc_id, content in test_documents.items():
            rag.add_document(
                text=content.strip(),
                doc_id=doc_id,
                metadata={"category": "技术文档", "source": "test"}
            )
        
        # 保存索引
        rag.save_index()
        
        # 显示统计信息
        stats = rag.get_stats()
        print(f"\n📊 索引统计:")
        print(f"  文档片段数: {stats['total_documents']}")
        print(f"  索引大小: {stats['index_size']}")
        print(f"  嵌入维度: {stats['embedding_dimension']}")
        
        print("\n" + "=" * 60)
        print("FAISS-RAG问答测试")
        print("=" * 60)
        
        # 测试问题
        test_questions = [
            "Python有什么特点？",
            "机器学习有哪些主要类型？", 
            "深度学习在哪些领域表现突出？",
            "什么是监督学习和无监督学习？"
        ]
        
        for question in test_questions:
            print(f"\n{'='*50}")
            answer = rag.ask(question, k=2)  # 减少检索文档数量
            print(f"💬 回答: {answer}")
        
        print("\n" + "=" * 60)
        print("搜索测试")
        print("=" * 60)
        
        # 测试搜索功能
        search_query = "神经网络"
        results = rag.search_documents(search_query, k=3)
        print(f"搜索查询: {search_query}")
        print(f"找到 {len(results)} 个结果:")
        
        for i, (doc, score, metadata) in enumerate(results, 1):
            print(f"\n{i}. 相似度: {score:.3f}")
            print(f"   来源: {metadata.get('source_doc', 'unknown')}")
            print(f"   内容: {doc[:100]}...")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()