import os
import logging
import time
from typing import List, Dict, Tuple, Any, Optional
from operator import itemgetter

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableMap
from langchain.schema.output_parser import StrOutputParser
from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.load import dumps, loads

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置环境变量
os.environ["USER_AGENT"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

class MultiQueryRAG:
    def __init__(self, model_name="deepseek-r1:7b", embedding_model="BAAI/bge-small-zh-v1.5"):
        self.llm = OllamaLLM(model=model_name)
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vect_store = None
        self.retriever = None
        
    def prepare_data(self, urls: List[str], chunk_size: int = 500, chunk_overlap: int = 50):
        """加载并处理网页数据"""
        try:
            all_docs = []
            for url in urls:
                logger.info(f"加载URL: {url}")
                loader = WebBaseLoader(url)
                docs = loader.load()
                all_docs.extend(docs)
                time.sleep(1)  # 避免请求过于频繁
                
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_documents(all_docs)
            logger.info(f"数据处理完成，共生成{len(chunks)}个文本块")
            return chunks
        except Exception as e:
            logger.error(f"数据准备过程出错: {e}")
            raise
    
    def embed_data(self, chunks, persist_directory='./chroma_db'):
        """嵌入数据并创建向量存储"""
        try:
            logger.info("开始嵌入数据...")
            self.vect_store = Chroma.from_documents(
                documents=chunks, 
                embedding=self.embedding_model, 
                persist_directory=persist_directory
            )
            self.retriever = self.vect_store.as_retriever()
            logger.info("数据嵌入完成")
            return self.vect_store, self.retriever
        except Exception as e:
            logger.error(f"数据嵌入过程出错: {e}")
            raise
    
    def retrieval_and_rank(self, queries: List[str], k: int = 10) -> Dict[str, List[Tuple[Any, float]]]:
        """执行检索并返回排序结果"""
        if not self.retriever:
            raise ValueError("检索器未初始化，请先嵌入数据")
            
        all_results = {}
        for query in queries:
            if query:
                try:
                    logger.info(f"执行查询: {query}")
                    search_results = self.retriever.get_relevant_documents(query)
                    results = []
                    for doc in search_results[:k]:
                        results.append((doc, doc.metadata.get('score', 0.0)))
                    all_results[query] = results
                except Exception as e:
                    logger.warning(f"查询 '{query}' 出错: {e}")
                    all_results[query] = []
        return all_results
    
    def reciprocal_rank_fusion(self, document_ranks: List[List[Any]], k: int = 60) -> List[Tuple[Any, float]]:
        """使用互反排序融合算法合并多个查询结果"""
        fused_scores = {}
        for docs in document_ranks:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + k)
                
        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        return reranked_results
    
    def get_multiple_queries(self, question: str) -> Tuple[Any, List[Tuple[Any, float]]]:
        """生成多个查询变体并获取融合结果"""
        template = """生成与用户问题相关的五个不同版本的查询，以从向量数据库中检索相关文档。
每个查询应该从不同角度解决原始问题。
请用换行符分隔这些查询。
原始问题: {question}"""
        
        prompt = ChatPromptTemplate.from_template(template)
        generate_queries_chain = (
            prompt 
            | self.llm 
            | StrOutputParser() 
            | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
        )
        
        queries = generate_queries_chain.invoke({"question": question})
        logger.info(f"生成的查询变体: {queries}")
        
        retrieval_results = self.retrieval_and_rank(queries)
        document_ranks = [
            [doc for doc, _ in sorted(results, key=lambda x: x[1], reverse=True)]
            for query, results in retrieval_results.items()
        ]
        
        reranked_results = self.reciprocal_rank_fusion(document_ranks)
        return generate_queries_chain, reranked_results
    
    def get_unique_union(self, documents: List[List[Any]]) -> List[Any]:
        """获取文档列表的唯一并集"""
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        unique_docs = list(set(flattened_docs))
        return [loads(doc) for doc in unique_docs]
    
    def multi_query_generate_answer(self, question: str) -> str:
        """使用多查询方法生成答案"""
        generate_queries_chain, reranked_results = self.get_multiple_queries(question)
        
        # 取前5个相关文档
        top_docs = [doc for doc, _ in reranked_results[:5]]
        
        # 显示检索到的相关文档
        print("\n检索到的相关文档:")
        for i, doc in enumerate(top_docs, 1):
            print(f"{i}. {doc.page_content[:100]}...")
        
        # 答案生成模板
        answer_template = """基于以下上下文回答问题：
        {context}
        
        问题：{question}
        
        请保持答案简洁，不超过100字。"""
        
        answer_prompt = ChatPromptTemplate.from_template(answer_template)
        
        # 构建回答链
        answer_chain = (
            {"context": lambda x: "\n\n".join([doc.page_content for doc in top_docs]), 
             "question": itemgetter("question")}
            | answer_prompt
            | self.llm
            | StrOutputParser()
        )
        
        answer = answer_chain.invoke({"question": question})
        logger.info(f"生成的答案: {answer}")
        return answer
    
    def single_query_generate_answer(self, question: str) -> str:
        """使用单查询方法生成答案"""
        if not self.retriever:
            raise ValueError("检索器未初始化，请先嵌入数据")
            
        # 获取相关文档
        docs = self.retriever.get_relevant_documents(question)[:5]
        
        # 显示检索到的相关文档
        print("\n检索到的相关文档:")
        for i, doc in enumerate(docs, 1):
            print(f"{i}. {doc.page_content[:100]}...")
            
        # 答案生成模板
        template = """基于以下上下文回答问题：
        {context}
        
        问题：{question}
        
        请保持答案简洁，不超过100字。"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # 构建回答链
        rag_chain = (
            {"context": lambda x: docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        answer = rag_chain.invoke(question)
        logger.info(f"单查询生成的答案: {answer}")
        return answer

def interactive_mode(rag_system):
    """交互式问答模式"""
    print("\n===== 交互式问答模式 =====")
    print("输入问题进行提问，输入 'q' 或 'quit' 退出，输入 'mode' 切换查询模式")
    
    query_mode = "multi"  # 默认使用多查询模式
    
    while True:
        user_input = input("\n问题 (输入 'q' 退出, 'mode' 切换查询模式): ").strip()
        
        if user_input.lower() in ['q', 'quit']:
            break
            
        if user_input.lower() == 'mode':
            query_mode = "single" if query_mode == "multi" else "multi"
            print(f"已切换到 {'多查询' if query_mode == 'multi' else '单查询'} 模式")
            continue
            
        if not user_input:
            continue
            
        print(f"\n{'多查询' if query_mode == 'multi' else '单查询'} 模式回答:")
        
        try:
            if query_mode == "multi":
                answer = rag_system.multi_query_generate_answer(user_input)
            else:
                answer = rag_system.single_query_generate_answer(user_input)
                
            print(f"\n答案: {answer}")
            
        except Exception as e:
            print(f"发生错误: {e}")

if __name__ == "__main__":
    # 初始化RAG系统
    rag_system = MultiQueryRAG(model_name="deepseek-r1:7b")
    
    # 准备数据
    urls = ["https://wingfeitsang.github.io/home"]
    print("正在加载和处理数据...")
    chunks = rag_system.prepare_data(urls)
    
    # 嵌入数据
    print("正在嵌入数据到向量数据库...")
    vect_store, retriever = rag_system.embed_data(chunks, persist_directory='./chroma_db')
    
    # 进入交互式模式
    interactive_mode(rag_system)    