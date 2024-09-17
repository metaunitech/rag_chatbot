import faiss
import tqdm
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_text_splitters import CharacterTextSplitter
from pathlib import Path
from glob import glob
from loguru import logger
import uuid
import json

from modules.text_extraction import TextExtraction
from modules.utils.ocr_handler import OCRHandler

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class DocSummary(BaseModel):
    summary: str = Field(
        description='本文档包含的内容，简介，文档概览。'
    )


class Answer(BaseModel):
    answer: str = Field(
        description='对于用户提出的问题的回答'
    )
    reason: str = Field(
        description='作出回答的原文依据。请把原文中的描述包含在依据中。格式如下：通过原文中：xxxx 中得知'
    )


class GeneralRAG:
    def __init__(self, llm_params, documents_path_dir=Path(__file__).parent / 'src' / 'inputs'):
        self.api_token = llm_params.get('api_key')
        self.model_name = llm_params.get('model_name')
        self.embedding = ZhipuAIEmbeddings(
            model="embedding-3",
            api_key=self.api_token
        )
        self.documents_path_dir = documents_path_dir
        self.stored_vector_path = documents_path_dir.parent / 'faiss_index'
        self.ocr_instance = OCRHandler()
        self.text_extractor = TextExtraction(ocr_instance=self.ocr_instance)
        if self.stored_vector_path.exists():
            try:
                self.vector_store = FAISS.load_local(str(self.stored_vector_path),
                                                     self.embedding,
                                                     allow_dangerous_deserialization=True)
            except Exception as e:
                logger.error(f"ERROR loading FAISS index: {self.stored_vector_path}:{str(e)}")
                self.vector_store = self.create_new_vector_store()
                self.load_inputs()
        else:
            self.vector_store = self.create_new_vector_store()
            self.load_inputs()

    def create_new_vector_store(self):
        logger.info("Starts to create new FAISS store instance.")
        index = faiss.IndexFlatL2(len(self.embedding.embed_query("hello world")))
        vector_store = FAISS(
            embedding_function=self.embedding,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        logger.success("FAISS store instance initiated.")
        return vector_store

    def create_llm_instance(self):
        return ChatOpenAI(temperature=0.95,
                          model=self.model_name,
                          openai_api_key=self.api_token,
                          openai_api_base="https://open.bigmodel.cn/api/paas/v4/")

    def summarize_doc(self, content_lines):
        llm_ins = self.create_llm_instance()
        parser = PydanticOutputParser(pydantic_object=DocSummary)
        retry_parser = OutputFixingParser.from_llm(parser=parser, llm=llm_ins)

        format_instruction = parser.get_format_instructions()
        content = '\n'.join(content_lines)
        prompt = f"""# Role: 你是一个非常有经验的文档助理，你擅长阅读文档并总结文档内容。
        # Task: 
        我会提供你一篇文档内容，你需要帮我总结一下文档的内容是。
        请根据我的要求返回我JSON格式的结果，具体要求如下：
        {format_instruction},

        # Doc Content:
        {content}

        # Output：
        YOUR ANSWER(请返回中文结果用JSON格式):
        """
        res_raw = llm_ins.invoke(prompt)
        res_content = res_raw.content
        summary = retry_parser.parse(res_content).summary
        return summary

    def load_inputs(self):
        logger.info(f"Starts to load input docs from {self.documents_path_dir}")
        files = glob(str(self.documents_path_dir / '*.*'))
        for file_path in tqdm.tqdm(files):
            if '_raw_res' in file_path or Path(file_path).name.startswith('~$'):
                continue
            doc_id = self.store_document(Path(file_path))
            logger.success(f"Loaded {file_path} to DOC_ID: {doc_id}")

    def parse_document(self, document_path: Path):
        return self.text_extractor(input_file_path=document_path)

    def store_document(self, document_path: Path, if_summarize=True):
        doc_raw_data_json = document_path.parent / (document_path.name + "_raw_res.json")
        if doc_raw_data_json.exists():
            logger.warning(f"{document_path} already parsed.")
            with open(doc_raw_data_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            line_contents = data.get("line_contents")
            metadata = data.get('metadata')
        else:
            line_contents, method_description, batch_dir = self.parse_document(document_path)
            metadata = {"file_path": str(document_path), 'method_description': method_description}

            with open(doc_raw_data_json, 'w', encoding='utf-8') as f:
                json.dump({'line_contents': line_contents, 'metadata': metadata}, f, indent=4, ensure_ascii=False)
            logger.success(f"{document_path} parsed. Stored to {doc_raw_data_json}")

        if if_summarize and 'summary' not in metadata:
            logger.warning("Should summarize doc into metadata.")
            doc_summary = self.summarize_doc(line_contents)
            metadata['summary'] = doc_summary
            logger.success(doc_summary)
            with open(doc_raw_data_json, 'w', encoding='utf-8') as f:
                json.dump({'line_contents': line_contents, 'metadata': metadata}, f, indent=4, ensure_ascii=False)
            logger.success(f"{document_path} parsed. Stored to {doc_raw_data_json}")

        text_splitter = CharacterTextSplitter(separator=",",
                                              chunk_size=100,
                                              chunk_overlap=20,
                                              length_function=len,
                                              is_separator_regex=False)

        docs = []
        for idx, content in enumerate(line_contents):
            current_meta = metadata.copy()
            current_meta['segment_idx'] = idx
            doc = Document(page_content=content,
                           metadata=current_meta)
            docs.append(doc)

        # docs = text_splitter.create_documents(texts=['\n'.join(line_contents)],
        #                                       metadatas=[metadata])

        # doc = Document(page_content='\n'.join(line_contents),
        #                metadata=metadata)
        # docs = [doc]
        # docs = [Document(page_content=content, metadata=metadata)]
        doc_ids = [uuid.uuid4() for _ in range(len(docs))]
        for doc, doc_id in zip(docs, doc_ids):
            logger.info(f"adding doc_id: {doc_id}")
            self.vector_store.add_documents(documents=[doc],
                                            ids=[doc_id])
        self.vector_store.save_local(str(self.stored_vector_path))
        return doc_ids

    def query_document(self, query, k=5):
        results = self.vector_store.similarity_search(
            query,
            k=k,
        )
        return results

    def answer_question_by_background(self, question, background_contents=None):
        llm_ins = self.create_llm_instance()
        parser = PydanticOutputParser(pydantic_object=Answer)
        retry_parser = OutputFixingParser.from_llm(parser=parser, llm=llm_ins)

        format_instruction = parser.get_format_instructions()
        content = '\n\n'.join(background_contents) if background_contents else '<客户未提供背景知识，请根据你的经验回答>'
        prompt = f"""# Role: 你是一个非常有经验的业务员，你擅长根据提供的文档上下文回答用户的问题。
# Task: 
我会提供你一些文档内容，你需要严格根据文档内容回答客户提供的问题并给出原文依据。。

# Doc Content(每一段都是HTML格式的内容):
{content}

# Question:
{question}

{format_instruction}

YOUR ANSWER(请返回中文结果。注意，请严格依据DocContent提供的页面内容回答。结合页面中的内容的结构，理解内容。Step by step 回答问题。不要杜撰答案。):
                """
        logger.debug(prompt)
        res_raw = llm_ins.invoke(prompt)
        res_content = res_raw.content
        logger.debug(res_content)
        answer_instance = retry_parser.parse(res_content)
        answer = answer_instance.answer
        reason = answer_instance.reason
        return answer, reason

    def qa_main(self, question, k=4):
        results = self.query_document(question, k)
        files = [f"{Path(r.metadata['file_path']).name}: Segment_index: {r.metadata['segment_idx']}" for r in results]
        contents = list(set(r.page_content for r in results))
        logger.success(f"Will answer according to {files}")
        answer, reason = self.answer_question_by_background(question, contents)
        return answer, reason, files


if __name__ == "__main__":
    import yaml

    backend_configs_path = Path(__file__).parent.parent / 'configs' / 'backend_configs.yaml'
    with open(backend_configs_path, 'r', encoding='utf-8') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    LLM_PARAMS = config_data.get('LLM', {}).get("llm_params", {})
    ins = GeneralRAG(LLM_PARAMS, Path(__file__).parent.parent / 'src' / 'inputs')
    answer, reason, files = ins.qa_main('体系部的职责是什么')
    logger.info(files)
    print('HERE')
