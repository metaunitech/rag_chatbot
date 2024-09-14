import os
from pathlib import Path
from loguru import logger
from modules.utils.file_utils import convert_pdf2img, is_scanned_pdf
from modules.utils.file_utils import transform_et2xlsx, transform_csv2xlsx, transform_xls2xlsx
from modules.utils.file_utils import extract_plaintext_from_xlsx, extract_plaintext_from_pdf
from modules.utils.ocr_handler import OCRHandler
from modules.utils.general_utils import method_descriptor, get_description
import fitz
import time
import shutil


class TextExtractionException(Exception):
    pass


class TextExtraction:
    def __init__(self, ocr_instance=None, batch_base=None):
        self.batch_dir = None
        if not ocr_instance:
            self.ocr_engine = OCRHandler()
        else:
            self.ocr_engine = ocr_instance
        if batch_base:
            if isinstance(batch_base, str):
                batch_base = Path(batch_base)
        else:
            batch_base = Path(__file__).parent
        self.batch_base = batch_base

    @method_descriptor({'raw_file_type': 'PDF',
                        'extraction_method': 'OCR',
                        'extraction_result_description': '结果中可能会出现错别字或多字，漏字的情况，请按上下文语意理解。'})
    def extract_pdf_with_ocr(self, input_file_path: Path, **kwargs):
        """
        Use OCR to extract plainText from PDF
        :param input_file_path:
        :return:
        """
        pic_result = convert_pdf2img(input_file_path, self.batch_dir)
        out_texts = []
        page_nums = kwargs.get('select_pages', None)
        page_nums = page_nums if page_nums else list(pic_result.keys())
        for page_num in page_nums:
            page_pic_path = pic_result[page_num]
            res = self.ocr_engine.get_ocr_result_by_row(page_pic_path, do_not_format=True, crop_method=3)
            logger.debug(res)
            out_texts.append(res)
        return out_texts

    @method_descriptor({'raw_file_type': 'PDF',
                        'extraction_method': 'OCR',
                        'extraction_result_description': '结果中可能会出现错别字或多字，漏字的情况，请按上下文语意理解。'})
    def extract_pdf_with_ocr(self, input_file_path: Path, **kwargs):
        """
        Use OCR to extract plainText from PDF
        :param input_file_path:
        :return:
        """
        pic_result = convert_pdf2img(input_file_path, self.batch_dir)
        out_texts = []
        page_nums = kwargs.get('select_pages', None)
        page_nums = page_nums if page_nums else list(pic_result.keys())
        for page_num in page_nums:
            page_pic_path = pic_result[page_num]
            res = self.ocr_engine.get_ocr_result_by_row(page_pic_path, do_not_format=True, crop_method=3)
            logger.debug(res)
            out_texts.append(res)
        return out_texts

    @method_descriptor({'raw_file_type': 'PDF',
                        'extraction_method': 'Fitz模块',
                        'extraction_result_description': '提取结果是文件中的文字原文内容'})
    def extract_pdf_with_fitz(self, input_file_path: Path, **kwargs):
        """
        Use Fitz module to extract plainText directly from PDF
        :param input_file_path:
        :return:
        """
        return extract_plaintext_from_pdf(input_file_path)

    @method_descriptor({'raw_file_type': 'EXCEL',
                        'extraction_method': '逐行提取表格',
                        'extraction_result_description': '提取结果是Excel文件逐行提取的文字信息，每一行的文字是通过用空格连接了该行的单元格，如果遇到空白单元格则为'
                                                         '" ", 表格中可能存在合并单元格等情况，请结合上下文理解整段语义'})
    def extract_excel(self, input_file_path: Path, **kwargs):
        """
        Use openpyxl to extract lines and converted to plainText from Excel.
        :param input_file_path:
        :return:
        """
        out_path = self.transform_2_xlsx(file_path=input_file_path)
        target_batch_file_name = self.batch_dir / out_path.name
        shutil.copy(out_path, target_batch_file_name)
        out_texts = extract_plaintext_from_xlsx(target_batch_file_name)
        return out_texts

    @method_descriptor({'raw_file_type': 'PDF',
                        'extraction_method': '按照版面的模块逐行提取',
                        'extraction_result_description': '结果中可能会出现错别字或多字，漏字的情况，请按上下文语意理解。'})
    def extract_pdf_with_structure(self, input_file_path: Path, **kwargs):
        """
        Use OCR to extract plainText from PDF
        :param input_file_path:
        :return:
        """
        pic_result = convert_pdf2img(input_file_path, self.batch_dir)
        out_texts = []
        page_nums = kwargs.get('select_pages', None)
        page_nums = page_nums if page_nums else list(pic_result.keys())
        for page_num in page_nums:
            page_pic_path = pic_result[page_num]
            res = self.ocr_engine.get_ocr_result_by_block(page_pic_path, if_debug=True, crop_method=0)
            logger.debug(res)
            out_texts.extend(res)
        return out_texts

    @staticmethod
    def transform_2_xlsx(file_path: Path):
        """
        Transform CSV or XLS file to XLSX format.
        :param file_path: Path to the input file (CSV or XLS).
        """
        if file_path.suffix == '.xlsx':
            out_path = file_path
        elif file_path.suffix == '.csv':
            out_path = transform_csv2xlsx(file_path)
        elif file_path.suffix == '.xls':
            out_path = transform_xls2xlsx(file_path)
        elif file_path.suffix == '.et':
            out_path = transform_et2xlsx(file_path)
        else:
            raise TextExtractionException("Unsupported file format for transformation.")
        logger.success(f'Resultant xlsx is stored at {out_path}')
        return out_path

    def __select_method(self, input_file_path: Path):
        if not input_file_path.exists():
            logger.error(f"Input file: {input_file_path} doesnt exist.")
            raise Exception(f"Input file: {input_file_path} doesnt exist.")
        if not input_file_path.is_file():
            logger.error(f"Input file: {input_file_path} should be a file.")
            raise Exception(f"Input file: {input_file_path} should be a file.")

        if input_file_path.suffix in ['.csv', '.xlsx', '.xls', '.et']:
            return self.extract_excel

        if input_file_path.suffix == '.pdf':
            is_scanned = is_scanned_pdf(str(input_file_path))
            with fitz.Document(input_file_path) as pdf_doc:
                page_number = pdf_doc.page_count
                if page_number == 0:
                    raise TextExtractionException(f"File might be broken. {input_file_path}")
                if is_scanned:
                    return self.extract_pdf_with_structure
                else:
                    return self.extract_pdf_with_structure

    def __call__(self, *args, **kwargs):
        """
        :param args:
        :param kwargs: input_file_path: str/pathlike,
        :return:
        """
        batch_id = kwargs.get('batch_id')
        batch_id = str(int(time.time() * 1000)) if not batch_id else batch_id
        self.batch_dir = self.batch_base / batch_id
        os.makedirs(self.batch_dir, exist_ok=True)
        logger.info(f"Initializing batch at: {self.batch_dir}")

        input_file_path = kwargs.get('input_file_path')
        if isinstance(input_file_path, str):
            logger.warning("Please input input_file_path as Path instance. Auto cast to Path instance.")
            input_file_path = Path(input_file_path)
        if not input_file_path:
            raise TextExtractionException("Fail to provide input file path.")
        select_pages = kwargs.get('select_pages')
        shutil.copy(input_file_path, self.batch_dir)

        method = self.__select_method(input_file_path=input_file_path)
        method_description = get_description(method)
        logger.info(f'Selected extraction method: <{method.__name__.upper()}>')

        out_lines = method(input_file_path=input_file_path, select_pages=select_pages)

        return out_lines, method_description, self.batch_dir


if __name__ == "__main__":
    pass
