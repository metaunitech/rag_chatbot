# encoding=utf-8
import json
import os
from pathlib import Path

import requests
from requests_toolbelt import MultipartEncoder
from loguru import logger

try:
    from .Feishu_base import FeishuApp
except:
    from Feishu_base import FeishuApp
import lark_oapi as lark
from lark_oapi.api.drive.v1 import *
from lark_oapi.api.wiki.v2 import *
from lark_oapi.api.docx.v1 import *

from typing import *


class FeishuMessageHandler(FeishuApp):
    def __init__(self, config_yaml_path, global_token_type=lark.AccessTokenType.TENANT):
        super().__init__(config_yaml_path)
        self.__global_token_type = global_token_type

    def send_message_by_template(self, receive_id, template_id, template_variable: dict):
        content = {'type': 'template', 'data': {'template_id':template_id, 'template_variable':template_variable}}
        card_json = {'receive_id': receive_id,
                     'msg_type': 'interactive',
                     'content': json.dumps(content, ensure_ascii=False)}
        # print(res.json())

if __name__ == "__main__":
    hack = 'https://open.feishu.cn/open-apis/bot/v2/hook/699721fe-7185-4d32-8fea-79b1970d85ec'
    webhook = 'https://open.feishu.cn/open-apis/bot/v2/hook/699721fe-7185-4d32-8fea-79b1970d85ec'
    ins = FeishuMessageHandler(r'W:\Personal_Project\NeiRelated\projects\rag_chatbot\configs\feishu_config.yaml')
    # ins.send_message_by_card_json(card_json=data_json, webhook_address=hack)
