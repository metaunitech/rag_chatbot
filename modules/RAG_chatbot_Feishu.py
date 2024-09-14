import lark_oapi as lark
from lark_oapi.api.drive.v1 import *
from lark_oapi.api.wiki.v2 import *
from lark_oapi.api.docx.v1 import *
from modules.Feishu.Feishu_messages import FeishuMessageHandler


class FeishuRAGBot(FeishuMessageHandler):
    def __init__(self, config_yaml_path, global_token_type=lark.AccessTokenType.TENANT):
        super().__init__(config_yaml_path)
        pass
