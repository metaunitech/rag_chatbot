import datetime

import lark_oapi as lark
from lark_oapi.api.drive.v1 import *
from lark_oapi.api.wiki.v2 import *
from lark_oapi.api.docx.v1 import *
import json
from pathlib import Path
from modules.Feishu.Feishu_messages import FeishuMessageHandler
from modules.rag_chatbot import GeneralRAG


class FeishuRAGBot(FeishuMessageHandler):
    def __init__(self, config_yaml_path, rag_chatbot_ins: GeneralRAG, global_token_type=lark.AccessTokenType.TENANT):
        super().__init__(config_yaml_path, global_token_type)
        self.rag_chatbot_ins = rag_chatbot_ins

    def process_msg_dict(self, msg_dicts):
        msgs = []
        for msg_dict in msg_dicts:
            event = msg_dict.get('event', {})
            sender_id = event.get('sender', {}).get('sender_id', {}).get('open_id')
            message = event.get('message', {})
            message_type = message.get('message_type')
            if message_type == 'text':
                content_str = message.get('content')
                content = json.loads(content_str) if content_str else {}
                question = content.get('text')
                message_id = message.get('message_id')
                answer, reason, files = self.rag_chatbot_ins.qa_main(question)
                rich_text_log = (
                    f'<b>【答案原因】</b>\n'
                    f'{reason}\n'
                    f'<b>【参考文献】</b>\n'
                    f'{files}'
                )
                self.send_message_by_template(receive_id=sender_id,
                                              template_id='AAqC5c9997YMX',
                                              template_variable={'question': question,
                                                                 'question_answer': answer,
                                                                 'reference': rich_text_log})
                msgs.append(rich_text_log)
            elif message_type == 'file':
                message_id = message.get('message_id')
                content_str = message.get('content')
                content = json.loads(content_str) if content_str else {}
                file_key = content.get('file_key')
                file_path = self.retrieve_file(message_id, file_key, Path(__file__).parent.parent/'src'/'inputs')
                self.rag_chatbot_ins.load_inputs()
                # SEND LOADED.
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                rich_text_log = (
                    f'<b>【已下载知识文档】</b>\n'
                    f'<i>{file_path.name}</i>\n'
                    f'<b>【时间】</b>: {current_time}\n'
                    '<b>【状态】</b>: 已录入知识库'
                    '<font color="green"><b>【提示】</b>: 可以开始问问题了</font>'
                )
                self.send_message_by_template(receive_id=sender_id,
                                              template_id='AAq7OhvOhSJB2', # Hardcoded.
                                              template_variable={'log_rich_text': rich_text_log})
                msgs.append(rich_text_log)
        return msgs

if __name__ == "__main__":
    demo = [{'schema': '2.0',
             'header': {'event_id': 'e0072e8cccf47802c01784ba854506fa', 'token': '0RA4IXx4vLtaJi4betQ4AfF6ToL8Rttb',
                        'create_time': '1726483524479', 'event_type': 'im.message.receive_v1',
                        'tenant_key': '16c5228b4a88575e', 'app_id': 'cli_a6681e9c2eb4d00d'},
             'event': {
            'message': {'chat_id': 'oc_7c1443de640303e5e9f47d02eb57ac1c', 'chat_type': 'p2p',
                        'content': '{"text":";;\\\\"}', 'create_time': '1726483524097',
                        'message_id': 'om_93751cf1efe431de5f644f5b6c3c3113', 'message_type': 'text',
                        'update_time': '1726483524097'}, 'sender': {
                'sender_id': {'open_id': 'ou_2401613ed164502ea6a20417c20dffee',
                              'union_id': 'on_c9326f5070c64b47c4f143961ccc1960', 'user_id': '5246a146'},
                'sender_type': 'user', 'tenant_key': '16c5228b4a88575e'}}}]

    ins = FeishuRAGBot(r'W:\Personal_Project\NeiRelated\projects\rag_chatbot\configs\feishu_config.yaml')
    ins.process_msg_dict(demo)
