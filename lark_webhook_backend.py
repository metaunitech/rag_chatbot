from flask import Flask, request, jsonify
from loguru import logger
from modules.RAG_chatbot_Feishu import FeishuRAGBot
from modules.rag_chatbot import GeneralRAG
from threading import Lock
from pathlib import Path
import yaml

app = Flask(__name__)


CONFIG_PATH = Path(__file__).parent / 'configs' / 'backend_configs.yaml'
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config_data = yaml.load(f, Loader=yaml.FullLoader)
LLM_PARAMS = config_data.get('LLM', {}).get("llm_params", {})
LLM_PLATFORM = LLM_PARAMS.get('platform')
logger.info(f"LLM_PLATFORM : {LLM_PLATFORM}")
logger.info(f"LLM_PARAMS: {LLM_PARAMS}")

FEISHU_CONFIG_PATH = Path(__file__).parent / 'configs' / 'feishu_config.yaml'
RAG_CHATBOT = GeneralRAG(llm_params=LLM_PARAMS,
                         documents_path_dir=Path(__file__).parent/'src'/'inputs')
FEISHU_CHATBOT = FeishuRAGBot(FEISHU_CONFIG_PATH,
                              RAG_CHATBOT)

previous_id_lock = Lock()
app.config['data_queue'] = []
app.config['previous_id'] = None


@app.route('/api/lark_event', methods=['POST'])
def receive_data():
    # 获取POST请求中的JSON数据
    data = request.get_json()

    # 打印接收到的数据
    logger.info(f"Received Data: {data}")
    inserted_msgs = []
    if data.get('header', {}).get('event_type') == 'im.message.receive_v1':
        with previous_id_lock:  # 使用锁保护previous_id的读写操作
            msg_id = data.get('event', {}).get('message', {}).get('message_id')
            logger.info(f"Received Message. MSG_ID: {msg_id}")
            if msg_id != app.config['previous_id']:
                app.config['data_queue'].append(data)
            if app.config['data_queue'] and msg_id != app.config['previous_id']:
                msg_dicts = app.config['data_queue']
                inserted_msgs = FEISHU_CHATBOT.process_msg_dict(msg_dicts)
                logger.info(msg_dicts)
                app.config['data_queue'] = []
            app.config['previous_id'] = msg_id

    logger.warning("DATA QUEUE:")
    logger.info(app.config['data_queue'])
    logger.warning("PREVIOUS ID:")
    logger.info(app.config['previous_id'])

    # 返回一个响应给客户端
    response = {"challenge": data.get('challenge')} if data.get('challenge') else {'note': 'success',
                                                                                   'data': inserted_msgs,
                                                                                   'data_string': f'PASS'}
    return jsonify(response), 200


if __name__ == '__main__':
    app.run("0.0.0.0", port=6162)
