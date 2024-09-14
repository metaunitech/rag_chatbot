from flask import Flask, request, jsonify
from loguru import logger
from modules.Feishu.Feishu_space import FeishuSpaceHandler
from threading import Lock
from pathlib import Path
import yaml

app = Flask(__name__)
FEISHU_CONFIG_PATH = Path(__file__).parent / 'configs' / 'feishu_config.yaml'
feishu_ins = FeishuSpaceHandler(config_yaml_path=str(FEISHU_CONFIG_PATH))

CONFIG_PATH = Path(__file__).parent / 'configs' / 'backend_configs.yaml'
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config_data = yaml.load(f, Loader=yaml.FullLoader)
LLM_PARAMS = config_data.get('LLM', {}).get("llm_params", {})
LLM_PLATFORM = LLM_PARAMS.get('platform')
logger.info(f"LLM_PLATFORM : {LLM_PLATFORM}")
logger.info(f"LLM_PARAMS: {LLM_PARAMS}")

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


# @app.route('/api/ios_lark_event', methods=['POST'])
# def receive_ios_data():
#     data = request.get_json()
#
#     # 打印接收到的数据
#     logger.info(f"Received Data: {data}")
#     msg_str_part = data.get('msg')
#     if not msg_str_part:
#         response = {'note': 'failed',
#                     'data': [],
#                     'data_string': '输入为空'}
#         return jsonify(response), 200
#     msg_note_part = data.get('note')
#     msg_str = f'消息主体：\n{msg_str_part}\n【备注】:{msg_note_part}'
#     response = {"challenge": data.get('challenge')} if data.get('challenge') else {'note': 'success',
#                                                                                    'data': inserted_msgs,
#                                                                                    'data_string': f'成功将标题：{",".join([i.get("标题") for i in inserted_msgs])}其归类为{",".join([i.get("记录分类") for i in inserted_msgs])}' if inserted_msgs else "没成功归类"}
#     return jsonify(response), 200
#
#
# @app.route('/api/shellProbe_lark_event', methods=['GET', 'POST'])
# def receive_shellProbe_data():
#     logger.info(f"received: {request.args}")
#     # 从请求的查询参数中获取数据
#     title = request.args.get('title')
#     platform = request.args.get('platform')
#     link = request.args.get('link')
#     content = request.args.get('content')
#
#     entry_dict = {'标题': '[Arxiv]' + title,
#                   '平台': platform,
#                   '链接': {'link': link},
#                   '内容': content,
#                   '记录分类': '稍后阅读'}
#
#     inserted_msgs = SNAP_ANTHONY_INSTANCE.upload_entry(entry_dict)
#     response = {'note': 'success',
#                 'data': inserted_msgs,
#                 'data_string': f'成功将标题：{",".join([i.get("标题") for i in inserted_msgs])}其归类为{",".join([i.get("记录分类") for i in inserted_msgs])}' if inserted_msgs else "没成功归类"}
#     return jsonify(response), 200


if __name__ == '__main__':
    app.run("0.0.0.0", port=6162)
