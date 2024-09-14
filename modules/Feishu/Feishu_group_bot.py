import requests


class FeishuBotHandler:
    @staticmethod
    def _add_at_list(current_text=None, at_list=None, mention_text=None):
        if not at_list:
            return current_text
        current_text = current_text if current_text else ''
        mention_text = mention_text if mention_text else '[新更新提醒]'
        at_final_message_raw_list = []
        if at_list == 'ALL':
            at_final_message_raw_list.append(f'<at user_id="all">所有ShellProber</at>')
        elif isinstance(at_list, list):
            for mention in at_list:
                if isinstance(mention, tuple):
                    at_final_message_raw_list.append(f'<at user_id="{mention[0]}">{mention[1]}</at>')
                else:
                    at_final_message_raw_list.append(f'<at user_id="{mention}">{mention}</at>')
        else:
            return current_text
        return mention_text + '\n' + current_text + '\n' + ''.join(at_final_message_raw_list)

    def create_message(self, text=None, at_list=None):
        pass

    def test(self):
        webhook = "https://open.feishu.cn/open-apis/bot/v2/hook/21c2eaf6-4a87-40f5-8d4a-23217f66b52b"
        data = {"msg_type": "text",
                "content": {"text": self._add_at_list("Arxiv request example", ['ou_ghi'])}}
        res = requests.post(webhook, json=data)
        print(res)
        print(res.json())


if __name__ == "__main__":
    ins = FeishuBotHandler()
    ins.test()
