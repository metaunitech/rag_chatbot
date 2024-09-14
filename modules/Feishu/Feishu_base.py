import hashlib
import json
import os
import time

import requests
from selenium.webdriver.chrome.options import Options
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pathlib import Path
from loguru import logger

import yaml

class FeishuApp:
    def __init__(self, config_yaml_path):
        with open(config_yaml_path, 'r') as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
        self.__app_id = configs.get('app_id')
        self.__app_secret = configs.get('app_secret')
        self.__chromedriver_path = configs.get('chromedriver_executable')
        self.__chromedriver_version = configs.get('chromedriver_version')
        self.__cookie_path = configs.get('cookie_path')
        self.__account = configs.get('account')
        self._user_access_token = None

    def get_tenant_access_token(self):
        """
        :return: {'code': 0, 'expire': xxx, 'msg': 'ok', 'tenant_access_token': 'xxx'}
        """
        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
        }
        data = {'app_id': self.__app_id,
                'app_secret': self.__app_secret}
        res = requests.post(url, headers=headers, json=data)
        return res.json().get('tenant_access_token'), res.json()

    def get_app_access_token(self):
        """
        :return: { "app_access_token": "xxx", "code": 0, "expire": xxx, "msg": "ok", "app_access_token": "xxx"}
        """
        url = "https://open.feishu.cn/open-apis/auth/v3/app_access_token/internal"
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
        }
        data = {'app_id': self.__app_id,
                'app_secret': self.__app_secret}
        res = requests.post(url, headers=headers, json=data)
        return res.json().get('app_access_token'), res.json()

    def get_authen_code(self):
        url = 'https://open.feishu.cn/open-apis/authen/v1/authorize?app_id={APPID}&redirect_uri={REDIRECT_URI}'
        # "&scope={SCOPE}&state={STATE}"
        APPID = self.__app_id
        REDIRECT_URI = 'https%3A%2F%2Fwww.baidu.com'
        url = url.format(APPID=APPID,
                         REDIRECT_URI=REDIRECT_URI)
        print(url)
        response = requests.get(url)
        print(response.text)
        return response

    def get_user_access_token(self, headless=True):
        if not self._user_access_token or (
                self._user_access_token and time.time() - self._user_access_token[1] >= 2 * 60 * 60):
            logger.error("User Access Token expired.")
            if headless:
                try:
                    logger.info("Will try headless first.")
                    token = self.rpa_get_user_access_token(account=self.__account, if_headless=True)
                except Exception as e:
                    logger.error(str(e))
                    logger.error('Headless rpa get user access code failed. Will try normal rpa.')
                    token = self.rpa_get_user_access_token(account=self.__account)
                self._user_access_token = [token, time.time()]
                logger.success("User Access Token updated.")
            else:
                token = self.rpa_get_user_access_token(account=self.__account)
                self._user_access_token = [token, time.time()]
                logger.success("User Access Token updated.")

        return self._user_access_token[0]

    def store_current_cookie(self, driver, account):
        # Starts to store cookies
        cookies = driver.get_cookies()
        cookie_name = self.md5_hash(account)
        cookie_path = Path(self.__cookie_path) / f'{cookie_name}.json'
        os.makedirs(Path(self.__cookie_path), exist_ok=True)
        with open(cookie_path, 'w', encoding='utf-8') as f:
            json.dump(cookies, f, ensure_ascii=False, indent=4)
        logger.success("Cookie stored.")
        return cookie_path

    @staticmethod
    def md5_hash(input_string):
        # 创建 MD5 哈希对象
        md5_hash_obj = hashlib.md5()

        # 使用 update 方法逐步更新哈希对象
        # 这样可以处理较长的输入字符串而不必一次性加载整个字符串到内存中
        md5_hash_obj.update(input_string.encode('utf-8'))

        # 获取十六进制表示的哈希值
        md5_hash_value = md5_hash_obj.hexdigest()

        return md5_hash_value

    def rpa_get_user_access_token(self, account, if_headless=False, timeout=20):
        url = 'https://open.feishu.cn/api-explorer/'
        chrome_options = Options()
        # options = {}
        # chrome_options = ChromeOptions()
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--lang=zh_CN.UTF-8")
        # chrome_options.add_argument("--incognito")
        chrome_options.add_argument("--disable-dev-shm-usage")
        if if_headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument("--disable-popup-blocking")
        driver = uc.Chrome(options=chrome_options,
                           driver_executable_path=self.__chromedriver_path,
                           version_main=self.__chromedriver_version)
        driver.execute_cdp_cmd(
            "Network.setUserAgentOverride",
            {
                "userAgent": driver.execute_script(
                    "return navigator.userAgent"
                ).replace("Headless", "")
            },
        )
        driver.get(url)
        cookie_path = Path(self.__cookie_path) / f'{self.md5_hash(account)}.json'
        if cookie_path.exists():
            with open(cookie_path, 'r', encoding='utf-8') as f:
                cookie_dict = json.load(f)
            driver.get(url)
            for c in cookie_dict:
                ccc = {
                    'domain': c.get('domain'),
                    'name': c.get('name'),
                    'value': c.get('value'),
                    # "expires": '',
                    'path': '/',
                    'httpOnly': c.get('httpOnly'),
                    'HostOnly': c.get('HostOnly'),
                    'Secure': c.get('Secure')
                }

                logger.debug(f"Insert: {ccc}")
                driver.add_cookie(ccc)
            driver.refresh()
        time_interval = time.time()
        while 1:
            if time.time() - time_interval > timeout * 3:
                logger.warning("Login failed. Timeout.")
                raise Exception("Login failed. Timeout.")
            try:
                WebDriverWait(driver, timeout, 0.1).until(
                    EC.presence_of_element_located((By.XPATH, "//*[@class='op-header-profile-info']"))
                )
                logger.success("Login success.")
                self.store_current_cookie(driver, account)
                break
            except:
                logger.warning("Need to login.")
        logger.info("Starts to request for user_access_token.")
        WebDriverWait(driver, timeout, 0.1).until(
            EC.presence_of_all_elements_located((By.XPATH, "//button[text()='点击获取']"))
        )
        button_request = driver.find_elements(By.XPATH, "//button[text()='点击获取']")[1]
        button_request.click()
        try:
            button_oauthen = WebDriverWait(driver, timeout, 0.1).until(
                EC.presence_of_element_located((By.XPATH, "//*[text()='授权']"))
            )
            button_oauthen.click()
        except:
            pass
        WebDriverWait(driver, timeout, 0.1).until(
            EC.presence_of_all_elements_located(
                (By.XPATH, "//span[@class='universe-icon api-explorer-token-block__handle__linkable']"))
        )
        button_envisible = \
            driver.find_elements(By.XPATH, "//span[@class='universe-icon api-explorer-token-block__handle__linkable']")[
                0]
        button_envisible.click()
        code = WebDriverWait(driver, timeout, 0.1).until(
            EC.presence_of_element_located((By.XPATH, "//*[@class='api-explorer-token-block__handle__token']"))
        ).text
        driver.close()
        logger.success("Chrome killed.")
        return code


if __name__ == "__main__":
    ins = FeishuApp(r'W:\Personal_Project\metaunitech\shellProbe_manager\configs\feishu_config.yaml')
    code = ins.get_user_access_token()
    print(code)
    # print(ins.get_tenant_access_token())
