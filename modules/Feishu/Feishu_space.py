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


class FeishuSpaceHandler(FeishuApp):
    def __init__(self, config_yaml_path, global_token_type=lark.AccessTokenType.TENANT, backend_ip='127.0.0.1:5000'):
        super().__init__(config_yaml_path)
        self.__folder_map = {}
        self.__global_token_type = global_token_type
        self.__backend_ip = backend_ip
        logger.success(f'GLOBAL TOKEN TYPE is: {self.__global_token_type}')

    def get_auth_token(self):
        token_type = self.__global_token_type
        if token_type == lark.AccessTokenType.USER:
            logger.info("Getting USER Token")
            bearer = self.get_user_access_token()
        elif token_type == lark.AccessTokenType.TENANT:
            logger.info("Getting TENANT Token")
            bearer, _ = self.get_tenant_access_token()
        else:
            logger.info("Getting APP Token")
            bearer, _ = self.get_app_access_token()
        return bearer

    @staticmethod
    def slice_list(input_list, slice_length):
        return [input_list[i:i + slice_length] for i in range(0, len(input_list), slice_length)]

    def _sdk_general(self,
                     uri,
                     request_type: Union[lark.HttpMethod],
                     path_params: Union[Dict, None] = None,
                     token_type: Union[lark.AccessTokenType] = None,
                     queries_params: Union[List[Tuple], None] = None,
                     bodies_params: Union[Dict, None] = None):
        if not token_type:
            token_type = self.__global_token_type
        client = lark.Client.builder() \
            .enable_set_token(True) \
            .log_level(lark.LogLevel.DEBUG) \
            .build()
        if token_type == lark.AccessTokenType.USER:
            bearer = self.get_user_access_token()
            option = lark.RequestOption.builder().user_access_token(bearer).build()

        elif token_type == lark.AccessTokenType.TENANT:
            bearer, _ = self.get_tenant_access_token()
            option = lark.RequestOption.builder().tenant_access_token(bearer).build()

        else:
            bearer, _ = self.get_app_access_token()
            option = lark.RequestOption.builder().app_access_token(bearer).build()

        request: lark.BaseRequest = lark.BaseRequest.builder() \
            .http_method(request_type) \
            .uri(uri) \
            .paths(path_params) \
            .headers({"Authorization": f'Bearer {bearer}'}) \
            .token_types({token_type}) \
            .queries(queries_params) \
            .body(bodies_params) \
            .build()
        response: lark.BaseResponse = client.request(request, option)
        # 处理失败返回
        if not response.success():
            lark.logger.error(
                f"{uri} failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
            return {}
        response = json.loads(response.raw.content)

        # 处理业务结果
        lark.logger.info(lark.JSON.marshal(response['data'], indent=4))
        return response

    """
    Handle spaces
    """

    def get_drive_files(self, page_size=None, page_token=None, folder_token=None, order_by=None, direction=None,
                        user_id_type=None):
        """
        # TODO
        :param page_size:
        :param page_token:
        :param order_by:
        :param direction:
        :param user_id_type:
        :return:
        """
        uri = '/open-apis/drive/v1/files'
        next_page_token = None
        variables = {
            'page_size': page_size,
            'page_token': page_token,
            'folder_token': folder_token,
            'order_by': order_by,
            'direction': direction,
            'user_id_type': user_id_type
        }
        files = []
        while 1:
            if next_page_token:
                variables["page_token"] = next_page_token
            queries = []
            for var_name, var_value in variables.items():
                if var_value:
                    queries.append((var_name, var_value))
            response = self._sdk_general(uri=uri,
                                         request_type=lark.HttpMethod.GET,
                                         token_type=self.__global_token_type,
                                         queries_params=queries)

            if response.get('code', -1) != 0:
                logger.error(response.get('msg'))
                break
            logger.debug(response['data']['files'])
            files.extend(response['data']['files'])
            if not response['data']['has_more']:
                break
            next_page_token = response['data']['next_page_token']
        return files

    def get_root_folder_token(self):
        url = "https://open.feishu.cn/open-apis/drive/explorer/v2/root_folder/meta"
        bearer = self.get_auth_token()
        headers = {
            'Authorization': f'Bearer {bearer}'
        }
        response = requests.get(url, headers=headers)
        if response.json().get('code', -1) != 0:
            logger.error(response.json().get('msg'))
            return None
        token = response.json().get('data', {}).get('token')
        return token

    def create_folder(self, folder_name='新建文件夹', parent_folder_token=None):
        """

        :param folder_name:
        :param parent_folder_token:
        :return: folder token, folder url
        """
        # 创建client
        # 使用 user_access_token 需开启 token 配置, 并在 request_option 中配置 token
        if not parent_folder_token:
            parent_folder_token = self.get_root_folder_token()
        folder_token = self.__folder_map.get(parent_folder_token, {}).get(folder_name)
        if folder_token:
            return folder_token, None

        client = lark.Client.builder() \
            .enable_set_token(True) \
            .log_level(lark.LogLevel.DEBUG) \
            .build()

        # 构造请求对象
        request: CreateFolderFileRequest = CreateFolderFileRequest.builder() \
            .request_body(CreateFolderFileRequestBody.builder()
                          .name(folder_name)
                          .folder_token(parent_folder_token)
                          .build()) \
            .build()

        user_access_token = self.get_auth_token()

        # 发起请求
        option = lark.RequestOption.builder().user_access_token(
            user_access_token).build()
        response: CreateFolderFileResponse = client.drive.v1.file.create_folder(request, option)

        # 处理失败返回
        if not response.success():
            lark.logger.error(
                f"client.drive.v1.file.create_folder failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
            return None, None

        # 处理业务结果
        lark.logger.info(lark.JSON.marshal(response.data, indent=4))
        if parent_folder_token not in self.__folder_map:
            self.__folder_map[parent_folder_token] = {}
        self.__folder_map[parent_folder_token][folder_name] = response.data.token
        return response.data.token, response.data.url

    def get_spaces(self):
        # 创建client
        # 使用 user_access_token 需开启 token 配置, 并在 request_option 中配置 token
        client = lark.Client.builder() \
            .enable_set_token(True) \
            .log_level(lark.LogLevel.DEBUG) \
            .build()
        bearer = self.get_auth_token()

        # 构造请求对象
        request: ListSpaceRequest = ListSpaceRequest.builder() \
            .build()

        # 发起请求
        option = lark.RequestOption.builder().tenant_access_token(bearer).build()
        response: ListSpaceResponse = client.wiki.v2.space.list(request, option)

        # 处理失败返回
        if not response.success():
            lark.logger.error(
                f"client.wiki.v2.space.list failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
            return

        # 处理业务结果
        lark.logger.info(lark.JSON.marshal(response.data, indent=4))

    def upload_file(self, file_path, parent_node, parent_type='explorer', file_name=None):
        file_size = os.path.getsize(file_path)
        file_name = file_name if file_name else Path(file_path).name
        url = "https://open.feishu.cn/open-apis/drive/v1/files/upload_all"

        form = {'file_name': file_name,
                'parent_type': parent_type,
                'parent_node': parent_node,
                'size': str(file_size),
                'file': (open(file_path, 'rb'))}
        multi_form = MultipartEncoder(form)
        bearer = self.get_auth_token()

        headers = {'Authorization': f'Bearer {bearer}', 'Content-Type': multi_form.content_type}
        response = requests.request("POST", url, headers=headers, data=multi_form)
        if response.json().get('code', -1) != 0:
            logger.error(response.json().get('msg'))
            return None
        file_token = response.json().get('data', {}).get('file_token')
        return file_token

    def move_doc_to_space(self, doc_token, space_token):
        pass

    """
    Handle docs
    """

    def create_document(self, document_title, folder_token=None):
        client = lark.Client.builder() \
            .enable_set_token(True) \
            .log_level(lark.LogLevel.DEBUG) \
            .build()

        # 构造请求对象
        request: CreateDocumentRequest = CreateDocumentRequest.builder() \
            .request_body(CreateDocumentRequestBody.builder()
                          .folder_token(folder_token)
                          .title(document_title)
                          .build()) \
            .build()
        bearer = self.get_auth_token()
        # 发起请求
        option = lark.RequestOption.builder().user_access_token(bearer).build()
        response: CreateDocumentResponse = client.docx.v1.document.create(request, option)

        # 处理失败返回
        if not response.success():
            lark.logger.error(
                f"client.docx.v1.document.create failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
            return

        # 处理业务结果
        lark.logger.info(lark.JSON.marshal(response.data, indent=4))
        return response.data.document.document_id

    def get_doc_permission(self, doc_token, doc_type):

        return self._sdk_general(f"/open-apis/drive/v2/permissions/:token/public",
                                 request_type=lark.HttpMethod.GET,
                                 path_params={"token": doc_token},
                                 queries_params=[('type', doc_type)])

    def set_doc_permission(self, doc_token, file_type, permission=None):
        if not permission:
            permission = {
                "external_access_entity": "open",
                "security_entity": "anyone_can_view",
                "comment_entity": "anyone_can_view",
                "share_entity": "anyone",
                "manage_collaborator_entity": "collaborator_full_access",
                "link_share_entity": "anyone_readable",
                "copy_entity": "only_full_access"
            }
        uri = '/open-apis/drive/v2/permissions/:token/public'
        response = self._sdk_general(uri=uri,
                                     request_type=lark.HttpMethod.PATCH,
                                     path_params={'token': doc_token},
                                     queries_params=[('type', file_type)],
                                     bodies_params=permission)
        return response

    def list_doc_all_blocks(self, document_id):
        uri = '/open-apis/docx/v1/documents/:document_id/blocks'
        blocks = []
        queries_params = []
        while 1:
            response = self._sdk_general(uri=uri,
                                         request_type=lark.HttpMethod.GET,
                                         path_params={'document_id': document_id},
                                         queries_params=queries_params)

            if not response['data']['has_more']:
                blocks.extend(response['data']['items'])
                break
            queries_params = [('page_token', response['data']['page_token'])]
        return blocks

    def get_document_base_block_id(self, document_id):
        parent_block_id = self.list_doc_all_blocks(document_id)
        if not parent_block_id:
            raise Exception("Fail to get all blocks in document.")
        parent_block_id = parent_block_id[0]['block_id']
        return parent_block_id

    """
    Handle single doc
    """

    def add_blocks(self,
                   document_token,
                   parent_block_id,
                   block_type_idx,
                   block_type_name,
                   block_type_args: List[Dict],
                   index=-1,
                   window_size=5):
        uri = f"/open-apis/docx/v1/documents/:document_id/blocks/:block_id/children"
        children = []
        responses = []
        for block_type_arg in block_type_args:
            children.append({'block_type': block_type_idx,
                             block_type_name: block_type_arg})
        for segment in self.slice_list(children, window_size):
            response = self._sdk_general(uri=uri,
                                         request_type=lark.HttpMethod.POST,
                                         path_params={'document_id': document_token,
                                                      'block_id': parent_block_id},
                                         bodies_params={'children': segment,
                                                        'index': index})
            if not response:
                logger.error(f"Failed to insert {block_type_name}: {str(segment)}")
            responses.append(response)
        return responses

    def del_block_children(self, document_id, block_id, start_index, end_index):
        uri = '/open-apis/docx/v1/documents/:document_id/blocks/:block_id/children/batch_delete'
        return self._sdk_general(uri=uri,
                                 request_type=lark.HttpMethod.DELETE,
                                 path_params={"block_id": block_id,
                                              'document_id': document_id},
                                 bodies_params={'start_index': start_index, 'end_index': end_index})

    def del_block(self, document_id, block_id):
        parent_id = self.get_block_parent(document_id=document_id, block_id=block_id)
        if not parent_id:
            logger.error(f"Fail to del block {block_id}. Cannot find parent block.")
            return
        parent_block_children = self.get_block_children(document_id=document_id, block_id=block_id)
        if block_id not in parent_block_children:
            logger.error(f"Fail to del block {block_id}. {block_id} not in parent block children [{parent_id}].")
            return
        block_index = parent_block_children.index(block_id)
        response = self.del_block_children(document_id=document_id,
                                           block_id=parent_id,
                                           start_index=block_index,
                                           end_index=block_index + 1)
        if response['code']:
            logger.error(f"Fail to del block {block_id}. {str(response)}")
            return False
        else:
            logger.success(f"{block_id} DELETED.")
            return True

    def modify_image_block(self, document_id, target_block_id, replaced_image_token,
                           image_kwargs: Union[Dict, None] = None):
        uri = '/open-apis/docx/v1/documents/:document_id/blocks/:block_id'
        if not image_kwargs:
            image_kwargs = {'token': replaced_image_token}
        return self._sdk_general(uri=uri,
                                 request_type=lark.HttpMethod.PATCH,
                                 path_params={"block_id": target_block_id,
                                              'document_id': document_id},
                                 bodies_params={'replace_image': image_kwargs})

    def get_block_info(self, document_id, block_id):
        uri = '/open-apis/docx/v1/documents/:document_id/blocks/:block_id'
        return self._sdk_general(uri=uri,
                                 request_type=lark.HttpMethod.GET,
                                 path_params={"block_id": block_id,
                                              'document_id': document_id})

    def get_block_children(self, document_id, block_id):
        return self.get_block_info(document_id=document_id,
                                   block_id=block_id).get('data', {}).get('block', {}).get('children', [])

    def get_block_parent(self, document_id, block_id):
        return self.get_block_info(document_id=document_id,
                                   block_id=block_id).get('data', {}).get('block', {}).get('parent_id')

    def add_callouts(self, document_id, callouts_num=1, parent_block_id=None, background_color_idx=5, border_color=2,
                     text_color=7):
        if not parent_block_id:
            parent_block_id = self.get_document_base_block_id(document_id)
        response = self.add_blocks(document_token=document_id,
                                   parent_block_id=parent_block_id,
                                   block_type_idx=19,
                                   block_type_name='callout',
                                   block_type_args=[{'background_color': background_color_idx,
                                                     'border_color': border_color,
                                                     'text_color': text_color}] * callouts_num)
        block_ids = []
        for res in response:
            children = res['data']['children']
            block_ids.extend([i['block_id'] for i in children])
        return block_ids

    def add_section(self,
                    document_id,
                    section_name,
                    parent_block_id=None,
                    header_level: int = 0,
                    if_bold: bool = True,
                    background_idx: int = None):
        if not parent_block_id:
            parent_block_id = self.get_document_base_block_id(document_id)
        self.add_titles(document_id=document_id,
                        titles=[section_name],
                        parent_block_id=parent_block_id,
                        header_level=header_level,
                        if_bold=if_bold,
                        background_idx=background_idx)
        callout_id = self.add_callouts(document_id=document_id,
                                       callouts_num=1,
                                       parent_block_id=parent_block_id)[0]
        return callout_id

    def add_callouts_with_titles(self,
                                 document_id,
                                 titles: list,
                                 urls: Union[List, None] = None,
                                 parent_block_id=None,
                                 header_level: int = 0,
                                 if_bold: bool = True,
                                 background_idx: int = None):
        callout_ids = self.add_callouts(document_id=document_id,
                                        callouts_num=len(titles),
                                        parent_block_id=parent_block_id)
        if not urls:
            block_type_args = [{'elements': [{'text_run': {"content": title,
                                                           'text_element_style': {'bold': if_bold,
                                                                                  'background_color': background_idx}
                                                           }}]} for title in titles] if background_idx else [
                {'elements': [{'text_run': {"content": title,
                                            'text_element_style': {'bold': if_bold}
                                            }}]} for title in titles]
        else:
            block_type_args = [{'elements': [{'text_run': {"content": title,
                                                           'text_element_style': {'bold': if_bold,
                                                                                  'background_color': background_idx,
                                                                                  'link': {'url': url}}
                                                           }}]} for title, url in
                               zip(titles, urls)] if background_idx else [{'elements': [{'text_run': {"content": title,
                                                                                                      'text_element_style': {
                                                                                                          'bold': if_bold,
                                                                                                          'link': {
                                                                                                              'url': url}}
                                                                                                      }}]} for
                                                                          title, url in zip(titles, urls)]
        for block_type_arg, callout_id in zip(block_type_args, callout_ids):
            self.add_blocks(document_token=document_id,
                            parent_block_id=callout_id,
                            block_type_idx=3 + header_level,
                            block_type_name=f'heading{header_level + 1}',
                            block_type_args=[block_type_arg],
                            index=0)
            res = self.del_block_children(document_id=document_id,
                                          block_id=callout_id,
                                          start_index=1,
                                          end_index=2)
            if not res:
                logger.warning("Failed to remove empty block.")
        return callout_ids

    def add_titles(self,
                   document_id,
                   titles: list,
                   urls: Union[List, None] = None,
                   parent_block_id=None,
                   header_level: int = 0,
                   if_bold: bool = True,
                   background_idx: int = None):
        if not parent_block_id:
            parent_block_id = self.get_document_base_block_id(document_id)
        if not urls:
            block_type_args = [{'elements': [{'text_run': {"content": title,
                                                           'text_element_style': {'bold': if_bold,
                                                                                  'background_color': background_idx}
                                                           }}]} for title in titles] if background_idx else [
                {'elements': [{'text_run': {"content": title,
                                            'text_element_style': {'bold': if_bold}
                                            }}]} for title in titles]
        else:
            block_type_args = [{'elements': [{'text_run': {"content": title,
                                                           'text_element_style': {'bold': if_bold,
                                                                                  'background_color': background_idx,
                                                                                  'link': {'url': url}}
                                                           }}]} for title, url in
                               zip(titles, urls)] if background_idx else [{'elements': [{'text_run': {"content": title,
                                                                                                      'text_element_style': {
                                                                                                          'bold': if_bold,
                                                                                                          'link': {
                                                                                                              'url': url}}
                                                                                                      }}]} for
                                                                          title, url in zip(titles, urls)]
        block_ids = []
        for block_type_arg in block_type_args:
            response = self.add_blocks(document_token=document_id,
                                       parent_block_id=parent_block_id,
                                       block_type_idx=3 + header_level,
                                       block_type_name=f'heading{header_level + 1}',
                                       block_type_args=[block_type_arg])
            for res in response:
                children = res['data']['children']
                block_ids.extend([i['block_id'] for i in children])
        if len(block_ids) != len(titles):
            raise Exception(
                f"Title assertion failed. Target title length: {len(titles)}. Current title length: {len(block_ids)}")
        return block_ids

    def add_title_texts(self,
                        document_id,
                        title: str,
                        url: Union[str, None] = None,
                        parent_block_id=None,
                        header_level: int = 0,
                        if_bold: bool = True,
                        background_idx: int = None,
                        texts: Union[List, None] = None
                        ):
        if not parent_block_id:
            parent_block_id = self.get_document_base_block_id(document_id)
        if not url:
            block_type_arg = {'elements':
                                  [{'text_run': {"content": title,
                                                 'text_element_style': {'bold': if_bold,
                                                                        'background_color': background_idx}
                                                 }
                                    }]
                              } if background_idx else {'elements':
                                                            [{'text_run': {"content": title,
                                                                           'text_element_style': {'bold': if_bold}}
                                                              }]
                                                        }
        else:
            block_type_arg = {'elements': [{'text_run': {"content": title,
                                                         'text_element_style': {'bold': if_bold,
                                                                                'background_color': background_idx,
                                                                                'link': {'url': url}}
                                                         }}
                                           ]} if background_idx else {'elements': [{'text_run':
                                                                                        {"content": title,
                                                                                         'text_element_style': {
                                                                                             'bold': if_bold,
                                                                                             'link': {'url': url}}}
                                                                                    }]}

        response = self.add_blocks(document_token=document_id,
                                   parent_block_id=parent_block_id,
                                   block_type_idx=3 + header_level,
                                   block_type_name=f'heading{header_level + 1}',
                                   block_type_args=[block_type_arg])

        children = response[0]['data']['children']
        block_id = children[0]['block_id']
        if texts:
            block_type_args = [{'elements':
                                    [{'text_run': {"content": unit_text}}]} for unit_text in texts]
            self.add_blocks(document_token=document_id,
                            parent_block_id=parent_block_id,
                            block_type_idx=2,
                            block_type_name='text',
                            block_type_args=block_type_args)
        return block_id

    def add_image(self, document_id, image_paths: List[str], parent_block_id=None):
        if not parent_block_id:
            parent_block_id = self.get_document_base_block_id(document_id)
        # folder_token, folder_url = self.sdk_create_folder(folder_name='temp_images', parent_folder_token=folder_token)
        image_token_map = {}
        responses = self.add_blocks(document_token=document_id,
                                    parent_block_id=parent_block_id,
                                    block_type_idx=27,
                                    block_type_name='image',
                                    block_type_args=[{'token': ''}] * len(image_paths))
        empty_img_block_slots_ids = []
        for res in responses:
            block_ids = [i['block_id'] for i in res['data']['children']]
            empty_img_block_slots_ids.extend(block_ids)
        for image_path, empty_slot_id in zip(image_paths, empty_img_block_slots_ids):
            if Path(image_path).exists():
                image_token = self.upload_file(file_path=image_path,
                                               parent_node=empty_slot_id,
                                               parent_type='docx_image')
                self.modify_image_block(document_id=document_id,
                                        target_block_id=empty_slot_id,
                                        replaced_image_token=image_token)
                image_token_map[image_path] = empty_slot_id
        # image_tokens = list(image_token_map.values())
        #
        # if len(empty_img_block_slots_ids) < len(image_tokens):
        #     logger.warning("Image slots less than to be inserted images.")
        #     return {}
        # for block_id, image_token in zip(empty_img_block_slots_ids, image_tokens):
        #     self.modify_image_block(document_id=document_id,
        #                             target_block_id=block_id,
        #                             replaced_image_token=image_token)
        #
        # asserted_image_tokens = self.get_block_children(document_id=document_id,
        #                                                 block_id=parent_block_id)
        # failed_image_tokens = list(set(image_tokens) - set(asserted_image_tokens))
        # if not failed_image_tokens:
        #     logger.success("All image added.")
        # else:
        #     logger.warning(f"Failed_image_tokens: {failed_image_tokens}")
        #     logger.warning(str(image_token_map))
        return image_token_map

    def add_read_later_btn(self, document_id, info_dict, parent_block_id=None, header_level=0, if_bold=True):
        if not parent_block_id:
            parent_block_id = self.get_document_base_block_id(document_id)
        url = f"http://{self.__backend_ip}/api/shellProbe_lark_event"
        # 使用 requests.utils.quote() 对参数值进行URL编码
        encoded_params = {k: requests.utils.quote(v) for k, v in info_dict.items()}
        query_string = '&'.join([f"{k}={v}" for k, v in encoded_params.items()])
        url = f'{url}?{query_string}'
        block_ids = self.add_titles(document_id=document_id,
                                    titles=['添加到稍后阅读'],
                                    urls=[url],
                                    parent_block_id=parent_block_id,
                                    header_level=header_level,
                                    if_bold=if_bold,
                                    background_idx=4)
        return block_ids[0]

    """
    Bitable handler
    """

    def get_bitable_meta(self, app_token):
        uri = '/open-apis/bitable/v1/apps/:app_token'
        response = self._sdk_general(uri=uri,
                                     request_type=lark.HttpMethod.GET,
                                     path_params={'app_token': app_token})
        return response['data']

    def get_bitable_all_tables(self, app_token):
        uri = '/open-apis/bitable/v1/apps/:app_token/tables'
        tables = []
        next_page_token = None
        variables = {
            'page_token': None,
            'page_size': 20
        }
        while 1:
            if next_page_token:
                variables["page_token"] = next_page_token
            queries = []
            for var_name, var_value in variables.items():
                if var_value:
                    queries.append((var_name, var_value))
            response = self._sdk_general(uri=uri,
                                         request_type=lark.HttpMethod.GET,
                                         token_type=self.__global_token_type,
                                         path_params={'app_token': app_token},
                                         queries_params=queries)

            if response.get('code', -1) != 0:
                logger.error(response.get('msg'))
                break
            logger.debug(response['data']['items'])
            tables.extend(response['data']['items'])
            if not response['data']['has_more']:
                break
            next_page_token = response['data']['page_token']

        return tables

    def get_bitable_table_views(self, app_token, table_id):
        uri = '/open-apis/bitable/v1/apps/:app_token/tables/:table_id/views'
        tables = []
        next_page_token = None
        variables = {
            'page_token': None,
            'page_size': 20
        }
        while 1:
            if next_page_token:
                variables["page_token"] = next_page_token
            queries = []
            for var_name, var_value in variables.items():
                if var_value:
                    queries.append((var_name, var_value))
            response = self._sdk_general(uri=uri,
                                         request_type=lark.HttpMethod.GET,
                                         token_type=self.__global_token_type,
                                         path_params={'app_token': app_token,
                                                      'table_id': table_id},
                                         queries_params=queries)

            if response.get('code', -1) != 0:
                logger.error(response.get('msg'))
                break
            logger.debug(response['data']['items'])
            tables.extend(response['data']['items'])
            if not response['data']['has_more']:
                break
            next_page_token = response['data']['page_token']

        return tables

    def insert_bitable_table_records(self, app_token, table_id, records: List[Dict]):
        uri = '/open-apis/bitable/v1/apps/:app_token/tables/:table_id/records'
        responses = []
        for record in records:
            response = self._sdk_general(uri,
                                         request_type=lark.HttpMethod.POST,
                                         path_params={'app_token': app_token,
                                                      'table_id': table_id},
                                         bodies_params={'fields': record})
            responses.append(response)
        return responses

    def get_bitable_table_fields(self, app_token, table_id, view_id=None):
        uri = '/open-apis/bitable/v1/apps/:app_token/tables/:table_id/fields'
        fields = []
        next_page_token = None
        variables = {
            'page_token': None,
            'page_size': 20,
            'view_id': view_id
        }
        while 1:
            if next_page_token:
                variables["page_token"] = next_page_token
            queries = []
            for var_name, var_value in variables.items():
                if var_value:
                    queries.append((var_name, var_value))
            response = self._sdk_general(uri=uri,
                                         request_type=lark.HttpMethod.GET,
                                         token_type=self.__global_token_type,
                                         path_params={'app_token': app_token,
                                                      'table_id': table_id},
                                         queries_params=queries)

            if response.get('code', -1) != 0:
                logger.error(response.get('msg'))
                break
            logger.debug(response['data']['items'])
            fields.extend(response['data']['items'])
            if not response['data']['has_more']:
                break
            next_page_token = response['data']['page_token']

        return fields

    def get_bitable_table_records(self, app_token, table_id, view_id=None, field_names: Union[List, None] = None,
                                  sort: Union[List[Dict], None] = None, filter: Union[Dict, None] = None,
                                  automatic_fields=False):
        uri = '/open-apis/bitable/v1/apps/:app_token/tables/:table_id/records/search'
        body_params = {}
        if view_id:
            body_params['view_id'] = view_id
        if field_names:
            body_params['field_names'] = field_names
        if sort:
            body_params['sort'] = sort
        if filter:
            body_params['filter'] = filter
        body_params['automatic_fields'] = automatic_fields
        items = []
        next_page_token = None
        variables = {
            'page_token': None,
            'page_size': 20,
        }
        while 1:
            if next_page_token:
                variables["page_token"] = next_page_token
            queries = []
            for var_name, var_value in variables.items():
                if var_value:
                    queries.append((var_name, var_value))
            response = self._sdk_general(uri=uri,
                                         request_type=lark.HttpMethod.POST,
                                         path_params={'app_token': app_token,
                                                      'table_id': table_id},
                                         token_type=self.__global_token_type,
                                         queries_params=queries,
                                         bodies_params=body_params)

            if response.get('code', -1) != 0:
                logger.error(response.get('msg'))
                break
            logger.debug(response['data']['items'])
            items.extend(response['data']['items'])
            if not response['data']['has_more']:
                break
            next_page_token = response['data']['page_token']
        return items


if __name__ == "__main__":
    ins = FeishuSpaceHandler(
        config_yaml_path=r'W:\Personal_Project\metaunitech\shellProbe_manager\configs\feishu_config.yaml')
    # ins.sdk_create_folder('Nihao')
    # print(ins.sdk_list_drive_files())
    # res = ins.load_xmind(
    #     'W:\Personal_Project\metaunitech\shellProbe_manager\src\shellProbe_demo_data_finance\india economy_印度经济情况-搜索[india economy]的结果合集_youtube总结.xmind')
    # from pprint import pprint
    #
    # pprint(res)
    # ins.get_doc_permission('Jr0pdaP4FoVdG5xU6x4cdf7Xn8b')
    # res = ins.sdk_general(f"/open-apis/drive/v2/permissions/:token/public",
    #                       request_type=lark.HttpMethod.GET,
    #                       path_params={"token": 'Jr0pdaP4FoVdG5xU6x4cdf7Xn8b'},
    #                       queries_params=[('type', 'docx')])
    # print(res)
    # res = ins.list_doc_all_blocks('Jr0pdaP4FoVdG5xU6x4cdf7Xn8b')
    # print(res)
    # res = ins.sdk_create_document('HERE')
