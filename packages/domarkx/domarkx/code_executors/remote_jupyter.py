import asyncio
import json
import ssl
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Self
from urllib.parse import urlparse

import aiohttp
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.jupyter import JupyterCodeExecutor, JupyterCodeResult
from packaging.version import Version
from pydantic import BaseModel, Field, field_validator


class JupyterClientConfig(BaseModel):
    remote_url: str = Field(..., description="Jupyter服务器地址")
    token: Optional[str] = Field(None, description="API访问令牌")
    verify_ssl: bool = Field(True, description="是否验证SSL证书")
    reuse_kernels: bool = Field(
        True,
        description="是否复用同名空闲内核",
    )
    kernel_idle_timeout: int = Field(60000, description="复用内核的空闲时间阈值（秒）", gt=0)
    min_server_version: Optional[str] = Field(
        None,
        description="最低要求的服务器版本号 (格式: '6.0.0')",
    )

    @field_validator("min_server_version")
    def validate_version(cls, v):
        if v is not None:
            Version(v)
        return v


class RemoteJupyterExecutorConfig(BaseModel):
    client_config: JupyterClientConfig = Field(..., description="Jupyter客户端配置")
    kernel_name: str = "python3"
    timeout: int = 60
    output_dir: str = "."


class JupyterClient:
    def __init__(self, config: JupyterClientConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._kernel_id: Optional[str] = None
        self._ws_url: Optional[str] = None
        self.server_version: Optional[Version] = None
        self.server_info: dict = {}
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._msg_cache: Dict[str, asyncio.Queue] = {}
        self._xsrf_token: Optional[str] = None

    @property
    def _ssl_context(self):
        if not self.config.verify_ssl:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx
        return None

    async def _fetch_server_info(self):
        """获取服务器元数据"""
        # 获取服务器版本信息
        info_url = f"{self.config.remote_url}/api"
        async with self._session.get(info_url) as resp:
            await self._validate_response(resp)
            self.server_info = await resp.json()
            self.server_version = Version(self.server_info.get("version", "0.0.0"))

            if self.config.min_server_version:
                min_ver = Version(self.config.min_server_version)
                if self.server_version < min_ver:
                    raise RuntimeError(f"Server version {self.server_version} below minimum {min_ver}")

    async def _fetch_xsrf_token(self):
        # 获取XSRF令牌
        async with self._session.get(self.config.remote_url):
            for cookie in self._session.cookie_jar:
                if cookie.key == "_xsrf":
                    self._xsrf_token = cookie.value
                    break

    @property
    def headers(self):
        headers = {"User-Agent": "JupyterCodeExecutor/1.0", "X-XSRFToken": self._xsrf_token or ""}
        if self.config.token:
            headers["Authorization"] = f"Token {self.config.token}"
        return headers

    async def _validate_response(self, response: aiohttp.ClientResponse):
        if response.status >= 400:
            raise RuntimeError(f"API request failed {response.status}: {await response.text()}")

    async def _send_execute_request(self, code: str) -> str:
        msg_id = str(uuid.uuid4())
        self._msg_cache[msg_id] = asyncio.Queue()

        execute_request = {
            "header": {
                "msg_id": msg_id,
                "username": "anonymous",
                "session": str(uuid.uuid4()),
                "msg_type": "execute_request",
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
            },
        }

        await self.ws.send_json(execute_request)
        return msg_id

    async def execute_code(self, code: str) -> list[dict]:
        """执行代码并返回原始消息列表"""
        handler_task = asyncio.create_task(self._handle_messages())
        try:
            msg_id = await self._send_execute_request(code)
            messages = []
            try:
                while True:
                    msg = await self._msg_cache[msg_id].get()
                    messages.append(msg)
                    if msg["header"]["msg_type"] == "execute_reply":
                        break
            finally:
                del self._msg_cache[msg_id]
            return messages
        finally:
            handler_task.cancel()

    async def _handle_messages(self):
        """异步处理WebSocket消息"""
        async for msg in self.ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                msg_id = data.get("parent_header", {}).get("msg_id")
                if msg_id in self._msg_cache:
                    await self._msg_cache[msg_id].put(data)

    async def create_kernel(self, kernel_name: str):
        """创建或复用内核"""
        if self.config.reuse_kernels:
            kernels = await self.list_kernels()
            current_time = time.time()
            for kernel in kernels:
                last_activity = datetime.fromisoformat(kernel["last_activity"]).timestamp()
                if kernel["name"] == kernel_name and (current_time - last_activity) < self.config.kernel_idle_timeout:
                    self._kernel_id = kernel["id"]
                    return

        create_url = f"{self.config.remote_url}/api/kernels"
        async with self._session.post(create_url, json={"name": kernel_name}, headers=self.headers) as resp:
            await self._validate_response(resp)
            kernel_info = await resp.json()
            self._kernel_id = kernel_info["id"]

    async def list_kernels(self) -> List[dict]:
        """获取所有运行中的内核"""
        list_url = f"{self.config.remote_url}/api/kernels"
        async with self._session.get(list_url) as resp:
            await self._validate_response(resp)
            return await resp.json()

    async def connect_kernel(self):
        """建立WebSocket连接"""
        parsed = urlparse(self.config.remote_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        self._ws_url = f"{scheme}://{parsed.netloc}{parsed.path}api/kernels/{self._kernel_id}/channels"
        self.ws = await self._session.ws_connect(self._ws_url, headers=self.headers)

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=self._ssl_context),
            headers={"Authorization": f"Token {self.config.token}"} if self.config.token else {},
        )
        if self._xsrf_token is None:
            await self._fetch_xsrf_token()
        # await self._fetch_server_info()
        return self

    async def __aexit__(self, *args):
        await self._session.close()


class RemoteJupyterCodeExecutor(JupyterCodeExecutor):
    component_config_schema = RemoteJupyterExecutorConfig
    component_provider_override = "domarkx.code_executors.remote_jupyter.RemoteJupyterCodeExecutor"

    def __init__(
        self,
        remote_url: str,
        token: Optional[str] = None,
        kernel_name: str = "python3",
        timeout: int = 60,
        output_dir: Path = Path("."),
    ):
        if timeout < 1:
            raise ValueError("Timeout must be greater than or equal to 1.")

        self._kernel_name = kernel_name
        self._timeout = timeout
        self._output_dir = output_dir
        self.client_config = JupyterClientConfig(
            remote_url=remote_url,
            token=token,
            verify_ssl=True,  # 根据实际情况调整
            reuse_kernels=True,
        )
        self.client = JupyterClient(self.client_config)

    async def _execute_code_block(self, code_block: CodeBlock, cancellation_token: CancellationToken) -> JupyterCodeResult:
        messages = await self.client.execute_code(code_block.code)
        return self._process_messages(messages)

    def _process_messages(self, messages: list[dict]) -> JupyterCodeResult:
        """处理原始消息并生成执行结果"""
        exit_code = 0
        outputs = []
        output_files = []

        for msg in messages:
            msg_type = msg["header"]["msg_type"]
            content = msg.get("content", {})

            if msg_type == "stream":
                outputs.append(content.get("text", ""))
            elif msg_type == "error":
                outputs.append("\n".join(content.get("traceback", [])))
                exit_code = 1
            elif msg_type == "execute_result":
                data = content.get("data", {})
                if "text/plain" in data:
                    outputs.append(data["text/plain"])
            elif msg_type == "display_data":
                data = content.get("data", {})
                if "image/png" in data:
                    output_files.append(self._save_image(data["image/png"]))
                elif "text/html" in data:
                    output_files.append(self._save_html(data["text/html"]))
            elif msg_type == "execute_reply" and content.get("status") == "error":
                exit_code = 1

        return JupyterCodeResult(exit_code=exit_code, output="\n".join(outputs), output_files=output_files)

    async def start(self) -> None:
        await self.client.__aenter__()
        await self.client.create_kernel(self._kernel_name)
        await self.client.connect_kernel()

    async def stop(self) -> None:
        await self.client.__aexit__(None, None, None)

    def _to_config(self) -> RemoteJupyterExecutorConfig:
        return RemoteJupyterExecutorConfig(
            client_config=self.client_config,
            kernel_name=self._kernel_name,
            timeout=self._timeout,
            output_dir=str(self._output_dir),
        )

    @classmethod
    def _from_config(cls, config: RemoteJupyterExecutorConfig) -> Self:
        return cls(
            remote_url=config.client_config.remote_url,
            token=config.client_config.token,
            kernel_name=config.kernel_name,
            timeout=config.timeout,
            output_dir=Path(config.output_dir),
        )
