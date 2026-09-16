import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import requests
from time import perf_counter
from typing import List

from loguru import logger
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from app.config import config
from app.models.llm_provider import DEFAULT_LLM_PROVIDER_ID, get_llm_provider
from app.utils import utils

_max_retries = 5
MIN_SCRIPT_PARAGRAPH_NUMBER = 1
MAX_SCRIPT_PARAGRAPH_NUMBER = 10
MAX_SCRIPT_PROMPT_LENGTH = 2000
MAX_SCRIPT_SYSTEM_PROMPT_LENGTH = 8000
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
_URL_USERINFO_RE = re.compile(
    r"((?:https?|wss?)://)([^/\s?#@]*:[^/\s?#@]*@)", re.IGNORECASE
)
_SENSITIVE_QUERY_RE = re.compile(
    r"([?&](?:api[_-]?key|access[_-]?token|token|key|secret|password)=)([^&#\s]+)",
    re.IGNORECASE,
)

DEFAULT_SCRIPT_SYSTEM_PROMPT = """
# Role: Video Script Generator

## Goals:
Generate a script for a video, depending on the subject of the video.

## Constrains:
1. the script is to be returned as a string with the specified number of paragraphs.
2. do not under any circumstance reference this prompt in your response.
3. get straight to the point, don't start with unnecessary things like, "welcome to this video".
4. you must not include any type of markdown or formatting in the script, never use a title.
5. only return the raw content of the script.
6. do not include "voiceover", "narrator" or similar indicators of what should be spoken at the beginning of each paragraph or line.
7. you must not mention the prompt, or anything about the script itself. also, never talk about the amount of paragraphs or lines. just write the script.
8. respond in the same language as the video subject.
""".strip()

# Claude Code CLI 默认使用编码 agent 的系统提示词，其中大量约束与文案写作
# 无关，会让脚本和关键词生成偏离要求，因此调用时整体替换掉。
CLAUDE_CODE_SYSTEM_PROMPT = (
    "You are a concise copywriter. Follow the user's instructions and output "
    "format exactly, and output nothing else."
)
CLAUDE_CODE_DEFAULT_TIMEOUT = 300.0
# `--tools ""` 关闭全部内置工具，`--safe-mode` 关闭 CLAUDE.md、skills、hooks、
# plugins、MCP 等所有用户级定制，同时保持鉴权、模型选择和权限正常工作。
# 二者需要较新的 CLI；低版本会以 "unknown option" 退出，由调用处转成明确提示。
CLAUDE_CODE_MIN_CLI_VERSION = "2.1.260"
# 这些环境变量会让 CLI 改用 API Key 或第三方供应商（Bedrock、Vertex、Foundry、
# Mantle、Gateway 等），从而绕过订阅登录并产生额外计费。逐个列举容易漏项，
# 而且 CLI 后续还会新增供应商，因此按前缀整类剔除：
#   ANTHROPIC_*           API Key、Auth Token、Base URL、各家供应商端点和 Profile
#   CLAUDE_CODE_USE_*     供应商开关
#   CLAUDE_CODE_SKIP_*_AUTH  跳过供应商鉴权的开关
CLAUDE_CODE_CONFLICTING_ENV_PREFIXES = ("ANTHROPIC_", "CLAUDE_CODE_USE_")
CLAUDE_CODE_CONFLICTING_ENV_VARS = (
    "AWS_BEARER_TOKEN_BEDROCK",
    "CLAUDE_CODE_GATEWAY_TOKEN_FILE_DESCRIPTOR",
)
# 这两类变量不能剔除：
#   CLAUDE_CODE_OAUTH_TOKEN 是容器内唯一的订阅鉴权方式（不匹配上面的前缀）；
#   *_CONFIG_DIR 只是指出凭证存放位置，剔除后反而会让已登录的订阅失效。
CLAUDE_CODE_PRESERVED_ENV_VARS = (
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_CONFIG_DIR",
    "CLAUDE_CONFIG_DIR",
)


def _is_conflicting_claude_code_env(name: str) -> bool:
    """判断某个环境变量是否会把 CLI 从订阅登录切换到别的鉴权方式。"""
    if name in CLAUDE_CODE_PRESERVED_ENV_VARS:
        return False
    if name in CLAUDE_CODE_CONFLICTING_ENV_VARS:
        return True
    if name.startswith(CLAUDE_CODE_CONFLICTING_ENV_PREFIXES):
        return True
    return name.startswith("CLAUDE_CODE_SKIP_") and name.endswith("_AUTH")


def coerce_claude_code_timeout(value, config_key: str = "claude_code_timeout"):
    """
    把配置里的超时值解析成正的有限秒数。

    TOML 既可能写成 `claude_code_timeout = 300`（int/float），也可能写成
    `"300"`（字符串），因此不能直接调用 `strip()`。nan / inf 会让
    `subprocess.run(timeout=...)` 永久阻塞，这里一并拒绝。
    """
    if value is None:
        return CLAUDE_CODE_DEFAULT_TIMEOUT

    if isinstance(value, bool):
        # bool 是 int 的子类，但 True 秒显然不是用户想要的超时配置。
        raise ValueError(f"{config_key} must be a number of seconds, got {value!r}")

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return CLAUDE_CODE_DEFAULT_TIMEOUT
        try:
            seconds = float(text)
        except ValueError:
            raise ValueError(
                f"{config_key} must be a number of seconds, got {value!r}"
            ) from None
    elif isinstance(value, (int, float)):
        seconds = float(value)
    else:
        raise ValueError(f"{config_key} must be a number of seconds, got {value!r}")

    if not math.isfinite(seconds):
        raise ValueError(f"{config_key} must be a finite number, got {value!r}")
    if seconds <= 0:
        raise ValueError(f"{config_key} must be greater than 0, got {value!r}")
    return seconds


def _resolve_provider_field_value(raw_value, default_value):
    """
    只有「未配置」时才回退到 Registry 默认值。

    之前用 `raw or default_value`，会把 0 和 false 这类合法取值也当成未配置
    替换掉：`claude_code_timeout = 0` 被静默改成 300，而 `"0"` 却报错。默认值
    只在 None 或空白字符串时生效，配置校验才能对所有写法保持一致。
    """
    if raw_value is None:
        return default_value
    if isinstance(raw_value, str) and not raw_value.strip():
        return default_value
    return raw_value


def build_claude_code_env(base_env=None):
    """
    构造只依赖订阅登录的子进程环境。

    返回 (环境变量字典, 被剔除的变量名列表)。剔除的是会切换鉴权方式或供应商
    的变量，`CLAUDE_CODE_OAUTH_TOKEN` 必须保留：容器内没有 keychain，CLI 只能
    靠它完成订阅鉴权。
    """
    env = dict(os.environ if base_env is None else base_env)
    removed = sorted(name for name in env if _is_conflicting_claude_code_env(name))
    for name in removed:
        env.pop(name, None)
    return env, removed


def _normalize_text_response(content, llm_provider: str) -> str:
    # 不同 LLM SDK 在异常或被拦截场景下，可能返回 None、空字符串，
    # 甚至返回非字符串对象。这里统一做兜底校验，避免后续直接调用
    # `.replace()` 时抛出 `NoneType` 之类的属性错误。
    if content is None:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    if not isinstance(content, str):
        raise TypeError(
            f"[{llm_provider}] returned non-text content: {type(content).__name__}"
        )

    # MiniMax M3、DeepSeek R1 这类 reasoning 模型可能会把内部推理包在
    # `<think>...</think>` 中返回。视频脚本和关键词只需要最终可朗读文本，
    # 如果不在服务层统一清理，WebUI、字幕和配音都会把思考过程当正文处理。
    content = _THINK_BLOCK_RE.sub("", content)
    content = _UNCLOSED_THINK_BLOCK_RE.sub("", content).strip()
    if not content:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    # 前面的 ``strip()`` 已经清理首尾空白。这里必须保留正文中的单换行和
    # 双换行：脚本生成依赖双换行区分段落，字幕处理也会按行读取用户文案。
    return content


def _sanitize_error_message(error: object) -> str:
    """
    清理返回给 WebUI/API 的错误信息，避免自定义 base_url 中的凭据泄露。

    一些 OpenAI-compatible SDK 会把请求 URL 原样拼进异常信息。如果用户为了
    代理网关配置了 `https://user:pass@example.com/v1`，直接返回 `str(e)`
    就会把密码暴露给页面、API 调用方或后续日志。这里仅处理错误文案，不改变
    实际请求地址，避免影响正常调用链路。
    """
    message = str(error)
    message = _URL_USERINFO_RE.sub(r"\1***:***@", message)
    message = _SENSITIVE_QUERY_RE.sub(r"\1***", message)
    return message


def _extract_chat_completion_text(response, llm_provider: str) -> str:
    # OpenAI 兼容接口在异常场景下，可能返回没有 choices、
    # 或者 choices/message/content 为空的响应对象。
    # 这里统一做结构校验，避免出现 `NoneType is not subscriptable`
    # 这类底层属性访问错误。
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError(f"[{llm_provider}] returned empty choices")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise ValueError(f"[{llm_provider}] returned empty message")

    content = getattr(message, "content", None)
    return _normalize_text_response(content, llm_provider)


def _get_response_field(value, key: str):
    """兼容 dict 和 SDK 响应对象的字段读取。"""
    if isinstance(value, dict):
        return value.get(key)

    try:
        return value[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(value, key, None)


def _extract_qwen_generation_text(response) -> str:
    """
    从 DashScope Generation 响应中提取文本。

    Qwen 使用 `messages` 调用时返回的是 chat 结构：
    `output.choices[0].message.content`；旧 completion 形态才会返回
    `output.text`。这里两个路径都兼容，避免 `output.text` 为 None 时
    继续 `.replace()` 触发不可诊断的 AttributeError。
    """
    output = _get_response_field(response, "output")
    choices = _get_response_field(output, "choices") if output else None
    if choices is not None:
        if not choices:
            logger.warning("Qwen returned an empty choices list")
            raise ValueError("[qwen] returned empty choices")

        first_choice = choices[0]
        message = _get_response_field(first_choice, "message")
        content = _get_response_field(message, "content") if message else None
        if content is not None:
            return _normalize_text_response(content, "qwen")

    text = _get_response_field(output, "text") if output else None
    return _normalize_text_response(text, "qwen")


def _ollama_native_base_url(base_url: str) -> str:
    """Convert an Ollama OpenAI-compatible base URL into the native API root."""
    value = str(base_url or "").strip().rstrip("/")
    if value.endswith("/v1"):
        value = value[:-3]
    return value.rstrip("/")


def _ollama_think_value(model_name: str):
    """
    Select the native Ollama thinking mode.

    GPT-OSS cannot fully disable thinking, so use its lowest supported level.
    Qwen and other local utility models use thinking disabled.
    """
    normalized = str(model_name or "").strip().lower()
    if normalized.startswith("gpt-oss"):
        return "low"
    return False


def _generate_response(
    prompt: str,
    app_config=None,
    json_mode: bool = False,
) -> str:
    try:
        # WebUI 在视频生成期间允许用户准备下一条文案。调用方可以传入提交瞬间
        # 的配置快照，确保模型请求重试期间不会因为后台任务结束并应用新配置，
        # 而切换到另一个 Provider、Base URL 或模型。
        runtime_app_config = app_config if app_config is not None else config.app
        llm_provider = str(
            runtime_app_config.get("llm_provider", DEFAULT_LLM_PROVIDER_ID)
        ).lower()
        provider = get_llm_provider(llm_provider)
        if provider is None:
            raise ValueError(f"{llm_provider}: unsupported llm provider")

        logger.info(f"llm provider: {llm_provider}")
        api_key = runtime_app_config.get(provider.config_key("api_key"), "")
        configured_model = runtime_app_config.get(provider.config_key("model_name"), "")
        model_name = provider.resolve_model_name(configured_model)
        if configured_model and model_name != configured_model:
            logger.warning(
                f"{llm_provider} model '{configured_model}' is deprecated, "
                f"fallback to '{model_name}'"
            )
        configured_base_url = runtime_app_config.get(
            provider.config_key("base_url"), ""
        )
        base_url = provider.resolve_base_url(configured_base_url)
        if configured_base_url and configured_base_url.strip().rstrip("/") in {
            url.rstrip("/") for url in provider.deprecated_base_urls
        }:
            logger.warning(
                f"{llm_provider} base URL '{configured_base_url}' is deprecated, "
                f"fallback to '{base_url}'"
            )
        adapter = provider.adapter
        api_version = ""

        # Ollama 的默认地址依赖当前是否运行在容器中，无法作为静态 Registry
        # 值保存；Registry 仍负责模型和必填规则，运行环境差异在这里解析。
        if llm_provider == "ollama":
            api_key = "ollama"
            if not base_url:
                base_url = config.get_default_ollama_base_url()

        if adapter == "azure":
            api_version = runtime_app_config.get(
                provider.config_key("api_version"), "2024-02-15-preview"
            )

        extra_values = {
            field.config_suffix: _resolve_provider_field_value(
                runtime_app_config.get(provider.config_key(field.config_suffix)),
                field.default_value,
            )
            for field in provider.extra_fields
        }

        if provider.requires_api_key and not api_key:
            raise ValueError(
                f"{llm_provider}: api_key is not set, please set it in the config.toml file."
            )
        if provider.requires_model_name and not model_name:
            raise ValueError(
                f"{llm_provider}: model_name is not set, please set it in the config.toml file."
            )
        if provider.requires_base_url and not base_url:
            raise ValueError(
                f"{llm_provider}: base_url is not set, please set it in the config.toml file."
            )

        for field in provider.extra_fields:
            if field.required and not extra_values[field.config_suffix]:
                raise ValueError(
                    f"{llm_provider}: {field.config_suffix} is not set, "
                    "please set it in the config.toml file."
                )

        if adapter == "qwen":
            import dashscope
            from dashscope.api_entities.dashscope_response import GenerationResponse

            dashscope.api_key = api_key
            response = dashscope.Generation.call(
                model=model_name, messages=[{"role": "user", "content": prompt}]
            )
            if response:
                if isinstance(response, GenerationResponse):
                    status_code = response.status_code
                    if status_code != 200:
                        raise Exception(
                            f'[{llm_provider}] returned an error response: "{response}"'
                        )

                    return _extract_qwen_generation_text(response)
                else:
                    raise Exception(
                        f'[{llm_provider}] returned an invalid response: "{response}"'
                    )
            else:
                raise Exception(f"[{llm_provider}] returned an empty response")

        if adapter == "gemini":
            from google import genai
            from google.genai import types

            http_options = types.HttpOptions(base_url=base_url) if base_url else None
            generation_config = types.GenerateContentConfig(
                temperature=0.5,
                top_p=1,
                top_k=1,
                max_output_tokens=2048,
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                ],
            )

            try:
                # 新版 google-genai 通过统一 Client 暴露模型服务。上下文管理器
                # 会在请求结束后关闭底层 HTTP 连接，避免频繁生成时积累连接资源。
                with genai.Client(
                    api_key=api_key,
                    http_options=http_options,
                ) as client:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=generation_config,
                    )
                generated_text = response.text
            except (AttributeError, IndexError, ValueError) as e:
                logger.warning(f"gemini returned invalid response content: {str(e)}")
                raise ValueError(f"[{llm_provider}] returned invalid response content")

            return _normalize_text_response(generated_text, llm_provider)

        if adapter == "cloudflare_ai_gateway":
            account_id = extra_values["account_id"]
            gateway_id = extra_values["gateway_id"]
            # Cloudflare 当前推荐的 AI Gateway REST API 兼容 OpenAI SDK。
            # Account ID 用于构造统一端点，Gateway ID 通过请求头选择；这里
            # 不再调用 Workers AI 的 /ai/run/{model} 专用接口。
            client = OpenAI(
                api_key=api_key,
                base_url=(
                    f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
                ),
                default_headers={"cf-aig-gateway-id": gateway_id},
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return _extract_chat_completion_text(response, llm_provider)

        if adapter == "litellm":
            import litellm

            if not model_name:
                raise ValueError(
                    f"{llm_provider}: model_name is not set, please set it in the config.toml file."
                )

            response = litellm.completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                drop_params=True,
            )

            if not response:
                raise ValueError(f"[{llm_provider}] returned empty response")
            if not getattr(response, "choices", None):
                raise ValueError(f"[{llm_provider}] returned empty response")

            return _extract_chat_completion_text(response, llm_provider)

        if adapter == "azure":
            # Azure OpenAI SDK 使用 `azure_endpoint` 和 `api_version` 生成专用请求地址，
            # 不能继续复用下面普通 OpenAI-compatible 的 `base_url` 初始化逻辑。
            # 这里在 Azure 分支内完成请求并立即返回，避免客户端被后续 fallback
            # 覆盖，导致用户配置的 Azure 凭证通过校验但实际请求没有被使用。
            logger.info(f"requesting azure chat completion, model: {model_name}")
            client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=base_url,
            )
            response = client.chat.completions.create(
                model=model_name, messages=[{"role": "user", "content": prompt}]
            )
            if response:
                if isinstance(response, ChatCompletion):
                    return _extract_chat_completion_text(response, llm_provider)
                else:
                    raise Exception(
                        f'[{llm_provider}] returned an invalid response: "{response}", please check your network '
                        f"connection and try again."
                    )
            else:
                raise Exception(
                    f"[{llm_provider}] returned an empty response, please check your network connection and try again."
                )

        if adapter == "claude_code":
            # Claude 订阅（Pro / Max / Team）不签发 API Key，其凭证只能由
            # Claude Code 官方客户端自己使用。这里不直接请求 Anthropic API，
            # 而是以 headless 模式调用本机已登录的 claude CLI（`claude -p`），
            # 由 CLI 完成鉴权，脚本生成只消费它返回的文本。
            configured_cli = (extra_values.get("cli_path") or "").strip() or "claude"
            cli_path = shutil.which(configured_cli)
            if not cli_path and os.path.isfile(configured_cli):
                cli_path = configured_cli
            if not cli_path:
                raise ValueError(
                    f"{llm_provider}: claude CLI not found ('{configured_cli}'), "
                    f"install it in the runtime or set "
                    f"{provider.config_key('cli_path')} in the config.toml file."
                )

            try:
                timeout_seconds = coerce_claude_code_timeout(
                    extra_values.get("timeout"), provider.config_key("timeout")
                )
            except ValueError as timeout_error:
                raise ValueError(f"{llm_provider}: {timeout_error}") from None

            command = [
                cli_path,
                "-p",
                prompt,
                "--output-format",
                "json",
                "--system-prompt",
                CLAUDE_CODE_SYSTEM_PROMPT,
                # 关闭全部内置工具，保证只做文本生成。
                "--tools",
                "",
                # 关闭 CLAUDE.md、skills、hooks、plugins、MCP 等用户级定制；
                # 鉴权与模型选择不受影响（不能用 --bare，它会禁用 OAuth）。
                "--safe-mode",
            ]
            # 模型名留空时沿用 CLI 自己的默认模型，避免这里硬编码的模型 ID
            # 随订阅可用模型变化而失效。
            if model_name:
                command += ["--model", model_name]

            cli_env, removed_env = build_claude_code_env()
            if removed_env:
                # 只记录变量名，不记录取值，避免把密钥写进日志。
                logger.warning(
                    f"{llm_provider}: ignoring conflicting environment variables "
                    f"so the subscription login is used: {', '.join(removed_env)}"
                )

            logger.info(f"invoking claude cli, model: {model_name or 'cli default'}")
            # CLI 会读取工作目录下的 CLAUDE.md 和项目设置，这些内容会污染
            # 文案结果，因此固定在一个临时空目录中执行。
            with tempfile.TemporaryDirectory() as work_dir:
                try:
                    completed = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        timeout=timeout_seconds,
                        cwd=work_dir,
                        env=cli_env,
                    )
                except subprocess.TimeoutExpired:
                    raise Exception(
                        f"[{llm_provider}] claude cli timed out after "
                        f"{timeout_seconds:.0f}s"
                    )

            # 未登录、用量耗尽这类失败同样会返回 JSON（`is_error` 为真，
            # `result` 是可读原因），只是退出码非 0。因此先解析 stdout，
            # 只有在拿不到 JSON 时才回退到退出码和 stderr。
            stdout = (completed.stdout or "").strip()
            try:
                payload = json.loads(stdout) if stdout else None
            except json.JSONDecodeError:
                payload = None

            if payload is None:
                detail = (completed.stderr or stdout or "").strip()
                if "unknown option" in detail.lower():
                    raise Exception(
                        f"[{llm_provider}] the installed claude CLI does not support "
                        f"the required isolation flags; upgrade to "
                        f"{CLAUDE_CODE_MIN_CLI_VERSION} or newer: {detail[:300]}"
                    )
                if completed.returncode != 0:
                    raise Exception(
                        f"[{llm_provider}] claude cli exited with code "
                        f"{completed.returncode}: {detail[:500]}"
                    )
                raise Exception(
                    f'[{llm_provider}] returned an invalid response: "{detail[:500]}"'
                )

            if payload.get("is_error") or completed.returncode != 0:
                reason = str(payload.get("result") or "").strip() or (
                    f"claude cli exited with code {completed.returncode}"
                )
                # 容器里无法执行交互式 /login，这里直接给出可用的鉴权方式。
                if "login" in reason.lower():
                    reason += (
                        " (run `claude setup-token` on the host and pass the token "
                        "to the container as CLAUDE_CODE_OAUTH_TOKEN)"
                    )
                raise Exception(
                    f'[{llm_provider}] returned an error response: "{reason[:500]}"'
                )

            return _normalize_text_response(payload.get("result"), llm_provider)

        if adapter == "modelscope":
            content = ""
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"enable_thinking": False},
                stream=True,
            )
            if response:
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        content += delta.content

                if not content.strip():
                    raise ValueError("Empty content in stream response")

                return _normalize_text_response(content, llm_provider)
            else:
                raise Exception(f"[{llm_provider}] returned an empty response")

        if llm_provider == "ollama":
            native_base_url = _ollama_native_base_url(base_url)
            native_url = f"{native_base_url}/api/chat"

            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": _ollama_think_value(model_name),
            }

            if json_mode:
                payload["format"] = "json"
                payload["options"] = {"temperature": 0}

            logger.info(
                "requesting Ollama native chat: "
                f"model={model_name}, json_mode={json_mode}, "
                f"think={payload['think']}"
            )

            response = requests.post(
                native_url,
                json=payload,
                timeout=300,
            )
            response.raise_for_status()

            data = response.json()
            message = data.get("message")
            if not isinstance(message, dict):
                raise ValueError("[ollama] returned empty message")

            content = message.get("content")
            return _normalize_text_response(content, llm_provider)

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        request_kwargs = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
        }

        response = client.chat.completions.create(**request_kwargs)
        if response:
            if isinstance(response, ChatCompletion):
                return _extract_chat_completion_text(response, llm_provider)
            else:
                raise Exception(
                    f'[{llm_provider}] returned an invalid response: "{response}", please check your network '
                    f"connection and try again."
                )
        else:
            raise Exception(
                f"[{llm_provider}] returned an empty response, please check your network connection and try again."
            )

    except Exception as e:
        return f"Error: {_sanitize_error_message(e)}"


def test_connection() -> tuple[bool, str, float]:
    """
    使用当前 Provider 配置发起一次最小请求，验证实际生成链路是否可用。

    连接测试直接复用 `_generate_response()`，因此会覆盖 API Key、Base URL、
    模型名称和 Provider 专用字段，但不会进入脚本生成的重试逻辑，也不会发送
    用户的视频主题或文案。返回值依次为成功状态、错误信息和请求耗时。
    """
    started_at = perf_counter()
    response = _generate_response(prompt="Reply with exactly: OK")
    elapsed = perf_counter() - started_at

    if not response:
        error_message = "LLM returned an empty response"
        logger.warning(f"llm connection test failed: {error_message}")
        return False, error_message, elapsed

    if response.startswith("Error:"):
        error_message = response.removeprefix("Error:").strip()
        logger.warning(f"llm connection test failed: {error_message}")
        return False, error_message, elapsed

    logger.info(f"llm connection test succeeded, elapsed: {elapsed:.2f}s")
    return True, "", elapsed


def _limit_script_text(text: str | None, max_length: int, field_name: str) -> str:
    value = (text or "").strip()
    if len(value) <= max_length:
        return value

    # API 层已经用 Pydantic 做长度校验；这里继续兜底，是为了保护
    # WebUI 或内部服务直接调用 generate_script 时不会把超长提示词发送给模型，
    # 避免 token 成本异常和请求失败。
    logger.warning(
        f"{field_name} is too long and will be truncated to {max_length} characters."
    )
    return value[:max_length]


def _normalize_script_paragraph_number(paragraph_number: int | None) -> int:
    try:
        value = int(paragraph_number or MIN_SCRIPT_PARAGRAPH_NUMBER)
    except (TypeError, ValueError):
        value = MIN_SCRIPT_PARAGRAPH_NUMBER

    if value < MIN_SCRIPT_PARAGRAPH_NUMBER or value > MAX_SCRIPT_PARAGRAPH_NUMBER:
        # WebUI 和 API 都会限制范围；这里兜底处理内部调用，避免异常参数直接扩大
        # LLM 生成成本或生成空结果。
        logger.warning(
            f"script paragraph_number is out of range and will be clamped: {value}"
        )
        return max(MIN_SCRIPT_PARAGRAPH_NUMBER, min(value, MAX_SCRIPT_PARAGRAPH_NUMBER))

    return value


def build_script_prompt(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
) -> str:
    paragraph_number = _normalize_script_paragraph_number(paragraph_number)
    video_script_prompt = _limit_script_text(
        video_script_prompt, MAX_SCRIPT_PROMPT_LENGTH, "video_script_prompt"
    )
    custom_system_prompt = _limit_script_text(
        custom_system_prompt, MAX_SCRIPT_SYSTEM_PROMPT_LENGTH, "custom_system_prompt"
    )

    # 将“脚本生成规则”和“运行时上下文”分开拼接。这样高级用户即使覆盖默认
    # system prompt，也不会漏掉视频主题、语言、段落数这些每次生成都必须带上的参数。
    prompt = custom_system_prompt or DEFAULT_SCRIPT_SYSTEM_PROMPT
    prompt += f"""

# Initialization:
- video subject: {video_subject}
- number of paragraphs: {paragraph_number}
""".rstrip()
    if language:
        prompt += f"\n- language: {language}"
    if video_script_prompt:
        prompt += f"""

# Additional User Requirements:
{video_script_prompt}
""".rstrip()

    return prompt


def generate_script(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
    app_config=None,
) -> str:
    paragraph_number = _normalize_script_paragraph_number(paragraph_number)
    video_script_prompt = _limit_script_text(
        video_script_prompt, MAX_SCRIPT_PROMPT_LENGTH, "video_script_prompt"
    )
    custom_system_prompt = _limit_script_text(
        custom_system_prompt, MAX_SCRIPT_SYSTEM_PROMPT_LENGTH, "custom_system_prompt"
    )
    prompt = build_script_prompt(
        video_subject=video_subject,
        language=language,
        paragraph_number=paragraph_number,
        video_script_prompt=video_script_prompt,
        custom_system_prompt=custom_system_prompt,
    )
    final_script = ""
    logger.info(
        "generating video script: "
        f"subject={video_subject}, paragraph_number={paragraph_number}, "
        f"has_custom_prompt={bool(video_script_prompt.strip())}, "
        f"has_custom_system_prompt={bool(custom_system_prompt.strip())}"
    )

    def format_response(response):
        # Clean the script
        # Remove asterisks, hashes
        response = response.replace("*", "")
        response = response.replace("#", "")

        # Remove markdown syntax.  Use non-greedy .*? so each bracket/paren
        # group is removed independently; the greedy form would eat all text
        # between the first opener and the last closer on the same line.
        response = re.sub(r"\[.*?\]", "", response)
        response = re.sub(r"\(.*?\)", "", response)

        # Split the script into paragraphs
        paragraphs = response.split("\n\n")

        # Select the specified number of paragraphs
        # selected_paragraphs = paragraphs[:paragraph_number]

        # Join the selected paragraphs into a single string
        return "\n\n".join(paragraphs)

    for i in range(_max_retries):
        try:
            if app_config is None:
                response = _generate_response(prompt=prompt)
            else:
                response = _generate_response(prompt=prompt, app_config=app_config)
            if response:
                final_script = format_response(response)
            else:
                logging.error("gpt returned an empty response")

            # Some upstream providers may return quota errors as plain text.
            if final_script and "当日额度已消耗完" in final_script:
                raise ValueError(final_script)

            if final_script:
                break
        except Exception as e:
            logger.error(f"failed to generate script: {e}")

        if i < _max_retries - 1:
            logger.warning(f"failed to generate video script, trying again... {i + 1}")
    if "Error: " in final_script:
        logger.error(f"failed to generate video script: {final_script}")
    else:
        logger.success(f"completed: \n{final_script}")
    return final_script.strip()


def _strip_code_fence(text: str) -> str:
    """Strip a surrounding markdown code fence from an LLM response.

    Non-OpenAI providers (Claude, Gemini, …) frequently wrap JSON output in a
    ```json … ``` fence even when asked to return raw JSON. Removing it lets the
    first json.loads() succeed instead of falling through to the regex recovery
    path (and spuriously logging a warning). Mirrors the DOTALL handling already
    used in _parse_social_metadata().
    """
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


AI_VISUAL_PROMPT_SOURCES = frozenset(
    {
        "wavespeed",
        "volcengine_seedance",
        "ofox",
        "metaso_minimax",
        "loomloom",
        "openai_image",
        "comfyui_t2i",
        "comfyui_mage",
        "comfyui_video",
    }
)


def should_generate_visual_prompts(video_source: str | None) -> bool:
    """Return whether a material source consumes generative visual prompts."""
    return str(video_source or "").strip() in AI_VISUAL_PROMPT_SOURCES


def calculate_visual_prompt_count(
    duration_min: float,
    duration_max: float,
    clip_duration: float,
) -> int:
    """Return the recommended prompt count from the midpoint narration duration."""
    try:
        minimum = float(duration_min)
        maximum = float(duration_max)
        per_clip = float(clip_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("duration values must be numeric") from exc

    if not all(math.isfinite(value) for value in (minimum, maximum, per_clip)):
        raise ValueError("duration values must be finite")
    if minimum < 0 or maximum < 0:
        raise ValueError("duration values cannot be negative")
    if maximum < minimum:
        minimum, maximum = maximum, minimum
    if per_clip <= 0:
        raise ValueError("clip_duration must be greater than zero")

    midpoint = (minimum + maximum) / 2.0
    return max(1, math.ceil(midpoint / per_clip))


_VISUAL_ANCHOR_TAG_RE = re.compile(r"^(CHAR|LOC|OBJ|VEH|CREATURE)_[1-9]\d*$")
_VISUAL_ANCHOR_REFERENCE_RE = re.compile(r"\[([A-Z][A-Z0-9_]*)\]")
_VISUAL_ANCHOR_CATEGORY_ORDER = {
    "CHAR": 0,
    "LOC": 1,
    "OBJ": 2,
    "VEH": 3,
    "CREATURE": 4,
}


def _normalize_visual_prompt_lines(visual_prompts) -> List[str]:
    """Normalize the current visual-prompt field without splitting internal commas."""
    if isinstance(visual_prompts, str):
        prompts = [line.strip() for line in visual_prompts.splitlines() if line.strip()]
        if not prompts and visual_prompts.strip():
            prompts = [visual_prompts.strip()]
    elif isinstance(visual_prompts, (list, tuple)):
        prompts = [str(prompt).strip() for prompt in visual_prompts if str(prompt).strip()]
    else:
        raise ValueError("visual_prompts must be a string or a list of strings")

    if not prompts:
        raise ValueError("at least one visual prompt is required")
    return prompts


def _load_json_object_response(response: str) -> dict:
    """Parse a JSON object, tolerating a provider-added code fence or short wrapper text."""
    cleaned = _strip_code_fence(response)
    try:
        data = json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError("response does not contain a JSON object")
        data = json.loads(match.group())
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")
    return data


def _normalize_visual_anchor_plan(data: dict, expected_scene_count: int) -> dict:
    """Validate and normalize the LLM continuity plan before it reaches Streamlit."""
    raw_anchors = data.get("subject_anchors")
    raw_scenes = data.get("scene_prompts")
    if not isinstance(raw_anchors, list):
        raise ValueError("subject_anchors must be a JSON array")
    if not isinstance(raw_scenes, list):
        raise ValueError("scene_prompts must be a JSON array")
    if len(raw_scenes) != expected_scene_count:
        raise ValueError(
            "scene prompt count changed during anchor generation: "
            f"expected={expected_scene_count}, received={len(raw_scenes)}"
        )

    anchors = []
    defined_tags = set()
    for raw_anchor in raw_anchors:
        if not isinstance(raw_anchor, dict):
            raise ValueError("each subject anchor must be a JSON object")
        tag = str(raw_anchor.get("tag") or "").strip()
        if tag.startswith("[") and tag.endswith("]"):
            tag = tag[1:-1].strip()
        description = str(raw_anchor.get("description") or "").strip()
        if not _VISUAL_ANCHOR_TAG_RE.fullmatch(tag):
            raise ValueError(f"invalid Subject Anchor tag: [{tag or '?'}]")
        if tag in defined_tags:
            raise ValueError(f"Subject Anchor [{tag}] is defined more than once")
        if not description:
            raise ValueError(f"Subject Anchor [{tag}] has an empty description")
        if _VISUAL_ANCHOR_REFERENCE_RE.search(description):
            raise ValueError(
                f"Subject Anchor [{tag}] description must not contain anchor references"
            )
        defined_tags.add(tag)
        anchors.append({"tag": tag, "description": description})

    scene_prompts = []
    referenced_tags = set()
    expected_indices = list(range(1, expected_scene_count + 1))
    received_indices = []
    for position, raw_scene in enumerate(raw_scenes, start=1):
        if isinstance(raw_scene, dict):
            index = raw_scene.get("index")
            prompt = str(raw_scene.get("prompt") or "").strip()
            try:
                index = int(index)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("each scene prompt index must be an integer") from exc
        elif isinstance(raw_scene, str):
            # Preserve compatibility with models that obey array order but omit index objects.
            index = position
            prompt = raw_scene.strip()
        else:
            raise ValueError("each scene prompt must be a JSON object or string")

        if not prompt:
            raise ValueError(f"scene prompt {position} is empty")

        received_indices.append(index)
        references = set(_VISUAL_ANCHOR_REFERENCE_RE.findall(prompt))
        undefined = sorted(references - defined_tags)
        if undefined:
            raise ValueError(
                "scene prompt references undefined Subject Anchor(s): "
                + ", ".join(f"[{tag}]" for tag in undefined)
            )
        referenced_tags.update(references)
        scene_prompts.append(prompt)

    if received_indices != expected_indices:
        raise ValueError(
            "scene prompt indices must preserve the original order exactly: "
            f"expected={expected_indices}, received={received_indices}"
        )

    unused = sorted(defined_tags - referenced_tags)
    if unused:
        raise ValueError(
            "Subject Anchors must be used by at least one Scene Prompt: "
            + ", ".join(f"[{tag}]" for tag in unused)
        )
    if anchors and not referenced_tags:
        raise ValueError("Scene Prompts did not use the generated Subject Anchors")

    def anchor_sort_key(anchor):
        prefix, number = anchor["tag"].rsplit("_", 1)
        return (_VISUAL_ANCHOR_CATEGORY_ORDER[prefix], int(number))

    anchors.sort(key=anchor_sort_key)
    anchor_text = "\n\n".join(
        f"[{anchor['tag']}]\n{anchor['description']}" for anchor in anchors
    )
    return {
        "subject_anchors": anchor_text,
        "scene_prompts": scene_prompts,
        "anchor_count": len(anchors),
        "scene_count": len(scene_prompts),
    }



def _expand_visual_anchor_scene_prompts(
    scene_prompts,
    anchors,
) -> List[str]:
    """
    Expand validated [TAG] references using the fixed Subject Anchor library.

    Expansion is intentionally literal. Any grammar artifacts produced by
    inserting a long noun phrase are handled by the refinement stage below.
    """
    anchor_map = {
        str(anchor["tag"]): str(anchor["description"]).rstrip(
            " \t\r\n.,;:!?"
        )
        for anchor in anchors
    }

    expanded_prompts = []

    for prompt in scene_prompts:
        expanded = _VISUAL_ANCHOR_REFERENCE_RE.sub(
            lambda match: anchor_map.get(
                match.group(1),
                match.group(0),
            ),
            str(prompt),
        ).strip()

        expanded_prompts.append(expanded)

    return expanded_prompts



def _find_expanded_prompt_artifacts(
    prompt: str,
    anchors=None,
) -> List[str]:
    """
    Detect objective grammar artifacts introduced by literal anchor expansion.

    This validator does not judge story quality or creative choices.
    """
    text = str(prompt or "").strip()
    issues = []

    for anchor in anchors or []:
        if not isinstance(anchor, dict):
            continue

        description = str(
            anchor.get("description") or ""
        ).strip().rstrip(
            " \t\r\n.,;:!?"
        )

        if not description:
            continue

        lowered = text.lower()

        for suffix in ("'s", "\u2019s"):
            bad_phrase = (
                description + suffix
            )

            if bad_phrase.lower() in lowered:
                issues.append(
                    "full expanded anchor used as "
                    f"possessive: {bad_phrase}"
                )

    malformed_determiner = re.search(
        r"\b(?:the|a|an)\s+"
        r"(?:(?:open|opened|closed)\s+)?"
        r"(?:a|an)\b",
        text,
        flags=re.IGNORECASE,
    )

    if malformed_determiner:
        issues.append(
            "malformed duplicated determiner: "
            + malformed_determiner.group(0)
        )

    return issues


def refine_expanded_scene_prompts(
    expanded_prompts,
    *,
    source_scene_prompts=None,
    anchors=None,
    video_subject: str = "",
    app_config=None,
) -> List[str]:
    """
    Clean linguistic artifacts introduced by Subject Anchor expansion.

    The model is only a copy editor here. It must preserve the visual and
    narrative content. If cleanup cannot be validated, the original expanded
    prompt remains usable and the workflow continues.
    """
    prompts = _normalize_visual_prompt_lines(
        expanded_prompts
    )

    source_prompts = (
        _normalize_visual_prompt_lines(
            source_scene_prompts
        )
        if source_scene_prompts
        else list(prompts)
    )

    if len(source_prompts) != len(prompts):
        raise ValueError(
            "source_scene_prompts must match "
            "expanded_prompts count"
        )

    batch_size = 8
    refined_prompts = []

    def call_json(prompt_text: str) -> str:
        if app_config is None:
            return _generate_response(
                prompt_text,
                json_mode=True,
            )

        return _generate_response(
            prompt_text,
            app_config=app_config,
            json_mode=True,
        )

    batch_count = math.ceil(
        len(prompts) / batch_size
    )

    for batch_number, batch_start in enumerate(
        range(0, len(prompts), batch_size),
        start=1,
    ):
        batch_end = min(
            batch_start + batch_size,
            len(prompts),
        )

        source_batch = source_prompts[
            batch_start:batch_end
        ]
        expanded_batch = prompts[
            batch_start:batch_end
        ]

        indexed_batch = [
            {
                "index": batch_start + offset + 1,
                "source_scene_prompt": source_prompt,
                "expanded_prompt": expanded_prompt,
            }
            for offset, (
                source_prompt,
                expanded_prompt,
            ) in enumerate(
                zip(
                    source_batch,
                    expanded_batch,
                )
            )
        ]

        expected_indices = [
            item["index"]
            for item in indexed_batch
        ]

        generation_prompt = f"""
# Role: Expanded Visual Prompt Copy Editor

## Goal
Rewrite the expanded prompts into natural, fluent English while preserving
their exact visual meaning.

These prompts were created by mechanically replacing Subject Anchor [TAG]
references with full descriptive noun phrases. That mechanical substitution
can create awkward grammar. Your job is to actively repair those artifacts.

Do NOT reinterpret the story and do NOT make creative changes.

## Typical Expansion Artifacts

BAD:
a man with short dark hair wearing blue jeans's hand

GOOD:
the hand of a man with short dark hair wearing blue jeans

BAD:
a vintage telephone with a brass dial's internal mechanism

GOOD:
the internal mechanism of a vintage telephone with a brass dial

BAD:
the open a wooden drawer

GOOD:
the open wooden drawer

BAD:
a dimly lit bedroom, a dimly lit bedroom with a wooden table

GOOD:
merge the accidental immediate repetition into one natural description

BAD:
a photograph showing a man opening a door lying on the floor

GOOD:
a photograph lying on the floor, showing a man opening a door

## Output Format
Return ONLY one valid JSON object with exactly one key:

"scene_prompts": an array of objects with exactly:
- "index"
- "prompt"

## Constraints
1. Return exactly {len(indexed_batch)} prompts.
2. Preserve these exact indices and order:
   {", ".join(str(index) for index in expected_indices)}
3. Preserve every subject, object, action, spatial relation, camera framing,
   lighting condition, time cue, weather detail, atmosphere, and narrative fact.
4. Actively fix grammar produced by anchor expansion, especially long
   possessive noun phrases, malformed articles, modifier attachment,
   punctuation, agreement, and immediate duplicated wording.
5. Do not add visual information.
6. Do not remove visual information.
7. Do not change characters, objects, locations, chronology, camera,
   lighting, weather, actions, or story events.
8. Do not summarize the scenes.
9. Final prompts must contain no Subject Anchor [TAG] references.
10. Keep every prompt in English.
11. Return no markdown, commentary, code fences, or explanations.

## Video Subject
{video_subject}

## Prompts
{json.dumps(indexed_batch, ensure_ascii=False)}
""".strip()

        logger.info(
            "refining expanded Scene Prompts "
            f"batch {batch_number}/{batch_count}: "
            f"scenes "
            f"{expected_indices[0]}-"
            f"{expected_indices[-1]}"
        )

        current_prompt = generation_prompt
        refined_batch = None
        last_error = None

        # One normal copy-editing attempt plus one focused repair.
        for attempt in range(1, 3):
            parsed_data = None

            try:
                response = call_json(
                    current_prompt
                )

                if response.startswith("Error: "):
                    raise ValueError(
                        response.removeprefix(
                            "Error: "
                        ).strip()
                        or "unknown LLM error"
                    )

                parsed_data = (
                    _load_json_object_response(
                        response
                    )
                )

                raw_scenes = parsed_data.get(
                    "scene_prompts"
                )

                if not isinstance(
                    raw_scenes,
                    list,
                ):
                    raise ValueError(
                        "scene_prompts must be "
                        "a JSON array"
                    )

                if len(raw_scenes) != len(
                    indexed_batch
                ):
                    raise ValueError(
                        "refinement changed scene count: "
                        f"expected={len(indexed_batch)}, "
                        f"received={len(raw_scenes)}"
                    )

                candidate_prompts = []
                received_indices = []
                artifact_errors = []

                for raw_scene in raw_scenes:
                    if not isinstance(
                        raw_scene,
                        dict,
                    ):
                        raise ValueError(
                            "each refined Scene Prompt "
                            "must be a JSON object"
                        )

                    try:
                        index = int(
                            raw_scene.get("index")
                        )
                    except (
                        TypeError,
                        ValueError,
                        OverflowError,
                    ) as exc:
                        raise ValueError(
                            "each refined Scene Prompt "
                            "index must be an integer"
                        ) from exc

                    prompt_text = str(
                        raw_scene.get("prompt")
                        or ""
                    ).strip()

                    if not prompt_text:
                        raise ValueError(
                            "refined Scene Prompt "
                            f"{index} is empty"
                        )

                    if (
                        _VISUAL_ANCHOR_REFERENCE_RE.search(
                            prompt_text
                        )
                    ):
                        raise ValueError(
                            "refined Scene Prompt must "
                            "not contain Subject Anchor "
                            "references"
                        )

                    issues = (
                        _find_expanded_prompt_artifacts(
                            prompt_text,
                            anchors=anchors,
                        )
                    )

                    for issue in issues:
                        artifact_errors.append(
                            f"scene {index}: {issue}"
                        )

                    received_indices.append(
                        index
                    )
                    candidate_prompts.append(
                        prompt_text
                    )

                if (
                    received_indices
                    != expected_indices
                ):
                    raise ValueError(
                        "refinement changed scene "
                        "indices: "
                        f"expected={expected_indices}, "
                        f"received={received_indices}"
                    )

                if artifact_errors:
                    raise ValueError(
                        "linguistic expansion artifacts "
                        "remain: "
                        + " | ".join(
                            artifact_errors
                        )
                    )

                refined_batch = (
                    candidate_prompts
                )
                break

            except Exception as exc:
                last_error = exc

                logger.warning(
                    "expanded Scene Prompt refinement "
                    f"batch "
                    f"{batch_number}/"
                    f"{batch_count} failed: "
                    f"{exc}"
                )

                if attempt < 2:
                    if parsed_data is not None:
                        current_prompt = f"""
# Role: Expansion Artifact Repair Editor

## Goal
The previous copy-edit still contains a grammar artifact caused by mechanical
Subject Anchor expansion.

Repair ONLY the reported linguistic problem. Do not regenerate the scenes.

## Validation Error
{str(exc)}

## Previous Refined JSON
{json.dumps(parsed_data, ensure_ascii=False)}

## Original Expanded Prompts
{json.dumps(indexed_batch, ensure_ascii=False)}

## Repair Rules
1. Preserve exactly {len(indexed_batch)} scenes.
2. Preserve these exact indices and order:
   {", ".join(str(index) for index in expected_indices)}
3. Fix every linguistic artifact named in Validation Error.
4. A long expanded subject description must not directly receive 's or ?s.
   Rephrase naturally using constructions such as "the hand of ...",
   "the face of ...", "the shoulder of ...", or another grammatically
   equivalent construction.
5. Fix duplicated or incompatible articles such as "the open a ...".
6. Remove only accidental immediate repetition caused by expansion.
7. Preserve all subjects, objects, actions, camera framing, lighting,
   atmosphere, chronology, and story facts.
8. Add nothing and remove nothing semantically.
9. Return ONLY the complete corrected JSON object.
10. Do not add markdown, commentary, code fences, or explanations.
""".strip()

                        logger.warning(
                            "expanded Scene Prompt "
                            "refinement "
                            f"batch "
                            f"{batch_number}/"
                            f"{batch_count}, "
                            "repairing previous "
                            "response... 1"
                        )

                    else:
                        current_prompt = (
                            generation_prompt
                        )

                        logger.warning(
                            "expanded Scene Prompt "
                            "refinement "
                            f"batch "
                            f"{batch_number}/"
                            f"{batch_count}, "
                            "trying again... 1"
                        )

        if refined_batch is None:
            logger.warning(
                "expanded Scene Prompt refinement "
                f"batch "
                f"{batch_number}/"
                f"{batch_count} fell back to "
                "the unrefined expanded prompts: "
                f"{last_error}"
            )

            refined_batch = expanded_batch

        refined_prompts.extend(
            refined_batch
        )

    return refined_prompts

def generate_subject_anchors_and_scene_prompts(
    visual_prompts,
    video_subject: str = "",
    video_script: str = "",
    app_config=None,
) -> dict:
    """
    Build one global Subject Anchor library, then rewrite Scene Prompts in
    small batches against that fixed library.

    Separating anchor discovery from scene rewriting keeps identity global
    while avoiding one very large structured LLM response.
    """
    prompts = _normalize_visual_prompt_lines(visual_prompts)
    indexed_prompts = [
        {"index": index, "prompt": prompt}
        for index, prompt in enumerate(prompts, start=1)
    ]
    clean_script = utils.remove_pause_tags(video_script or "").strip()
    batch_size = 8

    def call_json(prompt_text: str) -> str:
        if app_config is None:
            return _generate_response(
                prompt_text,
                json_mode=True,
            )
        return _generate_response(
            prompt_text,
            app_config=app_config,
            json_mode=True,
        )

    def request_json_with_retries(
        prompt_text: str,
        *,
        label: str,
        validator=None,
        repair_prompt_builder=None,
    ):
        """
        Generate structured JSON, validate it, and repair semantic failures.

        Parsing failures retry the original generation prompt. Once valid JSON
        exists, validator failures can instead feed the previous response and
        exact validation error into a focused repair pass.
        """
        last_error = None
        current_prompt = prompt_text

        for attempt in range(1, _max_retries + 1):
            parsed_data = None

            try:
                response = call_json(current_prompt)

                if response.startswith("Error: "):
                    raise ValueError(
                        response.removeprefix("Error: ").strip()
                        or "unknown LLM error"
                    )

                parsed_data = _load_json_object_response(response)
                data = parsed_data

                if validator is not None:
                    data = validator(data)

                return data

            except Exception as exc:
                last_error = exc
                logger.warning(
                    f"{label} failed: {exc}"
                )

                if attempt < _max_retries:
                    if (
                        repair_prompt_builder is not None
                        and parsed_data is not None
                    ):
                        try:
                            current_prompt = repair_prompt_builder(
                                parsed_data,
                                exc,
                            )
                            logger.warning(
                                f"{label}, repairing previous response... "
                                f"{attempt}"
                            )
                        except Exception as repair_exc:
                            logger.warning(
                                f"{label} could not build repair prompt: "
                                f"{repair_exc}; regenerating original request"
                            )
                            current_prompt = prompt_text
                    else:
                        current_prompt = prompt_text
                        logger.warning(
                            f"{label}, trying again... {attempt}"
                        )

        logger.error(
            f"{label} failed after {_max_retries} attempts: "
            f"{last_error}"
        )
        return None

    # --------------------------------------------------------
    # Stage 1: build the global anchor library only.
    # --------------------------------------------------------

    anchor_prompt = f"""
# Role: Visual Continuity Anchor Editor

## Goal
Create one compact global Subject Anchor library for the supplied Visual
Prompts. Do NOT rewrite the scenes yet.

## Output Format
Return ONLY one valid JSON object with exactly one key:

"subject_anchors": an array of objects with exactly:
- "tag"
- "description"

Example:
{{
  "subject_anchors": [
    {{
      "tag": "CHAR_1",
      "description": "a middle-aged man with short dark hair wearing a gray sleep shirt"
    }},
    {{
      "tag": "OBJ_1",
      "description": "an old red rotary telephone with worn paint and a heavy receiver"
    }}
  ]
}}

## Constraints
1. Allowed tags are only CHAR_n, LOC_n, OBJ_n, VEH_n, and CREATURE_n.
2. Number each category consecutively starting at 1.
3. Create anchors for recurring or story-critical visible elements whose stable
   identity materially improves continuity.
4. Prioritize recurring characters, important locations, distinctive objects,
   vehicles, creatures, doors, structures, and other visually important props.
5. Do not create anchors for generic disposable details.
6. Every anchor you create must correspond to something that can actually be
   referenced in at least one supplied Visual Prompt.
7. Anchor descriptions contain only stable visible identity: appearance,
   materials, colors, architecture, clothing, shape, age range, wear, or layout.
8. Do not include temporary actions, poses, expressions, camera angles,
   lighting, damage states, or scene-specific conditions.
9. Do not include carried or held items in a character anchor unless they are
   permanently defining.
10. Descriptions must be compact noun phrases and must not contain [TAG]
    references.
11. Do not add markdown, commentary, code fences, or explanations.
12. Do not rewrite or return Scene Prompts in this step.

## Video Subject
{video_subject}

## Video Script Context
{clean_script}

## Visual Prompts
{json.dumps(indexed_prompts, ensure_ascii=False)}
""".strip()

    logger.info(
        "generating global Subject Anchors: "
        f"scene_count={len(prompts)}"
    )

    def validate_anchor_data(data):
        raw_anchors = data.get("subject_anchors")

        if not isinstance(raw_anchors, list):
            raise ValueError(
                "subject_anchors must be a JSON array"
            )

        validated_anchors = []
        validated_tags = set()

        for raw_anchor in raw_anchors:
            if not isinstance(raw_anchor, dict):
                raise ValueError(
                    "each Subject Anchor must be a JSON object"
                )

            tag = str(
                raw_anchor.get("tag") or ""
            ).strip()

            if tag.startswith("[") and tag.endswith("]"):
                tag = tag[1:-1].strip()

            description = str(
                raw_anchor.get("description") or ""
            ).strip()

            if not _VISUAL_ANCHOR_TAG_RE.fullmatch(tag):
                raise ValueError(
                    f"invalid Subject Anchor tag: [{tag or '?'}]"
                )

            if tag in validated_tags:
                raise ValueError(
                    f"Subject Anchor [{tag}] is defined more than once"
                )

            if not description:
                raise ValueError(
                    f"Subject Anchor [{tag}] has an empty description"
                )

            if _VISUAL_ANCHOR_REFERENCE_RE.search(description):
                raise ValueError(
                    f"Subject Anchor [{tag}] description must not "
                    "contain anchor references"
                )

            validated_tags.add(tag)
            validated_anchors.append(
                {
                    "tag": tag,
                    "description": description,
                }
            )

        return {
            "subject_anchors": validated_anchors
        }

    anchor_data = request_json_with_retries(
        anchor_prompt,
        label="global Subject Anchor generation",
        validator=validate_anchor_data,
    )

    if anchor_data is None:
        return {}

    anchors = anchor_data["subject_anchors"]
    defined_tags = {
        anchor["tag"] for anchor in anchors
    }

    logger.info(
        "global Subject Anchors created: "
        f"anchor_count={len(anchors)}"
    )

    # --------------------------------------------------------
    # Stage 2: rewrite scenes in small batches while keeping
    # the exact same global anchor library for every batch.
    # --------------------------------------------------------

    scene_objects = []
    batch_count = math.ceil(len(indexed_prompts) / batch_size)

    for batch_number, batch_start in enumerate(
        range(0, len(indexed_prompts), batch_size),
        start=1,
    ):
        batch = indexed_prompts[
            batch_start : batch_start + batch_size
        ]

        first_index = batch[0]["index"]
        last_index = batch[-1]["index"]

        scene_prompt = f"""
# Role: Visual Continuity Scene Editor

## Goal
Rewrite ONLY the supplied batch of Visual Prompts using the fixed global
Subject Anchor library below.

The Subject Anchor library is authoritative. Do not create, rename, delete,
renumber, or redefine anchors.

## Output Format
Return ONLY one valid JSON object with exactly one key:

"scene_prompts": an array of objects with exactly:
- "index"
- "prompt"

Preserve the original global scene indices exactly.

## Fixed Subject Anchors
{json.dumps(anchors, ensure_ascii=False)}

## Constraints
1. Return exactly {len(batch)} Scene Prompts.
2. Preserve these exact scene indices and order:
   {", ".join(str(item["index"]) for item in batch)}
3. Never merge, split, omit, duplicate, or reorder scenes.
4. Use [TAG] references whenever a fixed Subject Anchor represents an element
   appearing in that scene.
5. Use only tags defined in the Fixed Subject Anchors above.
6. Do not invent new anchors.
7. Replace repeated identity descriptions with [TAG], while preserving the
   scene's action, composition, framing, lighting, atmosphere, and transient
   state.
8. Treat every [TAG] as a complete noun phrase and keep the surrounding
   sentence grammatically natural.
9. Keep every Scene Prompt in English.
10. Do not add a global art style.
11. Do not add markdown, commentary, code fences, captions, subtitles, logos,
    or explanatory text.
12. The source Visual Prompts are authoritative. Do not invent a new story.

## Video Subject
{video_subject}

## Source Visual Prompts for this batch
{json.dumps(batch, ensure_ascii=False)}
""".strip()

        logger.info(
            "generating anchored Scene Prompts "
            f"batch {batch_number}/{batch_count}: "
            f"scenes {first_index}-{last_index}"
        )

        def validate_scene_batch_data(data):
            raw_scenes = data.get("scene_prompts")

            if not isinstance(raw_scenes, list):
                raise ValueError(
                    "scene_prompts must be a JSON array"
                )

            if len(raw_scenes) != len(batch):
                raise ValueError(
                    "Scene Prompt batch changed scene count: "
                    f"expected={len(batch)}, "
                    f"received={len(raw_scenes)}"
                )

            expected_indices = [
                item["index"] for item in batch
            ]
            received_indices = []
            validated_scenes = []

            for raw_scene in raw_scenes:
                if not isinstance(raw_scene, dict):
                    raise ValueError(
                        "each Scene Prompt must be a JSON object"
                    )

                try:
                    index = int(raw_scene.get("index"))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        "each Scene Prompt index must be an integer"
                    ) from exc

                prompt_text = str(
                    raw_scene.get("prompt") or ""
                ).strip()

                if not prompt_text:
                    raise ValueError(
                        f"Scene Prompt {index} is empty"
                    )

                references = set(
                    _VISUAL_ANCHOR_REFERENCE_RE.findall(
                        prompt_text
                    )
                )

                undefined = sorted(
                    references - defined_tags
                )

                if undefined:
                    raise ValueError(
                        "Scene Prompt references undefined "
                        "Subject Anchor(s): "
                        + ", ".join(
                            f"[{tag}]"
                            for tag in undefined
                        )
                    )

                received_indices.append(index)
                validated_scenes.append(
                    {
                        "index": index,
                        "prompt": prompt_text,
                    }
                )

            if received_indices != expected_indices:
                raise ValueError(
                    "Scene Prompt indices changed: "
                    f"expected={expected_indices}, "
                    f"received={received_indices}"
                )

            return {
                "scene_prompts": validated_scenes
            }

        def build_scene_batch_repair_prompt(
            previous_data,
            validation_error,
        ):
            """
            Repair an already generated scene batch instead of regenerating it.

            The repair model receives the exact invalid JSON and validator
            feedback so it can make the smallest possible correction.
            """
            return f"""
# Role: Scene Prompt Repair Editor

## Goal
Repair the previous JSON response so it passes validation.

The previous response already represents the intended scenes. Do NOT regenerate
the batch from scratch. Preserve everything that is already valid and change
only what is necessary to resolve the validation error.

## Validation Error
{str(validation_error)}

## Previous JSON Response
{json.dumps(previous_data, ensure_ascii=False)}

## Fixed Subject Anchors
{json.dumps(anchors, ensure_ascii=False)}

## Required Scene Indices
{", ".join(str(item["index"]) for item in batch)}

## Repair Rules
1. Preserve exactly {len(batch)} scenes and the exact indices shown above.
2. Preserve the meaning, actions, objects, composition, framing, lighting,
   atmosphere, chronology, and continuity of every scene.
3. Preserve all valid Subject Anchor tags. Do not create, rename, renumber,
   delete, or redefine anchors.
4. Treat the Validation Error above as authoritative and fix only that error.
5. Do not simplify or rewrite unrelated parts of a Scene Prompt.
6. Preserve valid anchor references exactly as they appear.
7. Return ONLY the complete corrected JSON object.
8. Do not add markdown, commentary, code fences, or explanations.
""".strip()

        scene_data = request_json_with_retries(
            scene_prompt,
            label=(
                "anchored Scene Prompt batch "
                f"{batch_number}/{batch_count}"
            ),
            validator=validate_scene_batch_data,
            repair_prompt_builder=build_scene_batch_repair_prompt,
        )

        if scene_data is None:
            return {}

        scene_objects.extend(
            scene_data["scene_prompts"]
        )

    # --------------------------------------------------------
    # Stage 3: discard anchors that no generated scene uses.
    #
    # An unused anchor has no downstream effect and keeping it
    # would make the existing global validator reject an
    # otherwise valid continuity plan.
    # --------------------------------------------------------

    referenced_tags = set()

    for scene in scene_objects:
        referenced_tags.update(
            _VISUAL_ANCHOR_REFERENCE_RE.findall(
                scene["prompt"]
            )
        )

    used_anchors = [
        anchor
        for anchor in anchors
        if anchor["tag"] in referenced_tags
    ]

    removed_count = len(anchors) - len(used_anchors)

    if removed_count:
        logger.info(
            "discarded unused Subject Anchors after "
            f"scene rewriting: count={removed_count}"
        )

    try:
        plan = _normalize_visual_anchor_plan(
            {
                "subject_anchors": used_anchors,
                "scene_prompts": scene_objects,
            },
            expected_scene_count=len(prompts),
        )
    except Exception as exc:
        logger.error(
            "combined visual continuity plan failed "
            f"final validation: {exc}"
        )
        return {}

    logger.success(
        "completed visual continuity plan: "
        f"anchors={plan['anchor_count']}, "
        f"scenes={plan['scene_count']}, "
        f"batches={batch_count}"
    )

    return plan


def generate_terms(
    video_subject: str,
    video_script: str,
    amount: int = 5,
    match_script_order: bool = False,
    visual_prompt_mode: bool = False,
    app_config=None,
) -> List[str]:
    video_script = utils.remove_pause_tags(video_script or "").strip()
    amount = max(int(amount or 1), 1)

    if visual_prompt_mode:
        if match_script_order:
            goal = (
                f"Generate exactly {amount} chronological visual scene prompts that "
                "follow the narration from beginning to end."
            )
            ordering_rule = (
                "8. keep the prompts in the same order as the script narration; "
                "earlier prompts must describe earlier visual moments."
            )
        else:
            goal = (
                f"Generate exactly {amount} distinct visual scene prompts covering "
                "the most useful visible moments from the video script."
            )
            ordering_rule = ""

        prompt_kind_rules = f"""
# Role: Visual Scene Prompt Generator

## Goal
{goal}

## Constraints
1. Return ONLY one valid JSON array containing exactly {amount} strings.
2. Each string must be a complete English visual prompt, normally 12-35 words long.
3. Describe visible content: subject, action, environment, composition, lighting, and mood when useful.
4. Prefer concrete visual details over abstract concepts, narration, explanations, or keywords.
5. Do not add numbering, bullets, labels, markdown, or commentary inside or outside the JSON array.
6. Do not force a global art style; style can be added later by the image/video generation template.
7. When a character, object, vehicle, creature, or location recurs, repeat stable visible descriptors to improve continuity.
{ordering_rule}
9. Do not request on-screen text, captions, subtitles, logos, or watermarks unless the script specifically requires visible text.
10. Use English for every visual prompt, even when the source script is in another language.

## Context
### Video Subject
{video_subject}

### Video Script
{video_script}
""".strip()
        prompt = prompt_kind_rules
    else:
        if match_script_order:
            goal = (
                f"Generate exactly {amount} chronological stock-video search terms that "
                "follow the order of topics in the video script."
            )
            ordering_rule = (
                "6. keep the terms in the same order as the script narration; "
                "earlier terms must describe earlier visual moments."
            )
            example_terms = [
                "opening visual topic",
                *[f"script visual topic {index}" for index in range(2, max(amount, 1))],
                "final visual topic",
            ]
            output_example = json.dumps(example_terms[:amount], ensure_ascii=False)
        else:
            goal = (
                f"Generate exactly {amount} search terms for stock videos, depending on "
                "the subject of a video."
            )
            ordering_rule = ""
            output_example = json.dumps(
                [f"search term {index}" for index in range(1, amount + 1)],
                ensure_ascii=False,
            )

        prompt = f"""
# Role: Video Search Terms Generator

## Goals
{goal}

## Constraints
1. Return ONLY one valid JSON array containing exactly {amount} strings.
2. Each search term should consist of 1-3 words and remain suitable for stock-footage search.
3. Do not return the script, markdown, numbering, explanations, or any text outside the JSON array.
4. The search terms must be related to the subject and script.
5. Use English search terms only.
{ordering_rule}

## Output Example
{output_example}

## Context
### Video Subject
{video_subject}

### Video Script
{video_script}
""".strip()

    logger.info(
        f"subject: {video_subject}, match_script_order: {match_script_order}, "
        f"visual_prompt_mode: {visual_prompt_mode}, amount: {amount}"
    )

    search_terms = []
    response = ""
    for i in range(_max_retries):
        try:
            if app_config is None:
                response = _generate_response(prompt)
            else:
                response = _generate_response(prompt, app_config=app_config)
            if response.startswith("Error: "):
                logger.error(f"failed to generate video terms: {response}")
                return []

            search_terms = json.loads(_strip_code_fence(response))
            if not isinstance(search_terms, list) or not all(
                isinstance(term, str) for term in search_terms
            ):
                logger.error("response is not a list of strings.")
                search_terms = []
                continue

            search_terms = [term.strip() for term in search_terms if term.strip()]
            if len(search_terms) < amount:
                logger.warning(
                    "LLM returned fewer visual inputs than requested: "
                    f"requested={amount}, received={len(search_terms)}"
                )
                search_terms = []
                continue
            if len(search_terms) > amount:
                search_terms = search_terms[:amount]

        except Exception as e:
            logger.warning(f"failed to generate video terms: {str(e)}")
            search_terms = []
            if response:
                match = re.search(r"\[.*]", response, re.DOTALL)
                if match:
                    try:
                        recovered = json.loads(match.group())
                        if isinstance(recovered, list) and all(
                            isinstance(term, str) for term in recovered
                        ):
                            recovered = [term.strip() for term in recovered if term.strip()]
                            if len(recovered) >= amount:
                                search_terms = recovered[:amount]
                    except Exception as recovery_error:
                        logger.warning(
                            f"failed to recover video terms JSON: {str(recovery_error)}"
                        )

        if len(search_terms) == amount:
            break
        if i < _max_retries - 1:
            logger.warning(f"failed to generate video terms, trying again... {i + 1}")

    logger.success(f"completed: \n{search_terms}")
    return search_terms


# =============================================================================
# Social publishing metadata
#
# 根据视频主题和脚本生成发布到短视频平台时常用的 title、caption 和 hashtags。
# 这块能力只复用现有 LLM provider，不接入任何外部发布服务，也不影响视频生成主链路。
# =============================================================================

# 不同平台的文案长度和 hashtag 数量偏好不同。这里使用保守上限，避免模型返回
# 过长内容后调用方还需要二次裁剪。
SOCIAL_PLATFORMS = {
    "tiktok": {"title_max": 100, "caption_max": 2200, "hashtag_count": 5},
    "youtube_shorts": {"title_max": 100, "caption_max": 5000, "hashtag_count": 3},
    "instagram_reels": {"title_max": 125, "caption_max": 2200, "hashtag_count": 8},
    "facebook_reels": {"title_max": 125, "caption_max": 2200, "hashtag_count": 5},
}
DEFAULT_SOCIAL_PLATFORM = "tiktok"
DEFAULT_SOCIAL_LANGUAGE = "auto"
MAX_SOCIAL_SUBJECT_LENGTH = 500
MAX_SOCIAL_SCRIPT_LENGTH = 8000
MAX_SOCIAL_LANGUAGE_LENGTH = 64

SOCIAL_PLATFORM_LABELS = {
    "tiktok": "TikTok",
    "youtube_shorts": "YouTube Shorts",
    "instagram_reels": "Instagram Reels",
    "facebook_reels": "Facebook Reels",
}

# LLM 不可用时的通用兜底标签。这里故意不绑定某个国家或语种，保证 API
# 对中文、英文、越南语等不同场景都能返回可用结构。
DEFAULT_SOCIAL_HASHTAGS = [
    "#shorts",
    "#viral",
    "#trending",
    "#fyp",
    "#video",
    "#reels",
    "#creator",
    "#content",
]


def _resolve_social_platform(platform: str | None) -> str:
    value = (platform or "").strip().lower()
    return value if value in SOCIAL_PLATFORMS else DEFAULT_SOCIAL_PLATFORM


def _normalize_social_language(language: str | None) -> str:
    value = (language or DEFAULT_SOCIAL_LANGUAGE).strip()
    if len(value) > MAX_SOCIAL_LANGUAGE_LENGTH:
        logger.warning(
            "social metadata language is too long and will be truncated to "
            f"{MAX_SOCIAL_LANGUAGE_LENGTH} characters."
        )
        value = value[:MAX_SOCIAL_LANGUAGE_LENGTH]
    return value or DEFAULT_SOCIAL_LANGUAGE


def _limit_social_text(text: str | None, max_length: int, field_name: str) -> str:
    value = (text or "").strip()
    if len(value) <= max_length:
        return value

    # API 层会限制长度；这里继续兜底，是为了保护内部调用或未来 WebUI
    # 直接调用时不会把超长内容发送给模型，避免 token 成本异常。
    logger.warning(
        f"{field_name} is too long and will be truncated to {max_length} characters."
    )
    return value[:max_length]


def _social_language_instruction(language: str | None) -> str:
    language = _normalize_social_language(language)
    if language.lower() == DEFAULT_SOCIAL_LANGUAGE:
        return (
            "Use the same language as the video subject and script. If the subject "
            "and script use different languages, prefer the script language."
        )

    return f'Write "title" and "caption" in this language: {language}.'


def _clamp_text(text, max_length: int) -> str:
    value = ("" if text is None else str(text)).strip()
    if max_length and len(value) > max_length:
        return value[:max_length].rstrip()
    return value


def _normalize_hashtags(raw, count: int) -> List[str]:
    """
    将 LLM 返回的 hashtag 统一整理成 `#tag` 格式。

    LLM 可能返回字符串、数组、带空格的词组、重复标签或包含标点的内容。
    这里集中清洗，可以让接口响应结构稳定，也避免平台发布时出现空标签、
    重复标签或不符合常见格式的 hashtag。
    """
    if isinstance(raw, str):
        candidates = re.split(r"[\s,]+", raw)
    elif isinstance(raw, (list, tuple)):
        # 数组里的每一项视为一个完整标签，因此 "du lich" 会变成
        # "#dulich"，而不是拆成两个标签。
        candidates = [str(entry) for entry in raw]
    else:
        candidates = []

    seen = set()
    result: List[str] = []
    for item in candidates:
        tag = re.sub(r"[^\w]", "", item, flags=re.UNICODE)
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(f"#{tag}")
        if count and len(result) >= count:
            break
    return result


def build_social_metadata_prompt(
    video_subject: str,
    video_script: str = "",
    language: str = DEFAULT_SOCIAL_LANGUAGE,
    platform: str = DEFAULT_SOCIAL_PLATFORM,
) -> str:
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    video_script = _limit_social_text(
        video_script, MAX_SOCIAL_SCRIPT_LENGTH, "video_script"
    )
    platform = _resolve_social_platform(platform)
    spec = SOCIAL_PLATFORMS[platform]
    label = SOCIAL_PLATFORM_LABELS.get(platform, platform)
    language_instruction = _social_language_instruction(language)

    prompt = f"""
# Role: Short-Video Social Media Copywriter

## Goal
Write engaging publishing metadata for a short video that will be posted on {label}.

## Constraints
1. Respond ONLY with a single valid minified JSON object. No markdown, no code fences, no commentary.
2. The JSON must contain exactly these keys: "title", "caption", "hashtags".
3. "title": a catchy hook, at most {spec["title_max"]} characters.
4. "caption": an engaging description that ends with a call to action, at most {spec["caption_max"]} characters. Do not put hashtags inside the caption.
5. "hashtags": a JSON array of exactly {spec["hashtag_count"]} strings. Each must start with "#", contain no spaces, and be relevant to the topic and to {label}.
6. {language_instruction}

## Output Example
{{"title":"...","caption":"...","hashtags":["#example","#video"]}}

## Context
### Video Subject
{video_subject}

### Video Script
{video_script}
""".strip()
    return prompt


def _parse_social_metadata(response: str, platform: str) -> dict:
    spec = SOCIAL_PLATFORMS[_resolve_social_platform(platform)]

    data = None
    try:
        data = json.loads(_strip_code_fence(response))
    except Exception:
        # 部分模型会在 JSON 外层包一段说明文字或 markdown fence。
        # API 调用方只需要稳定结构，所以这里尝试提取第一个 JSON object。
        match = re.search(r"\{.*\}", response or "", re.DOTALL)
        if match:
            data = json.loads(match.group())

    if not isinstance(data, dict):
        raise ValueError("social metadata response is not a JSON object")

    title = _clamp_text(data.get("title", ""), spec["title_max"])
    caption = _clamp_text(data.get("caption", ""), spec["caption_max"])
    hashtags = _normalize_hashtags(data.get("hashtags", []), spec["hashtag_count"])

    if not title and not caption:
        raise ValueError("social metadata response is missing both title and caption")

    return {"title": title, "caption": caption, "hashtags": hashtags}


def _fallback_social_metadata(
    video_subject: str, video_script: str, platform: str
) -> dict:
    spec = SOCIAL_PLATFORMS[_resolve_social_platform(platform)]
    subject = (video_subject or "").strip()
    script = (video_script or "").strip()

    title = subject
    if not title and script:
        # 没有主题时，用脚本第一句兜底生成 title，避免接口返回空标题。
        title = re.split(r"(?<=[.!?。！？])\s+", script)[0]

    return {
        "title": _clamp_text(title, spec["title_max"]),
        "caption": _clamp_text(script or subject, spec["caption_max"]),
        "hashtags": _normalize_hashtags(DEFAULT_SOCIAL_HASHTAGS, spec["hashtag_count"]),
    }


def generate_social_metadata(
    video_subject: str,
    video_script: str = "",
    language: str = DEFAULT_SOCIAL_LANGUAGE,
    platform: str = DEFAULT_SOCIAL_PLATFORM,
) -> dict:
    """
    生成短视频发布文案元数据。

    返回结构固定为 `{"title": str, "caption": str, "hashtags": List[str]}`。
    如果 LLM 不可用或返回格式异常，会降级为通用启发式结果，保证 API
    调用方始终拿到可展示、可发布前编辑的数据结构。
    """
    platform = _resolve_social_platform(platform)
    language = _normalize_social_language(language)
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    video_script = _limit_social_text(
        video_script, MAX_SOCIAL_SCRIPT_LENGTH, "video_script"
    )
    prompt = build_social_metadata_prompt(
        video_subject=video_subject,
        video_script=video_script,
        language=language,
        platform=platform,
    )
    logger.info(f"generating social metadata: platform={platform}, language={language}")

    response = ""
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt)
            if isinstance(response, str) and "Error: " in response:
                logger.error(f"failed to generate social metadata: {response}")
                break
            metadata = _parse_social_metadata(response, platform)
            logger.success(f"completed: \n{metadata}")
            return metadata
        except Exception as e:
            logger.warning(f"failed to parse social metadata: {str(e)}")

        if i < _max_retries - 1:
            logger.warning(
                f"failed to generate social metadata, trying again... {i + 1}"
            )

    logger.warning("falling back to heuristic social metadata")
    return _fallback_social_metadata(video_subject, video_script, platform)


if __name__ == "__main__":
    video_subject = "生命的意义是什么"
    script = generate_script(
        video_subject=video_subject, language="zh-CN", paragraph_number=1
    )
    print("######################")
    print(script)
    search_terms = generate_terms(
        video_subject=video_subject, video_script=script, amount=5
    )
    print("######################")
    print(search_terms)
