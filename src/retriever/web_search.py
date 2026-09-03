"""
Vietnamese Law MCP Retriever / Web Search
==========================================
Tích hợp Model Context Protocol (MCP) Server tra cứu dữ liệu pháp luật Việt Nam
(Dựa trên đặc tả chuẩn của @ansvar/vietnamese-law-mcp & Ansvar Open Law MCP:
 https://mcpmarket.com/server/vietnamese-law & https://www.npmjs.com/package/@ansvar/vietnamese-law-mcp).

Bộ công cụ (Tools) theo chuẩn Ansvar Law MCP:
- search_legislation: Tìm kiếm full-text (BM25) qua các văn bản, điều khoản luật Việt Nam.
- get_provision: Lấy chính xác toàn văn một điều khoản (Điều/Khoản) theo tên văn bản luật.
- validate_citation: Xác thực trích dẫn điều luật để phòng chống hallucination.
- check_currency: Kiểm tra tính hiệu lực của văn bản quy phạm pháp luật (còn hiệu lực/hết hiệu lực).
- list_sources: Liệt kê danh mục các nguồn luật có trong cơ sở dữ liệu MCP.
"""

import asyncio
import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional

from src.config import settings

logger = logging.getLogger(__name__)


class VietnameseLawMCPClient:
	"""Client kết nối tới Vietnamese Law MCP Server thông qua Model Context Protocol (stdio / JSON-RPC)."""

	def __init__(
		self,
		command: Optional[str] = None,
		args: Optional[List[str]] = None,
		env: Optional[Dict[str, str]] = None,
	):
		raw_cmd = command or os.getenv("VIETNAMESE_LAW_MCP_COMMAND", "npx")
		self.command = shutil.which(raw_cmd) or raw_cmd
		self.args = args or (
			["-y", "@ansvar/vietnamese-law-mcp"]
			if "npx" in raw_cmd
			else []
		)
		self.env = env or dict(os.environ)

	def is_available(self) -> bool:
		"""Kiểm tra command thực thi (ví dụ npx/node) có sẵn trong PATH hệ thống hay không."""
		return shutil.which(self.command) is not None or os.path.exists(self.command)

	async def call_tool_async(
		self,
		tool_name: str,
		arguments: Dict[str, Any],
		timeout: float = 30.0,
	) -> Any:
		"""Gọi một tool MCP thông qua thư viện `mcp` chính thức hoặc fallback JSON-RPC 2.0."""
		try:
			from mcp import ClientSession, StdioServerParameters
			from mcp.client.stdio import stdio_client

			server_params = StdioServerParameters(
				command=self.command,
				args=self.args,
				env=self.env,
			)

			async with stdio_client(server_params) as (read, write):
				async with ClientSession(read, write) as session:
					await session.initialize()
					result = await asyncio.wait_for(
						session.call_tool(tool_name, arguments=arguments),
						timeout=timeout,
					)
					return result
		except Exception as e:
			logger.debug(f"Gọi tool '{tool_name}' qua mcp package gặp exception ({e}), chuyển sang fallback JSON-RPC 2.0 subprocess...")
			return await self._call_tool_raw_jsonrpc(tool_name, arguments, timeout=timeout)

	async def _call_tool_raw_jsonrpc(
		self,
		tool_name: str,
		arguments: Dict[str, Any],
		timeout: float = 30.0,
	) -> Any:
		"""Thực thi JSON-RPC 2.0 trực tiếp qua process stdin/stdout."""
		if not self.is_available():
			logger.error(f"Command '{self.command}' không tồn tại trong hệ thống.")
			return None

		cmd = [self.command] + self.args
		try:
			proc = await asyncio.create_subprocess_exec(
				*cmd,
				stdin=asyncio.subprocess.PIPE,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				env=self.env,
			)

			# MCP JSON-RPC 2.0 Handshake & Tool Call
			init_req = {
				"jsonrpc": "2.0",
				"id": 1,
				"method": "initialize",
				"params": {
					"protocolVersion": "2024-11-05",
					"capabilities": {},
					"clientInfo": {"name": "rag-textmining-client", "version": "1.0.0"},
				},
			}
			call_req = {
				"jsonrpc": "2.0",
				"id": 2,
				"method": "tools/call",
				"params": {
					"name": tool_name,
					"arguments": arguments,
				},
			}

			payload = f"{json.dumps(init_req)}\n{json.dumps(call_req)}\n".encode("utf-8")
			stdout, stderr = await asyncio.wait_for(proc.communicate(input=payload), timeout=timeout)

			lines = stdout.decode("utf-8", errors="ignore").strip().split("\n")
			for line in lines:
				try:
					data = json.loads(line.strip())
					if data.get("id") == 2 and "result" in data:
						return data["result"]
				except json.JSONDecodeError:
					continue

		except Exception as e:
			logger.error(f"Lỗi khi thực thi MCP process: {e}")
			return None

		return None

	def call_tool(self, tool_name: str, arguments: Dict[str, Any], timeout: float = 30.0) -> Any:
		"""Synchronous wrapper cho call_tool_async."""
		try:
			loop = asyncio.get_event_loop()
			if loop.is_running():
				import nest_asyncio
				nest_asyncio.apply()
				return loop.run_until_complete(self.call_tool_async(tool_name, arguments, timeout))
			return loop.run_until_complete(self.call_tool_async(tool_name, arguments, timeout))
		except RuntimeError:
			return asyncio.run(self.call_tool_async(tool_name, arguments, timeout))


class VietnameseLawWebSearch:
	"""Module Web Search & Retrieval pháp luật Việt Nam qua Ansvar Vietnamese Law MCP Server."""

	def __init__(
		self,
		mcp_client: Optional[VietnameseLawMCPClient] = None,
		default_limit: int = 5,
	):
		self.client = mcp_client or VietnameseLawMCPClient()
		self.default_limit = default_limit

	def search_legislation(self, query: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
		"""Tìm kiếm full-text các văn bản pháp luật, điều khoản liên quan qua tool `search_legislation`."""
		k = limit or self.default_limit

		# Danh sách tool name theo chuẩn Ansvar MCP và fallback tương thích
		tool_candidates = [
			"search_legislation",
			"search",
			"search_laws",
			"search_statutes",
			"find_legal_provisions",
		]

		for tool in tool_candidates:
			res = self.client.call_tool(tool, {"query": query, "limit": k})
			if res:
				parsed = self._parse_mcp_result(res)
				if parsed:
					return parsed

		logger.info(f"Không tìm thấy kết quả từ MCP server cho truy vấn: '{query}'")
		return []

	def get_provision(self, statute: str, provision: str | int) -> Optional[Dict[str, Any]]:
		"""Trích xuất chính xác nội dung của một Điều/Khoản luật cụ thể qua tool `get_provision`.

		Parameters
		----------
		statute : str
			Tên hoặc mã văn bản luật (ví dụ: 'Luật Đất đai 2024', 'Bộ luật Dân sự 2015').
		provision : str | int
			Số điều / khoản cần lấy (ví dụ: 79, 'Điều 79').
		"""
		args_candidates = [
			{"statute": statute, "provision": str(provision)},
			{"law_name": statute, "article": str(provision)},
			{"document": statute, "section": str(provision)},
		]

		for tool in ["get_provision", "get_article", "read_article"]:
			for args in args_candidates:
				res = self.client.call_tool(tool, args)
				if res:
					parsed = self._parse_mcp_single(res)
					if parsed:
						return parsed
		return None

	def validate_citation(self, citation: str) -> Dict[str, Any]:
		"""Xác thực tính chính xác của trích dẫn luật qua tool `validate_citation` (chống hallucination)."""
		res = self.client.call_tool("validate_citation", {"citation": citation})
		if res:
			return self._parse_mcp_single(res) or {"valid": True, "raw": str(res)}
		return {"valid": False, "reason": "Không thể kết nối hoặc không tìm thấy trích dẫn"}

	def check_currency(self, statute: str) -> Dict[str, Any]:
		"""Kiểm tra hiệu lực của văn bản quy phạm pháp luật qua tool `check_currency`."""
		for tool in ["check_currency", "check_validity", "is_in_force"]:
			res = self.client.call_tool(tool, {"statute": statute})
			if res:
				return self._parse_mcp_single(res) or {"statute": statute, "raw": str(res)}
		return {"statute": statute, "status": "unknown"}

	def list_sources(self) -> List[Dict[str, Any]]:
		"""Liệt kê các nguồn luật, bộ luật hiện có trên MCP server qua tool `list_sources`."""
		for tool in ["list_sources", "get_sources", "list_statutes"]:
			res = self.client.call_tool(tool, {})
			if res:
				return self._parse_mcp_result(res)
		return []

	def search(self, query: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
		"""Alias ngắn gọn cho search_legislation."""
		return self.search_legislation(query, limit=limit)

	def retrieve(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
		"""Giao diện retrieve tiêu chuẩn đồng bộ với toàn bộ RAG Pipeline trong dự án.

		Returns
		-------
		Dict[str, Any]
			{'query': str, 'context_chunks': List[Dict], 'source': 'vietnamese_law_mcp'}
		"""
		k = top_k or self.default_limit
		raw_items = self.search_legislation(query, limit=k)

		formatted_chunks = []
		for idx, item in enumerate(raw_items):
			formatted_chunks.append({
				"id": item.get("id") or item.get("provision_id") or f"mcp_law_{idx+1}",
				"content": item.get("content") or item.get("text") or item.get("provision_text", ""),
				"metadata": {
					"source": item.get("source") or item.get("statute") or item.get("law_name", "Vietnamese Law MCP"),
					"article_no": item.get("article_no") or item.get("provision") or item.get("article"),
					"chapter": item.get("chapter"),
					"title": item.get("title"),
					"status": item.get("status", "in_force"),
				},
				"dense_score": None,
				"rerank_score": item.get("score") or item.get("relevance"),
				"source": "vietnamese_law_mcp",
			})

		return {
			"query": query,
			"context_chunks": formatted_chunks,
			"source": "vietnamese_law_mcp",
		}

	def _parse_mcp_result(self, raw_result: Any) -> List[Dict[str, Any]]:
		"""Parse và chuẩn hóa danh sách kết quả trả về từ MCP response."""
		if isinstance(raw_result, dict):
			if "content" in raw_result and isinstance(raw_result["content"], list):
				for block in raw_result["content"]:
					if isinstance(block, dict) and block.get("type") == "text":
						text = block.get("text", "")
						try:
							parsed = json.loads(text)
							if isinstance(parsed, list):
								return parsed
							if isinstance(parsed, dict) and "results" in parsed:
								return parsed["results"]
						except Exception:
							return [{"content": text, "source": "vietnamese_law_mcp"}]
			elif "results" in raw_result:
				return raw_result["results"]
		elif isinstance(raw_result, list):
			return raw_result
		return []

	def _parse_mcp_single(self, raw_result: Any) -> Optional[Dict[str, Any]]:
		"""Parse kết quả đơn lẻ trả về từ MCP response."""
		if isinstance(raw_result, dict):
			if "content" in raw_result and isinstance(raw_result["content"], list):
				for block in raw_result["content"]:
					if isinstance(block, dict) and block.get("type") == "text":
						try:
							return json.loads(block.get("text", ""))
						except Exception:
							return {"content": block.get("text", "")}
			return raw_result
		return {"content": str(raw_result)}
