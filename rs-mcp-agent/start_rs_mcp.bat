@echo off
set HTTP_PROXY=
set HTTPS_PROXY=
set ALL_PROXY=
set http_proxy=
set https_proxy=
set all_proxy=

set NO_PROXY=127.0.0.1,localhost,::1
set no_proxy=127.0.0.1,localhost,::1
set RS_SERVICE_URL=http://127.0.0.1:18765

cd /d D:\program_myself\Myagent\rs-mcp-agent

C:\Users\45376\anaconda3\envs\rs-mcp-client\python.exe -m rs_mcp.server 2>> D:\program_myself\Myagent\rs-mcp-agent\mcp_stderr.log