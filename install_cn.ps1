Set-Location $PSScriptRoot

$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1

if (!(Test-Path -Path "venv")) {
    Write-Output  "创建python虚拟环境venv..."
    python -m venv venv
}
.\venv\Scripts\activate

Write-Output "安装依赖..."
pip install -U -r requirements-windows.txt -i https://mirror.baidu.com/pypi/simple

$SOURCEFILE="photomaker/wrappers.py"

$TARGETFILE="venv/Lib/site-packages/spaces/zero/wrappers.py"

Copy-Item -Path $SOURCEFILE -Destination $TARGETFILE -Force

Write-Output "安装完毕"
Read-Host | Out-Null ;
