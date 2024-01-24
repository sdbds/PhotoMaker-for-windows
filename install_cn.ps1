Set-Location $PSScriptRoot

$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1
$Env:PIP_INDEX_URL = "https://mirror.baidu.com/pypi/simple"

if (!(Test-Path -Path "venv")) {
    Write-Output  "����python���⻷��venv..."
    python -m venv venv
}
.\venv\Scripts\activate

Write-Output "��װ����..."
pip install -U -r requirements-windows.txt

$SOURCEFILE="photomaker/wrappers.py"

$TARGETFILE="venv/Lib/site-packages/spaces/zero/wrappers.py"

Copy-Item -Path $SOURCEFILE -Destination $TARGETFILE -Force

Write-Output "��װ���"
Read-Host | Out-Null ;
