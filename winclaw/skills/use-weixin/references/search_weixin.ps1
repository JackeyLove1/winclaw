# may need scan all user files to find wechat.exe
function Find-WeChatExe {
    $searchDirs = @(
        "$env:ProgramFiles",
        "$env:ProgramFiles(x86)",
        "$env:LOCALAPPDATA",
        "$env:APPDATA"
    )

    foreach ($dir in $searchDirs) {
        if (Test-Path $dir) {
            $result = Get-ChildItem -Path $dir -Filter "WeChat.exe" -Recurse -ErrorAction SilentlyContinue -Force
            if ($result) {
                return $result | Select-Object -ExpandProperty FullName
            }
        }
    }

    return $null
}

$wechatPath = Find-WeChatExe

if ($wechatPath) {
    Write-Host "微信路径: $wechatPath" -ForegroundColor Green
} else {
    Write-Host "未找到微信可执行文件" -ForegroundColor Red
}