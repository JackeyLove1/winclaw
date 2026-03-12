---
name: use-weixin
version: 1.0.0
description: use weixin in windows platform for send messages and complete work
author: JackyFan
---

# use-weixin agent

use python tools and powsershell script for send weixin messages and complete job

## references
references code in ~/.winclaw/skills/references

```
use-weixin/
├── SKILL.md              # use-weixin skill file
└── references/
    ├── send_message.py  # use python pywinauto to operate weixin and send messages to target people
    └── search_weixin.ps1 # find weixin exe powershell script
    └── send_files.ps1    # send files to target group
└── scripts/              # more scripts you can store it for next quick use

```


## process
0. use rg or powershell to scan user disk find wechat.exe, if the path is in the SKILL file, please append it in the file 
1. you can **record your experiences in the `lessons` chapter**, like weixin exe path: C:\Program Files\Tencent\WeChat\WeChat.exe for next time to quickly to run it.
2. use **append only** on the SKILL.md for minimal modify.
3. before every job, **use powershell screen snapshot to read the weixin window** get more infomations 
4. **DONNOT MODIFY THE references FILES**

## lessons