#!/usr/bin/env bash
# 一键检测 ANTHROPIC / Claude Code / 网关相关配置，避免 401 与「环境覆盖 settings」问题。
# 用法: bash scripts/check-anthropic-claude-env.sh
# 或:   curl -sSL https://...  # 若你自行托管再下载；本地直接跑即可

set -u

# ---- 脱敏：只显示长度 + 后 6 位 ----
mask() {
  local s="${1:-}"
  local n=${#s}
  if [ "$n" -le 8 ]; then
    echo "<len=$n, hidden>"
  else
    echo "<len=$n, ...${s: -6}>"
  fi
}

# ---- 有问题的历史 token / 主机 ----
RISK_PATTERNS=(
  "anyrouter"
  "sk-yoEhW1QZ1sF"
)

section() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

ISSUES=0
warn() { echo "  [WARN] $*"; ISSUES=$((ISSUES + 1)); }
ok()   { echo "  [OK]   $*"; }

section "0) 当前 shell 中的 ANTHROPIC / 相关环境"
FOUND=
while IFS= read -r line; do
  [ -z "$line" ] && continue
  FOUND=1
  k="${line%%=*}"
  v="${line#*=}"
  echo "  $k=$(mask "$v")"
  for p in "${RISK_PATTERNS[@]}"; do
    if [[ "$line" == *"$p"* ]]; then
      warn "环境变量行命中历史风险项: $p"
    fi
  done
done < <(env 2>/dev/null | sort | grep -iE '^(ANTHROPIC_|ANTHROPIC|CLAUDE|OPENAI).*=' || true)
[ -z "$FOUND" ] && ok "无 ANTHROPIC_/CLAUDE_/OPENAI_ 类变量（对「只认 ~/.claude/settings」较干净）"

section "1) 登录/交互 shell 可能注入的文件（仅对「非注释」行报 WARN）"
# 从 grep -n 结果中筛掉「行内容为注释」
non_comment_hits() {
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    rest="${line#*:}"
    [[ "$rest" =~ ^[[:space:]]*# ]] && continue
    echo "$line"
  done
}

for f in "$HOME/.bash_profile" "$HOME/.profile" "$HOME/.bashrc" "$HOME/.zshrc"; do
  if [ -f "$f" ]; then
    hits_all=$(grep -nE 'anyrouter|ANTHROPIC|yoEhW1' "$f" 2>/dev/null || true)
    hits_code=$(echo "$hits_all" | non_comment_hits)
    if [ -n "$hits_all" ]; then
      echo "  -- $f（命中行，含说明性注释时仅作参考）--"
      echo "$hits_all" | head -25 | sed 's/^\(.\{0,120\}\).*/\1/'
    fi
    if [ -n "$hits_code" ]; then
      # 只把「真在 export/赋值」的行当风险，忽略 case/清理脚本里的字面量
      bad_assign=$(echo "$hits_code" | grep -E '^\s*export[[:space:]]+ANTHRO|^\s*ANTHRO[^=]*=|[[:space:]]=[[:space:]]*.*(anyrouter|sk-yoEh)' || true)
      if [ -n "$bad_assign" ]; then
        for p in "${RISK_PATTERNS[@]}"; do
          if echo "$bad_assign" | grep -qF "$p"; then
            warn "文件 $f 的 export/赋值 仍含风险项: $p"
          fi
        done
      fi
    fi
  fi
done
if [ -f "$HOME/.bash_profile" ] || [ -f "$HOME/.profile" ] || [ -f "$HOME/.bashrc" ] || [ -f "$HOME/.zshrc" ]; then
  _merged=$(cat "$HOME/.bash_profile" "$HOME/.profile" "$HOME/.bashrc" "$HOME/.zshrc" 2>/dev/null)
  if ! echo "$_merged" | grep -E '^[[:space:]]*[^#[:space:]].*' | grep -qE 'anyrouter|sk-yoEhW1QZ1sF' 2>/dev/null; then
    ok "常见 rc 非注释行未再出现 anyrouter / 旧 sk-yoEh 的 export"
  fi
fi

section "2) ~/.claude/settings.json"
SF="$HOME/.claude/settings.json"
if [ -f "$SF" ]; then
  if command -v python3 &>/dev/null; then
    PYOUT=$(python3 - "$SF" <<'PY' || true
import json, re, sys
p = sys.argv[1]
issues = 0
with open(p) as f:
    d = json.load(f)
env = d.get("env") or {}
keys = [k for k in env if k.startswith("ANTHRO")]
print("  文件:", p)
for k in sorted(keys):
    v = str(env.get(k, ""))
    n = len(v)
    tail = v[-6:] if n > 6 else ""
    print(f"  {k} = <len={n} ...{tail}>" if n else f"  {k} = <empty>")
if "ANTHROPIC_API_KEY" not in env and "ANTHROPIC_AUTH_TOKEN" in env:
    print("  [INFO] 仅设置了 ANTHROPIC_AUTH_TOKEN，未设 ANTHROPIC_API_KEY（部分工具只认 API_KEY；可两者同值或二选一以文档为准）")
base = str(env.get("ANTHROPIC_BASE_URL", ""))
# newcli: .../codex 常为 OpenAI 兼容，CC 的 Messages 易失败
if base and re.search(r"/codex(/|$)", base):
    print("  [WARN] ANTHROPIC_BASE_URL 含 /codex 路径：Claude Code 使用 /v1/messages 时，不少网关会 400/401；可试 https://code.newcli.com/claude/ultra 等已支持 Messages 的 base。")
    issues += 1
m = d.get("model", "")
if m in ("", "haiku", "sonnet", "opus"):
    print("  [WARN] 顶层 model 为别名/空: " + repr(m) + "，建议写完整 id（如 claude-opus-4-7 / claude-haiku-4-5-20251001）。")
    issues += 1
else:
    print("  [OK]   顶层 model 字段:", m)
print("__SCRIPT_ISSUES__", issues, sep=" ", flush=True)
PY
    )
    echo "$PYOUT" | grep -v '^__SCRIPT_ISSUES__'
    extra=$(echo "$PYOUT" | sed -n 's/^__SCRIPT_ISSUES__ //p')
    if [ -n "${extra:-}" ] && [ "${extra:-0}" -eq "${extra}" ] 2>/dev/null; then
      ISSUES=$((ISSUES + extra))
    fi
  else
    echo "  未找到 python3，请手动检查: $SF"
  fi
else
  warn "未找到 $SF"
fi

section "3) ccswitch（若已安装）"
for d in "" "$HOME/.local/bin" "$HOME/go/bin"; do
  [ -n "$d" ] && export PATH="$d:$PATH"
  command -v ccswitch &>/dev/null && break
done
if command -v ccswitch &>/dev/null; then
  ccswitch list 2>/dev/null || ccswitch profiles 2>/dev/null || true
else
  echo "  ccswitch 不在 PATH（可将 ~/.local/bin 加入 PATH），跳过"
fi

section "4) Cursor User/settings（含 claudeCode.environmentVariables，可能与 ~/.claude 冲突）"
CURSOR_PATHS=(
  "$HOME/.cursor-server/data/User/settings.json"
  "$HOME/.cursor/User/settings.json"
  "$HOME/.config/Cursor/User/settings.json"
  "$HOME/Library/Application Support/Cursor/User/settings.json"
)
if command -v python3 &>/dev/null; then
  PY4=$(python3 "$HOME" <<'PY'
import json, os, sys

def flat_claude_env(obj):
    """从 Cursor settings 取出 claudeCode 注入的环境变量 -> dict"""
    if not isinstance(obj, dict):
        return {}
    raw = obj.get("claudeCode.environmentVariables")
    if raw is None and "claudeCode" in obj and isinstance(obj["claudeCode"], dict):
        raw = obj["claudeCode"].get("environmentVariables")
    out = {}
    if isinstance(raw, dict):
        out.update(raw)
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            k = item.get("name") or item.get("key")
            v = item.get("value")
            if k and v is not None:
                out[str(k)] = str(v)
    return out

home = sys.argv[1]
sf = os.path.join(home, ".claude", "settings.json")
base_claude = ""
if os.path.isfile(sf):
    try:
        with open(sf, encoding="utf-8") as f:
            base_claude = (json.load(f).get("env") or {}).get("ANTHROPIC_BASE_URL", "") or ""
    except Exception as e:
        print(f"  [WARN] 读取 {sf}: {e}")

paths = [
    os.path.join(home, ".cursor-server", "data", "User", "settings.json"),
    os.path.join(home, ".cursor", "User", "settings.json"),
    os.path.join(home, ".config", "Cursor", "User", "settings.json"),
    os.path.join(home, "Library", "Application Support", "Cursor", "User", "settings.json"),
]
issues = 0
found_any = False
for p in paths:
    if not os.path.isfile(p):
        continue
    found_any = True
    try:
        with open(p, encoding="utf-8") as f:
            st = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  [WARN] JSON 无效，请修复: {p}\n         {e}")
        issues += 1
        continue
    except OSError as e:
        print(f"  [WARN] 无法读: {p} ({e})")
        issues += 1
        continue
    ce = flat_claude_env(st)
    inh = st.get("terminal.integrated.inheritEnv")
    if inh is not None:
        print(f"  -- {p} --")
        print(f"  terminal.integrated.inheritEnv = {inh!r}")
    if ce:
        print(f"  -- {p} (claudeCode 注入) --")
        for k in sorted(ce):
            v = ce[k]
            tail = v[-6:] if len(v) > 6 else v
            print(f"    {k} = <len={len(v)} ...{tail}>")
        bcur = ce.get("ANTHROPIC_BASE_URL", "")
        if "claudedev.com" in bcur or "claudedev" in bcur:
            print("  [WARN] 上述 BASE_URL 含 claudedev：若你实际使用 newcli/自有网关，请与 ~/.claude/settings.json 只保留一致的一套，避免扩展覆盖导致 401/错路由。")
            issues += 1
        if base_claude and bcur and base_claude.rstrip("/") != bcur.rstrip("/"):
            print("  [WARN] Cursor 中 ANTHROPIC_BASE_URL 与 ~/.claude/settings.json 不一致；Claude Code 以扩展注入为准，请删一方或改一致。")
            issues += 1
if not found_any:
    print("  未找到常见路径下的 Cursor User/settings.json，跳过。")
print("__SCRIPT_ISSUES__", issues, sep=" ", flush=True)
PY
  )
  echo "$PY4" | grep -v '^__SCRIPT_ISSUES__'
  e4=$(echo "$PY4" | sed -n 's/^__SCRIPT_ISSUES__ //p')
  if [ -n "${e4:-}" ] && [ "${e4:-0}" -eq "${e4}" ] 2>/dev/null; then
    ISSUES=$((ISSUES + e4))
  fi
else
  for p in "${CURSOR_PATHS[@]}"; do
    [ -f "$p" ] || continue
    echo "  -- $p --"
    grep -E 'inheritEnv|ANTHROPIC|claude|claudedev' "$p" 2>/dev/null | head -25 || true
  done
fi

section "5) SSH 本机向远端传环境（在「你开 Cursor 的那台机」上跑）"
for p in "$HOME/.ssh/config" "$HOME/.ssh/environment"; do
  if [ -f "$p" ]; then
    if grep -qEi 'SendEnv|AcceptEnv|ANTHROPIC|anyrouter' "$p" 2>/dev/null; then
      echo "  -- $p（命中与 env 传递相关，请人工核对）--"
      grep -nEi 'SendEnv|SetEnv|ANTHROPIC|anyrouter' "$p" | head -15 || true
    fi
  fi
done

section "6) 小结与建议"
if [ "$ISSUES" -gt 0 ]; then
  echo "  共 $ISSUES 条 WARN。建议："
  echo "  - 删掉 shell rc / SSH 中 anyrouter 与误配的 ANTHROPIC_*；"
  echo "  - Cursor「设置」里 claudeCode.environmentVariables 会覆盖/合并进扩展进程，与 ~/.claude/settings.json 冲突时以扩展为准，请只保留一致的一套或删空其中一方；"
  echo "  - 用 ~/.claude/settings.json + ccswitch 作唯一信源，或本机/远端只保留一份；"
  echo "  - 改完后：完全退出 Cursor，必要时「Kill Remote Server」或重连 SSH，再试 Claude Code。"
  exit 1
else
  echo "  未发现脚本能识别的明显风险行；若仍 401，请把本脚本完整输出打码后发给维护者，并查 Claude VSCode 扩展日志中 [API REQUEST] 路径与 401 前后几行。"
  exit 0
fi
