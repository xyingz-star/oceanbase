#!/bin/bash
#
# 采集 observer 进程的 CPU 火焰图（用于分析 kmeans 等）。
# 用法：
#   1. 启动 observer 并确认已在运行
#   2. 执行本脚本: ./script/kmeans_flamegraph.sh
#   3. 脚本会启动 perf 采样，请在【另一终端】执行建索引等 kmeans 相关操作
#   4. 采样足够后回到本终端按 Ctrl+C 结束
#   5. 脚本会自动生成 kmeans_flame.svg
#

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OB_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FLAMEGRAPH_DIR="${FLAMEGRAPH_DIR:-$HOME/FlameGraph}"
OUTPUT_SVG="${1:-$OB_ROOT/kmeans_flame.svg}"
OB_BINARY="${OB_BINARY:-$OB_ROOT/build_release/src/observer/observer}"

# 确保 FlameGraph 在 PATH 中
if [ -d "$FLAMEGRAPH_DIR" ]; then
  export PATH="$FLAMEGRAPH_DIR:$PATH"
else
  echo "[WARN] FlameGraph 目录不存在: $FLAMEGRAPH_DIR"
  echo "       请先执行: git clone https://github.com/brendangregg/FlameGraph.git $FLAMEGRAPH_DIR"
  echo "       或设置环境变量 FLAMEGRAPH_DIR 指向 FlameGraph 目录"
fi

# 查找 observer 进程（排除 obshell）
OBSERVER_PID=$(ps aux | grep '[o]bserver' | grep -v obshell | awk '{print $2}' | head -1)

if [ -z "$OBSERVER_PID" ]; then
  echo "[ERROR] 未找到 observer 进程，请先启动 observer 后再运行本脚本"
  exit 1
fi

echo "=========================================="
echo "observer PID: $OBSERVER_PID"
echo "输出文件:     $OUTPUT_SVG"
echo "observer 二进制(符号): $OB_BINARY"
echo "=========================================="
echo "即将开始 perf 采样。请到【另一终端】执行建索引等 kmeans 操作，"
echo "采样约 30–60 秒后回到本终端按 Ctrl+C 结束采样。"
echo "=========================================="

# 采样（会阻塞，直到用户 Ctrl+C）。Ctrl+C 会使 perf 返回非零，用 || true 避免 set -e 导致脚本直接退出
sudo perf record -F 99 -p "$OBSERVER_PID" -g || true

echo ""
if [ ! -f perf.data ]; then
  echo "[ERROR] 未找到 perf.data，无法生成火焰图（请确认在采样目录下执行）。"
  exit 1
fi
echo "正在生成火焰图 ..."

# 若希望 perf script 使用指定二进制解析符号，可加: --symfs 或 确保当前目录有 perf.data
sudo perf script | stackcollapse-perf.pl | flamegraph.pl > "$OUTPUT_SVG"

echo "完成。火焰图已写入: $OUTPUT_SVG"
echo "可用浏览器打开查看。"
