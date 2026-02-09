#!/bin/bash
# 场景处理完整流水线脚本
# 依次执行：渲染(batch) -> 评估筛选(evaluate) -> 生成对话(generate)

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 默认参数（用于控制流程，不传递给子脚本）
SKIP_BATCH=false
SKIP_EVAL=false
SKIP_GEN=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-batch)
            SKIP_BATCH=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --skip-gen)
            SKIP_GEN=true
            shift
            ;;
        -h|--help)
            echo "场景处理完整流水线脚本"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --skip-batch             跳过渲染阶段"
            echo "  --skip-eval              跳过评估筛选阶段"
            echo "  --skip-gen               跳过生成对话阶段"
            echo "  -h, --help               显示此帮助信息"
            echo ""
            echo "说明:"
            echo "  此脚本使用每个Python脚本内部定义的默认参数。"
            echo "  如需自定义参数，请直接运行相应的Python脚本。"
            echo ""
            echo "示例:"
            echo "  $0                       # 运行完整流水线（使用所有默认参数）"
            echo "  $0 --skip-batch          # 跳过渲染，只运行评估和生成"
            echo "  $0 --skip-batch --skip-eval  # 只运行生成对话阶段"
            echo ""
            echo "各阶段脚本的默认参数请查看对应Python文件内的main()函数。"
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 打印配置信息
echo ""
echo "=========================================="
echo "  场景处理完整流水线"
echo "=========================================="
echo "说明: 使用各Python脚本内部的默认参数"
echo "=========================================="
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# ============================================================
# 阶段1: 批量渲染场景 (batch_process_scenes.py)
# ============================================================
if [ "$SKIP_BATCH" = false ]; then
    log_info "阶段 1/3: 批量渲染场景"
    echo "----------------------------------------"
    
    BATCH_CMD="python batch_process_scenes.py --resume --skip-processed"
    
    log_info "执行命令: $BATCH_CMD"
    
    if eval $BATCH_CMD; then
        log_success "阶段 1/3: 批量渲染完成"
    else
        log_error "阶段 1/3: 批量渲染失败 (退出码: $?)"
        exit 1
    fi
    echo ""
else
    log_warning "跳过阶段 1/3: 批量渲染"
    echo ""
fi

# ============================================================
# 阶段2: 评估和筛选编辑链 (evaluate_and_filter_chains.py)
# ============================================================
if [ "$SKIP_EVAL" = false ]; then
    log_info "阶段 2/3: 评估和筛选编辑链"
    echo "----------------------------------------"
    
    EVAL_CMD="python evaluate_and_filter_chains.py"
    
    log_info "执行命令: $EVAL_CMD"
    
    if eval $EVAL_CMD; then
        log_success "阶段 2/3: 评估筛选完成"
    else
        log_error "阶段 2/3: 评估筛选失败 (退出码: $?)"
        exit 1
    fi
    echo ""
else
    log_warning "跳过阶段 2/3: 评估筛选"
    echo ""
fi

# ============================================================
# 阶段3: 生成最终对话数据 (generate_final_conversations_v3.py)
# ============================================================
if [ "$SKIP_GEN" = false ]; then
    log_info "阶段 3/3: 生成最终对话数据"
    echo "----------------------------------------"
    
    GEN_CMD="python generate_final_conversations_v3.py"
    
    log_info "执行命令: $GEN_CMD"
    
    if eval $GEN_CMD; then
        log_success "阶段 3/3: 生成对话完成"
    else
        log_error "阶段 3/3: 生成对话失败 (退出码: $?)"
        exit 1
    fi
    echo ""
else
    log_warning "跳过阶段 3/3: 生成对话"
    echo ""
fi

# 计算总耗时
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

# 打印总结
echo ""
echo "=========================================="
log_success "所有阶段完成！"
echo "=========================================="
echo "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
echo ""
echo "各Python脚本使用其内部定义的默认参数执行。"
echo "如需查看输出位置等详细信息，请查看各脚本的执行日志。"
echo "=========================================="
echo ""
