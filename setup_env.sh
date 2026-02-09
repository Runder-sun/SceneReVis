#!/bin/bash
# 设置资产检索所需的环境变量

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 3D资产目录
export PTH_3DFUTURE_ASSETS="/path/to/datasets/3d-front/3D-FUTURE-model"

# 元数据文件
export PTH_ASSETS_METADATA="${PROJECT_ROOT}/metadata/model_info_3dfuture_assets.json"
export PTH_ASSETS_METADATA_SCALED="${PROJECT_ROOT}/metadata/model_info_3dfuture_assets_scaled.json"
export PTH_ASSETS_EMBED="${PROJECT_ROOT}/metadata/model_info_3dfuture_assets_embeds.pickle"

echo "✓ 环境变量已设置:"
echo "  PTH_3DFUTURE_ASSETS: $PTH_3DFUTURE_ASSETS"
echo "  PTH_ASSETS_METADATA: $PTH_ASSETS_METADATA"
echo "  PTH_ASSETS_METADATA_SCALED: $PTH_ASSETS_METADATA_SCALED"
echo "  PTH_ASSETS_EMBED: $PTH_ASSETS_EMBED"
echo ""
echo "验证文件是否存在..."

if [ -d "$PTH_3DFUTURE_ASSETS" ]; then
    echo "  ✓ 3D资产目录存在"
else
    echo "  ✗ 3D资产目录不存在: $PTH_3DFUTURE_ASSETS"
fi

if [ -f "$PTH_ASSETS_METADATA" ]; then
    echo "  ✓ 资产元数据文件存在"
else
    echo "  ✗ 资产元数据文件不存在: $PTH_ASSETS_METADATA"
fi

if [ -f "$PTH_ASSETS_METADATA_SCALED" ]; then
    echo "  ✓ 缩放资产元数据文件存在"
else
    echo "  ✗ 缩放资产元数据文件不存在: $PTH_ASSETS_METADATA_SCALED"
fi

if [ -f "$PTH_ASSETS_EMBED" ]; then
    echo "  ✓ 资产嵌入文件存在"
else
    echo "  ✗ 资产嵌入文件不存在: $PTH_ASSETS_EMBED"
fi

echo ""
echo "环境设置完成！现在可以运行测试了。"
