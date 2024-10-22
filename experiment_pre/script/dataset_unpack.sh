#!/bin/bash

# 定义目标解压目录
MAPS_DIR="/home2/DATASET_AD/wd2/uniad/data/nuscenes/maps"
CAN_BUS_DIR="/home2/DATASET_AD/wd2/uniad/data/nuscenes"
FULL_DATASET_DIR="/home2/DATASET_AD/wd2/uniad/data/nuscenes"

# 创建目标目录（如果不存在）
mkdir -p "$MAPS_DIR"
mkdir -p "$CAN_BUS_DIR"
mkdir -p "$FULL_DATASET_DIR"

# 遍历目标文件夹中的所有压缩文件
for archive in /home2/DATASET_AD/wd2/uniad/origin/**/*.{zip,tgz}; do
    # 检查文件是否存在
    if [[ -f "$archive" ]]; then
        case "$archive" in
            # Map(v1.3)
            *nuScenes-map-expansion-v1.3.zip)
                echo "正在解压 $archive 到 $MAPS_DIR..."
                unzip -o "$archive" -d "$MAPS_DIR"
                ;;
            # CAN_bus
            *can_bus.zip)
                echo "正在解压 $archive 到 $CAN_BUS_DIR..."
                unzip -o "$archive" -d "$CAN_BUS_DIR"
                ;;
            # nuScenes V1.0 full dataset data
            *v1.0-trainval*.tgz | *v1.0-test*.tgz)
                echo "正在解压 $archive 到 $FULL_DATASET_DIR..."
                tar -xzf "$archive" -C "$FULL_DATASET_DIR"
                ;;
            *)
                echo "不支持的文件格式或未指定的文件: $archive"
                ;;
        esac

        # 检查解压是否成功
        if [[ $? -eq 0 ]]; then
            echo "解压完成！文件保存在：$MAPS_DIR $CAN_BUS_DIR $FULL_DATASET_DIR"
        else
            echo "解压失败：$archive"
        fi
    else
        echo "没有找到匹配的压缩文件。"
    fi
done