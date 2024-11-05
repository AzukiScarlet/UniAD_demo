#!/bin/bash

# 定义下载链接和对应的目标文件夹
declare -A datasets

datasets=(
    # CAN_bus
    ["https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/can_bus.zip"]="/home2/DATASET_AD/wd1/uniad/origin/extensions"
    # Map(v1.3)
    ["https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/nuScenes-map-expansion-v1.3.zip"]="/home2/DATASET_AD/wd1/uniad/origin/extensions"

    # nuScenes V1.0 full dataset data
    # train
    ["https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval_meta.tgz"]="/home2/DATASET_AD/wd1/uniad/origin/fulldataset"
    ["https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval01_blobs.tgz"]="/home2/DATASET_AD/wd1/uniad/origin/fulldataset"
    ["https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval02_blobs.tgz"]="/home2/DATASET_AD/wd1/uniad/origin/fulldataset"
    ["https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval03_blobs.tgz"]="/home2/DATASET_AD/wd1/uniad/origin/fulldataset"
    ["https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval04_blobs.tgz"]="/home2/DATASET_AD/wd1/uniad/origin/fulldataset"
    ["https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval05_blobs.tgz"]="/home2/DATASET_AD/wd1/uniad/origin/fulldataset"
    ["https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval06_blobs.tgz"]="/home2/DATASET_AD/wd1/uniad/origin/fulldataset"
    ["https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval07_blobs.tgz"]="/home2/DATASET_AD/wd1/uniad/origin/fulldataset"
    ["https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval08_blobs.tgz"]="/home2/DATASET_AD/wd1/uniad/origin/fulldataset"
    ["https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval09_blobs.tgz"]="/home2/DATASET_AD/wd1/uniad/origin/fulldataset"
    ["https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval10_blobs.tgz"]="/home2/DATASET_AD/wd1/uniad/origin/fulldataset"
    # test
    ["https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-test_meta.tgz"]="/home2/DATASET_AD/wd1/uniad/origin/fulldataset"
    ["https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-test_blobs.tgz"]="/home2/DATASET_AD/wd1/uniad/origin/fulldataset"
    # mini数据集 可视化要用
    ["https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz"]="/home2/DATASET_AD/wd1/uniad/origin/fulldataset"
)

# 遍历所有的数据集链接
for url in "${!datasets[@]}"; do
    folder="${datasets[$url]}"
    output_file="${folder}/$(basename "$url")"

    # 创建目录（如果不存在）
    mkdir -p "$folder"

    # 使用 wget 进行下载，-c 表示断点续传
    echo "下载 ${url} 到 ${output_file}..."
    wget --limit-rate=10m  -c "$url" -O "$output_file"

    # 检查下载是否成功
    if [[ $? -eq 0 ]]; then
        echo "下载完成！文件保存在：$output_file"
    else
        echo "下载失败：${url}"
    fi
done
