#!/bin/bash
json_file="config.json"

modify_json_params() {
    while [ $# -ge 2 ]; do
        local param_name="$1"
        local new_value="$2"

        jq --arg param_name "$param_name" --argjson new_value "$new_value" \
           '.[$param_name] = $new_value' "$json_file" > tmp.json

        mv tmp.json "$json_file"

        shift 2  # 移除已处理的参数名和值
    done
}

# 使用示例：同时修改多个参数
# config_file="config.json"
modify_json_params "learning_rate" 0.001 "batch_size" 64 "epochs" 215 "model" '"GAT"'
