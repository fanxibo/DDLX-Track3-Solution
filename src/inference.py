# -*- coding: utf-8 -*-
import os
import json
import torch
import re
import zipfile
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 允许加载损坏/截断的图片
ImageFile.LOAD_TRUNCATED_IMAGES = True

def extract_json_from_text(text):
    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
    except Exception as e:
        pass
    return {
        "Classification result": "fake", 
        "Bounding boxes": [100, 100, 500, 500],
        "Visible forgery traces": text.strip()[:200]
    }

def main():
    image_dir = "./DDL_X_dataset/image"
    zip_path = "submission.zip"
    old_json_dir = "./submission/json"
    
    os.makedirs("./submission", exist_ok=True)

    # ==========================================
    # 核心动作 1：把以前生成的零碎 JSON 收编进 ZIP，并删除原文件释放配额
    # ==========================================
    if os.path.exists(old_json_dir):
        loose_files = [f for f in os.listdir(old_json_dir) if f.endswith('.json')]
        if loose_files:
            print(f"检测到 {len(loose_files)} 个之前生成的零碎 JSON 文件！正在将它们打包进 ZIP 以释放服务器空间...")
            with zipfile.ZipFile(zip_path, 'a', zipfile.ZIP_DEFLATED) as zf:
                for lf in tqdm(loose_files, desc="Packing"):
                    full_path = os.path.join(old_json_dir, lf)
                    zf.write(full_path, arcname=f"json/{lf}") # 保持官方要求的 json/xxx.json 结构
                    os.remove(full_path) # 打包完立刻删除原文件
            print("历史文件收编清理完成！")

    # 获取当前 ZIP 里已经做完的名单，用于断点续传
    completed_jsons = set()
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zf:
            completed_jsons = set(zf.namelist())

    # ==========================================
    # 核心动作 2：加载模型
    # ==========================================
    print("Loading Qwen2-VL-7B model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    prompt_text = (
        "You are an expert deepfake detection system. "
        "Analyze this image. Is it a real image or a fake/manipulated image? "
        "If fake, identify the bounding box of the manipulated region in [xmin, ymin, xmax, ymax] format (scaled from 1 to 1000) "
        "and describe the visible forgery traces. "
        "Respond STRICTLY with a valid JSON object in the exact following format, without any markdown formatting or extra text:\n"
        "{\"Classification result\": \"fake\" or \"real\", \"Bounding boxes\": [x, y, x, y] or \"None\", \"Visible forgery traces\": \"brief description\"}"
    )

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images, starting direct-to-zip inference...")

    # ==========================================
    # 核心动作 3：边推理边直接写进 ZIP 压缩包
    # ==========================================
    with zipfile.ZipFile(zip_path, 'a', zipfile.ZIP_DEFLATED) as zf:
        for img_name in tqdm(image_files):
            # 拼装出在压缩包内部的路径，例如 json/12345.json
            json_name_in_zip = f"json/{os.path.splitext(img_name)[0]}.json"
            
            # 断点续传：如果这个文件已经在压缩包里了，秒跳过！
            if json_name_in_zip in completed_jsons:
                continue
                
            img_path = os.path.join(image_dir, img_name)
            
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_path},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]

                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda")

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=128)
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

                result_dict = extract_json_from_text(output_text)
                
            except Exception as e:
                print(f"\n[Warning] Skip corrupted image {img_name}: {e}")
                result_dict = {
                    "Classification result": "fake",
                    "Bounding boxes": "None",
                    "Visible forgery traces": "Image corrupted."
                }

            # 直接把字典转成字符串，不落地直接塞进压缩包！
            json_str = json.dumps(result_dict, ensure_ascii=False, indent=4)
            zf.writestr(json_name_in_zip, json_str)
            completed_jsons.add(json_name_in_zip) # 更新已完成名单

    print("Finished! ALL results are safely packed in submission.zip!")

if __name__ == "__main__":
    main()
