import json
import zipfile
import os
from tqdm import tqdm

def rescue_submission(old_zip="submission.zip", new_zip="submission_fixed.zip"):
    if not os.path.exists(old_zip):
        print(f"❌ 找不到旧压缩包 {old_zip}！")
        return

    print("🚀 开始极速修复 JSON 格式...")
    
    with zipfile.ZipFile(old_zip, 'r') as zf_in, \
         zipfile.ZipFile(new_zip, 'w', zipfile.ZIP_DEFLATED) as zf_out:
        
        # 找出所有的 json 文件
        json_files = [f for f in zf_in.namelist() if f.endswith('.json')]
        
        for file_name in tqdm(json_files, desc="修复进度"):
            # 1. 读取原来错误的 JSON
            raw_content = zf_in.read(file_name).decode('utf-8')
            
            try:
                data = json.loads(raw_content)
                
                # 2. 强行转换键名
                result = data.get("Classification result", data.get("classification_result", "fake")).lower()
                boxes = data.get("Bounding boxes", data.get("bounding_boxes", []))
                traces = data.get("Visible forgery traces", data.get("visible_forgery_traces", "None"))
                
                # 3. 核心修复：一维列表转二维，真图转空列表
                fixed_boxes = []
                if result == "fake":
                    if isinstance(boxes, list) and len(boxes) == 4 and isinstance(boxes[0], (int, float)):
                        # 如果是 [250, 100, 750, 500]，套一层变 [[250, 100, 750, 500]]
                        fixed_boxes = [boxes]
                    elif isinstance(boxes, list) and len(boxes) > 0 and isinstance(boxes[0], list):
                        # 如果已经是二维了，保持原样
                        fixed_boxes = boxes
                
                # 4. 生成正确的官方格式
                new_data = {
                    "classification_result": result,
                    "bounding_boxes": fixed_boxes,
                    "visible_forgery_traces": traces
                }
                
                # 5. 直接写入新的压缩包
                new_json_str = json.dumps(new_data, ensure_ascii=False, indent=2)
                zf_out.writestr(file_name, new_json_str)
                
            except Exception as e:
                print(f"⚠️ 解析 {file_name} 失败，已跳过。报错: {e}")
                
    print(f"\n✅ 抢救成功！新文件已生成: {new_zip}")
    print("赶紧拿去官网提交吧，祝你好运！")

if __name__ == "__main__":
    rescue_submission()
