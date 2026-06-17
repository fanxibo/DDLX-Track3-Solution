from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download

print("正在登录魔搭账号...")
api = HubApi()
# 注入你的 Token
api.login('ms-7a7fb55b-cce0-4d1a-8585-113b72f85047')

print("登录成功，开始下载 DDL_X 数据集 (支持断点续传)...")
dataset_dir = snapshot_download('DDLteam/DDL_X', repo_type='dataset', local_dir='./DDL_X_dataset')

print(f"下载彻底完成！文件保存在: {dataset_dir}")
