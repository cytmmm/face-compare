# 🧠 FaceAngle REST API - 基于 FAISS 的 1:N 人脸识别系统

这是一个基于 Python、InsightFace 和 FAISS 实现的人脸识别 REST API 系统，支持百万级人脸比对能力，支持人脸注册、查询、删除、更新，并具备向量磁盘持久化能力。

---

## 🚀 功能特性

- ✅ 支持 **1:N 向量比对**（使用 FAISS L2 检索）
- ✅ 支持上传 **图像注册用户**
- ✅ 支持人脸库的注册 / 删除 / 更新 / 查询
- ✅ 向量支持 **磁盘持久化**
- ✅ 基于 FastAPI 封装为 REST API
- ✅ 使用 InsightFace 提取高质量人脸向量

---

## 🧱 技术栈

- Python 3.8+
- [InsightFace](https://github.com/deepinsight/insightface)
- FAISS 向量搜索库
- FastAPI（提供 RESTful 服务）

---

## 📦 安装依赖

推荐使用 `conda` 环境：

```bash
conda create -n faceenv python=3.10 -y
conda activate faceenv
pip install -r requirements.txt
如使用 GPU 可安装 faiss-gpu 和 insightface[all]
```

## 🏁 启动服务
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
或者使用 Gunicorn：
```bash
gunicorn -w 2 -k uvicorn.workers.UvicornWorker app:app
```

### 📸 接口说明
## 📝 接口总览

| 方法 | 接口路径       | 功能说明                 |
|------|----------------|--------------------------|
| POST | `/register`    | 注册用户（支持多图）     |
| POST | `/compare`     | 查询（1:N 人脸比对）      |
| DELETE | `/delete`    | 删除用户                 |
| PUT  | `/update`      | 更新用户（替换向量）     |
| GET  | `/users`       | 获取所有用户 ID          |

---