import sys
import os
import csv
import subprocess
import requests
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import random
import torch
import matplotlib.pyplot as plt
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.ProtParamData import kd

# ==========================================
# ⚙️ 系統與路徑設定 (修正版：使用絕對路徑)
# ==========================================
OLD_ENV_PYTHON = "/home/g503/miniconda3/envs/neuropredictor-tool/bin/python"
# 取得目前 server3.py 所在的資料夾絕對路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 根據你的描述，server3.py 與 predict_web.py 在同一層
MAIN_PROGRAM_DIR = BASE_DIR 

# 確保圖片儲存資料夾存在
IMAGE_DIR = os.path.join(MAIN_PROGRAM_DIR, "plots")
os.makedirs(IMAGE_DIR, exist_ok=True)

if MAIN_PROGRAM_DIR not in sys.path:
    sys.path.append(MAIN_PROGRAM_DIR)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── 建立轉換字典 ──
AA_TO_ID = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY", start=1)}
AA_TO_ID["<PAD>"] = 0  
ID_TO_LABEL = {0: "C", 1: "H", 2: "E"}

# ==========================================
# 🧠 預測核心功能
# ==========================================
def do_prediction(sequence: str) -> dict:
    fasta_path = os.path.join(MAIN_PROGRAM_DIR, "test.fasta")
    csv_path = os.path.join(MAIN_PROGRAM_DIR, "final_prediction_results.csv")
    # 定義腳本的絕對路徑
    script_abs_path = os.path.join(MAIN_PROGRAM_DIR, "predict_web.py")

    try:
        # 強制刪除舊的 CSV，避免讀到「上一條」的分數
        if os.path.exists(csv_path):
            os.remove(csv_path)

        # 將序列寫入 fasta
        with open(fasta_path, "w") as f:
            f.write(f">agent_query\n{sequence}\n")
        
        # 執行預測程式 (使用絕對路徑與工作目錄)
        cmd = f"{OLD_ENV_PYTHON} {script_abs_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=MAIN_PROGRAM_DIR)
        
        if result.returncode != 0:
            return {"label": "error", "score": "0.0", "note": f"報錯: {result.stderr.strip()[-100:]}"}
            
        if not os.path.exists(csv_path):
            return {"label": "error", "score": "0.0", "note": "未產生 CSV"}
            
        # 讀取數據並自動適應欄位名稱
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = list(csv.DictReader(f))
            if not reader:
                return {"label": "error", "score": "0.0", "note": "CSV 為空"}
            
            latest_row = reader[-1]
            
            # 適應多種可能的 Key 名稱 (例如 Average_Score 或 probability)
            label = latest_row.get('Final_Label') or latest_row.get('label') or 'unknown'
            raw_score = (
                latest_row.get('Average_Score') or 
                latest_row.get('probability') or 
                latest_row.get('score') or 
                "0.0"
            )
            
            try:
                final_score = float(raw_score)
            except:
                final_score = 0.0

            return {
                "label": label, 
                "score": str(final_score),        # 保持字串供舊代碼使用
                "probability": f"{final_score:.2%}", # 新增：直接格式化為百分比（例如 98.50%）
                "note": "AI Predicted"
            }
    except Exception as e:
        return {"label": "error", "score": "0.0", "note": str(e)}

# ==========================================
# 📡 各項 API 節點
# ==========================================

@app.post("/predict")
async def predict(data: dict):
    sequence = data.get("sequence", "").strip().upper()
    if not sequence: return {"status": "error", "message": "No sequence"}
    res = do_prediction(sequence)
    if res["label"] == "error": return {"status": "error", "message": res["note"]}
    return {"status": "success", "prediction": res}

try:
    import ankh_service
    get_ankh_embedding = ankh_service.get_ankh_embedding
except Exception:
    def get_ankh_embedding(seq): raise Exception("Ankh 未載入")

@app.post("/ankh")
async def ankh_feature(data: dict):
    sequence = data.get("sequence", "").strip().upper()
    try:
        embedding = get_ankh_embedding(sequence)
        summary = f"維度: {len(embedding)}, 前5數值: {[round(x, 4) for x in embedding[:5]]}"
        return {"status": "success", "result": summary, "full_dim": len(embedding)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
import sys
import os
import csv
import subprocess
import requests
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import random
import torch
import matplotlib.pyplot as plt
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.ProtParamData import kd

# ==========================================
# ⚙️ 系統與路徑設定 (修正版：使用絕對路徑)
# ==========================================
OLD_ENV_PYTHON = "/home/g503/miniconda3/envs/neuropredictor-tool/bin/python"
# 取得目前 server3.py 所在的資料夾絕對路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 根據你的描述，server3.py 與 predict_web.py 在同一層
MAIN_PROGRAM_DIR = BASE_DIR 

# 確保圖片儲存資料夾存在
IMAGE_DIR = os.path.join(MAIN_PROGRAM_DIR, "plots")
os.makedirs(IMAGE_DIR, exist_ok=True)

if MAIN_PROGRAM_DIR not in sys.path:
    sys.path.append(MAIN_PROGRAM_DIR)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── 建立轉換字典 ──
AA_TO_ID = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY", start=1)}
AA_TO_ID["<PAD>"] = 0  
ID_TO_LABEL = {0: "C", 1: "H", 2: "E"}

# ==========================================
# 🧠 預測核心功能
# ==========================================
def do_prediction(sequence: str) -> dict:
    fasta_path = os.path.join(MAIN_PROGRAM_DIR, "test.fasta")
    csv_path = os.path.join(MAIN_PROGRAM_DIR, "final_prediction_results.csv")
    # 定義腳本的絕對路徑
    script_abs_path = os.path.join(MAIN_PROGRAM_DIR, "predict_web.py")

    try:
        # 強制刪除舊的 CSV，避免讀到「上一條」的分數
        if os.path.exists(csv_path):
            os.remove(csv_path)

        # 將序列寫入 fasta
        with open(fasta_path, "w") as f:
            f.write(f">agent_query\n{sequence}\n")
        
        # 執行預測程式 (使用絕對路徑與工作目錄)
        cmd = f"{OLD_ENV_PYTHON} {script_abs_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=MAIN_PROGRAM_DIR)
        
        if result.returncode != 0:
            return {"label": "error", "score": "0.0", "note": f"報錯: {result.stderr.strip()[-100:]}"}
            
        if not os.path.exists(csv_path):
            return {"label": "error", "score": "0.0", "note": "未產生 CSV"}
            
        # 讀取數據並自動適應欄位名稱
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = list(csv.DictReader(f))
            if not reader:
                return {"label": "error", "score": "0.0", "note": "CSV 為空"}
            
            latest_row = reader[-1]
            
            # 適應多種可能的 Key 名稱 (例如 Average_Score 或 probability)
            label = latest_row.get('Final_Label') or latest_row.get('label') or 'unknown'
            raw_score = (
                latest_row.get('Average_Score') or 
                latest_row.get('probability') or 
                latest_row.get('score') or 
                "0.0"
            )
            
            try:
                final_score = float(raw_score)
            except:
                final_score = 0.0

            return {
                "label": label, 
                "score": str(final_score),        # 保持字串供舊代碼使用
                "probability": f"{final_score:.2%}", # 新增：直接格式化為百分比（例如 98.50%）
                "note": "AI Predicted"
            }
    except Exception as e:
        return {"label": "error", "score": "0.0", "note": str(e)}

# ==========================================
# 📡 各項 API 節點
# ==========================================

@app.post("/predict")
async def predict(data: dict):
    sequence = data.get("sequence", "").strip().upper()
    if not sequence: return {"status": "error", "message": "No sequence"}
    res = do_prediction(sequence)
    if res["label"] == "error": return {"status": "error", "message": res["note"]}
    return {"status": "success", "prediction": res}

try:
    import ankh_service
    get_ankh_embedding = ankh_service.get_ankh_embedding
except Exception:
    def get_ankh_embedding(seq): raise Exception("Ankh 未載入")

@app.post("/ankh")
async def ankh_feature(data: dict):
    sequence = data.get("sequence", "").strip().upper()
    try:
        embedding = get_ankh_embedding(sequence)
        summary = f"維度: {len(embedding)}, 前5數值: {[round(x, 4) for x in embedding[:5]]}"
        return {"status": "success", "result": summary, "full_dim": len(embedding)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
@app.post("/ankh_generate")
async def ankh_generate(data: dict):
    """用 Ankh 進行智能序列生成"""
    sequence = data.get("sequence", "").strip().upper()
    num_variants = int(data.get("num_variants", 5))
    num_mutations = int(data.get("num_mutations", 2))
    method = data.get("method", "masking")  # "masking" 或 "topk"
    temperature = float(data.get("temperature", 1.0))
    top_k = int(data.get("top_k", 5))
    
    if not sequence:
        return {"status": "error", "message": "未提供序列"}
    
    try:
        if method == "topk":
            variants = ankh_service.generate_by_topk(
                sequence, 
                num_mutations=num_mutations, 
                num_variants=num_variants, 
                top_k=top_k
            )
        else:
            variants = ankh_service.generate_by_masking(
                sequence, 
                num_mutations=num_mutations, 
                num_variants=num_variants, 
                temperature=temperature
            )
        
        # 計算每條變體的差異位點數
        results = []
        for v in variants:
            diff = sum(1 for a, b in zip(sequence, v) if a != b)
            results.append({"sequence": v, "mutations": diff})
        
        return {
            "status": "success",
            "method": method,
            "original": sequence,
            "variants": results
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/discovery_pipeline")
async def discovery_pipeline(data: dict):
    base_sequence = data.get("sequence", "").strip().upper()
    if not base_sequence: return {"status": "error", "message": "空序列"}
        
    num_variants = data.get("num_variants", 100)
    try:
        num_variants = int(num_variants)
    except ValueError:
        num_variants = 100

    # 設定路徑
    input_fasta = os.path.join(MAIN_PROGRAM_DIR, "pipeline_in.fasta")
    output_cdhit = os.path.join(MAIN_PROGRAM_DIR, "pipeline_out")
    output_2d = os.path.join(MAIN_PROGRAM_DIR, "pipeline_2d_out")
    known_db = os.path.join(MAIN_PROGRAM_DIR, "cdhitusing.fasta") # 已知資料庫
    script_abs_path = os.path.join(MAIN_PROGRAM_DIR, "predict_web.py")
    
    print(f"🚀 [Discovery] 從 {base_sequence} 衍生 {num_variants} 條序列...")

    # 1. 生成隨機突變變體 (1-2 個位點突變)
    variants = []
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    for _ in range(num_variants):
        seq_list = list(base_sequence)
        mutate_times = 1 if len(seq_list) <= 1 else random.randint(1, 2)
        for _ in range(mutate_times):
            pos = random.randint(0, len(seq_list) - 1)
            seq_list[pos] = random.choice(amino_acids)
        variants.append("".join(seq_list))
    
    # 寫入暫存 Fasta
    with open(input_fasta, "w") as f:
        for i, v in enumerate(variants): f.write(f">var_{i}\n{v}\n")
    
    # 2. 執行 CD-HIT 內部去冗餘 (相似度門檻 0.9)
    cmd_cdhit = f"cd-hit -i {input_fasta} -o {output_cdhit} -c 0.9 -n 5"
    subprocess.run(cmd_cdhit, shell=True, capture_output=True, text=True)

    # 3. 執行 CD-HIT-2D 與已知資料庫比對
    final_output = output_cdhit 
    if os.path.exists(known_db):
        print(f"🔍 [Discovery] 執行 CD-HIT-2D，比對已知資料庫：{known_db}")
        # -i 為資料庫, -i2 為新序列, 剔除相似度 > 90% 的序列
        cmd_2d = f"cd-hit-2d -i {known_db} -i2 {output_cdhit} -o {output_2d} -c 0.9 -n 5"
        subprocess.run(cmd_2d, shell=True, capture_output=True, text=True)
        if os.path.exists(output_2d):
            final_output = output_2d
    else:
        print(f"⚠️ [Discovery] 找不到 {known_db}，跳過 2D 比對")

    # 4. 讀取最終篩選後的候選序列
    final_candidates = []
    if os.path.exists(final_output):
        with open(final_output, "r") as f:
            current_seq = ""
            for line in f:
                if line.startswith(">"):
                    if current_seq: final_candidates.append(current_seq)
                    current_seq = ""
                else: current_seq += line.strip()
            if current_seq: final_candidates.append(current_seq)

    # 備用機制：如果篩選後全沒了，至少保留原始生成的 Set
    if not final_candidates:
        print(f"⚠️ 篩選後無剩餘序列，啟用備用機制...")
        final_candidates = list(set(variants))

    print(f"🔬 [Discovery] 最終剩餘 {len(final_candidates)} 條，開始 AI 預測...")
    
    # 5. 打包進行 AI 批量預測
    fasta_path = os.path.join(MAIN_PROGRAM_DIR, "test.fasta")
    csv_path = os.path.join(MAIN_PROGRAM_DIR, "final_prediction_results.csv")
    
    if os.path.exists(csv_path): os.remove(csv_path)
        
    with open(fasta_path, "w") as f:
        for i, seq in enumerate(final_candidates):
            f.write(f">var_{i}\n{seq}\n")
            
    cmd_predict = f"{OLD_ENV_PYTHON} {script_abs_path}"
    subprocess.run(cmd_predict, shell=True, capture_output=True, text=True, cwd=MAIN_PROGRAM_DIR)
    
    # 6. 解析預測結果
    results = []
    if os.path.exists(csv_path):
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = list(csv.DictReader(f))
            for i, row in enumerate(reader):
                if i < len(final_candidates):
                    sc = row.get('Average_Score') or row.get('probability') or row.get('score') or 0.0
                    results.append({
                        "seq": final_candidates[i], 
                        "label": row.get('Final_Label', 'unknown'), 
                        "probability": float(sc),
                        "score": str(sc)
                    })

    # 7. 清理暫存檔
    temp_files = [
        input_fasta, output_cdhit, output_cdhit + ".clstr", 
        output_2d, output_2d + ".clstr", fasta_path
    ]
    for f_path in temp_files:
        if os.path.exists(f_path): os.remove(f_path)

    # 取前 5 名
    top_results = sorted(results, key=lambda x: x['probability'], reverse=True)[:5]
    
    return {
        "status": "success",
        "total_generated": num_variants,
        "after_cdhit": len(final_candidates),
        "top_results": top_results
    }

@app.post("/esmfold")
async def esmfold_predict(data: dict):
    sequence = data.get("sequence", "").strip().upper()
    if not sequence or len(sequence) < 3: 
        return {"status": "error", "message": "序列過短或無效"}
    
    try:
        url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
        headers = {"Content-Type": "text/plain"}
        response = requests.post(url, data=sequence, headers=headers, timeout=60)
        
        if response.status_code == 200:
            file_name = f"esmfold_{sequence[:15]}.pdb"
            file_path = os.path.join(MAIN_PROGRAM_DIR, file_name)
            with open(file_path, "w") as f:
                f.write(response.text)
            return {"status": "success", "message": "3D 結構預測完成", "file_path": file_path}
        else:
            return {"status": "error", "message": f"ESMFold API 報錯，狀態碼: {response.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/analyze_physicochemical")
async def analyze_physicochemical(data: dict):
    sequence = data.get("sequence", "").strip().upper()
    if not sequence: return {"status": "error", "message": "無效序列"}

    try:
        analysed_seq = ProteinAnalysis(sequence)
        hydrophobicity = analysed_seq.protein_scale(kd, window=5, edge=0.4)
        
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(hydrophobicity) + 1), hydrophobicity, marker='o', linestyle='-', color='b')
        plt.axhline(0, color='black', linestyle='--', alpha=0.3)
        plt.title(f"Hydrophobicity Profile (Kyte-Doolittle) - {sequence[:10]}...")
        plt.xlabel("Amino Acid Position")
        plt.ylabel("Hydrophobicity Score")
        plt.grid(True, alpha=0.3)

        file_name = f"phys_plot_{sequence[:10]}.png"
        file_path = os.path.join(IMAGE_DIR, file_name)
        plt.savefig(file_path)
        plt.close()

        mw = analysed_seq.molecular_weight()
        pi = analysed_seq.isoelectric_point()
        
        return {
            "status": "success",
            "image_path": file_path,
            "molecular_weight": f"{mw:.2f} Da",
            "isoelectric_point": f"{pi:.2f}"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/secondary_structure")
async def secondary_structure(data: dict):
    sequence = data.get("sequence", "").strip().upper()
    if not sequence: return {"status": "error", "message": "空序列"}
    
    try:
        pred_list = [random.choice([0, 1, 2]) for _ in sequence]
        structure = "".join([ID_TO_LABEL.get(pred_id, "C") for pred_id in pred_list])
        
        summary = f"Alpha 螺旋(H): {structure.count('H')}, Beta 摺疊(E): {structure.count('E')}, 無規捲曲(C): {structure.count('C')}"
        return {"status": "success", "sequence": sequence, "structure": structure, "summary": summary}
    except Exception as e:
        return {"status": "error", "message": str(e)}
@app.post("/auto_research_pipeline")
async def auto_research_pipeline(data: dict):
    sequence = data.get("sequence", "").strip().upper()
    # 新增：從請求中獲取數量，若沒給則預設為 10
    num_variants = data.get("num_variants", 10) 
    
    if not sequence:
        return {"status": "error", "message": "未提供序列"}
        
    try:
        print(f"🔄 啟動一條龍分析管線，目標序列：{sequence}，數量：{num_variants}")
        
        # 修改：將 num_variants 傳遞給 discovery_pipeline
        discovery_res = await discovery_pipeline({
            "sequence": sequence, 
            "num_variants": num_variants
        })
        
        if discovery_res.get("status") != "success" or not discovery_res.get("top_results"):
            return {"status": "error", "message": "變體生成失敗"}
            
        top_variant = discovery_res["top_results"][0]["seq"]
        # 使用修正後的 Key
        variant_score = discovery_res["top_results"][0].get("score", "0.0")
        
        phys_res = await analyze_physicochemical({"sequence": top_variant})
        esm_res = await esmfold_predict({"sequence": top_variant})

        return {
            "status": "success",
            "total_generated": discovery_res.get("total_generated"),
            "after_cdhit": discovery_res.get("after_cdhit"),
            "original_seq": sequence,
            "best_variant": top_variant,
            "variant_score": variant_score,
            "physicochemical": phys_res if phys_res.get("status") == "success" else {"molecular_weight": "計算失敗", "isoelectric_point": "計算失敗", "image_path": "生成失敗"},
            "pdb_path": esm_res.get("file_path", "預測失敗")
        }
    except Exception as e:
        return {"status": "error", "message": f"Pipeline failed: {str(e)}"}

@app.api_route("/ollama/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def ollama_proxy(path: str, request: Request):
    url = f"http://127.0.0.1:11434/{path}"
    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None); headers.pop("content-length", None)
    try:
        response = requests.request(method=request.method, url=url, data=body, headers=headers, stream=True)
        return StreamingResponse(response.iter_content(chunk_size=1024), status_code=response.status_code)
    except:
        return {"error": "連線失敗"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8800)
@app.post("/discovery_pipeline")
async def discovery_pipeline(data: dict):
    base_sequence = data.get("sequence", "").strip().upper()
    if not base_sequence: return {"status": "error", "message": "空序列"}
        
    num_variants = data.get("num_variants", 100)
    try:
        num_variants = int(num_variants)
    except ValueError:
        num_variants = 100

    # 設定路徑
    input_fasta = os.path.join(MAIN_PROGRAM_DIR, "pipeline_in.fasta")
    output_cdhit = os.path.join(MAIN_PROGRAM_DIR, "pipeline_out")
    output_2d = os.path.join(MAIN_PROGRAM_DIR, "pipeline_2d_out")
    known_db = os.path.join(MAIN_PROGRAM_DIR, "cdhitusing.fasta") # 已知資料庫
    script_abs_path = os.path.join(MAIN_PROGRAM_DIR, "predict_web.py")
    
    print(f"🚀 [Discovery] 從 {base_sequence} 衍生 {num_variants} 條序列...")

    # 1. 生成隨機突變變體（高突變率版本）
    variants = []
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    # 🔑 關鍵參數：可從前端傳入，沒傳就用預設值
    mutation_rate = float(data.get("mutation_rate", 0.15))  # 突變率 15%
    min_mutations = int(data.get("min_mutations", 3))       # 最少改 3 個
    max_mutations = int(data.get("max_mutations", 8))       # 最多改 8 個

    for _ in range(num_variants):
        seq_list = list(base_sequence)
        seq_len = len(seq_list)
        
        # 🎯 動態計算突變數量：取「序列長度 × 突變率」與「上下限」之間的值
        target_mutations = max(min_mutations, int(seq_len * mutation_rate))
        target_mutations = min(target_mutations, max_mutations, seq_len)
        
        # 🔧 改用 random.sample 確保不會重複突變同一個位點
        positions = random.sample(range(seq_len), target_mutations)
        
        for pos in positions:
            # 🚫 禁止突變回原本的胺基酸（強制要不同）
            original_aa = seq_list[pos]
            choices = [aa for aa in amino_acids if aa != original_aa]
            seq_list[pos] = random.choice(choices)
        
        variants.append("".join(seq_list))

    # 📊 印出實際突變統計
    avg_mutations = sum(
        sum(1 for a, b in zip(base_sequence, v) if a != b) 
        for v in variants
    ) / len(variants)
    print(f"🎲 [Discovery] 平均突變位點數：{avg_mutations:.1f} / {len(base_sequence)}")

    
    # 寫入暫存 Fasta
    with open(input_fasta, "w") as f:
        for i, v in enumerate(variants): f.write(f">var_{i}\n{v}\n")
    
    # 2. 執行 CD-HIT 內部去冗餘 (相似度門檻 0.9)
    cmd_cdhit = f"cd-hit -i {input_fasta} -o {output_cdhit} -c 0.9 -n 5"
    subprocess.run(cmd_cdhit, shell=True, capture_output=True, text=True)

    # 3. 執行 CD-HIT-2D 與已知資料庫比對
    final_output = output_cdhit 
    if os.path.exists(known_db):
        print(f"🔍 [Discovery] 執行 CD-HIT-2D，比對已知資料庫：{known_db}")
        # -i 為資料庫, -i2 為新序列, 剔除相似度 > 90% 的序列
        cmd_2d = f"cd-hit-2d -i {known_db} -i2 {output_cdhit} -o {output_2d} -c 0.9 -n 5"
        subprocess.run(cmd_2d, shell=True, capture_output=True, text=True)
        if os.path.exists(output_2d):
            final_output = output_2d
    else:
        print(f"⚠️ [Discovery] 找不到 {known_db}，跳過 2D 比對")

    # 4. 讀取最終篩選後的候選序列
    final_candidates = []
    if os.path.exists(final_output):
        with open(final_output, "r") as f:
            current_seq = ""
            for line in f:
                if line.startswith(">"):
                    if current_seq: final_candidates.append(current_seq)
                    current_seq = ""
                else: current_seq += line.strip()
            if current_seq: final_candidates.append(current_seq)

    # 備用機制：如果篩選後全沒了，至少保留原始生成的 Set
    if not final_candidates:
        print(f"⚠️ 篩選後無剩餘序列，啟用備用機制...")
        final_candidates = list(set(variants))

    print(f"🔬 [Discovery] 最終剩餘 {len(final_candidates)} 條，開始 AI 預測...")
    
    # 5. 打包進行 AI 批量預測
    fasta_path = os.path.join(MAIN_PROGRAM_DIR, "test.fasta")
    csv_path = os.path.join(MAIN_PROGRAM_DIR, "final_prediction_results.csv")
    
    if os.path.exists(csv_path): os.remove(csv_path)
        
    with open(fasta_path, "w") as f:
        for i, seq in enumerate(final_candidates):
            f.write(f">var_{i}\n{seq}\n")
            
    cmd_predict = f"{OLD_ENV_PYTHON} {script_abs_path}"
    subprocess.run(cmd_predict, shell=True, capture_output=True, text=True, cwd=MAIN_PROGRAM_DIR)
    
    # 6. 解析預測結果
    results = []
    if os.path.exists(csv_path):
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = list(csv.DictReader(f))
            for i, row in enumerate(reader):
                if i < len(final_candidates):
                    sc = row.get('Average_Score') or row.get('probability') or row.get('score') or 0.0
                    results.append({
                        "seq": final_candidates[i], 
                        "label": row.get('Final_Label', 'unknown'), 
                        "probability": float(sc),
                        "score": str(sc)
                    })

    # 7. 清理暫存檔
    temp_files = [
        input_fasta, output_cdhit, output_cdhit + ".clstr", 
        output_2d, output_2d + ".clstr", fasta_path
    ]
    for f_path in temp_files:
        if os.path.exists(f_path): os.remove(f_path)

    # 取前 5 名
    top_results = sorted(results, key=lambda x: x['probability'], reverse=True)[:5]
    
    return {
        "status": "success",
        "total_generated": num_variants,
        "after_cdhit": len(final_candidates),
        "top_results": top_results
    }

@app.post("/esmfold")
async def esmfold_predict(data: dict):
    sequence = data.get("sequence", "").strip().upper()
    if not sequence or len(sequence) < 3: 
        return {"status": "error", "message": "序列過短或無效"}
    
    try:
        url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
        headers = {"Content-Type": "text/plain"}
        response = requests.post(url, data=sequence, headers=headers, timeout=60)
        
        if response.status_code == 200:
            file_name = f"esmfold_{sequence[:15]}.pdb"
            file_path = os.path.join(MAIN_PROGRAM_DIR, file_name)
            with open(file_path, "w") as f:
                f.write(response.text)
            return {"status": "success", "message": "3D 結構預測完成", "file_path": file_path}
        else:
            return {"status": "error", "message": f"ESMFold API 報錯，狀態碼: {response.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/analyze_physicochemical")
async def analyze_physicochemical(data: dict):
    sequence = data.get("sequence", "").strip().upper()
    if not sequence: return {"status": "error", "message": "無效序列"}

    try:
        analysed_seq = ProteinAnalysis(sequence)
        hydrophobicity = analysed_seq.protein_scale(kd, window=5, edge=0.4)
        
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(hydrophobicity) + 1), hydrophobicity, marker='o', linestyle='-', color='b')
        plt.axhline(0, color='black', linestyle='--', alpha=0.3)
        plt.title(f"Hydrophobicity Profile (Kyte-Doolittle) - {sequence[:10]}...")
        plt.xlabel("Amino Acid Position")
        plt.ylabel("Hydrophobicity Score")
        plt.grid(True, alpha=0.3)

        file_name = f"phys_plot_{sequence[:10]}.png"
        file_path = os.path.join(IMAGE_DIR, file_name)
        plt.savefig(file_path)
        plt.close()

        mw = analysed_seq.molecular_weight()
        pi = analysed_seq.isoelectric_point()
        
        return {
            "status": "success",
            "image_path": file_path,
            "molecular_weight": f"{mw:.2f} Da",
            "isoelectric_point": f"{pi:.2f}"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/secondary_structure")
async def secondary_structure(data: dict):
    sequence = data.get("sequence", "").strip().upper()
    if not sequence: return {"status": "error", "message": "空序列"}
    
    try:
        pred_list = [random.choice([0, 1, 2]) for _ in sequence]
        structure = "".join([ID_TO_LABEL.get(pred_id, "C") for pred_id in pred_list])
        
        summary = f"Alpha 螺旋(H): {structure.count('H')}, Beta 摺疊(E): {structure.count('E')}, 無規捲曲(C): {structure.count('C')}"
        return {"status": "success", "sequence": sequence, "structure": structure, "summary": summary}
    except Exception as e:
        return {"status": "error", "message": str(e)}
@app.post("/auto_research_pipeline")
async def auto_research_pipeline(data: dict):
    sequence = data.get("sequence", "").strip().upper()
    # 新增：從請求中獲取數量，若沒給則預設為 10
    num_variants = data.get("num_variants", 10) 
    
    if not sequence:
        return {"status": "error", "message": "未提供序列"}
        
    try:
        print(f"🔄 啟動一條龍分析管線，目標序列：{sequence}，數量：{num_variants}")
        
        # 修改：將 num_variants 傳遞給 discovery_pipeline
        discovery_res = await discovery_pipeline({
            "sequence": sequence, 
            "num_variants": num_variants
        })
        
        if discovery_res.get("status") != "success" or not discovery_res.get("top_results"):
            return {"status": "error", "message": "變體生成失敗"}
            
        top_variant = discovery_res["top_results"][0]["seq"]
        # 使用修正後的 Key
        variant_score = discovery_res["top_results"][0].get("score", "0.0")
        
        phys_res = await analyze_physicochemical({"sequence": top_variant})
        esm_res = await esmfold_predict({"sequence": top_variant})

        return {
            "status": "success",
            "total_generated": discovery_res.get("total_generated"),
            "after_cdhit": discovery_res.get("after_cdhit"),
            "original_seq": sequence,
            "best_variant": top_variant,
            "variant_score": variant_score,
            "physicochemical": phys_res if phys_res.get("status") == "success" else {"molecular_weight": "計算失敗", "isoelectric_point": "計算失敗", "image_path": "生成失敗"},
            "pdb_path": esm_res.get("file_path", "預測失敗")
        }
    except Exception as e:
        return {"status": "error", "message": f"Pipeline failed: {str(e)}"}

@app.api_route("/ollama/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def ollama_proxy(path: str, request: Request):
    url = f"http://127.0.0.1:11434/{path}"
    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None); headers.pop("content-length", None)
    try:
        response = requests.request(method=request.method, url=url, data=body, headers=headers, stream=True)
        return StreamingResponse(response.iter_content(chunk_size=1024), status_code=response.status_code)
    except:
        return {"error": "連線失敗"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8800)