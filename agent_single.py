import requests
import re
import os
import shutil
import subprocess
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from fpdf import FPDF

# ==========================================
# ⚙️ 基本設定
# ==========================================
BASE_TUNNEL_URL = "https://pellicular-becki-nastily.ngrok-free.dev"
print(f"🔧 正在連線至隧道: {BASE_TUNNEL_URL}")

GLOBAL_LOGS = []

def add_log(step_name, content):
    """將步驟紀錄加入全局日誌"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] --- {step_name} ---\n{content}\n"
    GLOBAL_LOGS.append(log_entry)
    print(f"📝 日誌已更新: {step_name}")

def sanitize_pdf_text(text: str) -> str:
    """將常見的 Emoji 替換為文字標籤，避免 Unicode 渲染報錯。"""
    replacements = {
        "✅": "[OK]", "❌": "[ERROR]", "📊": "[PLOT]",
        "🚀": "[START]", "📝": "[LOG]", "📡": "[SEND]",
        "🔧": "[CONFIG]", "⚠️": "[WARN]", "🧬": "[DNA]",
        "🔬": "[LAB]", "🎨": "[ART]", "🖼️": "[IMG]",
        "🛡️": "[SHIELD]", "🟢": "[GREEN]", "⏳": "[WAIT]"
    }
    for emoji, label in replacements.items():
        text = text.replace(emoji, label)
    return text

def clean_sequence(input_val):
    """只保留字母，移除數字、逗號、引號、空格"""
    s = str(input_val)
    if "value" in s:
        match = re.search(r"['\"]value['\"]:\s*['\"]([^'\"]+)['\"]", s)
        if match: s = match.group(1)
    clean = re.sub(r'[^a-zA-Z]', '', s).upper()
    return clean

# ==========================================
# 🎨 本機繪圖工具區塊
# ==========================================

def generate_local_plot(sequence: str, file_prefix: str = "plot") -> str:
    """
    🟢【新增】通用本機繪圖：簡單長條圖呈現氨基酸組成
    這是 full_auto_analysis_tool 與 protein_physicochemical_plot_tool 在呼叫的函式。
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        counter = Counter(sequence)
        aas = sorted(counter.keys())
        counts = [counter[a] for a in aas]

        plt.figure(figsize=(8, 4))
        plt.bar(aas, counts, color='steelblue', edgecolor='black')
        plt.title(f"Amino Acid Composition - {sequence[:15]}...")
        plt.xlabel("Amino Acid")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        img_filename = f"local_plot_{file_prefix}_{datetime.now().strftime('%H%M%S')}.png"
        img_path = os.path.join(current_dir, img_filename)
        plt.savefig(img_path, dpi=150)
        plt.close()

        print(f"🎨 [系統] 本機分析圖已生成: {img_path}")
        return img_path
    except Exception as e:
        print(f"❌ 本機繪圖失敗: {e}")
        return f"繪圖失敗:{e}"


def _download_pdb_to_local(sequence: str, remote_pdb_path: str, file_prefix: str) -> str:
    """🟢【共用工具】下載/讀取 PDB 並存到本機"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdb_data = ""

    if os.path.exists(remote_pdb_path):
        try:
            with open(remote_pdb_path, 'r') as f:
                pdb_data = f.read()
        except Exception:
            pass

    if not pdb_data:
        print("⚠️ [系統] 讀取遠端 PDB 失敗，改用 ESM API 獲取座標...")
        response = requests.post(
            "https://api.esmatlas.com/foldSequence/v1/pdb/",
            data=sequence, timeout=60
        )
        pdb_data = response.text

    local_pdb_path = os.path.join(current_dir, f"temp_{file_prefix}.pdb")
    with open(local_pdb_path, 'w') as f:
        f.write(pdb_data)
    return local_pdb_path


def generate_pymol_plots_triple(sequence: str, remote_pdb_path: str, file_prefix: str = "3D") -> list:
    """
    一口氣生成三種風格圖片：
    1. Ribbons (二級結構)
    2. All Atoms (原子交互作用)
    3. Hydrophobicity Surface (蛋白質表面特性)
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 🟢 修正：補上下載 PDB 的邏輯
    local_pdb_path = _download_pdb_to_local(sequence, remote_pdb_path, file_prefix)

    results = [
        {
            "title": "Interactive 1 (Ribbons) - 二級結構緞帶模型",
            "path": os.path.join(current_dir, f"render_{file_prefix}_1_ribbon.png"),
            "desc": "說明：此圖以緞帶模型展示蛋白質的二級結構（螺旋與折疊）。顏色按殘基順序呈彩虹漸層，有助於觀察整體的空間摺疊構型。"
        },
        {
            "title": "Interactive 2 (All Atoms) - 全原子棒狀模型",
            "path": os.path.join(current_dir, f"render_{file_prefix}_2_atoms.png"),
            "desc": "說明：此圖展示所有原子的棒狀模型。碳為灰色、氮為藍色、氧為紅色，適合觀察特定氨基酸側鏈的化學環境與交互作用。"
        },
        {
            "title": "Interactive 3 (Hydrophobicity Surface) - 親疏水表面模型",
            "path": os.path.join(current_dir, f"render_{file_prefix}_3_surface.png"),
            "desc": "說明：蛋白質表面根據親疏水性著色。紅色區域代表疏水性（Hydrophobic），藍色區域代表親水性（Hydrophilic），用於分析潛在的結合口袋。"
        }
    ]

    script_path = os.path.join(current_dir, f"render_triple.pml")
    with open(script_path, 'w') as f:
        f.write(f"load {local_pdb_path}, myprot\n")
        f.write("bg_color white\n")

        # Ribbon
        f.write("hide all\nshow cartoon, myprot\ncolor spectrum, myprot\n")
        f.write(f"png {results[0]['path']}, width=800, height=600, ray=1\n")

        # Sticks
        f.write("hide all\nshow sticks, myprot\ncolor cpk, myprot\n")
        f.write(f"png {results[1]['path']}, width=800, height=600, ray=1\n")

        # Surface
        f.write("hide all\nshow surface, myprot\n")
        f.write("set_color hydrophob, [1.0, 0.5, 0.5]\n")
        f.write("color hydrophob, resn ALA+VAL+LEU+ILE+PHE+TRP+MET+PRO\n")
        f.write("color slate, resn ASN+GLN+SER+THR+TYR+CYS\n")
        f.write("color marine, resn ARG+LYS+HIS+ASP+GLU\n")
        f.write(f"png {results[2]['path']}, width=800, height=600, ray=1\n")
        f.write("quit\n")

    pymol_cmd = ["xvfb-run", "-a", "python", "-m", "pymol", "-c", "-q", script_path]
    subprocess.run(pymol_cmd, check=True, capture_output=True)
    return results


def generate_pymol_plot(sequence: str, remote_pdb_path: str, file_prefix: str = "3D") -> str:
    """在本機端呼叫 PyMOL 生成專業黑底棒狀模型圖 (Stick Model)。"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_pdb_path = _download_pdb_to_local(sequence, remote_pdb_path, file_prefix)

    img_filename = f"pymol_render_{file_prefix}_{datetime.now().strftime('%H%M%S')}.png"
    img_path = os.path.join(current_dir, img_filename)
    script_path = os.path.join(current_dir, f"render_{file_prefix}.pml")

    with open(script_path, 'w') as f:
        f.write(f"load {local_pdb_path}, myprotein\n")
        f.write("hide all\n")
        f.write("show sticks, myprotein\n")
        f.write("set stick_radius, 0.25\n")
        f.write("set stick_ball, on\n")
        f.write("set stick_ball_ratio, 1.5\n")
        f.write("color cpk, myprotein\n")
        f.write("color gray70, elem c\n")
        f.write("bg_color black\n")
        f.write("set ray_shadows, 0\n")
        f.write("util.cbc myprotein\n")
        f.write("zoom\n")
        f.write(f"ray 800, 600\n")
        f.write(f"png {img_path}\n")
        f.write("quit\n")

    pymol_cmd = ["xvfb-run", "-a", "python", "-m", "pymol", "-c", "-q", script_path]
    try:
        print(f"⏳ [系統] 正在使用 PyMOL 渲染黑底棒狀模型...")
        subprocess.run(pymol_cmd, check=True, capture_output=True, text=True)
        if os.path.exists(script_path): os.remove(script_path)
        if os.path.exists(local_pdb_path): os.remove(local_pdb_path)
        print(f"✅ 圖片已生成: {img_path}")
        return img_path
    except subprocess.CalledProcessError as e:
        error_info = e.stderr if e.stderr else e.stdout
        raise Exception(f"PyMOL 渲染失敗。詳細資訊:\n{error_info}")


def generate_chimera_plot(sequence: str, remote_pdb_path: str, file_prefix: str = "3D") -> str:
    """在 Ubuntu 本機端呼叫原生 Chimera 1.19 生成 3D 渲染圖"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_pdb_path = _download_pdb_to_local(sequence, remote_pdb_path, file_prefix)

    img_filename = f"chimera_render_{file_prefix}_{datetime.now().strftime('%H%M%S')}.png"
    img_path = os.path.join(current_dir, img_filename)
    cmd_path = os.path.join(current_dir, f"render_{file_prefix}.com")

    with open(cmd_path, 'w') as f:
        f.write(f"open {local_pdb_path}\n")
        f.write("preset apply pub 1\n")
        f.write("color rainbow chain\n")
        f.write(f"copy file {img_path} width 800 height 600 supersample 3\n")
        f.write("stop\n")

    chimera_cmd = ["xvfb-run", "-a", "chimera", "--nogui", "--silent", cmd_path]
    try:
        print("⏳ [系統] 正在使用 Linux 原生 Chimera 渲染中...")
        subprocess.run(chimera_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if os.path.exists(cmd_path): os.remove(cmd_path)
        if os.path.exists(local_pdb_path): os.remove(local_pdb_path)
        return img_path
    except Exception as e:
        raise Exception(f"Chimera 執行失敗: {str(e)}")


def generate_local_3d_plot(sequence: str, remote_pdb_path: str, file_prefix: str = "3D") -> str:
    """在本機端生成蛋白質 3D 骨架(Backbone)預覽圖"""
    pdb_data = ""
    if os.path.exists(remote_pdb_path):
        try:
            with open(remote_pdb_path, 'r') as f:
                pdb_data = f.read()
        except Exception:
            pass

    if not pdb_data:
        print("⚠️ [系統] 無法讀取遠端 PDB，改為透過 API 獲取 3D 座標...")
        response = requests.post("https://api.esmatlas.com/foldSequence/v1/pdb/", data=sequence, timeout=30)
        pdb_data = response.text

    x, y, z = [], [], []
    for line in pdb_data.split('\n'):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                x.append(float(line[30:38]))
                y.append(float(line[38:46]))
                z.append(float(line[46:54]))
            except ValueError:
                pass

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, color='mediumpurple', linewidth=2.5)
    ax.scatter(x, y, z, color='darkslateblue', s=15)
    ax.set_title(f"Protein 3D Backbone Trace", fontsize=14)
    ax.axis('off')

    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_filename = f"local_3d_plot_{file_prefix}_{datetime.now().strftime('%H%M%S')}.png"
    img_path = os.path.join(current_dir, img_filename)
    plt.tight_layout()
    plt.savefig(img_path, dpi=150)
    plt.close()
    print(f"🎨 [系統] 本機 3D 預覽圖已生成: {img_path}")
    return img_path


# ==========================================
# 📄 PDF 報告工具
# ==========================================
@tool
def generate_pdf_report_tool(query: str = "生成報告") -> str:
    """彙整本次對話中所有工具的執行紀錄，生成最終 PDF 報告（支援自動插入圖片與防崩潰排版）。"""
    try:
        pdf = FPDF()
        pdf.add_page()
        font_path = "/home/tempadmin/agentic_ai/msjh.ttc"

        has_unicode_font = False
        if os.path.exists(font_path):
            pdf.add_font("msjh", style="", fname=font_path)
            pdf.set_font("msjh", size=16)
            has_unicode_font = True
        else:
            pdf.set_font("helvetica", size=16)

        pdf.cell(200, 10, text="Protein Research Comprehensive Report",
                 new_x="LMARGIN", new_y="NEXT", align='C')

        if has_unicode_font:
            pdf.set_font("msjh", size=10)
        else:
            pdf.set_font("helvetica", size=10)
        pdf.ln(10)

        if not GLOBAL_LOGS:
            pdf.multi_cell(0, 8, text="No logs recorded in this session.")
        else:
            raw_content = "\n".join(GLOBAL_LOGS)
            safe_content = sanitize_pdf_text(raw_content)
            if not has_unicode_font:
                safe_content = safe_content.encode('latin-1', 'replace').decode('latin-1')

            lines = safe_content.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    pdf.ln(2)
                    continue
                line = re.sub(r'-{15,}', '--------------------', line)
                pdf.set_x(pdf.l_margin)
                try:
                    pdf.multi_cell(0, 8, text=line)
                except Exception as cell_e:
                    pdf.set_x(pdf.l_margin)
                    pdf.multi_cell(0, 8, text=line[:60] + " ... (文字過長截斷保護)")
                    print(f"⚠️ [警告] 單行文字過長已截斷: {str(cell_e)}")

                # 🟢 改進：抓相對 + 絕對路徑都行
                png_match = re.search(r'([\w\./_-]+\.png)', line)
                if png_match:
                    img_path = png_match.group(1)
                    if os.path.exists(img_path):
                        pdf.ln(2)
                        try:
                            pdf.image(img_path, w=150)
                            pdf.ln(5)
                            print(f"🖼️ [系統] 成功插入圖片: {img_path}")
                        except Exception as img_e:
                            pdf.multi_cell(0, 8, text=f"[圖片載入失敗: {str(img_e)}]")

        report_name = f"Full_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(report_name)
        return f"✅ 完整報告已生成（包含分析圖表）：{report_name}"
    except Exception as e:
        return f"❌ PDF 生成失敗: {str(e)}"


# ==========================================
# 🤖 LLM 設定
# ==========================================
llm = ChatOllama(
    model="gemma2:9b",
    temperature=0,
    base_url=f"{BASE_TUNNEL_URL}/ollama"
)


# ==========================================
# 🛠️ 各項 AI 工具
# ==========================================

@tool
def full_auto_analysis_tool(query: str) -> str:
    """一鍵執行全自動化分析。輸入格式為 '序列,數量'。"""
    parts = str(query).split(',')
    target_seq = clean_sequence(parts[0])
    num_variants = int(re.search(r'\d+', parts[1]).group()) if len(parts) > 1 else 10
    url = f"{BASE_TUNNEL_URL}/auto_research_pipeline"
    try:
        response = requests.post(url, json={"sequence": target_seq, "num_variants": num_variants}, timeout=600)
        data = response.json()

        if data["status"] == "success":
            best_seq = data['best_variant']
            # 🟢 修正：呼叫已存在的 generate_local_plot
            local_img_path = generate_local_plot(best_seq, "BestVariant")

            res_summary = (f"原始序列: {data['original_seq']}\n"
                           f"最優變體: {best_seq}\n"
                           f"MW: {data['physicochemical']['molecular_weight']}\n"
                           f"pI: {data['physicochemical']['isoelectric_point']}\n"
                           f"本機繪圖路徑: {local_img_path}\n"
                           f"PDB路徑: {data['pdb_path']}")
            add_log("Full Auto Analysis", res_summary)
            return f"[OK] 全自動分析完成！\n{res_summary}"
        return f"[ERROR] 分析失敗：{data.get('message')}"
    except Exception as e:
        return f"異常: {str(e)}"


@tool
def neuropeptide_predictor_tool(sequence: str) -> str:
    """用於分析單一蛋白質序列是否為神經肽。"""
    clean_seq = clean_sequence(sequence)
    url = f"{BASE_TUNNEL_URL}/predict"
    try:
        print(f"📡 正在傳送序列 '{clean_seq}' 至 Server 進行單次預測...")
        response = requests.post(url, json={"sequence": clean_seq}, timeout=120)
        data = response.json()
        if data.get("status") == "error":
            return f"伺服器報錯：{data.get('message')}"

        res = data.get("prediction", {})
        # 🟢 改進：同時顯示原始 score 與百分比 probability
        result_text = (
            f"序列: {clean_seq}\n"
            f"預測結果: 【{res.get('label')}】\n"
            f"預測機率: 【{res.get('probability', '0.0')}】\n"
            # f"預測機率: 【{res.get('probability', 'N/A')}】\n"
            f"資料來源: 【{res.get('note')}】"
        )
        add_log("Neuropeptide Prediction", result_text)
        return f"預測完畢！\n{result_text}"
    except Exception as e:
        return f"❌ 工具呼叫失敗: {str(e)}"


@tool
def ankh_protein_language_tool(sequence: str) -> str:
    """當使用者詢問蛋白質的深層特徵、Embedding 或語言模型分析時使用。"""
    clean_seq = clean_sequence(sequence)
    # 🟢 修正：路徑必須與 Server 端 /ankh 一致
    url = f"{BASE_TUNNEL_URL}/ankh"
    try:
        print(f"📡 正在請求 Ankh 引擎分析序列 '{clean_seq}'...")
        response = requests.post(url, json={"sequence": clean_seq}, timeout=3000)
        data = response.json()
        if data.get("status") == "success":
            result_text = f"序列: {clean_seq}\nAnkh 特徵摘要：{data.get('result', '無摘要')}"
            add_log("Ankh Protein Language Tool", result_text)
            return f"✅ Ankh 分析完成！\n{result_text}"
        error_msg = data.get('message', '未知錯誤')
        add_log("Ankh Protein Language Tool (Error)", f"序列: {clean_seq}\n錯誤原因: {error_msg}")
        return f"❌ Ankh 工具報錯：{error_msg}"
    except Exception as e:
        return f"❌ 呼叫 Ankh 失敗: {str(e)}"


@tool
def protein_discovery_pipeline_tool(query: str) -> str:
    """執行蛋白質發現與進化管線。傳入格式：序列,數量 (例如 GPRLVRF,10)"""
    clean_query = str(query).replace("'", "").replace('"', '').replace('{', '').replace('}', '').strip()
    parts = clean_query.split(',')
    clean_seq = clean_sequence(parts[0])

    if not clean_seq or len(clean_seq) < 2:
        return "❌ 錯誤：請提供正確的蛋白質序列。"

    num_variants = 100
    if len(parts) > 1:
        try:
            num_match = re.search(r'\d+', parts[1])
            if num_match: num_variants = int(num_match.group())
        except Exception:
            num_variants = 100

    url = f"{BASE_TUNNEL_URL}/discovery_pipeline"
    try:
        print(f"🧬 啟動管線：針對序列 '{clean_seq}' 生成 {num_variants} 條變體並進行預測...")
        response = requests.post(url, json={"sequence": clean_seq, "num_variants": num_variants}, timeout=300)
        data = response.json()

        if data.get("status") == "success":
            results = data.get("top_results", [])
            if not results:
                result_text = "✅ 管線執行完畢，但未能找到任何候選序列。"
                add_log("Protein Discovery Pipeline", result_text)
                return result_text
            res_text = (f"✅ 管線執行成功！\n"
                        f"- 生成變體要求: {data['total_generated']} 條\n"
                        f"- 去冗餘後剩餘: {data['after_cdhit']} 條\n\n"
                        f"前 {len(results)} 名高分候選序列：\n")
            for i, r in enumerate(results, 1):
                res_text += f"{i}. 序列: {r['seq']} | 分數: {r.get('score', r.get('probability'))} | 標籤: {r['label']}\n"
            add_log("Protein Discovery Pipeline", res_text)
            return res_text
        return f"❌ 伺服器回傳錯誤: {data.get('message')}"
    except Exception as e:
        return f"❌ 連線失敗: {str(e)}"


@tool
def secondary_structure_tool(sequence: str) -> str:
    """分析二級結構 (Helix/Sheet/Coil)。"""
    clean_seq = clean_sequence(sequence)
    url = f"{BASE_TUNNEL_URL}/secondary_structure"
    try:
        response = requests.post(url, json={"sequence": clean_seq}, timeout=60)
        data = response.json()
        if data.get("status") == "success":
            result_text = (f"二級結構分析完成！\n序列: {data['sequence']}\n"
                           f"結構: {data['structure']}\n統計: {data['summary']}")
            add_log("Secondary Structure Analysis", result_text)
            return result_text
        error_msg = f"伺服器報錯：{data.get('message')}"
        add_log("Secondary Structure Analysis (Error)", error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"❌ 工具呼叫失敗: {str(e)}"
        add_log("Secondary Structure Analysis (Exception)", error_msg)
        return error_msg


@tool
def esmfold_3d_structure_tool(sequence: str) -> str:
    """預測立體結構，並使用 PyMOL 生成高畫質 3D 棒狀模型圖。"""
    clean_seq = clean_sequence(sequence)
    url = f"{BASE_TUNNEL_URL}/esmfold"
    try:
        response = requests.post(url, json={"sequence": clean_seq}, timeout=120)
        data = response.json()
        if data.get("status") == "success":
            remote_pdb_path = data['file_path']
            try:
                local_img_path = generate_pymol_plot(clean_seq, remote_pdb_path, "3DStruct")
                img_log_text = f"- 本機 3D 繪圖路徑: {local_img_path}"
            except Exception as draw_e:
                # 🟢 改進：PyMOL 失敗時退回 matplotlib 骨架圖，保證圖一定有
                print(f"⚠️ PyMOL 失敗，改用 matplotlib 骨架圖：{draw_e}")
                try:
                    fallback_path = generate_local_3d_plot(clean_seq, remote_pdb_path, "3DStruct")
                    img_log_text = f"- 本機 3D 繪圖路徑(備援): {fallback_path}"
                except Exception as e2:
                    img_log_text = f"- 本機 3D 繪圖失敗: {str(e2)}"

            result_text = (f"✅ 3D 結構預測大成功！\n"
                           f"- 遠端 PDB 檔案儲存於：【{remote_pdb_path}】\n"
                           f"{img_log_text}")
            add_log("ESMFold 3D Structure Tool", result_text)
            return result_text
        error_msg = f"伺服器報錯：{data.get('message')}"
        add_log("ESMFold 3D Structure Tool (Error)", error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"❌ 工具呼叫失敗: {str(e)}"
        add_log("ESMFold 3D Structure Tool (Exception)", error_msg)
        return error_msg


@tool
def protein_physicochemical_plot_tool(sequence: str) -> str:
    """疏水性、等電點與分子量分析。"""
    clean_seq = clean_sequence(sequence)
    url = f"{BASE_TUNNEL_URL}/analyze_physicochemical"
    try:
        response = requests.post(url, json={"sequence": clean_seq}, timeout=30)
        data = response.json()
        if data["status"] == "success":
            # 🟢 修正：呼叫已存在的 generate_local_plot
            local_img_path = generate_local_plot(clean_seq, "Physico")
            result_text = (f"📊 理化性質分析完成！\n"
                           f"- 分子量 (MW): {data['molecular_weight']}\n"
                           f"- 等電點 (pI): {data['isoelectric_point']}\n"
                           f"- 遠端原圖路徑: {data['image_path']}\n"
                           f"- 本機繪圖路徑: {local_img_path}")
            add_log("Protein Physicochemical Plot Tool", result_text)
            return result_text
        error_msg = f"錯誤：{data['message']}"
        add_log("Protein Physicochemical Plot Tool (Error)", error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"呼叫工具失敗: {str(e)}"
        add_log("Protein Physicochemical Plot Tool (Exception)", error_msg)
        return error_msg


# ==========================================
# 🤖 Agent 初始化
# ==========================================
tools = [
    full_auto_analysis_tool,
    neuropeptide_predictor_tool,
    ankh_protein_language_tool,
    protein_discovery_pipeline_tool,
    secondary_structure_tool,
    esmfold_3d_structure_tool,
    protein_physicochemical_plot_tool,
    generate_pdf_report_tool
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=2,
    handle_parsing_errors=True,
    agent_kwargs={
        "prefix": """你是一個自動化生物研究機器人。
        你的作業流程如下：
        1. 接收序列後，優先使用 full_auto_analysis_tool 進行全面分析；若使用者要求特定分析，則使用對應工具。
        2. 你所執行的「每一個工具」都會自動將詳細過程與數據存入後台日誌系統（GLOBAL_LOGS）。
        3. 分析完成後，你必須彙整得到的數據（如：序列、MW、pI、檔案路徑、預測結果等）。
        4. 最後，你必須主動呼叫 generate_pdf_report_tool。該工具會自動提取後台日誌，將你剛才「所有執行過的步驟」輸出成一份完整的 PDF 報告給使用者。
        請確保報告內容詳盡、包含所有工具的發現，且易於閱讀。"""
    }
)

if __name__ == "__main__":
    print("\n🚀 Agent 指揮中心啟動！")
    while True:
        user_input = input("\n請輸入指令 (exit 離開): ")
        if user_input.lower() in ['exit', 'quit']: break
        try:
            agent.invoke({"input": user_input})
            pdf_result = generate_pdf_report_tool.invoke("")
            print(f"\n{pdf_result}")
        except Exception as e:
            print(f"\n❌ 錯誤: {e}")
