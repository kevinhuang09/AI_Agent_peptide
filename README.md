# 🧬 Protein Research Agent — 自動化蛋白質研究機器人

以 **LangChain Agent + 本地 Ollama 多模型** 打造的自動化蛋白質分析系統。輸入一條蛋白質序列，系統會自動完成：神經肽預測、理化性質分析、二級結構分析、3D 結構預測與渲染、變體生成與優選，最後由 **三個 LLM 集成解讀 + 裁判整合**，並輸出一份完整的 **中文 PDF 研究報告**。

---

## ✨ 功能特色

| 功能 | 說明 |
|------|------|
| 🤖 **ReAct Agent 指揮官** | 以 `llama3.1:8b` 作為指揮官，自動判斷該呼叫哪個分析工具 |
| 🔬 **一鍵全自動分析** | `full_auto_analysis_tool`：序列 → 變體生成 → 優選 → 理化性質一條龍 |
| 🧠 **神經肽預測** | 判斷序列是否為神經肽，輸出標籤與機率（如 `0.7813`） |
| 🧬 **蛋白質發現管線** | 生成大量變體、CD-HIT 去冗餘、預測評分、輸出 Top-N 候選 |
| 📐 **二級結構分析** | Helix / Sheet / Coil 結構與統計 |
| 🎨 **3D 結構渲染** | ESMFold 預測 PDB，並用 **PyMOL / Chimera** 產生高畫質 3D 圖（含 matplotlib 骨架圖備援） |
| 📊 **理化性質分析** | 分子量 (MW)、等電點 (pI)、氨基酸組成長條圖 |
| 🌐 **Ankh 語言模型** | 蛋白質深層特徵 / Embedding 分析 |
| 🤝 **三模型集成解讀** | Gemma2 / Llama3.1 / Qwen2.5 各自解讀，Qwen2.5 裁判整合共識 |
| 📄 **中文 PDF 報告** | 彙整所有步驟日誌，自動插入分析圖表，輸出繁體中文報告 |

---

## 🏗️ 系統架構

```
使用者輸入序列
      │
      ▼
┌─────────────────────────────┐
│  LangChain ReAct Agent       │  ← 指揮官：llama3.1:8b
│  (自動選擇並呼叫工具)         │
└─────────────────────────────┘
      │  呼叫工具（打到 FastAPI 後端 / ngrok 隧道）
      ▼
┌─────────────────────────────┐
│  後端運算 (透過 BASE_TUNNEL) │
│  Ankh / LightGBM / ESMFold   │
│  理化性質 / 二級結構 / 管線   │
└─────────────────────────────┘
      │  每個工具自動寫入 GLOBAL_LOGS
      ▼
┌─────────────────────────────┐
│  generate_pdf_report_tool    │
│  1. 三模型集成解讀            │
│     Gemma2 / Llama3.1 /      │
│     Qwen2.5 → Qwen2.5 裁判   │
│  2. 本機 PyMOL/Chimera 渲染   │
│  3. 輸出中文 PDF 報告         │
└─────────────────────────────┘
      │
      ▼
   📄 Full_Report_YYYYMMDD_HHMMSS.pdf
```

---

## 📦 環境需求

### Python 套件

```bash
pip install requests matplotlib fpdf2 langchain langchain-ollama
# 選配：簡轉繁保底
pip install opencc-python-reimplemented
```

### 系統工具（3D 渲染用）

| 工具 | 用途 | 安裝 |
|------|------|------|
| **Ollama** | 本地 LLM 推論引擎 | https://ollama.com |
| **PyMOL** | 3D 蛋白質棒狀/表面模型渲染 | `conda install -c conda-forge pymol-open-source` |
| **Chimera 1.19** | 3D 發表級渲染（選配） | UCSF Chimera 官網 |
| **Xvfb** | 無頭伺服器虛擬顯示（PyMOL/Chimera 需要） | `sudo apt install xvfb` |

### 中文字型

PDF 報告需要中文字型檔（微軟正黑體）：

```python
font_path = "/home/tempadmin/agentic_ai/msjh.ttc"
```

> ⚠️ 請將 `msjh.ttc` 放到此路徑，或修改程式中的 `font_path`。找不到字型時會退回 `helvetica`（無法顯示中文）。

---

## 🚀 快速開始

### 1. 準備 Ollama 模型

```bash
ollama pull llama3.1:8b     # 指揮官 + 集成成員
ollama pull gemma2:9b       # 集成成員
ollama pull qwen2.5:7b      # 集成成員 + 裁判
```

### 2. 設定隧道位址

修改程式最上方的隧道 URL（指向你的 FastAPI 後端）：

```python
BASE_TUNNEL_URL = "https://pellicular-becki-nastily.ngrok-free.dev"
```

### 3. 啟動系統（分兩端）

本系統為 **前後端分離架構**：遠端主機跑運算 Server，本地端跑 Agent 指揮官，兩者透過 ngrok 隧道連接。**請務必先啟動遠端 Server，再啟動本地 Agent。**

#### 🖥️ Step A — 遠端主機：啟動運算 Server

負責 Ankh / LightGBM / ESMFold / Ollama 等重運算。

```bash
# 進入遠端專案資料夾
cd ai_agent0520

# 啟動後端運算 Server
python server.py
```

> 啟動後 Server 會監聽本機端口（如 `127.0.0.1:8000`），並由 ngrok 對外暴露成 `BASE_TUNNEL_URL`。請確認 Ollama 服務（`ollama serve`）也已在遠端運行。

#### 💻 Step B — 本地端：啟動 Agent 指揮官

負責理解指令、調度工具、生成 PDF 報告。

```bash
# 進入本地專案資料夾
cd NeuroP

# 啟動 Agent
python agent_ensemble.py
```

> ⚠️ 啟動前請確認 `agent_ensemble.py` 內的 `BASE_TUNNEL_URL` 已指向 Step A 遠端 Server 的 ngrok 網址（見上方步驟 2）。

| 端 | 資料夾 | 啟動指令 | 角色 |
|----|--------|----------|------|
| 🖥️ 遠端 | `ai_agent0520` | `python server.py` | 後端運算（模型推論） |
| 💻 本地 | `NeuroP` | `python agent_ensemble.py` | Agent 指揮官（LangChain） |

### 4. 互動範例

```
🚀 Agent 指揮中心啟動！

請輸入指令 (exit 離開): 分析序列 YGGFMTSEKSQTPLVT
```

系統會自動跑完分析流程，並在結束時生成 PDF 報告。輸入 `exit` 或 `quit` 離開。

---

## 🛠️ 工具清單

| 工具函式 | 對應後端路由 | 功能 |
|----------|-------------|------|
| `full_auto_analysis_tool` | `/auto_research_pipeline` | 一鍵全自動分析（序列,數量） |
| `neuropeptide_predictor_tool` | `/predict` | 神經肽預測 |
| `ankh_protein_language_tool` | `/ankh` | Ankh 蛋白質語言模型特徵 |
| `protein_discovery_pipeline_tool` | `/discovery_pipeline` | 變體生成與進化管線 |
| `secondary_structure_tool` | `/secondary_structure` | 二級結構分析 |
| `esmfold_3d_structure_tool` | `/esmfold` | ESMFold 3D 結構 + PyMOL 渲染 |
| `protein_physicochemical_plot_tool` | `/analyze_physicochemical` | 理化性質 + 組成圖 |
| `generate_pdf_report_tool` | — | 集成解讀 + 生成 PDF 報告 |

---

## 🤖 模型配置

```python
# 指揮官：ReAct 格式最穩定
llm = make_llm("llama3.1:8b", temp=0)

# 集成成員：三個模型各自解讀（temp=0.3 製造差異性）
ENSEMBLE_MODELS = {
    "gemma2":   make_llm("gemma2:9b",   temp=0.3),
    "llama3.1": make_llm("llama3.1:8b", temp=0.3),
    "qwen2.5":  make_llm("qwen2.5:7b",  temp=0.3),
}

# 裁判：整合三份解讀（中文總結用 qwen2.5 最佳）
JUDGE_MODEL = make_llm("qwen2.5:7b", temp=0)
```

---

## 🎨 3D 渲染模式

`esmfold_3d_structure_tool` 預設用 **PyMOL 黑底棒狀模型**，失敗時自動退回 matplotlib 骨架圖。另有以下可用函式：

- `generate_pymol_plot()` — 黑底棒狀模型 (Stick Model)
- `generate_pymol_plots_triple()` — 一次生成三種風格：緞帶模型 / 全原子模型 / 親疏水表面模型
- `generate_chimera_plot()` — UCSF Chimera 彩虹渲染
- `generate_local_3d_plot()` — matplotlib 3D 骨架 (Backbone) 備援圖

---

## 📄 PDF 報告內容

生成的 `Full_Report_*.pdf` 包含：

1. **各工具執行紀錄**（序列、MW、pI、預測機率、檔案路徑等）
2. **自動插入的分析圖表**（氨基酸組成圖、3D 結構圖）
3. **三模型集成解讀**
   - 最終共識結論（裁判整合）
   - 各模型（Gemma2 / Llama3.1 / Qwen2.5）原始意見

> 💡 報告使用微軟正黑體渲染繁體中文；Emoji 會自動轉為文字標籤（如 `✅ → [OK]`）避免渲染錯誤。

---

## ⚙️ 效能與資源建議

| 項目 | 建議 |
|------|------|
| **VRAM** | 三個 9B 模型同時常駐約需 20~24GB。不足時設定 `OLLAMA_MAX_LOADED_MODELS=1`（一次載一個，慢但省記憶體） |
| **量化** | 建議使用 `q4_K_M` 量化版模型加速推論 |
| **推論次數** | 集成需多跑 3+1=4 次 LLM，速度會變慢屬正常 |
| **保活** | 可設 `OLLAMA_KEEP_ALIVE` 避免模型頻繁卸載重載 |

---

## 🐛 常見問題

**Q：PDF 出現簡體字或字距爆開？**
A：在 `ensemble_interpret` 的 prompt 加入「嚴禁簡體中文」約束，並在 PDF 寫入時清除 Markdown 符號、`multi_cell` 使用 `align='L'`。必要時加 OpenCC 簡轉繁保底。

**Q：找不到中文，全部變亂碼？**
A：確認 `msjh.ttc` 字型檔路徑正確存在。

**Q：PyMOL / Chimera 渲染失敗？**
A：確認已安裝 `xvfb`，並用 `xvfb-run -a` 前綴執行（程式已內建）。PyMOL 失敗會自動退回 matplotlib 骨架圖。

**Q：Agent 沒自己生成報告？**
A：`__main__` 已在 Agent 執行後**手動**呼叫 `generate_pdf_report_tool`，所以報告一定會生成，不受 `max_iterations` 限制。

---

## 📁 專案結構（建議）

```
專案採前後端分離，分別部署於本地端與遠端主機：

# 💻 本地端（Agent 指揮官）
NeuroP/
├── agent_ensemble.py     # 主程式：LangChain Agent + 三模型集成
├── msjh.ttc              # 中文字型（PDF 報告用）
├── README.md             # 本說明文件
└── outputs/
    ├── Full_Report_*.pdf # 生成的報告
    ├── local_plot_*.png  # 分析圖表
    └── pymol_render_*.png# 3D 渲染圖

# 🖥️ 遠端主機（運算後端）
ai_agent0520/
└── server.py             # FastAPI 運算 Server（Ankh / LightGBM / ESMFold / Ollama proxy）
```

---

## 📜 授權

本專案僅供學術研究與教學使用。使用之第三方模型（Gemma2 / Llama3.1 / Qwen2.5）與工具（PyMOL / Chimera / ESMFold）請遵守各自授權條款。