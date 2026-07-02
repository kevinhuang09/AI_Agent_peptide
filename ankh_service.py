# ankh_service.py
import torch
import random
import torch.nn.functional as F
from transformers import AutoTokenizer, T5ForConditionalGeneration

model_name = "ElnaggarLab/ankh-base"

print(f"🚀 正在載入 Ankh 模型至 GPU...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"✅ Ankh 已載入到 {device}")

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# 預先計算 token id（加速）
AA_TOKEN_IDS = []
AA_LIST = []
for aa in AMINO_ACIDS:
    tid = tokenizer.convert_tokens_to_ids(aa)
    if tid != tokenizer.unk_token_id:
        AA_TOKEN_IDS.append(tid)
        AA_LIST.append(aa)
AA_TOKEN_IDS_TENSOR = torch.tensor(AA_TOKEN_IDS, device=device)

# 化學性質分組（forbid_similar 用）
SIMILAR_GROUPS = {
    'hydrophobic': set("AVLIMF"),
    'aromatic':    set("FWY"),
    'polar':       set("STNQ"),
    'positive':    set("KRH"),
    'negative':    set("DE"),
    'special':     set("CGP"),
}


# ==========================================
# 📊 特徵萃取
# ==========================================
def get_ankh_embedding(sequence):
    """回傳 768 維蛋白質向量（mean pooling）"""
    seq_with_space = " ".join(list(sequence))
    inputs = tokenizer(seq_with_space, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.encoder(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy().tolist()[0]


# ==========================================
# 🧬 單點預測（核心）
# ==========================================
def predict_amino_acid(seq_list, position, temperature=1.5, top_k=12,
                       forbid_original=True, forbid_similar=False):
    """
    用 Ankh 對指定位點做 masked language modeling 預測
    
    Args:
        temperature: 溫度（越高越隨機）
        top_k: 候選池大小
        forbid_original: 強制不選回原胺基酸
        forbid_similar: 連化學性質類似的也不選（極致多樣性）
    """
    masked_seq = seq_list.copy()
    original_aa = masked_seq[position]
    masked_seq[position] = "<extra_id_0>"

    input_text = " ".join(masked_seq)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]]).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=decoder_input_ids
        )

    logits = outputs.logits[0, -1, :] / temperature
    aa_logits = logits[AA_TOKEN_IDS_TENSOR].clone()

    # 🚫 不選原胺基酸
    if forbid_original and original_aa in AA_LIST:
        aa_logits[AA_LIST.index(original_aa)] = float('-inf')

    # 🚫 不選相似性質
    if forbid_similar:
        for group in SIMILAR_GROUPS.values():
            if original_aa in group:
                for aa in group:
                    if aa in AA_LIST:
                        aa_logits[AA_LIST.index(aa)] = float('-inf')
                break

    # Top-K 採樣
    valid_count = (aa_logits != float('-inf')).sum().item()
    k = min(top_k, valid_count)
    if k <= 0:
        candidates = [a for a in AA_LIST if a != original_aa]
        return random.choice(candidates) if candidates else original_aa

    top_values, top_indices = torch.topk(aa_logits, k)
    top_probs = F.softmax(top_values, dim=-1)
    chosen = torch.multinomial(top_probs, num_samples=1).item()
    return AA_LIST[top_indices[chosen].item()]


# ==========================================
# 🌟 通用智能突變生成器（內部使用）
# ==========================================
def generate_smart_variants(
    sequence,
    num_variants=100,
    mutation_rate=0.65,
    min_mutations=4,
    max_mutations=None,
    temperature=1.5,
    top_k=12,
    diversity_mode=None,
    forbid_similar=False,
    fixed_n_mutations=None,   # 🆕 若指定，每條剛好突變這麼多位點
):
    """
    🌟 高多樣性 Ankh 智能突變主函數
    
    diversity_mode 預設組合（會覆蓋上面的參數）：
        - "low":     mutation_rate=0.30, T=1.0, top_k=5
        - "medium":  mutation_rate=0.50, T=1.3, top_k=10
        - "high":    mutation_rate=0.65, T=1.5, top_k=12  ⭐
        - "extreme": mutation_rate=0.80, T=2.0, top_k=20, forbid_similar=True
    
    fixed_n_mutations: 若給定整數，會忽略 mutation_rate，每條剛好突變這麼多位點
    """
    presets = {
        "low":     dict(mutation_rate=0.30, temperature=1.0, top_k=5,  forbid_similar=False),
        "medium":  dict(mutation_rate=0.50, temperature=1.3, top_k=10, forbid_similar=False),
        "high":    dict(mutation_rate=0.65, temperature=1.5, top_k=12, forbid_similar=False),
        "extreme": dict(mutation_rate=0.80, temperature=2.0, top_k=20, forbid_similar=True),
    }
    if diversity_mode in presets:
        cfg = presets[diversity_mode]
        mutation_rate = cfg["mutation_rate"]
        temperature   = cfg["temperature"]
        top_k         = cfg["top_k"]
        forbid_similar = cfg["forbid_similar"]

    seq_len = len(sequence)

    # 決定每條要突變幾個位點
    if fixed_n_mutations is not None:
        target_mutations = max(1, min(int(fixed_n_mutations), seq_len))
    else:
        target_mutations = max(min_mutations, int(round(seq_len * mutation_rate)))
        if max_mutations is not None:
            target_mutations = min(target_mutations, max_mutations)
        target_mutations = min(target_mutations, seq_len)

    print(f"🧬 [Ankh 智能突變{f' / {diversity_mode}' if diversity_mode else ''}]")
    print(f"   目標：{num_variants} 條 × 每條突變 {target_mutations}/{seq_len} 位點 "
          f"({target_mutations/seq_len*100:.0f}%)")
    print(f"   參數：T={temperature}, top_k={top_k}, forbid_similar={forbid_similar}")

    variants = set()
    max_attempts = num_variants * 5
    attempts = 0

    while len(variants) < num_variants and attempts < max_attempts:
        attempts += 1
        seq_list = list(sequence)

        # 突變數加一點隨機性
        if fixed_n_mutations is not None:
            n_mut = target_mutations
        else:
            n_mut = target_mutations + random.randint(-1, 1)
            n_mut = max(min_mutations, min(n_mut, seq_len))

        positions = random.sample(range(seq_len), n_mut)
        for pos in positions:
            new_aa = predict_amino_acid(
                seq_list, pos,
                temperature=temperature,
                top_k=top_k,
                forbid_original=True,
                forbid_similar=forbid_similar,
            )
            seq_list[pos] = new_aa

        new_seq = "".join(seq_list)
        if new_seq != sequence:
            variants.add(new_seq)

        if len(variants) > 0 and len(variants) % 25 == 0:
            print(f"   ⏳ 已生成 {len(variants)}/{num_variants} (嘗試 {attempts} 次)")

    variants = list(variants)
    if variants:
        avg_mut = sum(
            sum(1 for a, b in zip(sequence, v) if a != b) for v in variants
        ) / len(variants)
        print(f"✅ 完成！共 {len(variants)} 條，平均突變 {avg_mut:.1f}/{seq_len} 位點 "
              f"({avg_mut/seq_len*100:.0f}%)")

    return variants


# ==========================================
# 🔌 對外 API（給 server.py 呼叫）
# ==========================================
def generate_by_masking(
    sequence,
    num_mutations=None,
    num_variants=50,
    temperature=1.5,
    top_k=12,
    diversity_mode=None,
    forbid_similar=False,
    **kwargs,
):
    """
    🎯 Masking-based 智能突變（server.py 主要呼叫）
    
    Args:
        sequence: 原始序列
        num_mutations: 每條要突變的位點數（若指定，會固定為此值）
        num_variants: 要生成幾條
        temperature: 溫度（預設 1.5）
        top_k: top-k 採樣（預設 12）
        diversity_mode: "low"/"medium"/"high"/"extreme"（會覆蓋 T/top_k）
    """
    print(f"🧠 [Ankh] generate_by_masking → "
          f"n={num_variants}, mut={num_mutations}, T={temperature}, top_k={top_k}")

    return generate_smart_variants(
        sequence=sequence,
        num_variants=num_variants,
        temperature=temperature,
        top_k=top_k,
        diversity_mode=diversity_mode,
        forbid_similar=forbid_similar,
        fixed_n_mutations=num_mutations,
        min_mutations=num_mutations if num_mutations else 4,
    )


def generate_by_topk(
    sequence,
    num_mutations=None,
    num_variants=50,
    top_k=12,
    temperature=1.2,   # topk 模式溫度稍低，更聚焦
    **kwargs,
):
    """
    🎯 Top-K 聚焦突變（比 masking 更保守、更聚焦於高機率胺基酸）
    """
    print(f"🧠 [Ankh] generate_by_topk → "
          f"n={num_variants}, mut={num_mutations}, top_k={top_k}, T={temperature}")

    return generate_smart_variants(
        sequence=sequence,
        num_variants=num_variants,
        temperature=temperature,
        top_k=top_k,
        diversity_mode=None,
        forbid_similar=False,
        fixed_n_mutations=num_mutations,
        min_mutations=num_mutations if num_mutations else 4,
    )


# 額外別名（容錯）
def mask_and_generate(*args, **kwargs):
    return generate_by_masking(*args, **kwargs)


def ankh_mutate(*args, **kwargs):
    return generate_by_masking(*args, **kwargs)


# ==========================================
# 🧪 測試入口
# ==========================================
if __name__ == "__main__":
    test_seq = "EDRLLGV"   # 您的 7 殘基測試序列
    print(f"\n📌 原始序列：{test_seq} (長度 {len(test_seq)})\n")

    print("=" * 60)
    print("【測試 1】generate_by_masking（server.py 實際呼叫方式）")
    print("=" * 60)
    variants = generate_by_masking(
        test_seq,
        num_mutations=4,
        num_variants=10,
        temperature=1.5,
        top_k=12,
    )
    for i, v in enumerate(variants[:10], 1):
        diff = sum(1 for a, b in zip(test_seq, v) if a != b)
        print(f"  變體 {i}: {v}  (突變 {diff} 個位點)")

    print("\n" + "=" * 60)
    print("【測試 2】extreme 模式（極致多樣性）")
    print("=" * 60)
    variants2 = generate_by_masking(
        test_seq,
        num_variants=10,
        diversity_mode="extreme",
    )
    for i, v in enumerate(variants2[:10], 1):
        diff = sum(1 for a, b in zip(test_seq, v) if a != b)
        print(f"  變體 {i}: {v}  (突變 {diff} 個位點)")
