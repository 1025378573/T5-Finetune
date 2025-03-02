import json
import os
import logging
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, get_scheduler,T5ForConditionalGeneration,T5Tokenizer
from sacrebleu.metrics import BLEU

#############################################
# 模型下载与数据处理部分
#############################################
def download_model():
    # 使用 pipeline 测试模型输出
    # 加载 tokenizer 和模型
    tokenizer = T5Tokenizer.from_pretrained("Langboat/mengzi-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("Langboat/mengzi-t5-base")
    return model, tokenizer

class T5Dataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

def collate_fn(batch_samples, tokenizer):
    """
    拼接问题和上下文构造输入，确保答案以 </s> 结尾，
    并对 padding 部分用 -100 替换，方便计算损失时忽略。
    """
    batch_inputs = []
    batch_labels = []
    references = []
    for sample in batch_samples:
        # 拼接输入：问题与上下文
        batch_inputs.append("问题：" + sample['question'] + " 原文：" + sample['context'])
        answer = sample['answer'].strip()
        if not answer.endswith("</s>"):
            answer = answer + " </s>"
        batch_labels.append(answer)
        references.append(sample['answer'])
    
    inputs = tokenizer(batch_inputs, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(batch_labels, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    # 将填充 token 替换为 -100
    labels["input_ids"][labels["input_ids"] == tokenizer.pad_token_id] = -100
    
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels["input_ids"],
        "reference": references
    }

#############################################
# BLEU 评估函数
#############################################
# def compute_bleu_scores(candidates, references):
#     # 分别计算不同 n-gram 下的 BLEU 分数
#     bleu1 = BLEU(tokenize='zh', max_ngram_order=1).corpus_score(candidates, references).score
#     bleu2 = BLEU(tokenize='zh', max_ngram_order=2).corpus_score(candidates, references).score
#     bleu3 = BLEU(tokenize='zh', max_ngram_order=3).corpus_score(candidates, references).score
#     bleu4 = BLEU(tokenize='zh', max_ngram_order=4).corpus_score(candidates, references).score
#     return bleu1, bleu2, bleu3, bleu4
def compute_bleu_scores(candidates, references):

    # 初始化累加器
    total_bleu1, total_bleu2, total_bleu3, total_bleu4 = 0.0, 0.0, 0.0, 0.0

    num_samples = len(candidates)
    # 遍历每组候选文本和参考文本
    for idx in range(num_samples):
        # 计算 BLEU-1 到 BLEU-4 分数
        bleu1 = BLEU(tokenize='zh', max_ngram_order=1).corpus_score([candidates[idx]], [references[idx]]).score
        bleu2 = BLEU(tokenize='zh', max_ngram_order=2).corpus_score([candidates[idx]], [references[idx]]).score
        bleu3 = BLEU(tokenize='zh', max_ngram_order=3).corpus_score([candidates[idx]], [references[idx]]).score
        bleu4 = BLEU(tokenize='zh', max_ngram_order=4).corpus_score([candidates[idx]], [references[idx]]).score
        # 累加分数
        total_bleu1 += bleu1
        total_bleu2 += bleu2
        total_bleu3 += bleu3
        total_bleu4 += bleu4

    # 计算平均值
    avg_bleu1 = total_bleu1 / num_samples
    avg_bleu2 = total_bleu2 / num_samples
    avg_bleu3 = total_bleu3 / num_samples
    avg_bleu4 = total_bleu4 / num_samples


    return avg_bleu1, avg_bleu2, avg_bleu3, avg_bleu4

#############################################
# 预测函数：给定 query 和 context，生成答案
#############################################
def predict_answer(query, context, model, tokenizer, device):
    input_text = "问题：" + query + " 原文：" + context
    inputs = tokenizer(input_text, max_length=512, truncation=True, return_tensors="pt").to(device)
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")
    outputs = model.generate(**inputs)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

#############################################
# 训练和评估代码
#############################################
if __name__ == "__main__":
    # 日志配置
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="train.log",
        filemode="w"
    )
    logger = logging.getLogger(__name__)
    
    # 超参数设置
    batch_size = 6
    total_epochs = 20
    warmup_ratio = 0.06
    train_data_ratio = 0.8

    # 下载模型和 tokenizer
    model, tokenizer = download_model()
    
    # 设备选择：优先 CUDA，其次 MPS，最后 CPU
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Using {device} device')
    model.to(device)
    
    # 加载数据集
    train_data = T5Dataset("train.json")
    dev_data = T5Dataset("dev.json")
    
    # 划分训练集与验证集
    train_size = int(train_data_ratio * len(train_data))
    val_size = len(train_data) - train_size
    train_dataset, val_dataset = random_split(train_data, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              collate_fn=lambda batch: collate_fn(batch, tokenizer))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=lambda batch: collate_fn(batch, tokenizer))
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False, 
                            collate_fn=lambda batch: collate_fn(batch, tokenizer))
    
    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    num_update_steps_per_epoch = len(train_loader)
    max_train_steps = total_epochs * num_update_steps_per_epoch
    warmup_steps = int(warmup_ratio * max_train_steps)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )
    
    # 用于记录训练过程的指标
    epoch_list = []
    loss_list = []
    val_bleu1_list, val_bleu2_list, val_bleu3_list, val_bleu4_list = [], [], [], []
    dev_bleu1_list, dev_bleu2_list, dev_bleu3_list, dev_bleu4_list = [], [], [], []
    best_bleu2 = 0.0  # 以 BLEU-2 作为保存最佳模型的依据
    best_model_path = "./best_model.pth"
    
    model.train()
    for epoch in range(1, total_epochs + 1):
        print(f"##### Epoch {epoch} #####")
        logger.info(f"##### Epoch {epoch} #####")
        epoch_list.append(epoch)
        
        batch_losses = []
        # tqdm 循环中实时显示当前 batch 的 loss
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} (Train)", leave=False)
        for idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            current_loss = loss.item()
            batch_losses.append(current_loss)
            # 更新 tqdm 显示当前 loss
            progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})
            # 记录每个 batch 的 loss 到日志
            if idx % 5 == 0:
                logger.info(f"Epoch {epoch} - Batch loss: {current_loss:.4f}")
            
        avg_loss = np.mean(batch_losses)
        loss_list.append(avg_loss)
        logger.info(f"Epoch {epoch} - Average Train Loss: {avg_loss:.4f}")
        print(f"Epoch {epoch} - Average Train Loss: {avg_loss:.4f}")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"./model_epoch_{epoch}.pth")
        
        # 每 2 个 epoch 进行验证和开发集评估
        if epoch % 2 == 0:
            model.eval()
            # 验证集评估
            val_candidates = []
            val_references = []
            val_progress = tqdm(val_loader, desc="Validation", leave=False)
            with torch.no_grad():
                for i, batch in enumerate(val_progress, start=1):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
                    decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    val_candidates.extend(decoded_preds)
                    for ref in batch["reference"]:
                        val_references.append([ref])
            bleu1, bleu2, bleu3, bleu4 = compute_bleu_scores(val_candidates, val_references)
            val_bleu1_list.append(bleu1)
            val_bleu2_list.append(bleu2)
            val_bleu3_list.append(bleu3)
            val_bleu4_list.append(bleu4)
            logger.info(f"Epoch {epoch} - Validation BLEU: BLEU-1 {bleu1:.2f}, BLEU-2 {bleu2:.2f}, BLEU-3 {bleu3:.2f}, BLEU-4 {bleu4:.2f}")
            print(f"Epoch {epoch} - Validation BLEU: BLEU-1 {bleu1:.2f}, BLEU-2 {bleu2:.2f}, BLEU-3 {bleu3:.2f}, BLEU-4 {bleu4:.2f}")
            
            # 若 BLEU-2 得分更高，则保存最佳模型
            if bleu2 > best_bleu2:
                best_bleu2 = bleu2
                torch.save(model.state_dict(), best_model_path)
                print(f"Epoch {epoch} - New best model saved with BLEU-2: {best_bleu2:.2f}")
                logger.info(f"Epoch {epoch} - New best model saved with BLEU-2: {best_bleu2:.2f}")
            
            # 开发集评估
            dev_candidates = []
            dev_references = []
            with torch.no_grad():
                for batch in dev_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
                    decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    dev_candidates.extend(decoded_preds)
                    for ref in batch["reference"]:
                        dev_references.append([ref])
            d_bleu1, d_bleu2, d_bleu3, d_bleu4 = compute_bleu_scores(dev_candidates, dev_references)
            dev_bleu1_list.append(d_bleu1)
            dev_bleu2_list.append(d_bleu2)
            dev_bleu3_list.append(d_bleu3)
            dev_bleu4_list.append(d_bleu4)
            logger.info(f"Epoch {epoch} - Dev BLEU: BLEU-1 {d_bleu1:.2f}, BLEU-2 {d_bleu2:.2f}, BLEU-3 {d_bleu3:.2f}, BLEU-4 {d_bleu4:.2f}")
            print(f"Epoch {epoch} - Dev BLEU: BLEU-1 {d_bleu1:.2f}, BLEU-2 {d_bleu2:.2f}, BLEU-3 {d_bleu3:.2f}, BLEU-4 {d_bleu4:.2f}")
            
            model.train()
    
    # 输出训练结束信息
    print("Training finished.")
    print("Train Loss per epoch:", loss_list)
    print("Validation BLEU-1 per eval epoch:", val_bleu1_list)
    print("Validation BLEU-2 per eval epoch:", val_bleu2_list)
    print("Validation BLEU-3 per eval epoch:", val_bleu3_list)
    print("Validation BLEU-4 per eval epoch:", val_bleu4_list)
    
    #############################################
    # 绘制收敛曲线图
    #############################################
    plt.figure(figsize=(12, 5))
    # 绘制训练 loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epoch_list, loss_list, marker='o', label='Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()

    # 绘制 BLEU 曲线（这里以验证集 BLEU 指标为例）
    plt.subplot(1, 2, 2)
    eval_epochs = [epoch for epoch in epoch_list if epoch % 2 == 0]
    plt.plot(eval_epochs, val_bleu1_list, marker='o', label='BLEU-1')
    plt.plot(eval_epochs, val_bleu2_list, marker='o', label='BLEU-2')
    plt.plot(eval_epochs, val_bleu3_list, marker='o', label='BLEU-3')
    plt.plot(eval_epochs, val_bleu4_list, marker='o', label='BLEU-4')
    plt.xlabel("Epoch")
    plt.ylabel("BLEU Score")
    plt.title("Validation BLEU Scores")
    plt.legend()

    plt.tight_layout()
    plt.savefig("./convergence_curve.png")
    plt.show()
