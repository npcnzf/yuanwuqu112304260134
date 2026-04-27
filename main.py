import pandas as pd
import numpy as np
import re
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# 加载环境变量
load_dotenv()
api_key = os.getenv('API_KEY')

# 英文停用词列表（保留否定词）
stop_words = [
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'which', 'this', 'that',
    'these', 'those', 'then', 'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
    'under', 'again', 'further', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'don', 'should', 'now'
]

# 文本预处理
def preprocess_text(text):
    # 1. 去HTML标签
    text = re.sub(r'<br />', ' ', text)
    
    # 2. 小写化
    text = text.lower()
    
    # 3. 标点处理：保留情感相关标点，处理否定词
    # 处理否定词
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'t", " not", text)
    text = re.sub(r"'ve", " have", text)
    
    # 4. 保留情感相关标点
    # 只移除不影响情感的标点
    text = re.sub(r'[\[\]{}()*+,;:/\\]', ' ', text)
    
    # 5. 移除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# 主函数
def main():
    # 加载数据
    labeled_train = pd.read_csv('../labeledTrainData.tsv/labeledTrainData.tsv', sep='\t', quoting=3)
    test_data = pd.read_csv('../testData.tsv/testData.tsv', sep='\t', quoting=3)
    
    # 预处理文本
    labeled_train['processed'] = labeled_train['review'].apply(preprocess_text)
    test_data['processed'] = test_data['review'].apply(preprocess_text)
    
    # 使用TF-IDF提取特征，使用短语模式（n-gram）
    vectorizer = TfidfVectorizer(
        max_features=10000, 
        stop_words=stop_words, 
        ngram_range=(1, 2),  # 使用1-gram和2-gram
        min_df=3,  # 最小文档频率
        max_df=0.9  # 最大文档频率
    )
    X_train = vectorizer.fit_transform(labeled_train['processed'])
    y_train = labeled_train['sentiment'].values
    X_test = vectorizer.transform(test_data['processed'])
    
    # 分割验证集
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 训练逻辑回归模型，使用正则化参数
    lr = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    lr.fit(X_train_split, y_train_split)
    
    # 计算AUC
    y_val_pred = lr.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_val_pred)
    
    # 记录AUC分数
    with open('auc_score.txt', 'w') as f:
        f.write(f"{auc:.4f}")
    
    # 在完整训练集上训练
    lr.fit(X_train, y_train)
    
    # 预测测试集
    test_pred = lr.predict_proba(X_test)[:, 1]
    
    # 生成提交文件
    # 移除id列中的额外引号
    submission = pd.DataFrame({'id': test_data['id'].str.replace('"', ''), 'sentiment': test_pred})
    # 确保保存时不添加额外的引号
    submission.to_csv('final_submission.csv', index=False, quoting=0, escapechar='\\')
    
    # 打印结果
    print(f"AUC score: {auc:.4f}")
    print("Submission file generated: final_submission.csv")

if __name__ == "__main__":
    main()
