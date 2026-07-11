# ==========================================
# AI学生就业竞争力分析系统
# train.py
# 数据清洗 + 模型训练
# ==========================================


import pandas as pd
import numpy as np
import os
import joblib

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    r2_score
)



# ==========================
# 创建文件夹
# ==========================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)



# ==========================
# 读取数据
# ==========================


print("正在读取数据...")


df = pd.read_csv(
    "data/student_placement_salary_elite_v2.csv"
)


print("\n数据大小:")
print(df.shape)


print("\n字段:")
print(df.columns)



# ==========================
# 数据清洗
# ==========================


print("\n开始清洗数据...")


# 删除重复

df.drop_duplicates(inplace=True)



# 缺失值处理

for col in df.columns:

    if df[col].dtype.name in ["object", "string"]:

        df[col] = df[col].fillna(
            df[col].mode()[0]
        )


    elif pd.api.types.is_numeric_dtype(df[col]):

        df[col] = df[col].fillna(
            df[col].median()
        )


print("清洗完成")



# ==========================
# 保存原始字段
# ==========================

original_columns = df.columns.tolist()



# ==========================
# 类别编码
# ==========================


encoder = {}


for col in df.select_dtypes(
        include=["object","string"]
):

    le = LabelEncoder()

    df[col] = le.fit_transform(
        df[col]
    )

    encoder[col] = le



joblib.dump(
    encoder,
    "models/encoder.pkl"
)


print("编码完成")



# ==========================
# 数据可视化
# ==========================


plt.figure(
    figsize=(12,8)
)


sns.heatmap(
    df.corr(),
    cmap="coolwarm"
)


plt.title(
    "Feature Correlation"
)


plt.tight_layout()


plt.savefig(
    "results/correlation.png",
    dpi=300
)


plt.close()


print("相关性图生成")



# =================================================
# 就业预测模型
# =================================================


print("\n训练就业预测模型...")


drop_columns = [

    "placed",

    "salary_lpa",

    "student_id",

    "job_role",

    "company_type"

]


X = df.drop(

    columns=drop_columns,

    errors="ignore"

)


y = df["placed"]



placement_features = X.columns.tolist()


joblib.dump(

    placement_features,

    "models/placement_features.pkl"

)



X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)



clf = RandomForestClassifier(

    n_estimators=200,

    random_state=42

)



clf.fit(

    X_train,

    y_train

)



prediction = clf.predict(

    X_test

)



accuracy = accuracy_score(

    y_test,

    prediction

)



print(
    "就业预测准确率:",
    accuracy
)


print(

    classification_report(

        y_test,

        prediction

    )

)



joblib.dump(

    clf,

    "models/placement_model.pkl"

)



print(
    "就业模型保存成功"
)



# ==========================
# 特征重要性
# ==========================


importance = pd.DataFrame({

    "feature":X.columns,

    "importance":clf.feature_importances_

})


importance.sort_values(

    "importance",

    ascending=False,

    inplace=True

)


print("\n影响就业Top10:")

print(
    importance.head(10)
)



plt.figure(
    figsize=(10,6)
)


sns.barplot(

    data=importance.head(10),

    x="importance",

    y="feature"

)


plt.title(
    "Feature Importance"
)


plt.tight_layout()


plt.savefig(

    "results/feature_importance.png",

    dpi=300

)


plt.close()




# =================================================
# 薪资预测模型
# =================================================


print(
    "\n训练薪资预测模型..."
)



salary_drop_columns=[

    "salary_lpa",

    "placed",

    "student_id",

    "job_role",

    "company_type"

]



X_salary = df.drop(

    columns=salary_drop_columns,

    errors="ignore"

)


y_salary=df["salary_lpa"]



salary_features = X_salary.columns.tolist()



joblib.dump(

    salary_features,

    "models/salary_features.pkl"

)



X_train, X_test, y_train, y_test = train_test_split(

    X_salary,

    y_salary,

    test_size=0.2,

    random_state=42

)



reg = RandomForestRegressor(

    n_estimators=200,

    random_state=42

)



reg.fit(

    X_train,

    y_train

)



salary_pred = reg.predict(

    X_test

)



print(

    "薪资MAE:",

    mean_absolute_error(

        y_test,

        salary_pred

    )

)



print(

    "薪资R2:",

    r2_score(

        y_test,

        salary_pred

    )

)



joblib.dump(

    reg,

    "models/salary_model.pkl"

)



print(
    "薪资模型保存成功"
)



print("\n====================")
print("训练全部完成！")
print("====================")