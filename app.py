# =====================================
# AI学生就业竞争力预测系统
# Streamlit版本
# =====================================


import streamlit as st
import pandas as pd
import joblib
import os
import zipfile


# 自动解压模型文件
if not os.path.exists("models"):

    with zipfile.ZipFile(
        "models.zip",
        "r"
    ) as zip_ref:

        zip_ref.extractall()


# ==========================
# 页面设置
# ==========================

st.set_page_config(
    page_title="AI就业预测系统",
    page_icon="🎓",
    layout="wide"
)


st.title(
    "🎓 AI学生就业竞争力分析系统"
)


st.write(
    "输入学生信息，预测就业概率和未来薪资"
)



# ==========================
# 加载模型
# ==========================


@st.cache_resource
def load_models():


    placement_model = joblib.load(
        "models/placement_model.pkl"
    )


    salary_model = joblib.load(
        "models/salary_model.pkl"
    )


    encoder = joblib.load(
        "models/encoder.pkl"
    )


    placement_features = joblib.load(
        "models/placement_features.pkl"
    )


    salary_features = joblib.load(
        "models/salary_features.pkl"
    )


    return (
        placement_model,
        salary_model,
        encoder,
        placement_features,
        salary_features
    )



(
    placement_model,
    salary_model,
    encoder,
    placement_features,
    salary_features

) = load_models()



# ==========================
# 输入区域
# ==========================


st.sidebar.header(
    "📌 学生信息输入"
)



cgpa = st.sidebar.slider(

    "CGPA",

    0.0,

    10.0,

    8.0

)



college_tier = st.sidebar.selectbox(

    "学校等级",

    [1,2,3]

)



python_skill = st.sidebar.slider(

    "Python能力",

    0,

    10,

    7

)



dsa_skill = st.sidebar.slider(

    "算法能力",

    0,

    10,

    7

)



ml_skill = st.sidebar.slider(

    "机器学习能力",

    0,

    10,

    5

)



web_dev_skill = st.sidebar.slider(

    "Web开发能力",

    0,

    10,

    5

)



coding_score = st.sidebar.slider(

    "编程评分",

    0,

    100,

    75

)



communication_score = st.sidebar.slider(

    "沟通能力",

    0,

    100,

    75

)



aptitude_score = st.sidebar.slider(

    "逻辑能力",

    0,

    100,

    75

)



internships = st.sidebar.number_input(

    "实习次数",

    0,

    10,

    1

)



projects = st.sidebar.number_input(

    "项目数量",

    0,

    20,

    3

)



backlogs = st.sidebar.number_input(

    "挂科数量",

    0,

    10,

    0

)



resume_score = st.sidebar.slider(

    "简历评分",

    0,

    100,

    80

)



skill_score = st.sidebar.slider(

    "综合技能评分",

    0,

    100,

    80

)



# ==========================
# 输入数据
# ==========================


input_data = pd.DataFrame({

    "cgpa":[cgpa],

    "college_tier":[college_tier],

    "python_skill":[python_skill],

    "dsa_skill":[dsa_skill],

    "ml_skill":[ml_skill],

    "web_dev_skill":[web_dev_skill],

    "coding_score":[coding_score],

    "communication_score":[communication_score],

    "aptitude_score":[aptitude_score],

    "internships":[internships],

    "projects":[projects],

    "backlogs":[backlogs],

    "resume_score":[resume_score],

    "skill_score":[skill_score]

})



# ==========================
# 编码
# ==========================


for col in encoder:


    if col in input_data.columns:


        input_data[col] = encoder[col].transform(

            input_data[col]

        )



# ==========================
# 自动匹配模型字段
# ==========================


def prepare_data(data, features):


    for col in features:


        if col not in data.columns:

            data[col] = 0


    data = data[features]


    return data



placement_input = prepare_data(

    input_data.copy(),

    placement_features

)



salary_input = prepare_data(

    input_data.copy(),

    salary_features

)




# ==========================
# 开始预测
# ==========================


if st.button(
    "🚀 开始AI预测"
):


    # 就业概率

    placement_prob = placement_model.predict_proba(

        placement_input

    )[0][1]



    # 薪资

    salary = salary_model.predict(

        salary_input

    )[0]



    st.divider()



    col1,col2 = st.columns(2)



    with col1:


        st.subheader(
            "🎯 就业预测"
        )


        st.success(

            f"就业成功概率：{placement_prob:.2%}"

        )



    with col2:


        st.subheader(
            "💰 薪资预测"
        )


        st.info(

            f"预计年薪：{salary:.2f} LPA"

        )




    st.divider()



    st.subheader(
        "🤖 AI职业建议"
    )



    advice=[]



    if python_skill < 7:

        advice.append(
            "加强Python工程能力"
        )



    if dsa_skill < 7:

        advice.append(
            "增加算法刷题量"
        )



    if internships < 2:

        advice.append(
            "建议增加企业实习经历"
        )



    if projects < 3:

        advice.append(
            "建议增加项目作品"
        )



    if resume_score < 80:

        advice.append(
            "优化个人简历"
        )



    if len(advice)==0:

        advice.append(
            "综合竞争力较强，可以冲击高薪岗位"
        )



    for a in advice:

        st.write(
            "✅",
            a
        )



st.sidebar.success(
    "模型加载成功"
)