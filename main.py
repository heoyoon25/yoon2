import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 1. 페이지 설정 및 제목
st.set_page_config(page_title="DB Insurance AI Marketing", layout="wide")
st.title("🛡️ 외국인 관광객 보험 가입 예측 AI 엔진")
st.markdown("---")

# 2. 분석 성과 요약 (Metric Cards)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Model AUC", "0.9996", "+0.515") # Baseline 대비 성능 향상
col2.metric("Precision", "1.000", "Perfect") # RF 정밀도 수치
col3.metric("Marketing Lift", "10.0x", "Top 10%") # 상위 10% 리프트 수치
col4.metric("Targeting ROI", "90%", "Cost Saving") # 비용 절감 기대 효과

st.sidebar.header("🔍 고객 데이터 입력 (Lead Scoring)")

# 3. 사이드바 - 실시간 가입 확률 시뮬레이터 (Scoring)
with st.sidebar:
    gender = st.selectbox("성별", ["여성", "남성"])
    age = st.slider("연령대", 10, 70, 25)
    purpose = st.selectbox("방문 목적", ["쇼핑/관광", "의료/시술", "비즈니스", "기타"])
    stay_duration = st.number_input("체류 기간 (일)", min_value=1, value=5)
    
    st.subheader("📝 샤오홍슈 게시글 분석")
    post_text = st.text_area("게시글 내용 복사/붙여넣기", 
                             placeholder="예: 면세점에서 가방 사고 택스리펀 받았어요!")
    
    predict_btn = st.button("가입 확률 예측하기")

# 4. 분석 엔진 작동 (Logic - 예시 데이터 기반)
if predict_btn:
    # 텍스트 내 키워드에 따른 가상의 토픽 가중치 계산 (실제 구현시 LDA 모델 로드 필요)
    shopping_score = 0.8 if "면세점" in post_text or "쇼핑" in post_text else 0.2
    risk_score = 0.7 if "경찰" in post_text or "분실" in post_text else 0.1
    
    # 가입 확률 계산 (Random Forest 로직 시뮬레이션)
    # $$P(Subscription) = \frac{1}{1 + e^{-z}}$$ (Logistic) 또는 RF의 가중 평균
    probability = (shopping_score * 0.5 + risk_score * 0.4 + (stay_duration / 30) * 0.1) * 100
    
    st.header("🎯 실시간 타겟팅 리드 분석 결과")
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.subheader("가입 확률")
        st.title(f"{probability:.1f}%")
        if probability > 70:
            st.success("🔥 고가치 타겟 고객 (High Intent)")
        else:
            st.warning("⚖️ 일반 관심 고객 (Moderate Intent)")
            
    with c2:
        # 토픽 분포 시각화 (사용자가 입력한 텍스트 분석 결과)
        topic_df = pd.DataFrame({
            "Topic": ["Shopping (T1)", "Leisure (T2)", "Medical (T3)", "Risk (T4)"],
            "Weight": [shopping_score, 0.3, 0.1, risk_score]
        })
        fig = px.bar(topic_df, x="Topic", y="Weight", color="Topic", 
                     title="사용자 관심사 분석 (LDA Topic Distribution)")
        st.plotly_chart(fig)

# 5. 비즈니스 가치 검증 (ROI Simulator)
st.markdown("---")
st.header("📈 마케팅 예산 최적화 시뮬레이터")
decile = st.slider("타겟팅 범위 선택 (상위 %)", 1, 100, 10)

# 리프트 데이터 기반 기대 효과 계산
# 상위 10%에서 리프트 10배 발생
expected_efficiency = 10.0 if decile <= 10 else 1.0 + (10 - 1) * (100 - decile) / 90
estimated_saving = (100 - decile)

ec1, ec2 = st.columns(2)
ec1.metric("기존 대비 마케팅 효율", f"{expected_efficiency:.1f}x")
ec2.metric("예상 마케팅 비용 절감액", f"{estimated_saving}%")

st.info("💡 모델 분석 결과, 전체 고객을 대상으로 하는 무작위 마케팅보다 쇼핑(T1)과 리스크(T4) 관심도가 높은 상위 10% 고객에게 집중하는 것이 가장 효율적입니다.")
