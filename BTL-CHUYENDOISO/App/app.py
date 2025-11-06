import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Any
from data_processing import process_data, process_personal_data
from clustering import find_best_clustering
from visualization import (
    create_roadmap_fig, create_radar_fig, create_bar_fig,
    create_boxplot_fig, create_pca_fig
)
from utils import convert_df_to_excel
from major_mapping import major_mapping
from unique_pages import unique_pages, unique_groups

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Score columns
score_columns = ['Lập trình C', 'Cấu trúc dữ liệu và giải thuật', 'Cơ sở dữ liệu',
                 'Toán cao cấp', 'Mạng máy tính', 'Hệ điều hành', 'Tiếng Anh P1']

# Session state initialization
if 'df' not in st.session_state:
    st.session_state.df: Optional[pd.DataFrame] = None
if 'features' not in st.session_state:
    st.session_state.features: Optional[np.ndarray] = None
if 'clusters' not in st.session_state:
    st.session_state.clusters: Optional[np.ndarray] = None
if 'tfidf_vectorizer' not in st.session_state:
    st.session_state.tfidf_vectorizer: Optional[Any] = None
if 'scaler' not in st.session_state:
    st.session_state.scaler: Optional[Any] = None
if 'best_model' not in st.session_state:
    st.session_state.best_model: Optional[Any] = None
if 'best_method' not in st.session_state:
    st.session_state.best_method: Optional[str] = None

st.title("Major Recommendation System")

menu = st.sidebar.selectbox("Chọn chế độ phân tích", ["Phân tích dữ liệu tổng hợp", "Phân tích cá nhân"])

if menu == "Phân tích dữ liệu tổng hợp":
    st.header("Phân tích dữ liệu tổng hợp")
    uploaded_file = st.file_uploader("Upload file", type="xlsx")

    if uploaded_file is not None:
        st.session_state.df = pd.read_excel(uploaded_file)
        df = st.session_state.df
        st.write("Dữ liệu đã upload:")
        st.dataframe(df)

        # Process data
        df, features, tfidf_vectorizer, scaler = process_data(df, score_columns)
        st.session_state.features = features
        st.session_state.tfidf_vectorizer = tfidf_vectorizer
        st.session_state.scaler = scaler

        # Find best clustering
        best_model, best_labels, best_method = find_best_clustering(features)
        if best_model is None:
            st.error("Không thể clustering dữ liệu. Vui lòng kiểm tra dữ liệu.")
        else:
            st.session_state.best_model = best_model
            st.session_state.best_method = best_method

            df['Cluster'] = best_labels
            st.session_state.clusters = best_labels

            st.write("Danh sách sinh viên:")
            for idx, row in df.iterrows():
                with st.expander(f"Sinh viên: **{row['Họ tên']}** (MSSV: {row['MSSV']})"):
                    cluster = row['Cluster']
                    mapping = major_mapping.get(cluster, {})

                    scores_df = pd.DataFrame({
                        'Môn học': score_columns,
                        'Điểm': [row[col] for col in score_columns]
                    })
                    st.write("**Điểm các môn:**")
                    st.dataframe(scores_df.style.format({'Điểm': '{:.1f}'}))

                    major = mapping.get('major', 'Không xác định')
                    st.markdown(
                        f"**Chuyên ngành đề xuất:** <span style='color: #4CAF50; font-weight: bold;'>{major}</span>",
                        unsafe_allow_html=True)
                    st.write("**Đề xuất môn học & kỹ năng bổ sung:**", mapping.get('supplementary', ''), ";",
                             mapping.get('skills', ''))

                    interests = row['Combined Interests'].split('; ')
                    interests_df = pd.DataFrame({'Sở thích': interests})
                    st.write("- Pages/groups:")
                    st.dataframe(
                        interests_df.style.set_properties(**{'text-align': 'left', 'border': '1px solid lightgray'}))

                    st.subheader("Sơ đồ lộ trình học tập")
                    roadmap = mapping.get('roadmap', [])
                    fig = create_roadmap_fig(roadmap)
                    st.plotly_chart(fig, key=f"roadmap_{idx}")

                    st.subheader("Biểu đồ radar điểm số")
                    fig_radar = create_radar_fig(row, score_columns)
                    st.plotly_chart(fig_radar, key=f"radar_{idx}")

                    st.subheader("So sánh điểm với trung bình lớp")
                    avg_scores = df[score_columns].mean()
                    fig_bar = create_bar_fig(row, avg_scores, score_columns)
                    st.plotly_chart(fig_bar, key=f"bar_{idx}")

            # Download button
            excel_data = convert_df_to_excel(df)
            st.download_button(
                label="Tải kết quả Excel",
                data=excel_data,
                file_name="RecommendationResults.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.subheader("Boxplots of Normalized Scores")
            fig_box = create_boxplot_fig(df, score_columns)
            st.pyplot(fig_box)

            st.subheader("Clusters Visualization (PCA)")
            fig_pca = create_pca_fig(st.session_state.features, df['Cluster'])
            st.pyplot(fig_pca)

elif menu == "Phân tích cá nhân":
    st.header("Phân Tích Cá Nhân")

    selected_pages = st.multiselect("Bạn quan tâm đến trang nào trên Facebook? (Chọn nhiều)", unique_pages)
    selected_groups = st.multiselect("Bạn quan tâm đến nhóm nào trên Facebook? (Chọn nhiều)", unique_groups)

    personal_upload = st.file_uploader("Upload file", type="xlsx")

    manual_pages = '; '.join(selected_pages)
    manual_groups = '; '.join(selected_groups)

    if personal_upload is not None:
        try:
            personal_df = pd.read_excel(personal_upload)
            if not personal_df.empty:
                row = personal_df.iloc[0]
                personal_name = row['Họ tên']
                mssv = row['MSSV']

                # Process personal data
                personal_df, personal_features = process_personal_data(
                    personal_df, score_columns, manual_pages, manual_groups,
                    st.session_state.tfidf_vectorizer, st.session_state.scaler
                )

                # Predict cluster
                if st.session_state.best_model is not None:
                    model = st.session_state.best_model
                    if hasattr(model, 'predict'):
                        personal_cluster = model.predict(personal_features)
                    else:
                        # Fallback to fit_predict with combined features
                        combined_features = np.vstack((st.session_state.features, personal_features))
                        personal_cluster = model.fit_predict(combined_features)[-1:]
                else:
                    # Fallback if no model
                    from sklearn.cluster import AgglomerativeClustering
                    hierarchical = AgglomerativeClustering(n_clusters=8)
                    personal_cluster = hierarchical.fit_predict(personal_features)

                cluster = personal_cluster[0]
                mapping = major_mapping.get(cluster, {})
                st.subheader(f"Sinh viên: **{personal_name}** (MSSV: {mssv})")

                scores_df = pd.DataFrame({
                    'Môn học': score_columns,
                    'Điểm': [row[col] for col in score_columns]
                })
                st.write("**Điểm các môn:**")
                st.dataframe(scores_df.style.format({'Điểm': '{:.1f}'}))

                major = mapping.get('major', 'Không xác định')
                st.markdown(
                    f"**Chuyên ngành đề xuất:** <span style='color: #4CAF50; font-weight: bold;'>{major}</span>",
                    unsafe_allow_html=True)
                st.write("**Đề xuất môn học & kỹ năng bổ sung:**", mapping.get('supplementary', ''), ";",
                         mapping.get('skills', ''))

                st.subheader("Biểu đồ radar điểm số")
                fig_radar = create_radar_fig(row, score_columns)
                st.plotly_chart(fig_radar, key="radar_personal")

                if st.session_state.df is not None:
                    st.subheader("So sánh điểm với trung bình lớp")
                    avg_scores = st.session_state.df[score_columns].mean()
                    fig_bar = create_bar_fig(row, avg_scores, score_columns)
                    st.plotly_chart(fig_bar, key="bar_personal")

        except Exception as e:
            st.error(f"Lỗi đọc file cá nhân: {e}")

st.sidebar.info("""
Copyright © DNU CNTT16-02
""")