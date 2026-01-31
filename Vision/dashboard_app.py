# -*- coding: utf-8 -*-
"""
Vision Analysis Dashboard
Comprehensive visualization of benchmark results, videos, and comparisons.
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Manually add project root to sys.path for Streamlit
PROJECT_ROOT = Path(__file__).resolve().parents[1] # Correct calculation for project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.utils.path_manager import path_manager

# Configuración de página
st.set_page_config(
    page_title="Sensor Fusion Dashboard",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# --- FUNCIONES DE UTILIDAD ---

@st.cache_data
def load_benchmark_results():
    """Load benchmark results from JSON."""
    json_path = path_manager.get("output") / "data" / "benchmark_results.json"
    
    if not json_path.exists():
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def load_prediction_stats():
    """Load prediction statistics."""
    preds_dir = path_manager.get("predictions")
    
    if not preds_dir.exists():
        return None
    
    stats = []
    for json_file in preds_dir.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if 'meta' in data and 'results' in data:
                model_name = json_file.stem
                results = data['results']
                
                # Calcular estadísticas
                total_detections = sum(len(r['detections']) for r in results)
                avg_detections = total_detections / len(results) if results else 0
                avg_inference = sum(r['inference_ms'] for r in results) / len(results) if results else 0
                
                # Contar por clase
                class_counts = {}
                for result in results:
                    for det in result['detections']:
                        class_name = det['class_name']
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                stats.append({
                    'model': model_name,
                    'total_images': len(results),
                    'total_detections': total_detections,
                    'avg_detections_per_image': round(avg_detections, 2),
                    'avg_inference_ms': round(avg_inference, 2),
                    'class_distribution': class_counts
                })
    
    return stats

def get_available_videos():
    """Get list of generated videos."""
    videos_dir = path_manager.get("videos")
    
    if not videos_dir.exists():
        return []
    
    videos = list(videos_dir.glob("*.mp4"))
    return sorted(videos, key=lambda x: x.stat().st_mtime, reverse=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("📊 Dashboard Control")
    
    page = st.radio(
        "Navigation",
        ["🏠 Overview", "📈 Benchmarks", "🎬 Videos", "🔍 Detailed Analysis"]
    )
    
    st.divider()
    
    # Project information
    st.subheader("ℹ️ Info")
    st.caption(f"**Base:** `{path_manager.BASE_DIR.name}`")
    
    # Quick statistics
    videos = get_available_videos()
    st.metric("Videos", len(videos))
    
    bench_data = load_benchmark_results()
    if bench_data:
        total_models = sum(len(results) for results in bench_data.get('datasets', {}).values())
        st.metric("Evaluated Models", total_models)

# --- MAIN PAGE ---
st.title("🚗 Vision Analysis Dashboard")

# ====================================================================
# PAGE 1: OVERVIEW
# ====================================================================
if page == "🏠 Overview":
    st.header("Project Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📦 Available Models")
        models_dir = path_manager.get("models")
        if models_dir.exists():
            models = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.pth"))
            st.metric("Total Models", len(models))
            
            with st.expander("View models"):
                for model in sorted(models):
                    size_mb = model.stat().st_size / 1e6
                    st.text(f"• {model.name} ({size_mb:.1f} MB)")
    
    with col2:
        st.subheader("🎬 Generated Videos")
        videos = get_available_videos()
        st.metric("Total Videos", len(videos))
        
        if videos:
            latest = videos[0]
            st.caption(f"Last: {latest.name}")
    
    with col3:
        st.subheader("📊 Datasets")
        val_dir = path_manager.get("bdd_images_val")
        train_dir = path_manager.get("bdd_images_train")
        
        val_count = len(list(val_dir.glob("*.jpg"))) if val_dir.exists() else 0
        train_count = len(list(train_dir.glob("*.jpg"))) if train_dir.exists() else 0
        
        st.metric("BDD100K Val", f"{val_count:,}")
        st.metric("BDD100K Train", f"{train_count:,}")
    
    # Prediction statistics
    st.divider()
    st.subheader("📊 Latest Prediction Statistics")
    
    pred_stats = load_prediction_stats()
    
    if pred_stats:
        df_stats = pd.DataFrame(pred_stats)
        
        # Display table
        st.dataframe(
            df_stats[['model', 'total_images', 'total_detections', 'avg_detections_per_image', 'avg_inference_ms']],
            width="stretch",
            hide_index=True
        )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                df_stats,
                x='model',
                y='avg_detections_per_image',
                title='Average Detections per Image',
                labels={'avg_detections_per_image': 'Detections', 'model': 'Model'}
            )
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            fig = px.bar(
                df_stats,
                x='model',
                y='avg_inference_ms',
                title='Average Inference Time',
                labels={'avg_inference_ms': 'ms', 'model': 'Model'},
                color='avg_inference_ms',
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig, width="stretch")

# ====================================================================
# PAGE 2: BENCHMARKS
# ====================================================================
elif page == "📈 Benchmarks":
    st.header("Benchmark Results")
    
    bench_data = load_benchmark_results()
    
    if not bench_data:
        st.warning("⚠️ No benchmark results available.")
        st.info("Run `python Vision/run_benchmark.py` to generate results.")
    else:
        st.success(f"✅ Last updated: {bench_data.get('timestamp', 'Unknown')}")
        
        # Select dataset
        datasets = bench_data.get('datasets', {})
        selected_dataset = st.selectbox("Dataset", list(datasets.keys()))
        
        if selected_dataset:
            results = datasets[selected_dataset]
            
            if not results:
                st.warning(f"No results for {selected_dataset}")
            else:
                df = pd.DataFrame(results)
                
                # Main metrics
                st.subheader("📊 Main Metrics")
                
                # Display table with highlighting
                st.dataframe(
                    df[['model', 'mAP50-95', 'mAP50', 'Precision', 'Recall', 'Inference_Time_ms']].style.highlight_max(
                        axis=0,
                        subset=['mAP50-95', 'mAP50', 'Precision', 'Recall'],
                        color='lightgreen'
                    ).highlight_min(
                        axis=0,
                        subset=['Inference_Time_ms'],
                        color='lightgreen'
                    ),
                    width="stretch",
                    hide_index=True
                )
                
                # Comparative charts
                st.subheader("📈 Visual Comparisons")
                
                tab1, tab2, tab3 = st.tabs(["mAP Comparison", "Speed vs Accuracy", "Per-Class Performance"])
                
                with tab1:
                    # mAP Comparison
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='mAP50-95',
                        x=df['model'],
                        y=df['mAP50-95'],
                        marker_color='lightblue'
                    ))
                    fig.add_trace(go.Bar(
                        name='mAP50',
                        x=df['model'],
                        y=df['mAP50'],
                        marker_color='lightcoral'
                    ))
                    fig.update_layout(
                        title='mAP Comparison',
                        barmode='group',
                        yaxis_title='Score',
                        xaxis_title='Model'
                    )
                    st.plotly_chart(fig, width="stretch")
                
                with tab2:
                    # Speed vs Accuracy trade-off
                    fig = px.scatter(
                        df,
                        x='Inference_Time_ms',
                        y='mAP50-95',
                        text='model',
                        size='Precision',
                        color='Recall',
                        title='Trade-off: Speed vs Accuracy',
                        labels={
                            'Inference_Time_ms': 'Inference Time (ms)',
                            'mAP50-95': 'mAP50-95'
                        }
                    )
                    fig.update_traces(textposition='top center')
                    st.plotly_chart(fig, width="stretch")
                    
                    st.caption("💡 **Ideal:** Top-left corner (high precision, low latency)")
                
                with tab3:
                    # Per-class performance
                    st.subheader("Per-Class Performance")
                    
                    selected_model = st.selectbox("Select Model", df['model'].tolist())
                    
                    model_data = df[df['model'] == selected_model].iloc[0]
                    
                    if 'per_class' in model_data and model_data['per_class']:
                        class_df = pd.DataFrame(
                            list(model_data['per_class'].items()),
                            columns=['Class', 'mAP']
                        ).sort_values('mAP', ascending=False)
                        
                        fig = px.bar(
                            class_df,
                            x='Class',
                            y='mAP',
                            title=f'mAP per Class - {selected_model}',
                            color='mAP',
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.info("No per-class data available")

# ====================================================================
# PAGE 3: VIDEOS
# ====================================================================
elif page == "🎬 Videos":
    st.header("Generated Video Gallery")
    
    videos = get_available_videos()
    
    if not videos:
        st.warning("⚠️ No videos generated yet.")
        st.info("Run `python Vision/main.py` to generate videos.")
    else:
        # Filters
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search = st.text_input("🔍 Search video", "")
        
        with col2:
            sort_by = st.selectbox("Sort by", ["Most recent", "Name", "Size"])
        
        # Filter videos
        filtered_videos = [v for v in videos if search.lower() in v.name.lower()] if search else videos
        
        # Sort
        if sort_by == "Name":
            filtered_videos = sorted(filtered_videos, key=lambda x: x.name)
        elif sort_by == "Size":
            filtered_videos = sorted(filtered_videos, key=lambda x: x.stat().st_size, reverse=True)
        
        st.write(f"**{len(filtered_videos)} videos found**")
        
        # Display videos in grid
        for video in filtered_videos:
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.subheader(f"🎬 {video.name}")
                
                with col2:
                    size_mb = video.stat().st_size / 1e6
                    st.metric("Size", f"{size_mb:.1f} MB")
                
                with col3:
                    st.caption(f"Modified: {pd.Timestamp(video.stat().st_mtime, unit='s').strftime('%Y-%m-%d %H:%M')}")
                
                # Video player
                st.video(str(video))
                
                # Download button
                with open(video, 'rb') as f:
                    st.download_button(
                        label="⬇️ Download",
                        data=f,
                        file_name=video.name,
                        mime="video/mp4"
                    )
                
                st.divider()

# ====================================================================
# PAGE 4: DETAILED ANALYSIS
# ====================================================================
elif page == "🔍 Detailed Analysis":
    st.header("Detailed Prediction Analysis")
    
    pred_stats = load_prediction_stats()
    
    if not pred_stats:
        st.warning("⚠️ No predictions available for analysis.")
    else:
        # Select model
        models = [s['model'] for s in pred_stats]
        selected_model = st.selectbox("Select Model", models)
        
        model_data = next(s for s in pred_stats if s['model'] == selected_model)
        
        # Model metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Images Processed", model_data['total_images'])
        
        with col2:
            st.metric("Total Detections", model_data['total_detections'])
        
        with col3:
            st.metric("Average per Image", f"{model_data['avg_detections_per_image']:.2f}")
        
        with col4:
            st.metric("Average Inference", f"{model_data['avg_inference_ms']:.1f} ms")
        
        # Class distribution
        st.subheader("📊 Detected Class Distribution")
        
        class_dist = model_data['class_distribution']
        
        if class_dist:
            # Create DataFrame
            class_df = pd.DataFrame(
                list(class_dist.items()),
                columns=['Class', 'Quantity']
            ).sort_values('Quantity', ascending=False)
            
            # Pie chart
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    class_df,
                    values='Quantity',
                    names='Class',
                    title='Class Distribution'
                )
                st.plotly_chart(fig, width="stretch")
            
            with col2:
                fig = px.bar(
                    class_df.head(10),
                    x='Quantity',
                    y='Class',
                    orientation='h',
                    title='Top 10 Most Detected Classes'
                )
                st.plotly_chart(fig, width="stretch")
            
            # Detailed table
            st.dataframe(class_df, width="stretch", hide_index=True)
        else:
            st.info("No class distribution data available")

# --- FOOTER ---
st.divider()
st.caption(f"🗂️ Project: `{path_manager.BASE_DIR}` | Dashboard v1.0")
