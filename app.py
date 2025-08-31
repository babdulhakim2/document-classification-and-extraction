import streamlit as st
import json
import os
from datetime import datetime
from model import OpenRouterClient
from processor import DocumentProcessor
from schemas import DocumentCategory
from dotenv import load_dotenv

load_dotenv()


st.set_page_config(page_title="Document Classifier", page_icon="📄", layout="wide")

@st.cache_resource
def load_client():
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        st.error("Please set OPENROUTER_API_KEY environment variable")
        st.stop()
    return OpenRouterClient(api_key)

@st.cache_resource
def load_processor():
    return DocumentProcessor()

def main():
    st.title("📄 Document Classification & Extraction")
    st.markdown("Upload documents to classify and extract structured content using LLMs")
    
    client = load_client()
    processor = load_processor()
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Settings")
        
        model_choice = st.selectbox(
            "Select Model",
            ["gemini-flash-2.5", "gpt-4o-mini", "gpt-4o"],
            index=0
        )
        
        st.subheader("Upload Files")
        uploaded_files = st.file_uploader(
            "Choose documents",
            accept_multiple_files=True,
            type=['pdf', 'png', 'jpg', 'jpeg']
        )
        
        if uploaded_files:
            st.write(f"**{len(uploaded_files)} files**")
        
        process_button = st.button("Process Documents", type="primary")
    
    with col2:
        if not uploaded_files and not process_button:
            st.info("Upload files to get started")
            
            st.markdown("""
            **Categories**: Invoice, Marketplace Screenshot, Chat Screenshot, Website Screenshot, Other
            
            **Models**: Gemini Flash 2.5, GPT-4o Mini, GPT-4o
            """)
    
        # Results section
        if process_button and uploaded_files:
            results = []
            files_data = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process all files first
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                try:
                    _, file_base64 = processor.process_uploaded_file(uploaded_file)
                    
                    result = client.classify_and_extract(
                        file_base64=file_base64,
                        model_name=model_choice
                    )
                    
                    result_dict = {
                        "filename": uploaded_file.name,
                        "category": result.category.value,
                        "confidence": result.confidence,
                        "entities": result.entities.model_dump(),
                        "processing_time": result.processing_time
                    }
                    
                    results.append(result_dict)
                    files_data.append((file_base64, result_dict))
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # LLM Judge evaluation in parallel
            if files_data:
                status_text.text("Evaluating predictions...")
                evaluations = client.evaluate_batch(files_data)
                
                # Update results with judge confidence and reasoning
                for i, eval_result in enumerate(evaluations):
                    if i < len(results):
                        results[i]["judge_confidence"] = eval_result["confidence"]
                        results[i]["judge_reasoning"] = eval_result["reasoning"]
            
            status_text.text("Processing complete!")
            
            if results:
                st.success(f"Processed {len(results)} files")
            
                for i, result in enumerate(results):
                    with st.expander(f"📄 {result['filename']} - {result['category'].replace('_', ' ').title()}", expanded=True):
                        
                        col_preview, col_results = st.columns([1, 3])
                        
                        with col_preview:
                            st.write("**Preview:**")
                            uploaded_file = uploaded_files[i]
                            
                            if uploaded_file.type.startswith('image/'):
                                st.image(uploaded_file, caption=result['filename'], width='stretch')
                            elif uploaded_file.type == 'application/pdf':
                                st.write("📄 PDF Document")
                                st.write(f"**{result['filename']}**")
                                try:
                                    # Try to display PDF using streamlit
                                    uploaded_file.seek(0)  # Reset file pointer
                                    with st.expander("View PDF", expanded=False):
                                        st.download_button("Download PDF", uploaded_file.read(), file_name=result['filename'])
                                except:
                                    st.write(f"Size: {uploaded_file.size:,} bytes")
                            else:
                                st.write(f"📎 {uploaded_file.type}")
                                st.write(f"**{result['filename']}**")
                        
                        with col_results:
                            st.write("**Classification:**")
                            st.write(f"Category: `{result['category']}`")
                            if result.get('judge_confidence'):
                                st.write(f"Judge Confidence: {result['judge_confidence']:.2%}")
                                if result.get('judge_reasoning'):
                                    st.write(f"*{result['judge_reasoning']}*")
                            st.write(f"Processing time: {result['processing_time']:.2f}s")
                            
                            st.write("**Extracted Entities:**")
                            entities = {k: v for k, v in result['entities'].items() if v is not None}
                            if entities:
                                st.json(entities)
                            else:
                                st.write("No entities extracted")
                            
                            # Individual download button
                            individual_result = {
                                "filename": result['filename'],
                                "category": result['category'],
                                "judge_confidence": result.get('judge_confidence'),
                                "judge_reasoning": result.get('judge_reasoning'),
                                "entities": entities,
                                "processing_time": result['processing_time']
                            }
                            st.download_button(
                                label=f"Download {result['filename']} result",
                                data=json.dumps(individual_result, indent=2),
                                file_name=f"{result['filename']}_result.json",
                                mime="application/json",
                                key=f"download_{i}"
                            )
                
                st.subheader("📥 Export All Results")
                col_download1, col_download2 = st.columns(2)
                
                with col_download1:
                    # Clean results for download (remove cost)
                    clean_results = []
                    for result in results:
                        clean_result = {
                            "filename": result['filename'],
                            "category": result['category'],
                            "judge_confidence": result.get('judge_confidence'),
                            "judge_reasoning": result.get('judge_reasoning'),
                            "entities": {k: v for k, v in result['entities'].items() if v is not None},
                            "processing_time": result['processing_time']
                        }
                        clean_results.append(clean_result)
                    
                    results_json = json.dumps(clean_results, indent=2)
                    st.download_button(
                        label="📄 Download All Results (JSON)",
                        data=results_json,
                        file_name=f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="download_all"
                    )
                
                with col_download2:
                    # Summary stats
                    categories = [r['category'] for r in results]
                    category_counts = {cat: categories.count(cat) for cat in set(categories)}
                    
                    summary = {
                        "processed_files": len(results),
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "model_used": model_choice,
                        "category_breakdown": category_counts,
                        "results": clean_results
                    }
                    
                    st.download_button(
                        label="📊 Download with Summary",
                        data=json.dumps(summary, indent=2),
                        file_name=f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="download_summary"
                    )
        
        elif process_button and not uploaded_files:
            st.warning("Please upload files first")

if __name__ == "__main__":
    main()