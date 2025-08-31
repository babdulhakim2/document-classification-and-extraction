import streamlit as st
import json
import time
from datetime import datetime
from typing import List
import concurrent.futures

from data_pipeline import DocumentPipeline
from config import settings, validate_file_size, validate_file_type, get_supported_extensions

# Configure page
st.set_page_config(
    page_title=settings.page_title, 
    page_icon=settings.page_icon, 
    layout="wide"
)

@st.cache_resource
def load_pipeline():
    """Load document processing pipeline"""
    return DocumentPipeline()

def show_file_preview(uploaded_file, filename):
    """Show preview of uploaded file with proper display"""
    if uploaded_file.type.startswith("image/"):
        st.image(uploaded_file, caption=filename, width="stretch")
        # Reset pointer after preview in case Streamlit consumed the buffer
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
    elif uploaded_file.type == "application/pdf":
        st.write("📄 PDF Document")
        st.write(f"**{filename}**")
        try:
            uploaded_file.seek(0)
            pdf_bytes = uploaded_file.read()
            import base64
            pdf_base64 = base64.b64encode(pdf_bytes).decode()
            pdf_display = f'<embed src="data:application/pdf;base64,{pdf_base64}" width="100%" height="400" type="application/pdf">'
            st.markdown(pdf_display, unsafe_allow_html=True)
            # Reset pointer so later processing reads the full file again
            try:
                uploaded_file.seek(0)
            except Exception:
                pass
        except:
            st.write(f"Size: {uploaded_file.size:,} bytes")
    else:
        st.write(f"📎 {uploaded_file.type}")
        st.write(f"**{filename}**")
        st.write(f"Size: {uploaded_file.size:,} bytes")

def validate_files(uploaded_files) -> List:
    """Validate uploaded files and show feedback"""
    if not uploaded_files:
        return []
    
    valid_files = []
    issues = []
    
    for file in uploaded_files:
        # Check file size
        if hasattr(file, 'size') and not validate_file_size(file.size):
            issues.append(f"❌ {file.name}: Too large ({file.size / 1024 / 1024:.1f}MB > {settings.max_file_size_mb}MB)")
        # Check file type  
        elif not validate_file_type(file.type):
            issues.append(f"❌ {file.name}: Unsupported type ({file.type})")
        else:
            valid_files.append(file)
    
    if issues:
        st.warning("\\n".join(issues))
    
    if len(uploaded_files) > settings.max_files_per_batch:
        st.warning(f"Only first {settings.max_files_per_batch} files will be processed")
        valid_files = valid_files[:settings.max_files_per_batch]
    
    return valid_files

def show_processing_results(results: List[dict], uploaded_files: List):
    """Display processing results with file previews"""
    if not results:
        st.error("No files were processed successfully")
        return
    
    st.success(f"✅ Processed {len(results)} files")
    
    for i, result in enumerate(results):
        category_display = result['category'].replace('_', ' ').title()
        
        with st.expander(f"📄 {result['filename']} - {category_display}", expanded=True):
            col_preview, col_results = st.columns([1, 3])
            
            with col_preview:
                st.write("**Preview:**")
                if i < len(uploaded_files):
                    show_file_preview(uploaded_files[i], result['filename'])
                else:
                    st.write(f"📎 {result['filename']}")
            
            with col_results:
                st.write("**Classification:**")
                st.write(f"Category: `{result['category']}`")
                
                if result.get('judge_confidence'):
                    confidence_pct = result['judge_confidence'] * 100
                    st.write(f"Judge Confidence: **{confidence_pct:.1f}%**")
                    
                    if result.get('judge_reasoning'):
                        st.write(f"*{result['judge_reasoning']}*")
                
                st.write(f"Processing time: **{result['processing_time']:.2f}s**")
                
                # Show entities
                st.write("**Extracted Entities:**")
                entities = {k: v for k, v in result['entities'].items() if v is not None}
                
                if entities:
                    st.json(entities)
                else:
                    st.write("No entities extracted")
                
                # Download button for individual result
                download_data = {
                    "filename": result['filename'],
                    "category": result['category'],
                    "judge_confidence": result.get('judge_confidence'),
                    "judge_reasoning": result.get('judge_reasoning'),
                    "entities": entities,
                    "processing_time": result['processing_time']
                }
                
                st.download_button(
                    label=f"📥 Download {result['filename']} result",
                    data=json.dumps(download_data, indent=2),
                    file_name=f"{result['filename']}_result.json",
                    mime="application/json",
                    key=f"download_{i}"
                )

def show_batch_download(results: List[dict]):
    """Show batch download options"""
    if not results:
        return
    
    st.subheader("📥 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        # Clean results for download
        clean_results = []
        for result in results:
            clean_result = {
                "filename": result['filename'],
                "category": result['category'],
                "confidence": result.get('judge_confidence'),
                "reasoning": result.get('judge_reasoning'),
                "entities": {k: v for k, v in result['entities'].items() if v},
                "processing_time": result['processing_time']
            }
            clean_results.append(clean_result)
        
        st.download_button(
            label="📄 Download All Results (JSON)",
            data=json.dumps(clean_results, indent=2),
            file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # Summary report
        categories = [r['category'] for r in results]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        
        summary = {
            "processed_files": len(results),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "category_breakdown": category_counts,
            "results": clean_results
        }
        
        st.download_button(
            label="📊 Download Summary Report",
            data=json.dumps(summary, indent=2),
            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def main():
    """Main Streamlit application"""
    st.title("📄 Document Classification & Extraction")
    st.markdown("Upload documents to classify and extract structured data.")
    
    pipeline = load_pipeline()
    accepted_exts = get_supported_extensions()
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Settings")
        
        # Model selection
        available_models = list(settings.available_models.keys())
        default_idx = available_models.index(settings.default_model) if settings.default_model in available_models else 0
        model_choice = st.selectbox("Select Model", available_models, index=default_idx)
        
        st.subheader("Upload Files")
        help_text = (
            f"Max: {settings.max_file_size_mb}MB per file, "
            f"{settings.max_files_per_batch} files total. "
            f"Accepted: {', '.join(accepted_exts)}"
        )
        uploaded_files = st.file_uploader(
            "Choose documents",
            accept_multiple_files=True,
            type=accepted_exts,
            help=help_text
        )
        
        # File validation and preview
        if uploaded_files:
            valid_files = validate_files(uploaded_files)
            
            if valid_files:
                st.write(f"**{len(valid_files)} valid files ready**")
                
                # Show file previews
                for file in valid_files:
                    with st.expander(f"📄 {file.name}"):
                        show_file_preview(file, file.name)
            else:
                valid_files = []
        else:
            valid_files = []
        
        process_button = st.button("Process Documents", type="primary") if valid_files else None
    
    with col2:
        if not uploaded_files and not process_button:
            st.info("Upload files to get started")

            models_list = ", ".join(settings.available_models.keys()) or "(none configured)"
            st.markdown(
                f"""
                **Categories**: Invoice, Marketplace Screenshot, Chat Screenshot, Website Screenshot, Other

                **Models**: {models_list}

                **Limits**: Max {settings.max_file_size_mb}MB per file, {settings.max_files_per_batch} files total

                **Accepted types**: {', '.join(accepted_exts)}

                **Features**:
                - Async processing for speed
                - Image optimization
                - Confidence evaluation
                - Structured JSON output
                """
            )
        
        # Results section
        elif process_button and valid_files:
            results = []
            files_data = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process all files first (parallelize classification)
            start_batch = time.time()
            def classify_worker(args):
                idx, up_file = args
                try:
                    file_start = time.time()
                    file_base64 = pipeline.preprocess(up_file)
                    result = pipeline.classify(file_base64, model_choice, up_file.name)
                    result_dict = {
                        "filename": up_file.name,
                        "category": result.category.value,
                        "entities": result.entities.model_dump(),
                        "processing_time": time.time() - file_start,
                    }
                    return idx, result_dict, file_base64, None
                except Exception as e:
                    return idx, None, None, f"Error processing {up_file.name}: {str(e)}"

            with concurrent.futures.ThreadPoolExecutor(max_workers=settings.max_concurrent_requests) as executor:
                futures = [executor.submit(classify_worker, (i, f)) for i, f in enumerate(valid_files)]
                done = 0
                total = len(futures)
                # Preallocate to preserve original ordering
                result_slots = [None] * total
                files_data_slots = [None] * total
                for fut in concurrent.futures.as_completed(futures):
                    idx, result_dict, file_base64, err = fut.result()
                    done += 1
                    status_text.text(f"Processing {valid_files[idx].name} ({done}/{total})...")
                    progress_bar.progress(done / total)
                    if err:
                        st.error(err)
                        # Put placeholder in the original index
                        result_slots[idx] = {
                            "filename": valid_files[idx].name,
                            "category": "other",
                            "entities": {},
                            "processing_time": 0.0,
                        }
                        files_data_slots[idx] = ("", result_slots[idx])
                    else:
                        result_slots[idx] = result_dict
                        files_data_slots[idx] = (file_base64, result_dict)

                # Materialize ordered lists
                results = result_slots
                files_data = files_data_slots

            # LLM Judge evaluation in parallel
            if files_data:
                status_text.text("Evaluating predictions...")
                judge_start = time.time()

                # Prepare tasks
                def run_judge(args):
                    file_base64, result_dict = args
                    if not file_base64:
                        return 0.0, "Skipped: classification error"
                    prediction = {
                        "category": result_dict["category"],
                        "entities": result_dict["entities"],
                    }
                    return pipeline.judge(file_base64, prediction, result_dict.get("filename", "document"))

                with concurrent.futures.ThreadPoolExecutor(max_workers=settings.max_concurrent_requests) as executor:
                    futures_map = {executor.submit(run_judge, files_data[i]): i for i in range(len(files_data))}
                    done = 0
                    total = len(futures_map)
                    eval_slots = [None] * total
                    for fut in concurrent.futures.as_completed(futures_map):
                        i = futures_map[fut]
                        status_text.text(f"Evaluating {results[i]['filename']} ({done + 1}/{total})...")
                        eval_slots[i] = fut.result()
                        done += 1
                        progress_bar.progress(done / total)
                    evaluations = eval_slots

                judge_time = time.time() - judge_start

                # Update results with judge confidence and reasoning
                for i, (confidence, reasoning) in enumerate(evaluations):
                    if i < len(results):
                        results[i]["judge_confidence"] = confidence
                        results[i]["judge_reasoning"] = reasoning
                
                total_time = time.time() - start_batch
                status_text.text(f"Complete! Total: {total_time:.1f}s | Judge: {judge_time:.1f}s")
            
            # Show results
            if results:
                show_processing_results(results, valid_files)
                show_batch_download(results)
        
        elif uploaded_files and not valid_files:
            accepted_exts = get_supported_extensions()
            st.warning(
                "Please upload valid files. Accepted types: "
                + ", ".join(accepted_exts)
                + f". Max {settings.max_file_size_mb}MB per file."
            )

if __name__ == "__main__":
    main()
