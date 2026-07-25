import streamlit as st
import requests
import pandas as pd
import json

st.set_page_config(
    page_title="SuperKart Prediction Dashboard",
    page_icon="🛒",
    layout="wide"
)

# Sidebar configuration for Backend API Connection
st.sidebar.header("🔌 API Configuration")
backend_url = "http://backend:7860"

# Main Title & Description
st.title("🛒 SuperKart Sales Prediction Dashboard")
st.markdown("Welcome to the interactive interface for the productionized **SuperKart XGBoost Model**. Inputs align strictly with `Batch_Data_SuperKart.csv` schema.")

# Check Backend Health Status
try:
    health_res = requests.get(f"{backend_url}/health", timeout=2)
    if health_res.status_code == 200:
        health_data = health_res.json()
        if health_data.get("model_loaded"):
            st.sidebar.success("🟢 Backend Connected & Model Loaded")
        else:
            st.sidebar.warning("🟡 Backend Connected, but Model not loaded.")
    else:
        st.sidebar.error("🔴 Backend returned an error status.")
except Exception as e:
    st.sidebar.error(f"🔴 Cannot reach backend at `{backend_url}`")

# Create Tabs for Dual Interface (/predict vs /predict-batch)
tab1, tab2 = st.tabs(["📊 Single Record Prediction", "📁 Batch File Prediction"])

with tab1:
    st.header("Single Record Inference (`/predict`)")
    st.markdown("Provide feature values matching `Batch_Data_SuperKart.csv` schema or raw JSON payload.")

    input_method = st.radio("Input Method:", ["Interactive Form", "Raw JSON Payload"], horizontal=True)

    if input_method == "Interactive Form":
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                product_weight = st.number_input("Product_Weight", value=12.35)
                product_allocated_area = st.number_input("Product_Allocated_Area", value=0.05)
                product_mrp = st.number_input("Product_MRP", value=141.61)
                store_age_years = st.number_input("Store_Age_Years", value=15, step=1)

            with col2:
                product_sugar_content = st.selectbox("Product_Sugar_Content", ["Regular","Low Sugar", "No Sugar"])
                product_id_char = st.selectbox("Product_Id_char", ["Food", "Drinks", "Non-Consumable"])
                product_type_category = st.selectbox("Product_Type_Category", ["Perishable", "Non-Perishable"])

            with col3:
                store_size = st.selectbox("Store_Size", ["Medium", "High", "Small"])
                store_location_city_type = st.selectbox("Store_Location_City_Type", ["Tier 1", "Tier 2", "Tier 3"])
                store_type = st.selectbox("Store_Type", ["Departmental Store","Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Food Mart"])

            submit_btn = st.form_submit_button("Generate Prediction")

            if submit_btn:
                payload = {
                    "Product_Weight": product_weight,
                    "Product_Sugar_Content": product_sugar_content,
                    "Product_Allocated_Area": product_allocated_area,
                    "Product_MRP": product_mrp,
                    "Store_Size": store_size,
                    "Store_Location_City_Type": store_location_city_type,
                    "Store_Type": store_type,
                    "Product_Id_char": product_id_char,
                    "Store_Age_Years": store_age_years,
                    "Product_Type_Category": product_type_category
                }

                try:
                    res = requests.post(f"{backend_url}/predict", json=payload)
                    if res.status_code == 200:
                        result = res.json()
                        pred_value = result['predictions'][0]
                        st.success(f"### Predicted Sales: **${pred_value:,.2f}**")
                    else:
                        st.error(f"Error from API: {res.text}")
                except Exception as err:
                    st.error(f"Connection failed: {err}")

    else:
        default_json = json.dumps({
            "Product_Weight": 12.35,
            "Product_Sugar_Content": "Low Sugar",
            "Product_Allocated_Area": 0.05,
            "Product_MRP": 141.61,
            "Store_Size": "Medium",
            "Store_Location_City_Type": "Tier 1",
            "Store_Type": "Supermarket Type1",
            "Product_Id_char": "Food",
            "Store_Age_Years": 15,
            "Product_Type_Category": "Perishable"
        }, indent=4)

        raw_json_input = st.text_area("Enter JSON payload matching schema:", value=default_json, height=220)

        if st.button("Submit JSON Payload"):
            try:
                parsed_json = json.loads(raw_json_input)
                res = requests.post(f"{backend_url}/predict", json=parsed_json)
                if res.status_code == 200:
                    result = res.json()
                    st.success("Predictions Generated Successfully!")
                    st.json(result)
                else:
                    st.error(f"Error from API: {res.text}")
            except json.JSONDecodeError:
                st.error("Invalid JSON format. Please check your syntax.")
            except Exception as err:
                st.error(f"Connection failed: {err}")

with tab2:
    st.header("Batch File Prediction (`/predict-batch`)")
    st.markdown("Upload `Batch_Data_SuperKart.csv` or a matching CSV file to run bulk inference through the backend API.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df_preview = pd.read_csv(uploaded_file)
        st.subheader("Data Preview:")
        st.dataframe(df_preview.head(), use_container_width=True)

        if st.button("Run Batch Prediction"):
            try:
                uploaded_file.seek(0)
                files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')}

                res = requests.post(f"{backend_url}/predict-batch", files=files)

                if res.status_code == 200:
                    batch_result = res.json()
                    st.success(f"Successfully processed {batch_result.get('record_count')} records from `{batch_result.get('filename')}`!")

                    df_preview['Predicted_Sales'] = batch_result.get('predictions')
                    st.subheader("Prediction Results:")
                    st.dataframe(df_preview, use_container_width=True)

                    csv_download = df_preview.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions CSV",
                        data=csv_download,
                        file_name="superkart_predictions_output.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"Batch inference error: {res.text}")
            except Exception as err:
                st.error(f"Batch request failed: {err}")
