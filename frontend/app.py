import streamlit as st
from datetime import datetime
import json
import pandas as pd
import requests

# Backend service URL configured for the Docker bridge network
BACKEND_URL = "http://backend:7860"

st.set_page_config(
    page_title="SuperKart Sales Predictor", page_icon="🛒", layout="wide"
)

st.title("🛒 SuperKart Sales Prediction Dashboard")
st.markdown(
    "Predict product outlet sales using the containerized microservices"
    " architecture (Flask + XGBoost + Streamlit)."
)

# Navigation tabs for testing modalities
tabs = st.tabs(["Single Record Form", "Batch JSON Input", "CSV File Upload"])

with tabs[0]:
  st.header("Single Record Inference")
  st.markdown(
      "Enter product and store attributes below. Derived fields and UI-only"
      " inferences update dynamically in real-time."
  )

  col1, col2 = st.columns(2)

  with col1:
    product_id = st.text_input(
        "Product ID",
        value="FDW58",
        help=(
            "Prefix determines Product Category (FD: Food, DR: Drinks, NC:"
            " Non-Consumable)"
        ),
    )
    product_weight = st.number_input(
        "Product Weight", min_value=0.0, max_value=50.0, value=12.66, step=0.01
    )
    product_sugar_content = st.selectbox(
        "Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"]
    )
    product_allocated_area = st.number_input(
        "Product Allocated Area",
        min_value=0.0,
        max_value=1.0,
        value=0.027,
        step=0.001,
    )
    product_type = st.selectbox(
        "Product Type",
        [
            "Frozen Foods",
            "Dairy",
            "Canned",
            "Baking Goods",
            "Meat",
            "Fruits and Vegetables",
            "Snack Foods",
            "Starchy Foods",
            "Breads",
            "Hard Drinks",
            "Seafood",
            "Soft Drinks",
            "Others",
        ],
    )
    product_mrp = st.number_input(
        "Product MRP", min_value=0.0, max_value=300.0, value=117.08, step=0.01
    )

  with col2:
    store_establishment_year = st.number_input(
        "Store Establishment Year",
        min_value=1950,
        max_value=2026,
        value=2009,
        step=1,
    )
    store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
    store_location_city_type = st.selectbox(
        "Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"]
    )
    store_type = st.selectbox(
        "Store Type",
        [
            "Supermarket Type1",
            "Supermarket Type2",
            "Supermarket Type3",
            "Grocery Store",
        ],
    )

    # --- UI-ONLY DYNAMIC DERIVED & INFERRED FIELDS ---
    # 1. Product Category based on Product ID prefix (FD, DR, NC)
    pid_prefix = product_id.strip()[:2].upper()
    if pid_prefix == "FD":
      product_category = "Food"
    elif pid_prefix == "DR":
      product_category = "Drinks"
    elif pid_prefix == "NC":
      product_category = "Non-Consumable"
    else:
      product_category = "General / Other"

    # 2. Product Type Category (Perishable vs Non-Perishable)
    perishable_types = [
        "Fruits and Vegetables",
        "Meat",
        "Dairy",
        "Seafood",
        "Breads",
    ]
    product_type_category = (
        "Perishable" if product_type in perishable_types else "Non-Perishable"
    )

    # 3. Price Range based on Product MRP ('Budget', 'Mid-Range', 'High-End', 'Luxury')
    if product_mrp < 70:
      price_range = "Budget"
    elif product_mrp < 140:
      price_range = "Mid-Range"
    elif product_mrp < 210:
      price_range = "High-End"
    else:
      price_range = "Luxury"

    # 4. Store Age calculation
    current_year = 2026
    store_age = current_year - int(store_establishment_year)

    st.markdown("### 📊 UI-Inferred Attributes (Read-Only)")
    st.text_input(
        "Product Category (from ID)",
        value=product_category,
        disabled=True,
        key="derived_category",
    )
    st.text_input(
        "Product Type Category",
        value=product_type_category,
        disabled=True,
        key="derived_type_category",
    )
    st.text_input(
        "Price Range",
        value=price_range,
        disabled=True,
        key="derived_price_range",
    )
    st.text_input(
        "Store Age (Years)",
        value=f"{store_age} years",
        disabled=True,
        key="derived_store_age",
    )

  submit_button = st.button("Predict Sales", type="primary")

  if submit_button:
    payload = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": product_sugar_content,
        "Product_Allocated_Area": product_allocated_area,
        "Product_Type": product_type,
        "Product_MRP": product_mrp,
        "Store_Establishment_Year": store_establishment_year,
        "Store_Size": store_size,
        "Store_Location_City_Type": store_location_city_type,
        "Store_Type": store_type,
    }

    try:
      response = requests.post(f"{BACKEND_URL}/predict", json=payload)
      if response.status_code == 200:
        result = response.json()
        sales_prediction = result.get(
            "prediction", result.get("predicted_sales", 0.0)
        )
        st.success("Prediction Successful!")
        st.metric(
            label="Predicted Product Store Sales Total",
            value=f"${sales_prediction:,.2f}",
        )
      else:
        st.error(f"Backend Error ({response.status_code}): {response.text}")
    except Exception as e:
      st.error(f"Connection failed to backend service: {e}")

with tabs[1]:
  st.header("Batch JSON Inference")
  st.markdown("Paste a JSON array of multiple records matching the schema.")

  default_json = json.dumps(
      [{
          "Product_Weight": 12.66,
          "Product_Sugar_Content": "Low Sugar",
          "Product_Allocated_Area": 0.027,
          "Product_Type": "Frozen Foods",
          "Product_MRP": 117.08,
          "Store_Establishment_Year": 2009,
          "Store_Size": "Medium",
          "Store_Location_City_Type": "Tier 2",
          "Store_Type": "Supermarket Type2",
      }],
      indent=4,
  )

  json_input = st.text_area("Raw JSON Payload", value=default_json, height=250)

  if st.button("Run Batch JSON Prediction"):
    try:
      parsed_json = json.loads(json_input)
      response = requests.post(f"{BACKEND_URL}/predict", json=parsed_json)
      if response.status_code == 200:
        st.success("Batch Prediction Successful!")
        st.json(response.json())
      else:
        st.error(f"Backend Error ({response.status_code}): {response.text}")
    except json.JSONDecodeError as jde:
      st.error(f"Invalid JSON format: {jde}")
    except Exception as e:
      st.error(f"Connection failed: {e}")

with tabs[2]:
  st.header("CSV File Upload Inference")
  st.markdown("Upload `Batch_Data_SuperKart.csv` to perform bulk predictions.")

  uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

  if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    if st.button("Process CSV Predictions"):
      files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
      try:
        response = requests.post(f"{BACKEND_URL}/predict-file", files=files)
        if response.status_code == 200:
          st.success("File Prediction Successful!")
          result_data = response.json()
          if "predictions" in result_data:
            df["Predicted_Product_Store_Sales_Total"] = result_data[
                "predictions"
            ]
            st.dataframe(df.head(10))
            csv_download = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Predictions CSV",
                data=csv_download,
                file_name="predicted_superkart_results.csv",
                mime="text/csv",
            )
          else:
            st.json(result_data)
        else:
          st.error(f"Backend Error ({response.status_code}): {response.text}")
      except Exception as e:
        st.error(f"Connection failed: {e}")
