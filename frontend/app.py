from datetime import datetime
import json
import pandas as pd
import requests
import streamlit as st

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
      "Enter product and store attributes below. Derived fields dynamically "
      "update in the designated read-only section based on your inputs."
  )

  # 3-column layout to distribute input fields evenly
  col1, col2, col3 = st.columns(3)

  with col1:
    st.subheader("Product Basics")
    pid_prefix = st.selectbox(
        "Product ID Prefix",
        options=["FD", "DR", "NC"],
        format_func=lambda x: {
            "FD": "FD (Food)",
            "DR": "DR (Drinks)",
            "NC": "NC (Non-Consumable)",
        }[x],
        help="Prefix determines Product Category",
    )
    product_weight = st.number_input(
        "Product Weight", min_value=0.0, max_value=50.0, value=15.30, step=0.01
    )
    product_sugar_content = st.selectbox(
        "Product Sugar Content", ["Regular", "Low Sugar", "No Sugar"]
    )

  with col2:
    st.subheader("Product Details")
    product_type = st.selectbox(
        "Product Type",
        [
            "Soft Drinks",
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
            "Others",
        ],
    )
    # Unrounded Product Allocated Area (retaining precision)
    product_allocated_area = st.number_input(
        "Product Allocated Area",
        min_value=0.0,
        max_value=1.0,
        value=0.054321,
        step=0.000001,
        format="%.6f",
    )
    product_mrp = st.number_input(
        "Product MRP", min_value=0.0, max_value=300.0, value=185.50, step=0.01
    )

  with col3:
    st.subheader("Store Attributes")
    # Store Type aligned exactly to SuperKart.csv distinct options
    store_type = st.selectbox(
        "Store Type",
        [
            "Departmental Store",
            "Supermarket Type 1",
            "Supermarket Type 2",
            "Supermarket Type 3",
            "Food Mart",
        ],
    )
    store_establishment_year = st.number_input(
        "Store Establishment Year",
        min_value=1950,
        max_value=2026,
        value=1999,
        step=1,
    )
    store_size = st.selectbox("Store Size", ["High", "Medium", "Small"])
    store_location_city_type = st.selectbox(
        "Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"]
    )

  st.divider()

  # --- SEPARATE SECTION: UI-ONLY DYNAMIC DERIVED & INFERRED FIELDS ---
  st.subheader("📊 UI-Inferred Attributes (Read-Only)")
  st.markdown("These fields are calculated automatically from your inputs above.")
  
  # Business Logic Calculations
  if pid_prefix == "FD":
    product_category = "Food"
  elif pid_prefix == "DR":
    product_category = "Drinks"
  else:
    product_category = "Non-Consumable"

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

  if product_mrp < 70:
    price_range = "Budget"
  elif product_mrp < 140:
    price_range = "Mid-Range"
  elif product_mrp < 210:
    price_range = "High-End"
  else:
    price_range = "Luxury"

  current_year = 2026
  store_age = current_year - int(store_establishment_year)

  # Display read-only fields cleanly in a 4-column row
  ro_col1, ro_col2, ro_col3, ro_col4 = st.columns(4)

  with ro_col1:
    st.text_input(
        "Product Category",
        value=product_category,
        disabled=True,
        key="derived_category",
    )
  with ro_col2:
    st.text_input(
        "Type Category",
        value=product_type_category,
        disabled=True,
        key="derived_type_category",
    )
  with ro_col3:
    st.text_input(
        "Price Range",
        value=price_range,
        disabled=True,
        key="derived_price_range",
    )
  with ro_col4:
    st.text_input(
        "Store Age",
        value=f"{store_age} years",
        disabled=True,
        key="derived_store_age",
    )

  st.write("") # small spacing
  submit_button = st.button("Predict Sales", type="primary", use_container_width=True)

  if submit_button:
    # Wrap in list since backend processes arrays of records
    payload = [{
        "Product_Weight": product_weight,
        "Product_Sugar_Content": product_sugar_content,
        "Product_Allocated_Area": product_allocated_area,
        "Product_Type": product_type,
        "Product_MRP": product_mrp,
        "Store_Establishment_Year": store_establishment_year,
        "Store_Size": store_size,
        "Store_Location_City_Type": store_location_city_type,
        "Store_Type": store_type,
    }]

    try:
      response = requests.post(f"{BACKEND_URL}/predict", json=payload)
      if response.status_code == 200:
        result = response.json()
        if result.get("status") == "success":
            # Predictions are returned as a list; extract the first one for the single record
            predictions = result.get("predictions", [0.0])
            sales_prediction = predictions[0] if predictions else 0.0
            
            st.success("Prediction Successful!")
            st.metric(
                label="Predicted Product Store Sales Total",
                value=f"${sales_prediction:,.2f}",
            )
        else:
            st.error("Backend successfully responded, but reported a failure status.")
            st.json(result)
      else:
        st.error(f"Backend Error ({response.status_code}): {response.text}")
    except Exception as e:
      st.error(f"Connection failed to backend service: {e}")

with tabs[1]:
  st.header("Batch JSON Inference")
  st.markdown("Paste a JSON array of multiple records matching the schema.")

  default_json = json.dumps(
      [
          {
              "Product_Weight": 15.30,
              "Product_Sugar_Content": "Regular",
              "Product_Allocated_Area": 0.054321,
              "Product_Type": "Soft Drinks",
              "Product_MRP": 185.50,
              "Store_Establishment_Year": 1999,
              "Store_Size": "High",
              "Store_Location_City_Type": "Tier 1",
              "Store_Type": "Supermarket Type 1",
          },
          {
              "Product_Weight": 8.21,
              "Product_Sugar_Content": "No Sugar",
              "Product_Allocated_Area": 0.012456,
              "Product_Type": "Snack Foods",
              "Product_MRP": 56.40,
              "Store_Establishment_Year": 2007,
              "Store_Size": "Small",
              "Store_Location_City_Type": "Tier 3",
              "Store_Type": "Departmental Store",
          },
      ],
      indent=4,
  )

  json_input = st.text_area("Raw JSON Payload", value=default_json, height=250)

  if st.button("Run Batch JSON Prediction"):
    try:
      parsed_json = json.loads(json_input)
      response = requests.post(f"{BACKEND_URL}/predict", json=parsed_json)
      if response.status_code == 200:
        result = response.json()
        if result.get("status") == "success":
            record_count = result.get("record_count", 0)
            st.success(f"Batch Prediction Successful! Processed {record_count} records.")
            
            # Combine the parsed input JSON with the returned predictions for a clean display
            df_results = pd.DataFrame(parsed_json)
            df_results["Predicted_Sales_Total"] = result.get("predictions", [])
            
            st.dataframe(df_results)
            with st.expander("View Raw Backend Response"):
                st.json(result)
        else:
             st.error("Backend successfully responded, but reported a failure status.")
             st.json(result)
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
          result_data = response.json()
          
          if result_data.get("status") == "success" and "predictions" in result_data:
            filename = result_data.get("filename", uploaded_file.name)
            record_count = result_data.get("record_count", 0)
            
            st.success(f"File '{filename}' Prediction Successful! Processed {record_count} records.")
            
            df["Predicted_Product_Store_Sales_Total"] = result_data["predictions"]
            st.dataframe(df.head(10))
            
            csv_download = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Predictions CSV",
                data=csv_download,
                file_name=f"predicted_{filename}",
                mime="text/csv",
            )
          else:
            st.error("The API response did not indicate success or was missing predictions.")
            st.json(result_data)
        else:
          st.error(f"Backend Error ({response.status_code}): {response.text}")
      except Exception as e:
        st.error(f"Connection failed: {e}")
