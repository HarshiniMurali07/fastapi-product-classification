
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load your machine learning pipeline
pipeline = joblib.load('product_classification_pipeline.pkl')

# Define input schema using Pydantic
class ProductInput(BaseModel):
    s_name: str
    s_description: str
    s_breadcrumb: str
    s_brand: str

# Define API endpoint
@app.post("/predict")
async def predict(product: ProductInput):
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([product.dict()])
        
        # Make prediction
        predictions = pipeline.predict(input_data)
        
        # Prepare response
        response = {
            "Level 1": predictions[0][0],
            "Level 2": predictions[0][1],
            "Level 3": predictions[0][2]
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
