
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load your machine learning pipeline
print("Loading pipeline...")
pipeline = joblib.load('product_classification_pipeline.pkl')
print("Pipeline loaded successfully!")

# Define input schema using Pydantic
class ProductInput(BaseModel):
    s_name: str
    s_description: str
    s_breadcrumb: str
    s_brand: str

# Define root endpoint
@app.get("/")
def read_root():
    print("Root endpoint accessed.")
    return {"message": "Welcome to the Product Classification API!"}

# Define prediction endpoint
@app.post("/predict")
async def predict(product: ProductInput):
    try:
        # Log input data
        print(f"Received input data: {product.dict()}")

        # Convert input to DataFrame
        input_data = pd.DataFrame([product.dict()])
        print(f"Converted input to DataFrame: {input_data}")

        # Make prediction
        predictions = pipeline.predict(input_data)
        print(f"Predictions: {predictions}")

        # Prepare response
        response = {
            "Level 1": predictions[0][0],
            "Level 2": predictions[0][1],
            "Level 3": predictions[0][2]
        }
        print(f"Response prepared: {response}")
        return response
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
