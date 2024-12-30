
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize FastAPI app
app = FastAPI(
    title="Product Classification API",
    description="An API for classifying products into hierarchical categories using a machine learning model.",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI path
    redoc_url="/redoc" # ReDoc documentation path
    )

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the machine learning pipeline
print("Loading pipeline...")
try:
    pipeline = joblib.load('product_classification_pipeline.pkl')
    print("Pipeline loaded successfully!")
except FileNotFoundError:
    print("Error: 'product_classification_pipeline.pkl' not found.")
    raise HTTPException(status_code=500, detail="Pipeline file not found.")

# Define input schema using Pydantic
class ProductInput(BaseModel):
    s_name: str
    s_description: str
    s_breadcrumb: str
    s_brand: str

# Root endpoint
@app.get("/")
def read_root():
    print("Root endpoint accessed.")
    return {"message": "Welcome to the Product Classification API!"}

# Prediction endpoint
@app.post("/predict")
async def predict(product: ProductInput):
    try:
        print(f"Received input: {product.dict()}")
        input_data = pd.DataFrame([product.dict()])
        print(f"Converted input to DataFrame: {input_data}")
        predictions = pipeline.predict(input_data)
        print(f"Predictions: {predictions}")
        response = {
            "Level 1": predictions[0][0],
            "Level 2": predictions[0][1],
            "Level 3": predictions[0][2],
        }
        print(f"Response prepared: {response}")
        return response
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

print("app.py has been successfully created and is ready to run!")
