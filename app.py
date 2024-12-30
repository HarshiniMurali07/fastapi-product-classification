from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import gzip
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
    allow_origins=["https://harshinimurali07.github.io"],  # Allow all origins for testing purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the compressed pipeline
try:
    print("Decompressing and loading pipeline...")
    with gzip.open('product_classification_pipeline.pkl.gz', 'rb') as f:
        pipeline = joblib.load(f)
    print("Pipeline loaded successfully!")
except FileNotFoundError:
    print("Error: Compressed pipeline file not found.")
    raise HTTPException(status_code=500, detail="Compressed pipeline file not found.")
except Exception as e:
    print(f"Error loading pipeline: {str(e)}")
    raise HTTPException(status_code=500, detail="Pipeline loading failed.")

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

        # Map input fields to match pipeline's expected column names
        input_data = pd.DataFrame([{
            's:name': product.s_name,
            's:description': product.s_description,
            's:breadcrumb': product.s_breadcrumb,
            's:brand': product.s_brand,
        }])
        print(f"Converted input to DataFrame: {input_data}")

        # Make predictions
        predictions = pipeline.predict(input_data)
        print(f"Predictions: {predictions}")

        # Prepare response
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

