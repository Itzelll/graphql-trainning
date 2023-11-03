from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]

app = FastAPI(title = 'Breast Cancer Prediction')

app.add_middleware(
   CORSMiddleware,
   allow_origins=origins,
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"]
)

model = load(pathlib.Path('model/breast_cancer_data-v1.joblib'))

class InputData(BaseModel):
    mean_radius:float=13.54
    mean_texture:float=14.36
    mean_perimeter:float=87.46
    mean_area:float=566.3
    mean_smoothness:float=0.09779


class OutputData(BaseModel):
    diagnosis:float=1

@app.post('/diagnosis', response_model = OutputData)
def diagnosis(data:InputData):
    model_input = np.array([v for k,v in data.dict().items()]).reshape(1,-1)
    result = model.predict_proba(model_input)[:,-1]

    return {'diagnosis':result}
