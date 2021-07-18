import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from ml_utils import load_model, predict, retrain
from typing import List
from datetime import date, datetime

app = FastAPI(title="Bug Predictor",docs_url="/")

app.add_event_handler("startup", load_model)

class QueryIn(BaseModel):
    mccabe_line_count_of_code: float
    mccabe_cyclomatic_complexity: float
    mccabe_essential_complexity: float
    mccabe_design_complexity: float
    halstead_total_operators_operands: float
    halstead_volume: float
    halstead_program_length: float
    halstead_difficulty: float
    halstead_intelligence: float
    halstead_effort: float
    halstead_b: float
    halstead_time_estimator: float
    halstead_line_count: float
    halstead_count_of_lines_of_comments: float
    halstead_count_of_blank_lines: float
    lOCodeAndComment: float
    unique_operators: float
    unique_operands: float
    total_operators: float
    total_operands: float
    branchCount: float
    

class QueryOut(BaseModel):
    defects: str
    timestamp : datetime

class FeedbackIn(BaseModel):
    mccabe_line_count_of_code: float
    mccabe_cyclomatic_complexity: float
    mccabe_essential_complexity: float
    mccabe_design_complexity: float
    halstead_total_operators_operands: float
    halstead_volume: float
    halstead_program_length: float
    halstead_difficulty: float
    halstead_intelligence: float
    halstead_effort: float
    halstead_b: float
    halstead_time_estimator: float
    halstead_line_count: float
    halstead_count_of_lines_of_comments: float
    halstead_count_of_blank_lines: float
    lOCodeAndComment: float
    unique_operators: float
    unique_operands: float
    total_operators: float
    total_operands: float
    branchCount: float
    defects: bool



@app.get("/ping")
def ping():
    return {"ping": "pong"}


@app.post("/predict", response_model=QueryOut, status_code=200)
def predict_bug(query_data: QueryIn):
    output = {'defects': predict(query_data),'timestamp': datetime.now()}
    return output

@app.post("/feedback",status_code=200)
def feedback(data: List[FeedbackIn]):
    retrain(data)
    return{"detail":"Feedback successful retrained the model"}

if __name__ == "__main__":
    uvicorn.run("main:app", host='127.0.0.1', port=8888, reload=True, debug=True)
