import json

from joblib import load
from helper import ImmoHelper
from django.http import HttpResponse

helper = ImmoHelper()
model = load('models/simple_model.joblib')

def index(request):
    try:
        data = helper.process_prediction(
            request.GET["living_space"],
            request.GET["type"],
            request.GET["rooms"],
            request.GET["gde_tax"]
        )
        return HttpResponse(json.dumps({
            "prediction": model.predict(data)[0]
        }))
    except Exception as e:
        return HttpResponse("Error occured, please check your input.", status = 500)


def test(request, testnum):
    return HttpResponse(f"Model 1 test. Value: {testnum}")
