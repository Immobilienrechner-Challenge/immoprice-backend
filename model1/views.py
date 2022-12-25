import json
import numpy as np

import model1.models as models
from django.http import HttpResponse


def index(request):
    try:
        model = models.PredictionHistGradientBoostingRegression(
            living_space=(
                request.GET["living_space"] if "living_space" in request.GET else np.nan
            ),
            floor_space=(
                request.GET["floor_space"] if "floor_space" in request.GET else np.nan
            ),
            type=(request.GET["type"] if "type" in request.GET else np.nan),
            plot_area=(
                request.GET["plot_area"] if "plot_area" in request.GET else np.nan
            ),
            rooms=(request.GET["rooms"] if "rooms" in request.GET else np.nan),
            zip_code=(request.GET["zip_code"] if "zip_code" in request.GET else np.nan),
            last_refurbishment=(
                request.GET["last_refurbishment"]
                if "last_refurbishment" in request.GET
                else np.nan
            ),
            year_built=(
                request.GET["year_built"] if "year_built" in request.GET else np.nan
            ),
        )

        prediction = model.predict()

        return HttpResponse(json.dumps({"prediction": prediction}))

    except Exception as e:
        return HttpResponse("Error occured, please check your input.", status=500)


def test(request, testnum):
    return HttpResponse(f"Model 1 test. Value: {testnum}")
