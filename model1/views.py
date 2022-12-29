import json
import numpy as np

import model1.models as models
from django.http import HttpResponse


def index(request):
    # try to get prediction
    try:
        # create model
        model = models.PredictionHistGradientBoostingRegression(
            # get living_space from request if it exists, else set to np.nan
            living_space=(
                request.GET["living_space"] if "living_space" in request.GET else np.nan
            ),
            # get floor_space from request if it exists, else set to np.nan
            floor_space=(
                request.GET["floor_space"] if "floor_space" in request.GET else np.nan
            ),
            # get type from request if it exists, else set to np.nan
            type=(request.GET["type"] if "type" in request.GET else np.nan),
            # get plot_area from request if it exists, else set to np.nan
            plot_area=(
                request.GET["plot_area"] if "plot_area" in request.GET else np.nan
            ),
            # get rooms from request if it exists, else set to np.nan
            rooms=(request.GET["rooms"] if "rooms" in request.GET else np.nan),
            # get zip_code from request if it exists, else set to np.nan
            zip_code=(request.GET["zip_code"] if "zip_code" in request.GET else np.nan),
            # get last_refurbishment from request if it exists, else set to np.nan
            last_refurbishment=(
                request.GET["last_refurbishment"]
                if "last_refurbishment" in request.GET
                else np.nan
            ),
            # get year_built from request if it exists, else set to np.nan
            year_built=(
                request.GET["year_built"] if "year_built" in request.GET else np.nan
            ),
        )

        # get prediction
        prediction = model.predict()

        # return prediction as json encoded string
        return HttpResponse(json.dumps({"prediction": prediction}))

    # if an error occurs, return error message and status code 500
    except Exception as e:
        return HttpResponse("Error occured, please check your input.", status=500)
