import numpy as np
import joblib as jb
import pandas as pd

data_plz = pd.read_parquet(
    "https://github.com/Immobilienrechner-Challenge/data/blob/main/plz_data.parquet?raw=true"
)
scaler = jb.load("model1/utils/scaler_histgradientboostingregressor.joblib")
model = jb.load("model1/utils/model_histgradientboostingregressor.joblib")
cols = [
    "floor_space",
    "last_refurbishment",
    "living_space",
    "plot_area",
    "rooms",
    "year_built",
    "zip_code",
    "type_attic-flat",
    "type_attic-room",
    "type_castle",
    "type_chalet",
    "type_detached-house",
    "type_detached-secondary-suite",
    "type_duplex-maisonette",
    "type_farmhouse",
    "type_flat",
    "type_furnished-residential-property",
    "type_loft",
    "type_penthouse",
    "type_rustico",
    "type_secondary-suite",
    "type_semi-detached-house",
    "type_single-room",
    "type_stepped-apartment",
    "type_stepped-house",
    "type_studio",
    "type_terrace-house",
    "type_villa",
    "gde_average_house_hold",
    "NoisePollutionRoadS",
    "PopulationDensityM",
    "NoisePollutionRoadL",
    "gde_workers_sector1",
    "ForestDensityL",
    "ForestDensityM",
    "gde_tax",
    "gde_politics_sp",
    "gde_social_help_quota",
    "RiversAndLakesS",
    "NoisePollutionRoadM",
    "gde_private_apartments",
    "ForestDensityS",
    "distanceToTrainStation",
    "NoisePollutionRailwayM",
    "gde_workers_total",
    "gde_workers_sector3",
    "PopulationDensityL",
    "gde_area_forest_percentage",
    "gde_politics_pda",
    "gde_politics_bdp",
    "RiversAndLakesM",
    "RiversAndLakesL",
    "NoisePollutionRailwayL",
    "gde_politics_evp",
    "WorkplaceDensityM",
    "gde_area_nonproductive_percentage",
    "gde_foreigners_percentage",
    "gde_politics_glp",
    "gde_pop_per_km2",
    "gde_politics_cvp",
    "gde_population",
    "WorkplaceDensityL",
    "Longitude",
    "gde_area_settlement_percentage",
    "NoisePollutionRailwayS",
    "gde_new_homes_per_1000",
    "gde_empty_apartments",
    "WorkplaceDensityS",
    "gde_area_agriculture_percentage",
    "gde_politics_rights",
    "gde_politics_fdp",
    "PopulationDensityS",
    "gde_politics_gps",
    "gde_workers_sector2",
    "Latitude",
    "gde_politics_svp",
]

# Create your models here.
class PredictionHistGradientBoostingRegression:
    def __init__(
        self,
        living_space=None,
        floor_space=None,
        type=None,
        plot_area=None,
        rooms=None,
        zip_code=None,
        last_refurbishment=None,
        year_built=None,
    ):
        self.living_space = np.float(living_space)
        self.floor_space = np.float(floor_space)
        self.type = type
        self.plot_area = np.float(plot_area)
        self.rooms = np.float(rooms)
        self.zip_code = int(zip_code)
        self.last_refurbishment = np.float(last_refurbishment)
        self.year_built = np.float(year_built)

    def predict(self):
        df = pd.DataFrame(columns=cols, index=[0])

        if self.living_space:
            df["living_space"] = self.living_space
        if self.floor_space:
            df["floor_space"] = self.floor_space
        if self.plot_area:
            df["plot_area"] = self.plot_area
        if self.rooms:
            df["rooms"] = self.rooms
        if self.last_refurbishment:
            df["last_refurbishment"] = self.last_refurbishment
        if self.year_built:
            df["year_built"] = self.year_built

        if self.type:
            for col in df.columns:
                if "type_" in col:
                    df[col] = 0
            df[f"type_{self.type}"] = 1

        if self.zip_code:
            df["zip_code"] = self.zip_code
            df[data_plz.columns] = data_plz.loc[self.zip_code]

        df = pd.DataFrame(scaler.transform(df[cols]), columns=cols)

        prediction = model.predict(df[cols])[0]
        prediction = np.exp(prediction)
        prediction = np.round(prediction, -3)
        prediction = np.max([0, prediction])
        prediction = int(prediction)

        return prediction
